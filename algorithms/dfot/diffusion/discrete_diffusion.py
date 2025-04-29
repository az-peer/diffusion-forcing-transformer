from typing import Optional, Callable, Literal
from collections import namedtuple
from omegaconf import DictConfig
import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange, reduce
from ..backbones import (
    Unet3D,
    DiT3D,
    DiT3DPose,
    UViT3D,
    UViT3DPose,
)
from .noise_schedule import make_beta_schedule
# from .clip_embeddings import get_clip_embeddings
import clip
from PIL import Image

clip_model, preprocess = clip.load("ViT-B/32", device="cuda")


def get_clip_embeddings(text: str):
    # Encode the text using CLIP
    text_input = clip.tokenize([text]).to("cuda")  # Tokenize text and move to device
    with torch.no_grad():
        text_features = clip_model.encode_text(
            text_input
        )  # Get the text features (embedding)
    return text_features / text_features.norm(
        dim=-1, keepdim=True
    )  # Normalize embedding


guided_text = "Produce a video that ends up with some sign of you near a bathroom.You can be creative with what a bathroom is but you must end up in a bathroom."

# grab the embeddings
clip_embeddings = get_clip_embeddings(guided_text)


def extract(a, t, x_shape):
    # a is a just a one d tensor maybe like precomputed coefficients
    # t is maybe a tensor for timesteps
    # x_shape
    shape = t.shape
    # use the t to index the coefficient by t
    out = a[t]
    # reshape out to match x but only in batch dimensions
    # match batch size and fill int the rest with 1s
    # makes this broadcastable
    return out.reshape(*shape, *((1,) * (len(x_shape) - len(shape))))


# when we call modelprediction we wil get obj.pred_noise will return the
# first thing that we make
# we may choose a different one depending on what our model architecture actually
# returns
ModelPrediction = namedtuple(
    "ModelPrediction", ["pred_noise", "pred_x_start", "model_out"]
)


class DiscreteDiffusion(nn.Module):
    def __init__(
        self,
        cfg: DictConfig,
        backbone_cfg: DictConfig,
        x_shape: torch.Size,
        max_tokens: int,
        external_cond_dim: int,
    ):
        super().__init__()
        self.cfg = cfg
        self.x_shape = x_shape
        self.max_tokens = max_tokens
        # if we are working with any embeddings
        # this could be the size of the actual embeddings
        self.external_cond_dim = external_cond_dim
        # we grab the timesteps from the config
        self.timesteps = cfg.timesteps
        # we grab the sampling timesteps
        self.sampling_timesteps = cfg.sampling_timesteps
        # we get the beta_scdule
        self.beta_schedule = cfg.beta_schedule

        self.schedule_fn_kwargs = cfg.schedule_fn_kwargs
        # which objective to use
        self.objective = cfg.objective
        self.loss_weighting = cfg.loss_weighting
        self.ddim_sampling_eta = cfg.ddim_sampling_eta
        self.clip_noise = cfg.clip_noise

        self.backbone_cfg = backbone_cfg
        self.use_causal_mask = cfg.use_causal_mask
        self._build_model()
        self._build_buffer()

    def _build_model(self):
        # match the model to the actual architecture
        match self.backbone_cfg.name:
            case "u_net3d":
                model_cls = Unet3D
            case "u_vit3d":
                model_cls = UViT3D
            case "u_vit3d_pose":
                model_cls = UViT3DPose
            case "dit3d":
                model_cls = DiT3D
            case "dit3d_pose":
                model_cls = DiT3DPose
            case _:
                raise ValueError(f"unknown model type {self.model_type}")
        # it uses the match to the model and then initiate the params for the actual model
        self.model = model_cls(
            cfg=self.backbone_cfg,
            x_shape=self.x_shape,
            max_tokens=self.max_tokens,
            external_cond_dim=self.external_cond_dim,
            use_causal_mask=self.use_causal_mask,
        )

    # this is what creates the actual noise schedule
    def _build_buffer(self):
        betas = make_beta_schedule(
            schedule=self.beta_schedule,
            timesteps=self.timesteps,
            # if we are predicting pred noise then we do something
            zero_terminal_snr=self.objective != "pred_noise",
            **self.schedule_fn_kwargs,
        )
        # we make the alphas
        alphas = 1.0 - betas
        # alpha ba r
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        # then we get alpha_bar-1 this is just something we need in diffusion
        # models
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        # sampling related parameters
        # we need the sampling timesteps to be less then the training
        assert self.sampling_timesteps <= self.timesteps
        # if sampling is strictly less then training, we run ddim
        self.is_ddim_sampling = self.sampling_timesteps < self.timesteps

        # helper function to register buffer from float64 to float32
        # helps prevent cluttered code for later
        register_buffer = lambda name, val: self.register_buffer(
            name, val.to(torch.float32), persistent=False
        )

        # we register the betas
        register_buffer("betas", betas)
        # the a_bars
        register_buffer("alphas_cumprod", alphas_cumprod)
        # the a_bars-1
        register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        """
            Basically buffers are things that we want to keep track of when we 
            have a model like when we move the model from cpu to gpu. And then we 
            want to actually move them as well but we do not want them to be 
            tracked with optimization. We use something like this. 
        """

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )
        register_buffer("log_one_minus_alphas_cumprod", torch.log(1.0 - alphas_cumprod))
        # if (
        #     self.objective == "pred_noise"
        #     or self.cfg.reconstruction_guidance is not None
        # ):
        register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        register_buffer(
            "sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1)
        )

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer("posterior_variance", posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer(
            "posterior_log_variance_clipped",
            torch.log(posterior_variance.clamp(min=1e-20)),
        )
        register_buffer(
            "posterior_mean_coef1",
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod),
        )

        # snr: signal noise ratio
        snr = alphas_cumprod / (1 - alphas_cumprod)
        register_buffer("snr", snr)
        if self.loss_weighting.strategy in {"min_snr", "fused_min_snr"}:
            clipped_snr = snr.clone()
            clipped_snr.clamp_(max=self.loss_weighting.snr_clip)
            register_buffer("clipped_snr", clipped_snr)
        elif self.loss_weighting.strategy == "sigmoid":
            register_buffer("logsnr", torch.log(snr))

    # this reshapes the tensor to match the actual shape of the videos
    def add_shape_channels(self, x):
        return rearrange(x, f"... -> ...{' 1' * len(self.x_shape)}")

    def model_predictions(self, x, k, external_cond=None, external_cond_mask=None):
        """
        Grabs the models predictions. Here this is the actual model
        from the configuration file.

        Params:
            x: The current noisy sample
            k: the current time step
            external_cond: Optional condition --> MAYBE add CLIP
        """
        # actually creates an instance of the model output
        model_output = self.model(x, k, external_cond, external_cond_mask)

        # if the model predicts how much noise is added to the clean image
        # then we clip it
        if self.objective == "pred_noise":
            pred_noise = torch.clamp(model_output, -self.clip_noise, self.clip_noise)
            # then we predict the clean image the current noisy sample
            # the time step
            # and the predicted nosie
            x_start = self.predict_start_from_noise(x, k, pred_noise)

        # then if the model we are using does pred_x0
        elif self.objective == "pred_x0":
            # we grab the clean image
            x_start = model_output
            # we grab the predicted nosie
            pred_noise = self.predict_noise_from_start(x, k, x_start)

        # if we are doing a flow map or the velocity field
        elif self.objective == "pred_v":
            # grab the velocity
            v = model_output
            # predict the clean image from the velocity
            x_start = self.predict_start_from_v(x, k, v)
            # and then get the noise
            pred_noise = self.predict_noise_from_v(x, k, v)

        # use the predicted noise, the clean image, and the model output
        # to get the model prediction
        # uses the named tuple convention to this
        model_pred = ModelPrediction(pred_noise, x_start, model_output)

        return model_pred

    # gets the clean image from the actual nosie
    def predict_start_from_noise(self, x_k, k, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, k, x_k.shape) * x_k
            - extract(self.sqrt_recipm1_alphas_cumprod, k, x_k.shape) * noise
        )

    # predcit the estimated noise from the actual clean image
    def predict_noise_from_start(self, x_k, k, x0):
        # return (
        #     extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0
        # ) / extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        return (x_k - extract(self.sqrt_alphas_cumprod, k, x_k.shape) * x0) / extract(
            self.sqrt_one_minus_alphas_cumprod, k, x_k.shape
        )

    # get v from start and the nosie
    def predict_v(self, x_start, k, noise):
        return (
            extract(self.sqrt_alphas_cumprod, k, x_start.shape) * noise
            - extract(self.sqrt_one_minus_alphas_cumprod, k, x_start.shape) * x_start
        )

    # get clean from the v
    def predict_start_from_v(self, x_k, k, v):
        return (
            extract(self.sqrt_alphas_cumprod, k, x_k.shape) * x_k
            - extract(self.sqrt_one_minus_alphas_cumprod, k, x_k.shape) * v
        )

    # get noise from the v
    def predict_noise_from_v(self, x_k, k, v):
        return (
            extract(self.sqrt_alphas_cumprod, k, x_k.shape) * v
            + extract(self.sqrt_one_minus_alphas_cumprod, k, x_k.shape) * x_k
        )

    # this is computing the q(x_t | x_0)
    # this is for the forward diffusion process
    def q_mean_variance(self, x_start, k):
        mean = extract(self.sqrt_alphas_cumprod, k, x_start.shape) * x_start
        variance = extract(1.0 - self.alphas_cumprod, k, x_start.shape)
        log_variance = extract(self.log_one_minus_alphas_cumprod, k, x_start.shape)
        return mean, variance, log_variance

    # this is estimating q(x_{t-1} | x_0, x_t)
    def q_posterior(self, x_start, x_k, k):
        posterior_mean = (
            extract(self.posterior_mean_coef1, k, x_k.shape) * x_start
            + extract(self.posterior_mean_coef2, k, x_k.shape) * x_k
        )
        posterior_variance = extract(self.posterior_variance, k, x_k.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, k, x_k.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    # this generates a noisy sample x_k from a schedule
    def q_sample(self, x_start, k, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
            noise = torch.clamp(noise, -self.clip_noise, self.clip_noise)

        return (
            extract(self.sqrt_alphas_cumprod, k, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, k, x_start.shape) * noise
        )

    # this is using the models predictions to get the posterioir distribution
    # while the one above is not
    def p_mean_variance(self, x, k, external_cond=None, external_cond_mask=None):
        model_pred = self.model_predictions(
            x=x, k=k, external_cond=external_cond, external_cond_mask=external_cond_mask
        )
        x_start = model_pred.pred_x_start
        return self.q_posterior(x_start=x_start, x_k=x, k=k)

    # this is how we should define how we weigh the importance of each time step
    def compute_loss_weights(
        self,
        k: torch.Tensor,
        strategy: Literal["min_snr", "fused_min_snr", "uniform", "sigmoid"],
    ) -> torch.Tensor:
        if strategy == "uniform":
            return torch.ones_like(k)
        snr = self.snr[k]
        epsilon_weighting = None
        match strategy:
            case "sigmoid":
                logsnr = self.logsnr[k]
                # sigmoid reweighting proposed by https://arxiv.org/abs/2303.00848
                # and adopted by https://arxiv.org/abs/2410.19324
                epsilon_weighting = torch.sigmoid(
                    self.cfg.loss_weighting.sigmoid_bias - logsnr
                )
            case "min_snr":
                # min-SNR reweighting proposed by https://arxiv.org/abs/2303.09556
                clipped_snr = self.clipped_snr[k]
                epsilon_weighting = clipped_snr / snr.clamp(min=1e-8)  # avoid NaN
            case "fused_min_snr":
                # fused min-SNR reweighting proposed by Diffusion Forcing v1
                # with an additional support for bi-directional Fused min-SNR for non-causal models
                snr_clip, cum_snr_decay = (
                    self.loss_weighting.snr_clip,
                    self.loss_weighting.cum_snr_decay,
                )
                clipped_snr = self.clipped_snr[k]
                normalized_clipped_snr = clipped_snr / snr_clip
                normalized_snr = snr / snr_clip

                def compute_cum_snr(reverse: bool = False):
                    new_normalized_clipped_snr = (
                        normalized_clipped_snr.flip(1)
                        if reverse
                        else normalized_clipped_snr
                    )
                    cum_snr = torch.zeros_like(new_normalized_clipped_snr)
                    for t in range(0, k.shape[1]):
                        if t == 0:
                            cum_snr[:, t] = new_normalized_clipped_snr[:, t]
                        else:
                            cum_snr[:, t] = (
                                cum_snr_decay * cum_snr[:, t - 1]
                                + (1 - cum_snr_decay) * new_normalized_clipped_snr[:, t]
                            )
                    cum_snr = F.pad(cum_snr[:, :-1], (1, 0, 0, 0), value=0.0)
                    return cum_snr.flip(1) if reverse else cum_snr

                if self.use_causal_mask:
                    cum_snr = compute_cum_snr()
                else:
                    # bi-directional cum_snr when not using causal mask
                    cum_snr = compute_cum_snr(reverse=True) + compute_cum_snr()
                    cum_snr *= 0.5
                clipped_fused_snr = 1 - (1 - cum_snr * cum_snr_decay) * (
                    1 - normalized_clipped_snr
                )
                fused_snr = 1 - (1 - cum_snr * cum_snr_decay) * (1 - normalized_snr)
                clipped_snr = clipped_fused_snr * snr_clip
                snr = fused_snr * snr_clip
                epsilon_weighting = clipped_snr / snr.clamp(min=1e-8)  # avoid NaN
            case _:
                raise ValueError(f"unknown loss weighting strategy {strategy}")

        match self.objective:
            case "pred_noise":
                return epsilon_weighting
            case "pred_x0":
                return epsilon_weighting * snr
            case "pred_v":
                return epsilon_weighting * snr / (snr + 1)
            case _:
                raise ValueError(f"unknown objective {self.objective}")

    #
    def forward(
        self,
        x: torch.Tensor,
        external_cond: Optional[torch.Tensor],
        k: torch.Tensor,
    ):
        # we first sampel from random gaussian noise
        noise = torch.randn_like(x)
        # then we clip the noise
        noise = torch.clamp(noise, -self.clip_noise, self.clip_noise)
        # we create a noisy version of the clean image
        # based off of our noise we just computed
        noised_x = self.q_sample(x_start=x, k=k, noise=noise)

        # then we take our model prediction
        # notice here that it takes in the external condition here
        model_pred = self.model_predictions(
            x=noised_x, k=k, external_cond=external_cond
        )

        # remember this is a collectable
        pred = model_pred.model_out
        # the models guess of clean version of the image
        x_pred = model_pred.pred_x_start

        # we want to actually get the ground truth
        if self.objective == "pred_noise":
            target = noise
        elif self.objective == "pred_x0":
            target = x
        elif self.objective == "pred_v":
            target = self.predict_v(x, k, noise)
        else:
            raise ValueError(f"unknown objective {self.objective}")

        # this is a per pixel loss
        loss = F.mse_loss(pred, target.detach(), reduction="none")

        # we do some strategic weighting
        loss_weight = self.compute_loss_weights(k, self.loss_weighting.strategy)
        # reshape the loss
        loss_weight = self.add_shape_channels(loss_weight)
        # rescarle the loss
        loss = loss * loss_weight

        return x_pred, loss

    def ddim_idx_to_noise_level(self, indices: torch.Tensor):
        """
        Takes indices and converts this into noise levels for our
        forward process of the diffusion process.

        """
        shape = indices.shape
        real_steps = torch.linspace(-1, self.timesteps - 1, self.sampling_timesteps + 1)
        real_steps = real_steps.long().to(indices.device)
        k = real_steps[indices.flatten()]
        return k.view(shape)

    """
        The prupose of this is to run one sampling step. There is a flag that 
        says self.ddim_sample_steps which will run DDIM vs. DDPM 

    
    """

    def clip_guidance_fn(
        self, xk, pred_x0, alpha_cumprod, external_cond, guidance_scale=2.0
    ):
        """
        CLIP guidance function that uses CLIP embeddings to guide the sampling process.

        Args:
            xk: Current noisy sample
            pred_x0: Predicted clean sample
            alpha_cumprod: Alpha cumulative product
            external_cond: CLIP embeddings
            guidance_scale: Scale of the guidance
        """
        # Ensure we have gradients for xk
        xk = xk.detach().requires_grad_()

        # Get CLIP image embeddings for the current sample
        with torch.no_grad():
            # Preprocess the image for CLIP
            # Assuming xk is in range [-1, 1], normalize to [0, 1]
            normalized_xk = (xk + 1) / 2
            # Resize to CLIP input size (224x224)
            resized_xk = F.interpolate(normalized_xk, size=(224, 224), mode="bilinear")
            # Get CLIP image features
            image_features = clip_model.encode_image(resized_xk)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # Calculate cosine similarity between image and text embeddings
        similarity = torch.cosine_similarity(image_features, external_cond, dim=-1)

        # Calculate guidance loss
        guidance_loss = (
            -similarity.mean()
        )  # Negative because we want to maximize similarity

        # Scale the guidance
        guidance_loss = guidance_scale * guidance_loss

        return guidance_loss

    def sample_step(
        self,
        x: torch.Tensor,
        curr_noise_level: torch.Tensor,
        next_noise_level: torch.Tensor,
        external_cond: Optional[torch.Tensor],
        external_cond_mask: Optional[torch.Tensor] = None,
        guidance_fn: Optional[Callable] = None,
        clip_text: Optional[str] = None,
    ):
        # Get CLIP embeddings if text is provided
        print("The CLIP TEXT IS ", clip_text)
        
        if clip_text is not None:
            print("Entering CLIP")
            clip_embeddings = get_clip_embeddings(clip_text)
            # If external_cond is None, use CLIP embeddings
            if external_cond is None:
                external_cond = clip_embeddings
            # If external_cond exists, concatenate with CLIP embeddings
            else:
                external_cond = torch.cat([external_cond, clip_embeddings], dim=-1)

            # Use CLIP guidance if enabled
            if self.cfg.diffusion.use_clip_guidance:
                guidance_fn = lambda xk, pred_x0, alpha_cumprod, external_cond: self.clip_guidance_fn(
                    xk,
                    pred_x0,
                    alpha_cumprod,
                    external_cond,
                    guidance_scale=self.cfg.diffusion.clip_guidance_scale,
                )

        if self.is_ddim_sampling:
            return self.ddim_sample_step(
                x=x,
                curr_noise_level=curr_noise_level,
                next_noise_level=next_noise_level,
                external_cond=external_cond,
                external_cond_mask=external_cond_mask,
                guidance_fn=guidance_fn,
            )

        # FIXME: temporary code for checking ddpm sampling
        assert torch.all(
            (curr_noise_level - 1 == next_noise_level)
            | ((curr_noise_level == -1) & (next_noise_level == -1))
        ), "Wrong noise level given for ddpm sampling."

        assert (
            self.sampling_timesteps == self.timesteps
        ), "sampling_timesteps should be equal to timesteps for ddpm sampling."

        return self.ddpm_sample_step(
            x=x,
            curr_noise_level=curr_noise_level,
            external_cond=external_cond,
            external_cond_mask=external_cond_mask,
            guidance_fn=guidance_fn,
        )

    def ddpm_sample_step(
        self,
        x: torch.Tensor,
        curr_noise_level: torch.Tensor,
        external_cond: Optional[torch.Tensor],
        external_cond_mask: Optional[torch.Tensor] = None,
        guidance_fn: Optional[Callable] = None,
    ):
        if guidance_fn is not None:
            raise NotImplementedError("guidance_fn is not yet implmented for ddpm.")

        clipped_curr_noise_level = torch.clamp(curr_noise_level, min=0)

        model_mean, _, model_log_variance = self.p_mean_variance(
            x=x,
            k=clipped_curr_noise_level,
            external_cond=external_cond,
            external_cond_mask=external_cond_mask,
        )

        noise = torch.where(
            self.add_shape_channels(clipped_curr_noise_level > 0),
            torch.randn_like(x),
            0,
        )
        noise = torch.clamp(noise, -self.clip_noise, self.clip_noise)
        x_pred = model_mean + torch.exp(0.5 * model_log_variance) * noise

        # only update frames where the noise level decreases
        return torch.where(self.add_shape_channels(curr_noise_level == -1), x, x_pred)

    def ddim_sample_step(
        self,
        x: torch.Tensor,
        curr_noise_level: torch.Tensor,
        next_noise_level: torch.Tensor,
        external_cond: Optional[torch.Tensor],
        external_cond_mask: Optional[torch.Tensor] = None,
        guidance_fn: Optional[Callable] = None,
    ):
        clipped_curr_noise_level = torch.clamp(curr_noise_level, min=0)

        # get alphas
        alpha = self.alphas_cumprod[clipped_curr_noise_level]
        alpha_next = torch.where(
            next_noise_level < 0,
            torch.ones_like(next_noise_level),
            self.alphas_cumprod[next_noise_level],
        )

        # Calculate sigma for DDIM sampling
        sigma = torch.where(
            next_noise_level < 0,
            torch.zeros_like(next_noise_level),
            self.ddim_sampling_eta
            * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt(),
        )

        # Calculate scaling parameters
        c = (1 - alpha_next - sigma**2).sqrt()

        # Add shape channels for broadcasting
        alpha = self.add_shape_channels(alpha)
        alpha_next = self.add_shape_channels(alpha_next)
        c = self.add_shape_channels(c)
        sigma = self.add_shape_channels(sigma)

        # Get model predictions with CLIP embeddings
        model_pred = self.model_predictions(
            x=x,
            k=clipped_curr_noise_level,
            external_cond=external_cond,
            external_cond_mask=external_cond_mask,
        )

        # Log current sampling step and CLIP text if available
        # if hasattr(self, 'cfg') and self.cfg.diffusion.use_clip_guidance:
        print(f"\nDDIM Sampling Step:")
       #  print(f"Current noise level: {clipped_curr_noise_level.mean().item():.2f}")
       #  print(f"Next noise level: {next_noise_level.mean().item():.2f}")
        # print(f"CLIP guidance text: {self.cfg.diffusion.clip_text}")
        # print(f"CLIP guidance scale: {self.cfg.diffusion.clip_guidance_scale}")

        # If using guidance (like CLIP guidance)
        if guidance_fn is not None:
            with torch.enable_grad():
                x = x.detach().requires_grad_()

                # Calculate guidance loss using CLIP embeddings
                guidance_loss = guidance_fn(
                    xk=x,
                    pred_x0=model_pred.pred_x_start,
                    alpha_cumprod=alpha,
                    external_cond=external_cond,  # Pass CLIP embeddings to guidance function
                )

                # Log guidance loss
                # if hasattr(self, 'cfg') and self.cfg.diffusion.use_clip_guidance:
                # print(f"CLIP guidance loss: {guidance_loss.mean().item():.4f}")

                # Calculate gradient for guidance
                grad = -torch.autograd.grad(guidance_loss, x)[0]
                grad = torch.nan_to_num(grad, nan=0.0)

                # Apply guidance to noise prediction
                pred_noise = model_pred.pred_noise + (1 - alpha).sqrt() * grad

                # Predict x_start with guided noise
                x_start = torch.where(
                    alpha > 0,  # Avoid NaN from zero terminal SNR
                    self.predict_start_from_noise(
                        x, clipped_curr_noise_level, pred_noise
                    ),
                    model_pred.pred_x_start,
                )
        else:
            # Use standard predictions without guidance
            x_start = model_pred.pred_x_start
            pred_noise = model_pred.pred_noise

        # Generate noise for sampling
        noise = torch.randn_like(x)
        noise = torch.clamp(noise, -self.clip_noise, self.clip_noise)

        # DDIM sampling step
        x_pred = x_start * alpha_next.sqrt() + pred_noise * c + sigma * noise

        # Only update frames where the noise level decreases
        mask = curr_noise_level == next_noise_level
        x_pred = torch.where(
            self.add_shape_channels(mask),
            x,
            x_pred,
        )

        return x_pred

    def estimate_noise_level(self, x, mu=None):
        # x ~ ( B, T, C, ...)
        if mu is None:
            mu = torch.zeros_like(x)
        x = x - mu
        mse = reduce(x**2, "b t ... -> b t", "mean")
        ll_except_c = -self.log_one_minus_alphas_cumprod[None, None] - mse[
            ..., None
        ] * self.alphas_cumprod[None, None] / (1 - self.alphas_cumprod[None, None])
        k = torch.argmax(ll_except_c, -1)
        return k
