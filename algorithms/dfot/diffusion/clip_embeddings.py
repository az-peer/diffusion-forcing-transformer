import clip
import torch
from PIL import Image


# Load pre-trained CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)


def get_clip_embeddings(text: str):
    # Encode the text using CLIP
    text_input = clip.tokenize([text]).to(device)  # Tokenize text and move to device
    with torch.no_grad():
        text_features = model.encode_text(
            text_input
        )  # Get the text features (embedding)
    return text_features / text_features.norm(
        dim=-1, keepdim=True
    )  # Normalize embeddings
