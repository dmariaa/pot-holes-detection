import torch.nn as nn
import torch.nn.functional as F

from potholes.detection.models.cnn import SpectrogramCNNEncoder


class ProjectionHead(nn.Module):
    def __init__(self, embedding_dim: int, hidden_dim: int = 256, projection_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, projection_dim),
        )

    def forward(self, embeddings):
        return self.net(embeddings)


class ContrastiveEncoderModel(nn.Module):
    def __init__(self, encoder: nn.Module, embedding_dim: int, hidden_dim: int = 256, projection_dim: int = 128):
        super().__init__()
        self.encoder = encoder
        self.projector = ProjectionHead(
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            projection_dim=projection_dim,
        )
        self.channel_names = getattr(encoder, "channel_names", None)

    def encode(self, pixel_values):
        return self.encoder(pixel_values)

    def forward(self, pixel_values):
        embeddings = self.encode(pixel_values)
        projections = F.normalize(self.projector(embeddings), dim=1)
        return embeddings, projections


def load_contrastive_model(model_config: dict | None = None):
    model_config = model_config or {}
    encoder_architecture = model_config.get("encoder_architecture", model_config.get("architecture", "cnn"))
    if encoder_architecture not in {"cnn", "spectrogram_cnn"}:
        raise ValueError(f"Contrastive pretraining currently supports only the CNN encoder, got: {encoder_architecture}")

    embedding_dim = model_config.get("embedding_dim", 256)
    encoder = SpectrogramCNNEncoder(
        channels=model_config.get("channels"),
        embedding_dim=embedding_dim,
        base_channels=model_config.get("base_channels", 32),
        dropout=model_config.get("dropout", 0.2),
    )
    model = ContrastiveEncoderModel(
        encoder=encoder,
        embedding_dim=embedding_dim,
        hidden_dim=model_config.get("projection_hidden_dim", embedding_dim),
        projection_dim=model_config.get("projection_dim", 128),
    )
    print("Input channels:", model.channel_names)
    print(f"Embedding dim: {embedding_dim}")
    print(f"Projection dim: {model_config.get('projection_dim', 128)}")
    return model
