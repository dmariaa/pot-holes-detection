import torch
import torch.nn as nn

from potholes.detection.models.channels import resolve_channel_indices


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(dropout),
        )

    def forward(self, x):
        return self.block(x)


class SpectrogramCNNEncoder(nn.Module):
    """
    Encodes selected sensor spectrogram channels into a compact embedding.

    Input shape: [B, 6, H, W]
    Output shape: [B, embedding_dim]
    """
    def __init__(
            self,
            channels: str | int | list[str | int] | None = None,
            embedding_dim: int = 256,
            base_channels: int = 32,
            dropout: float = 0.2,
    ):
        super().__init__()
        channel_indices, channel_names = resolve_channel_indices(channels)
        self.channel_names = channel_names
        self.embedding_dim = embedding_dim
        self.register_buffer("channel_indices", torch.tensor(channel_indices, dtype=torch.long), persistent=False)

        in_channels = len(channel_indices)
        widths = [base_channels, base_channels * 2, base_channels * 4, base_channels * 8]
        self.features = nn.Sequential(
            ConvBlock(in_channels, widths[0], dropout),
            ConvBlock(widths[0], widths[1], dropout),
            ConvBlock(widths[1], widths[2], dropout),
            ConvBlock(widths[2], widths[3], dropout),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.embedding = nn.Sequential(
            nn.Flatten(),
            nn.Linear(widths[-1], embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
        )

    def forward(self, pixel_values):
        x = pixel_values.index_select(dim=1, index=self.channel_indices)
        x = self.features(x)
        x = self.pool(x)
        return self.embedding(x)


class SpectrogramCNNClassifier(nn.Module):
    def __init__(
            self,
            channels: str | int | list[str | int] | None = None,
            num_labels: int = 6,
            embedding_dim: int = 256,
            base_channels: int = 32,
            dropout: float = 0.2,
            freeze_encoder: bool = False,
    ):
        super().__init__()
        self.freeze_encoder = freeze_encoder
        self.encoder = SpectrogramCNNEncoder(
            channels=channels,
            embedding_dim=embedding_dim,
            base_channels=base_channels,
            dropout=dropout,
        )
        self.channel_names = self.encoder.channel_names
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, num_labels),
        )

    def encode(self, pixel_values):
        return self.encoder(pixel_values)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_encoder:
            self.encoder.eval()
        return self

    def forward(self, pixel_values, **kwargs):
        embeddings = self.encode(pixel_values)
        return self.classifier(embeddings)


def _load_encoder_checkpoint(model: SpectrogramCNNClassifier, checkpoint_path: str):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.encoder.load_state_dict(state_dict)


def _freeze_encoder(model: SpectrogramCNNClassifier):
    for param in model.encoder.parameters():
        param.requires_grad = False
    model.encoder.eval()


def load_cnn_model(model_config: dict | None = None):
    model_config = model_config or {}
    model = SpectrogramCNNClassifier(
        channels=model_config.get("channels"),
        num_labels=model_config.get("num_labels", 6),
        embedding_dim=model_config.get("embedding_dim", 256),
        base_channels=model_config.get("base_channels", 32),
        dropout=model_config.get("dropout", 0.2),
        freeze_encoder=model_config.get("freeze_encoder", False),
    )

    encoder_checkpoint = model_config.get("encoder_checkpoint")
    model.encoder_checkpoint = encoder_checkpoint
    if encoder_checkpoint:
        _load_encoder_checkpoint(model, encoder_checkpoint)
        print(f"Loaded encoder checkpoint: {encoder_checkpoint}")

    if model_config.get("freeze_encoder", False):
        _freeze_encoder(model)
        print("Encoder frozen: true")

    print("Input channels:", model.channel_names)
    print(f"Embedding dim: {model.encoder.embedding_dim}")
    return model
