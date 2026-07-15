from potholes.detection.models.cnn import load_cnn_model
from potholes.detection.models.transformer import load_ast_model


def load_model(model_config: dict | None = None):
    model_config = model_config or {}
    architecture = model_config.get("architecture", "ast")

    if architecture == "ast":
        return load_ast_model(model_config)
    if architecture in {"cnn", "spectrogram_cnn"}:
        return load_cnn_model(model_config)

    raise ValueError(f"Unknown model architecture: {architecture}")
