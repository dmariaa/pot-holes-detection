from transformers import ASTConfig, ASTForAudioClassification, ViTMSNForImageClassification
import torch.nn as nn
import torch.nn.functional as F


class ASTWrapper(nn.Module):
    """
    Expects input: [B, 6, H, W]
    Converts to:   [B, 1, H, W] by direct reduction (no conv),
    then resizes to: [B, T=1024, F=128] for AST (time, num_mel_bins)
    """
    def __init__(self, ast_model: ASTForAudioClassification):
        super().__init__()
        self.ast = ast_model
        self.num_mel_bins = self.ast.config.num_mel_bins      # 128
        self.max_length   = self.ast.config.max_length        # 1024

    def forward(self, pixel_values, **kwargs):
        # pixel_values: [B, 6, H, W]

        # 1) 6 -> 1 channel WITHOUT conv (direct reduction)
        x = pixel_values.mean(dim=1, keepdim=True)    # [B, 1, H, W]
        # or: x = pixel_values[:, 0:1, :, :]

        # 2) Resize to AST's expected (freq, time) = (num_mel_bins, max_length)
        #    Shape: [B, 1, num_mel_bins=128, max_length=1024]
        x = F.interpolate(
            x,
            size=(self.num_mel_bins, self.max_length),
            mode="bilinear",
            align_corners=False,
        )  # [B, 1, 128, 1024]

        # 3) Convert to [B, time, num_mel_bins]
        x = x.squeeze(1)           # [B, 128, 1024]
        x = x.permute(0, 2, 1)     # [B, 1024, 128] = [B, T, F]

        # 4) Feed to AST
        out = self.ast(input_values=x, **kwargs)
        return out.logits          # [B, NUM_LABELS]

class ViTMSNWithStem(nn.Module):
    def __init__(self, vit_model: ViTMSNForImageClassification):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(6, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 3, kernel_size=1, bias=True),
        )
        self.vit = vit_model

    def forward(self, pixel_values, **kwargs):
        # pixel_values: [B, 6, H, W]
        x3 = self.stem(pixel_values)          # -> [B, 3, H, W]
        return self.vit(pixel_values=x3, **kwargs)

def load_model():
    NUM_LABELS = 6
    pretrained_model = "MIT/ast-finetuned-audioset-10-10-0.4593"
    config = ASTConfig.from_pretrained(pretrained_model)
    config.num_labels = NUM_LABELS

    ast_backbone = ASTForAudioClassification.from_pretrained(
        pretrained_model,
        config=config,
        ignore_mismatched_sizes=True,
    )

    # Freeze everything
    for p in ast_backbone.parameters():
        p.requires_grad = False

    # Unfreeze classifier head only
    for p in ast_backbone.classifier.parameters():
        p.requires_grad = True

    N = 3
    transformer_blocks = ast_backbone.audio_spectrogram_transformer.encoder.layer
    for block in transformer_blocks[-N:]:
        for p in block.parameters():
            p.requires_grad = True

    # Unfreeze global LayerNorm
    for name, param in ast_backbone.named_parameters():
        if "audio_spectrogram_transformer.layernorm" in name.lower():
            param.requires_grad = True

    # Build wrapped AST model with stem
    model = ASTWrapper(ast_backbone)
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    print("Trainable params:", trainable)

    def count_parameters(m):
        total = sum(p.numel() for p in m.parameters())
        trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
        return total, trainable

    total_params, trainable_params = count_parameters(model)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    return model