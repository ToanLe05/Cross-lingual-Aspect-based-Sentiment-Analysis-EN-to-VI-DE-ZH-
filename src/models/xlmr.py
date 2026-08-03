"""
src/models/xlmr.py
──────────────────
L2: XLM-R + Masked Mean Pooling + LLRD

Two key improvements over vanilla XLM-R fine-tuning:

1. Masked Mean Pooling (mathematically exact):
   Instead of slicing by position (fragile, wrong if category shifts),
   we use a category_mask built at dataset level.
   Only category token positions contribute to the aspect vector.

2. Layer-wise Learning Rate Decay (LLRD):
   Lower layers encode universal linguistic features — small LR, preserve.
   Upper layers + classifier encode task features — large LR, adapt fast.

Enhancements:
  - Label smoothing for CrossEntropy (helps few-shot)
  - Dropout before classifier input
  - Better weight decay handling (no decay on bias/LayerNorm)
  - Option to freeze lower layers for very few shots
"""

import torch
import torch.nn as nn
from transformers import AutoModel
from pathlib import Path


class XLMRForABSA(nn.Module):
    """
    XLM-R with aspect-aware classifier for Oracle ASC.

    Input  : [CLS] text [SEP] category [SEP]
    Pooling: concat([CLS], masked_mean(category_tokens))
    Output : logits over {positive, negative, neutral}
    """

    def __init__(
        self,
        model_name: str = "xlm-roberta-base",
        num_labels: int = 3,
        dropout: float = 0.1,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size   # 768
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden * 2, num_labels)
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def _masked_mean_pool(
        self,
        hidden: torch.Tensor,        # (B, T, H)
        category_mask: torch.Tensor, # (B, T) — 1 at category positions
    ) -> torch.Tensor:               # (B, H)
        """
        Exact mean pooling over category token positions only.
        Uses category_mask built at dataset level — no positional slicing.
        Zero-safe: clamp denominator to avoid div/0 for implicit aspects.
        """
        mask_exp = category_mask.unsqueeze(-1).float()           # (B, T, 1)
        sum_vec = (hidden * mask_exp).sum(dim=1)                # (B, H)
        count = mask_exp.sum(dim=1).clamp(min=1e-9)             # (B, 1)
        return sum_vec / count

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        category_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> dict:
        hidden = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).last_hidden_state                                       # (B, T, H)

        cls_vec = hidden[:, 0, :]                              # (B, H)
        aspect_vec = self._masked_mean_pool(hidden, category_mask)  # (B, H)
        combined = torch.cat([cls_vec, aspect_vec], dim=-1)    # (B, 2H)

        logits = self.classifier(self.dropout(combined))          # (B, 3)

        out = {"logits": logits}
        if labels is not None:
            out["loss"] = self.loss_fn(logits, labels)
        return out

    def freeze_lower_layers(self, num_layers: int = 6):
        """
        Freeze lower encoder layers for very few-shot scenarios.
        Helps preserve universal features and prevent overfitting.
        """
        for i, layer in enumerate(self.encoder.encoder.layer):
            if i < num_layers:
                for param in layer.parameters():
                    param.requires_grad = False
            else:
                break


# ── LLRD optimizer ─────────────────────────────────────────────────────────────

def build_llrd_optimizer(
    model: XLMRForABSA,
    lr_embeddings: float = 1e-5,
    lr_encoder_low: float = 1.5e-5,
    lr_encoder_high: float = 2e-5,
    lr_classifier: float = 3e-5,
    weight_decay: float = 0.01,
) -> torch.optim.Optimizer:
    """
    Layer-wise Learning Rate Decay.
    No weight_decay on bias and LayerNorm (standard practice).
    """
    no_decay = ["bias", "LayerNorm.weight"]

    def wd(name):
        return 0.0 if any(nd in name for nd in no_decay) else weight_decay

    groups = []

    # Embeddings
    for n, p in model.encoder.embeddings.named_parameters():
        if p.requires_grad:
            groups.append({"params": [p], "lr": lr_embeddings, "weight_decay": wd(n)})

    # Encoder layers
    for i, layer in enumerate(model.encoder.encoder.layer):
        lr = lr_encoder_low if i < 6 else lr_encoder_high
        for n, p in layer.named_parameters():
            if p.requires_grad:
                groups.append({"params": [p], "lr": lr, "weight_decay": wd(n)})

    # Classifier
    for n, p in model.classifier.named_parameters():
        if p.requires_grad:
            groups.append({"params": [p], "lr": lr_classifier, "weight_decay": wd(n)})

    return torch.optim.AdamW(groups)

def build_xlmr_model(cfg: dict) -> XLMRForABSA:
    paths = cfg.get("model_paths", {})
    model_path = paths.get("xlmr", "pretrained_models/xlmr")
    if not Path(model_path).exists():
        model_path = "xlm-roberta-base"
    return XLMRForABSA(
        model_name=model_path,
        num_labels=3,
        dropout=cfg.get("dropout", 0.1),
        label_smoothing=cfg.get("label_smoothing", 0.1)
    )



