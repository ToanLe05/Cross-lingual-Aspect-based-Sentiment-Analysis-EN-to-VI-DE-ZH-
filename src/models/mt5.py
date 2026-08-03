"""
src/models/mt5.py
─────────────────
L3: mT5-small with Prefix Constrained Decoding.

Constrained decoding forces output ∈ {positive, negative, neutral}
at the first decoder step — eliminates hallucination, 100% valid output.

Input  : "sentiment: {text} aspect: {category}"
Output : "positive" / "negative" / "neutral"
Post   : map string → label {0,1,2} for unified evaluation

Enhancements:
  - Generation parameters (temperature, top_p, num_beams) for flexible inference
  - Manual loss with guaranteed label_smoothing and class-weighted support
  - Better device handling
  - Explicit pad token setting
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

LABEL2ID = {"positive": 0, "negative": 1, "neutral": 2}
ID2LABEL = {0: "positive", 1: "negative", 2: "neutral"}
ALLOWED_LABELS = ["positive", "negative", "neutral"]


class MT5ForABSA:
    """
    Thin wrapper around mT5-small for Oracle ASC.
    Handles constrained decoding and label mapping.
    """

    def __init__(
        self,
        model_name: str = "google/mt5-small",
        use_constrained_decoding: bool = True,
        temperature: float = 1.0,
        top_p: float = 1.0,
        max_length: int = 2,
        num_beams: int = 1,
        do_sample: bool = False,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.use_constrained = use_constrained_decoding
        self.temperature = temperature
        self.top_p = top_p
        self.max_length = max_length
        self.num_beams = num_beams
        self.do_sample = do_sample

        # Ensure pad token is set (important for batch generation)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Pre-compute allowed token IDs for constrained decoding
        self._allowed_ids = [
            self.tokenizer.encode(label, add_special_tokens=False)[0]
            for label in ALLOWED_LABELS
        ]

    def _prefix_fn(self, batch_id: int, input_ids: torch.Tensor) -> list[int]:
        """
        Prefix constrained decoding function.
        Step 1 (only decoder start token): allow {positive, negative, neutral}
        Step 2+: force EOS — model outputs exactly one token.
        """
        if input_ids.shape[-1] == 1:
            return self._allowed_ids
        return [self.tokenizer.eos_token_id]

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        **kwargs,
    ) -> list[int]:
        """
        Generate sentiment labels as integers {0, 1, 2}.
        Uses constrained decoding if enabled.

        Args:
            input_ids: tokenized input ids
            attention_mask: attention mask
            **kwargs: override generation parameters (max_length, num_beams, etc.)
        """
        gen_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_length": kwargs.get("max_length", self.max_length),
            "num_beams": kwargs.get("num_beams", self.num_beams),
            "do_sample": kwargs.get("do_sample", self.do_sample),
            "temperature": kwargs.get("temperature", self.temperature),
            "top_p": kwargs.get("top_p", self.top_p),
        }
        if self.use_constrained:
            gen_kwargs["prefix_allowed_tokens_fn"] = self._prefix_fn

        gen_ids = self.model.generate(**gen_kwargs)
        texts = self.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
        # Lowercase to be safe (mT5 outputs lower case)
        return [LABEL2ID.get(t.strip().lower(), 2) for t in texts]

    @torch.no_grad()
    def predict_labels(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> list[int]:
        """
        Classify by scoring the only valid label strings.

        This is more reliable than free-form generation for ABSA because the
        model must choose among positive/negative/neutral and cannot return a
        malformed string. Lower average NLL means the label is more likely.
        """
        device = input_ids.device
        batch_size = input_ids.size(0)
        num_labels = len(ALLOWED_LABELS)

        try:
            label_enc = self.tokenizer(
                text_target=ALLOWED_LABELS,
                padding=True,
                return_tensors="pt",
            )
        except TypeError:
            label_enc = self.tokenizer(
                ALLOWED_LABELS,
                padding=True,
                return_tensors="pt",
            )

        candidate_labels = label_enc["input_ids"].to(device)
        candidate_labels[candidate_labels == self.tokenizer.pad_token_id] = -100
        label_len = candidate_labels.size(1)

        repeated_input_ids = (
            input_ids.unsqueeze(1)
            .expand(batch_size, num_labels, input_ids.size(1))
            .reshape(batch_size * num_labels, input_ids.size(1))
        )
        repeated_attention = (
            attention_mask.unsqueeze(1)
            .expand(batch_size, num_labels, attention_mask.size(1))
            .reshape(batch_size * num_labels, attention_mask.size(1))
        )
        repeated_labels = (
            candidate_labels.unsqueeze(0)
            .expand(batch_size, num_labels, label_len)
            .reshape(batch_size * num_labels, label_len)
        )

        outputs = self.model(
            input_ids=repeated_input_ids,
            attention_mask=repeated_attention,
            labels=repeated_labels,
        )
        logits = outputs.logits
        token_loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            repeated_labels.reshape(-1),
            ignore_index=-100,
            reduction="none",
        ).view(batch_size * num_labels, label_len)
        token_counts = repeated_labels.ne(-100).sum(dim=1).clamp(min=1)
        scores = (token_loss.sum(dim=1) / token_counts).view(batch_size, num_labels)
        return scores.argmin(dim=1).cpu().tolist()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        label_smoothing: float = 0.0,
        class_weights: torch.Tensor | None = None,
        ref_labels: torch.Tensor | None = None,
        **kwargs,
    ) -> dict:
        """
        Training forward — returns loss and logits.

        Computes loss manually via F.cross_entropy to:
        1. Guarantee label_smoothing works (T5Config doesn't reliably read
           config.label_smoothing_factor unlike BERT-family).
        2. Support class-weighted loss for parity with AG-CAN/XLM-R.
        """
        # Pass labels so HF auto-shifts decoder_input_ids correctly,
        # but we compute our own loss instead of using outputs.loss.
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        logits = outputs.logits

        per_token = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=-100,
            reduction="none",
            label_smoothing=label_smoothing,
        ).view(labels.size(0), -1)
        token_counts = (labels != -100).sum(dim=1).clamp(min=1)
        per_example = per_token.sum(dim=1) / token_counts          # (B,)

        if class_weights is not None and ref_labels is not None:
            w = class_weights[ref_labels]
            loss = (per_example * w).sum() / w.sum()
        else:
            loss = per_example.mean()

        return {"loss": loss, "logits": logits}

    def parameters(self):
        return self.model.parameters()

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def to(self, device):
        self.model.to(device)
        return self

    def state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, state):
        self.model.load_state_dict(state)

def build_mt5_model(cfg: dict) -> tuple[MT5ForABSA, AutoTokenizer]:
    paths = cfg.get("model_paths", {})
    model_path = paths.get("mt5", "pretrained_models/mt5")
    from pathlib import Path
    if not Path(model_path).exists():
        model_path = "google/mt5-small"
    model = MT5ForABSA(
        model_name=model_path,
        use_constrained_decoding=cfg.get("constrained_decoding", True),
        max_length=cfg.get("max_target_len", 10)
    )
    return model, model.tokenizer
