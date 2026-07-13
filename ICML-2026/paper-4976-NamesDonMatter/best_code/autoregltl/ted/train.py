import torch
from typing import Dict, Union, Any, Optional, Tuple, List
from autoregltl.train import LTLTrainer


class TedTrainer(LTLTrainer):
    """
    Transformer encoder-decoder trainer
    """
    def compute_loss(self, model, inputs, return_outputs=False):
        logits = model(inputs["input_ids"], inputs["target_ids"], inputs["pe"])
        labels = torch.where(inputs["target_ids"] == model.pad_id, -100, inputs["target_ids"]).to(logits.device)
        loss = self.loss_fct(logits, labels, model.training)
        return loss

    def prediction_step(
        self,
        model: torch.nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        with torch.no_grad():
            logits = model(inputs["input_ids"], inputs["target_ids"], inputs["pe"])
            labels = torch.where(inputs["target_ids"] == model.pad_id, -100, inputs["target_ids"]).to(logits.device)
            loss = self.loss_fct(logits, labels, model.training)
        if prediction_loss_only:
            return loss, None, None
        else:
            return loss, logits, labels

    def save_model(self, output_dir=None, _internal_call=None):
        if not output_dir:
            output_dir = self.args.output_dir
        self.model.save_pretrained(output_dir)