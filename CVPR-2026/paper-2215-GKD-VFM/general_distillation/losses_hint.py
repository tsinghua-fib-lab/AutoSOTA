# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Implements the knowledge distillation loss
"""
import torch
from torch.nn import functional as F


class DistillationLoss(torch.nn.Module):
    """
    This module wraps a standard criterion and adds an extra knowledge distillation loss by
    taking a teacher model prediction and using it as additional supervision.
    """
    def __init__(self, base_criterion: torch.nn.Module, teacher_model: torch.nn.Module,
                 distillation_type: str, lambda_token: float, lambda_fea: float):
        super().__init__()
        self.base_criterion = base_criterion
        self.teacher_model = teacher_model
        assert distillation_type in ['none', 'soft', 'hard']
        self.distillation_type = distillation_type
        self.lambda_token = lambda_token
        self.lambda_fea = lambda_fea
        self.soft_criterion = torch.nn.MSELoss()

    def forward(self, inputs, outputs, labels):
        """
        Args:
            inputs: The original inputs that are feed to the teacher model
            outputs: the outputs of the model to be trained. It is expected to be
                either a Tensor, or a Tuple[Tensor, Tensor], with the original output
                in the first position and the distillation predictions as the second output
            labels: the labels for the base criterion
        """
        outputs_token, outputs_fea = outputs

        # don't backprop throught the teacher
        with torch.no_grad():
            teacher_outputs = self.teacher_model.backbone.get_intermediate_layers(inputs, n=4, return_class_token=True)
            teacher_outputs_token = teacher_outputs[3][1]
            teacher_outputs_fea = torch.cat((teacher_outputs_token.unsqueeze(1),teacher_outputs[3][0]),dim=1)

        distillation_loss_token = self.soft_criterion(outputs_token, teacher_outputs_token)
        distillation_loss_fea = self.soft_criterion(outputs_fea, teacher_outputs_fea)

        token_loss = self.lambda_token * distillation_loss_token
        fea_loss = self.lambda_fea * distillation_loss_fea
        
        return token_loss, fea_loss
