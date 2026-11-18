"""
Knowledge distillation framework.

Total student loss (all three terms can be switched on/off from the CLI):

    L = (1 - alpha) * CE(z_s, y)                                   hard-label term
      + alpha * T^2 * KL( softmax(z_t / T) || softmax(z_s / T) )   soft-target term (Hinton et al., 2015)
      + beta * MSE( r(F_s), F_t )                                  feature term (FitNets, Romero et al., 2015)

z_s / z_t      : student / teacher logits
T              : temperature; T > 1 softens the distributions so the student also learns the teacher's
                 "dark knowledge" (relative probabilities of the wrong classes). The T^2 factor keeps the
                 gradient scale of the soft term independent of T.
F_s / F_t      : last convolutional feature maps of student and teacher (4x4 for all models here)
r(.)           : learnable 1x1 conv + BN regressor that maps student channels to teacher channels
                 (e.g. 1280 -> 2048); it is trained jointly with the student and discarded afterwards.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def softmax_with_temperature(logits: torch.Tensor, T: float) -> torch.Tensor:
    """Temperature-scaled softmax: exp(z_i / T) / sum_j exp(z_j / T)."""
    return F.softmax(logits / T, dim=1)


def soft_target_loss(student_logits, teacher_logits, T: float) -> torch.Tensor:
    log_p_s = F.log_softmax(student_logits / T, dim=1)
    p_t = softmax_with_temperature(teacher_logits, T)
    return F.kl_div(log_p_s, p_t, reduction="batchmean") * (T * T)


class FeatureAdapter(nn.Module):
    """1x1 conv regressor that projects student features into the teacher's feature space."""

    def __init__(self, student_ch: int, teacher_ch: int):
        super().__init__()
        self.proj = nn.Sequential(nn.Conv2d(student_ch, teacher_ch, 1, bias=False), nn.BatchNorm2d(teacher_ch))

    def forward(self, f_s: torch.Tensor, target_hw) -> torch.Tensor:
        f = self.proj(f_s)
        if f.shape[-2:] != tuple(target_hw):              # spatial alignment if resolutions differ
            f = F.adaptive_avg_pool2d(f, target_hw)
        return f


class DistillationLoss(nn.Module):
    """Combined hard-label + soft-target + feature distillation loss."""

    def __init__(self, alpha: float = 0.7, temperature: float = 4.0, beta: float = 1.0,
                 student_ch: int | None = None, teacher_ch: int | None = None):
        super().__init__()
        self.alpha, self.T, self.beta = alpha, temperature, beta
        self.adapter = FeatureAdapter(student_ch, teacher_ch) if beta > 0 else None

    def forward(self, s_logits, t_logits, y, f_s=None, f_t=None):
        hard = F.cross_entropy(s_logits, y)
        soft = soft_target_loss(s_logits.float(), t_logits.float(), self.T)
        total = (1 - self.alpha) * hard + self.alpha * soft
        parts = {"hard": hard.item(), "soft": soft.item()}
        if self.adapter is not None and f_s is not None:
            feat = F.mse_loss(self.adapter(f_s, f_t.shape[-2:]).float(), f_t.float())
            total = total + self.beta * feat
            parts["feat"] = feat.item()
        return total, parts


class Distiller(nn.Module):
    """
    Wraps a frozen teacher and a trainable student. `step(x, y)` runs both networks and returns
    (loss, student_logits, loss_parts) in the format expected by common.fit().
    """

    def __init__(self, student: nn.Module, teacher: nn.Module, criterion: DistillationLoss):
        super().__init__()
        self.student, self.teacher, self.criterion = student, teacher, criterion
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        self.teacher.eval()                               # teacher BN statistics must never change
        return self

    def step(self, x, y):
        with torch.no_grad():
            t_logits, f_t = self.teacher.forward_with_features(x)
        s_logits, f_s = self.student.forward_with_features(x)
        loss, parts = self.criterion(s_logits, t_logits, y, f_s, f_t)
        return loss, s_logits, parts

    def trainable_extra(self):
        """Parameters besides the student's that must be optimised (the feature adapter)."""
        return list(self.criterion.parameters())
