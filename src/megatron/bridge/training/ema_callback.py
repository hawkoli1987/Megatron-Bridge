# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""EMA / BEMA weight-averaging callback for Megatron-Bridge.

Supports two modes selected via the ``mode`` argument:

* ``"ema"`` — classical exponential moving average with constant decay:
      ``ema = decay * ema + (1 - decay) * param``
  applied every training step.

* ``"bema"`` — Bias-corrected EMA (Block & Zhang, 2025, arXiv:2508.00180,
  "EMA Without the Lag"). Stores an additional anchor ``param0`` captured at
  the end of a burn-in period ``tau``. Every ``phi`` steps, updates the EMA
  with a time-varying weight ``beta_t = (rho + gamma*t)^(-kappa)``. At
  evaluation, swaps in ``alpha_t * (param - param0) + ema`` where
  ``alpha_t = (rho + gamma*t)^(-eta)``.

In both modes the training trajectory is **unaffected**: the live model
parameters are only temporarily swapped during eval (``on_eval_start``) and
restored at ``on_eval_end``.

Usage::

    from megatron.bridge.training.ema_callback import EMACallback
    pretrain(cfg, forward_step, callbacks=[EMACallback(mode="bema", phi=400)])
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal

import torch

from megatron.bridge.training.callbacks import Callback


if TYPE_CHECKING:
    from megatron.bridge.training.callbacks import CallbackContext


logger: logging.Logger = logging.getLogger(__name__)

_EMA_KEY = "ema_params"
_ANCHOR_KEY = "ema_anchor_params"
_BACKUP_KEY = "ema_backup"


class EMACallback(Callback):
    """Maintain an EMA (or BEMA) shadow of model weights for evaluation."""

    def __init__(
        self,
        mode: Literal["ema", "bema"] = "ema",
        decay: float = 0.999,
        kappa: float = 0.2,
        eta: float = 0.2,
        rho: float = 1.0,
        gamma: float = 1.0,
        tau: int = 0,
        phi: int = 400,
        eval_with_avg: bool = True,
    ) -> None:
        if mode not in ("ema", "bema"):
            raise ValueError(f"mode must be 'ema' or 'bema', got {mode!r}")
        self.mode = mode
        self.decay = decay
        self.kappa = kappa
        self.eta = eta
        self.rho = rho
        self.gamma = gamma
        self.tau = tau
        self.phi = phi
        self.eval_with_avg = eval_with_avg

    @staticmethod
    def _iter_named_params(model_chunks):
        for chunk_idx, chunk in enumerate(model_chunks):
            for name, param in chunk.named_parameters():
                if not param.requires_grad:
                    continue
                yield (chunk_idx, name), param

    @torch.no_grad()
    def on_train_start(self, context: "CallbackContext") -> None:
        ema = {k: p.data.detach().clone() for k, p in self._iter_named_params(context.model)}
        context.user_state[_EMA_KEY] = ema
        context.user_state[_BACKUP_KEY] = {}
        if self.mode == "bema":
            context.user_state[_ANCHOR_KEY] = {k: v.clone() for k, v in ema.items()}
        logger.info(
            "EMACallback initialised: mode=%s, decay=%s, kappa=%s, eta=%s, "
            "rho=%s, gamma=%s, tau=%s, phi=%s, eval_with_avg=%s, params=%d",
            self.mode, self.decay, self.kappa, self.eta,
            self.rho, self.gamma, self.tau, self.phi, self.eval_with_avg, len(ema),
        )

    @torch.no_grad()
    def on_train_step_end(self, context: "CallbackContext") -> None:
        if context.skipped_iter:
            return

        ema = context.user_state[_EMA_KEY]
        step = int(context.state.train_state.step)

        if self.mode == "ema":
            for k, p in self._iter_named_params(context.model):
                shadow = ema.get(k)
                if shadow is None:
                    continue
                shadow.mul_(self.decay).add_(p.data, alpha=1.0 - self.decay)
            return

        # mode == "bema"
        anchor = context.user_state[_ANCHOR_KEY]

        if step <= self.tau:
            for k, p in self._iter_named_params(context.model):
                ema[k].copy_(p.data)
                anchor[k].copy_(p.data)
            return

        if (step - self.tau) % self.phi != 0:
            return

        beta_t = (self.rho + self.gamma * step) ** (-self.kappa)
        for k, p in self._iter_named_params(context.model):
            shadow = ema.get(k)
            if shadow is None:
                continue
            shadow.mul_(1.0 - beta_t).add_(p.data, alpha=beta_t)

    @torch.no_grad()
    def on_eval_start(self, context: "CallbackContext") -> None:
        if not self.eval_with_avg:
            return
        ema = context.user_state[_EMA_KEY]
        backup = context.user_state[_BACKUP_KEY]
        step = int(context.state.train_state.step)

        if self.mode == "bema":
            anchor = context.user_state[_ANCHOR_KEY]
            alpha_t = (self.rho + self.gamma * max(step, 1)) ** (-self.eta)

        for k, p in self._iter_named_params(context.model):
            shadow = ema.get(k)
            if shadow is None:
                continue
            backup[k] = p.data.detach().clone()
            if self.mode == "ema":
                p.data.copy_(shadow)
            else:
                p.data.copy_(alpha_t * (p.data - anchor[k]) + shadow)
        logger.info("EMACallback swapped %s weights into model for eval (step=%d)", self.mode.upper(), step)

    @torch.no_grad()
    def on_eval_end(self, context: "CallbackContext") -> None:
        if not self.eval_with_avg:
            return
        backup = context.user_state[_BACKUP_KEY]
        for k, p in self._iter_named_params(context.model):
            saved = backup.get(k)
            if saved is None:
                continue
            p.data.copy_(saved)
        backup.clear()
        logger.info("EMACallback restored training weights after eval")
