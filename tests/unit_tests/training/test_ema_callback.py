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

"""Unit tests for EMACallback (vanilla EMA + BEMA modes)."""

from types import SimpleNamespace
from unittest.mock import Mock

import torch
from torch import nn

from megatron.bridge.training.callbacks import CallbackContext
from megatron.bridge.training.ema_callback import EMACallback


def _make_model(weight: torch.Tensor) -> nn.Linear:
    m = nn.Linear(weight.shape[1], weight.shape[0], bias=False)
    with torch.no_grad():
        m.weight.copy_(weight)
    return m


def _make_context(model: nn.Module, step: int = 0, skipped_iter: bool = False) -> CallbackContext:
    """Build a CallbackContext with a fake GlobalState whose train_state.step is set."""
    state = SimpleNamespace(train_state=SimpleNamespace(step=step))
    return CallbackContext(
        state=state,
        model=[model],
        user_state={},
        skipped_iter=skipped_iter,
    )


def test_ema_update_math():
    """ema = decay*ema + (1-decay)*param after one step."""
    W0 = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    W1 = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
    model = _make_model(W0)
    cb = EMACallback(mode="ema", decay=0.9)

    ctx = _make_context(model, step=0)
    cb.on_train_start(ctx)

    with torch.no_grad():
        model.weight.copy_(W1)
    ctx2 = CallbackContext(
        state=SimpleNamespace(train_state=SimpleNamespace(step=1)),
        model=[model],
        user_state=ctx.user_state,
        skipped_iter=False,
    )
    cb.on_train_step_end(ctx2)

    expected = 0.9 * W0 + 0.1 * W1
    actual = ctx.user_state["ema_params"][(0, "weight")]
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)


def test_eval_swap_and_restore():
    """on_eval_start swaps in EMA; on_eval_end restores live weights bit-exact."""
    W_train = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    sentinel = torch.full_like(W_train, 7.0)
    model = _make_model(W_train)
    cb = EMACallback(mode="ema", decay=0.9)

    ctx = _make_context(model, step=10)
    cb.on_train_start(ctx)
    # Force the shadow to a known sentinel different from live weights.
    ctx.user_state["ema_params"][(0, "weight")].copy_(sentinel)

    cb.on_eval_start(ctx)
    assert torch.equal(model.weight.data, sentinel), "swap into model failed"

    cb.on_eval_end(ctx)
    assert torch.equal(model.weight.data, W_train), "restore was not bit-exact"
    assert ctx.user_state["ema_backup"] == {}, "backup not cleared after restore"


def test_skipped_iter_no_update():
    """When skipped_iter=True, the EMA shadow must not change."""
    W0 = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    W1 = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
    model = _make_model(W0)
    cb = EMACallback(mode="ema", decay=0.9)

    ctx = _make_context(model, step=0)
    cb.on_train_start(ctx)
    shadow_before = ctx.user_state["ema_params"][(0, "weight")].clone()

    with torch.no_grad():
        model.weight.copy_(W1)
    ctx2 = CallbackContext(
        state=SimpleNamespace(train_state=SimpleNamespace(step=1)),
        model=[model],
        user_state=ctx.user_state,
        skipped_iter=True,
    )
    cb.on_train_step_end(ctx2)

    assert torch.equal(
        ctx.user_state["ema_params"][(0, "weight")], shadow_before
    ), "EMA was updated on a skipped iter"


def test_bema_update_and_swap_math():
    """BEMA: β_1 = α_1 = 1/√2 with kappa=eta=0.5, rho=gamma=1, phi=1, tau=0."""
    W0 = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    W1 = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
    model = _make_model(W0)
    cb = EMACallback(
        mode="bema", kappa=0.5, eta=0.5, rho=1.0, gamma=1.0, tau=0, phi=1
    )

    ctx = _make_context(model, step=0)
    cb.on_train_start(ctx)
    # Sanity: anchor and ema both initialised to W0.
    assert torch.equal(ctx.user_state["ema_anchor_params"][(0, "weight")], W0)
    assert torch.equal(ctx.user_state["ema_params"][(0, "weight")], W0)

    # Step to t=1 with new weight.
    with torch.no_grad():
        model.weight.copy_(W1)
    ctx2 = CallbackContext(
        state=SimpleNamespace(train_state=SimpleNamespace(step=1)),
        model=[model],
        user_state=ctx.user_state,
        skipped_iter=False,
    )
    cb.on_train_step_end(ctx2)

    beta_1 = (1.0 + 1.0 * 1) ** (-0.5)  # 1/√2
    expected_ema = (1.0 - beta_1) * W0 + beta_1 * W1
    torch.testing.assert_close(
        ctx.user_state["ema_params"][(0, "weight")], expected_ema, rtol=1e-6, atol=1e-6
    )

    # Now eval-swap. Anchor is still W0; live model weight is W1.
    cb.on_eval_start(ctx2)
    alpha_1 = (1.0 + 1.0 * 1) ** (-0.5)  # 1/√2
    expected_swap = alpha_1 * (W1 - W0) + expected_ema
    torch.testing.assert_close(model.weight.data, expected_swap, rtol=1e-6, atol=1e-6)

    cb.on_eval_end(ctx2)
    assert torch.equal(model.weight.data, W1), "BEMA restore not bit-exact"


def test_bema_phi_skips_intermediate_steps():
    """With phi=5, the EMA must not update on steps that are not multiples of 5."""
    W0 = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    W1 = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
    model = _make_model(W0)
    cb = EMACallback(mode="bema", kappa=0.5, rho=1.0, gamma=1.0, tau=0, phi=5)

    ctx = _make_context(model, step=0)
    cb.on_train_start(ctx)
    snapshot = ctx.user_state["ema_params"][(0, "weight")].clone()

    with torch.no_grad():
        model.weight.copy_(W1)
    for step in (1, 2, 3, 4):
        ctx_i = CallbackContext(
            state=SimpleNamespace(train_state=SimpleNamespace(step=step)),
            model=[model],
            user_state=ctx.user_state,
            skipped_iter=False,
        )
        cb.on_train_step_end(ctx_i)
        assert torch.equal(
            ctx.user_state["ema_params"][(0, "weight")], snapshot
        ), f"BEMA updated on non-phi step {step}"

    # Step 5 should update.
    ctx5 = CallbackContext(
        state=SimpleNamespace(train_state=SimpleNamespace(step=5)),
        model=[model],
        user_state=ctx.user_state,
        skipped_iter=False,
    )
    cb.on_train_step_end(ctx5)
    assert not torch.equal(
        ctx.user_state["ema_params"][(0, "weight")], snapshot
    ), "BEMA did not update on a phi-aligned step"
