"""LoRA adapters for SGLD: they start at the MAP function and only they move."""
import torch
import torch.nn as nn

from solar_uq.lora import apply_lora, freeze_norm_stats, lora_parameters
from solar_uq.sgld import SGLD


def _model():
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(12, 32), nn.BatchNorm1d(32), nn.ReLU(),
                         nn.Linear(32, 32), nn.ReLU(), nn.Dropout(0.1), nn.Linear(32, 1))


def test_starts_at_map_and_freezes_base():
    m = _model().eval()
    x = torch.randn(16, 12)
    y0 = m(x).detach()
    info = apply_lora(m, rank=8, n_last=2)
    assert [d["layer"] for d in info] == ["3", "6"]
    assert [d["rank"] for d in info] == [8, 1]           # rank capped for Linear(32, 1)
    assert torch.equal(m(x), y0)                          # B = 0 -> same function
    trainable = {n for n, p in m.named_parameters() if p.requires_grad}
    assert trainable == {"3.lora_A", "3.lora_B", "6.lora_A", "6.lora_B"}
    assert len(lora_parameters(m)) == 4


def test_sgld_moves_only_adapters_and_bn_stats_stay():
    m = _model()
    apply_lora(m, rank=8, n_last=2)
    base = {n: p.detach().clone() for n, p in m.named_parameters() if not p.requires_grad}
    bn_mean = m[1].running_mean.clone()
    opt = SGLD(lora_parameters(m), lr=1e-3, weight_decay=10.0)
    x, y = torch.randn(64, 12), torch.randn(64)
    for _ in range(5):
        m.train()
        freeze_norm_stats(m)
        opt.zero_grad()
        nn.functional.mse_loss(m(x).squeeze(-1), y).backward()
        opt.step()
    for n, p in m.named_parameters():
        if n in base:
            assert torch.equal(p, base[n]), n
    assert torch.equal(m[1].running_mean, bn_mean)
    assert m[6].lora_B.abs().sum() > 0                    # the adapters did move
