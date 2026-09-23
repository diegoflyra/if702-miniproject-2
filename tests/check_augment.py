"""Verifica as transformações de data augmentation (sem treino)."""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import augment  # noqa: E402

torch.manual_seed(0)
B, L, F = 64, 60, 3
x, y = torch.randn(B, L, F), torch.randn(B)
base = {"target": "log_return", "aug_prob": 1.0, "aug_strength": "forte"}

xa, ya = augment.apply(x, y, {**base, "augment": "none"})
assert xa is x and ya is y, "'none' deve devolver o batch intacto"
xa, _ = augment.apply(x, y, {**base, "augment": "jitter", "aug_prob": 0.0})
assert torch.equal(xa, x), "aug_prob = 0 não pode alterar nada"
for m in augment.METHODS + ["jitter+scaling"]:
    xa, ya = augment.apply(x, y, {**base, "augment": m})
    assert xa.shape == x.shape and ya.shape == y.shape and torch.isfinite(xa).all(), m
    print(f"OK {m}: forma {tuple(xa.shape)}, mudou {(xa != x).float().mean():.0%} dos valores")
# permutação só reordena: os mesmos valores em cada janela
xa, _ = augment.apply(x, y, {**base, "augment": "permutation"})
assert torch.allclose(xa.sort(dim=1).values, x.sort(dim=1).values), "permutação deve só reordenar"
# time warping preserva o primeiro e o último dia (o dia da decisão)
xa, _ = augment.apply(x, y, {**base, "augment": "timewarp"})
assert torch.allclose(xa[:, -1], x[:, -1], atol=1e-5) and torch.allclose(xa[:, 0], x[:, 0], atol=1e-5)
# scaling com alvo em retorno escala o alvo pelo mesmo fator; com alvo em preço, não mexe no alvo
torch.manual_seed(1)
xa, ya = augment.apply(x, y, {**base, "augment": "scaling"})
f = xa[:, 0, 0] / x[:, 0, 0]
assert torch.allclose(ya, y * f, atol=1e-4)
_, ya = augment.apply(x, y, {**base, "augment": "scaling", "target": "close"})
assert torch.equal(ya, y)
try:
    augment.parse("rotacao")
    raise AssertionError("estratégia inválida deveria falhar")
except ValueError:
    pass
print("OK: augmentation (identidade, probabilidade, permutação, extremos do time warping, alvo no scaling).")
