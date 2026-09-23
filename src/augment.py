"""Data augmentation de séries temporais, aplicada só nos batches de treino, direto no device.

Validação, teste e as métricas de treino (medidas em modo avaliação) usam as janelas originais.
Cada amostra do batch é aumentada com probabilidade `aug_prob`; as janelas estão normalizadas (z-score), então as
intensidades abaixo são em desvios padrão.

Estratégias (`augment`; combine com "+", ex.: "jitter+scaling"):
  jitter       ruído gaussiano branco somado a cada ponto da janela
  scaling      multiplica a janela inteira por um fator aleatório ~ N(1, σ) (simula regimes de amplitude/volatilidade);
               com alvo em retorno, o alvo é escalado pelo mesmo fator, para manter a relação entrada → saída
  magwarp      magnitude warping: multiplica por uma curva suave aleatória (nós ~ N(1, σ), interpolados);
               o alvo acompanha o valor da curva no último passo
  timewarp     time warping: acelera/desacelera trechos da janela (velocidades ~ N(1, σ) suavizadas) e reamostra;
               o último ponto (o dia da decisão) é preservado
  permutation  divide a janela em n segmentos e embaralha a ordem deles (testa se o modelo usa a ordem temporal)
  slicing      window slicing: recorta um trecho contíguo (fração da janela) e o reamostra para o tamanho original
Intensidade (`aug_strength`): "fraca" ou "forte", com os valores de STRENGTH.
"""
import torch
import torch.nn.functional as F

METHODS = ["jitter", "scaling", "magwarp", "timewarp", "permutation", "slicing"]
STRENGTH = {
    "jitter": {"fraca": 0.03, "forte": 0.1},       # σ do ruído
    "scaling": {"fraca": 0.1, "forte": 0.2},       # σ do fator
    "magwarp": {"fraca": 0.1, "forte": 0.2},       # σ dos nós da curva
    "timewarp": {"fraca": 0.1, "forte": 0.2},      # σ das velocidades
    "permutation": {"fraca": 3, "forte": 6},       # nº de segmentos
    "slicing": {"fraca": 0.9, "forte": 0.7},       # fração da janela mantida
}
KNOTS = 4


def parse(spec):
    if not spec or spec == "none":
        return []
    methods = spec.split("+")
    unknown = [m for m in methods if m not in METHODS]
    if unknown:
        raise ValueError(f"augment desconhecido: {unknown} (opções: {METHODS}, combinadas com '+')")
    return methods


def _smooth_curve(B, L, sigma, device, knots=KNOTS):
    """Curva suave [B, L] com nós ~ N(1, σ) interpolados linearmente."""
    k = 1 + sigma * torch.randn(B, 1, knots + 2, device=device)
    return F.interpolate(k, size=L, mode="linear", align_corners=True)[:, 0]


def _resample(x, pos):
    """Reamostra x[B, L, F] nas posições fracionárias pos[B, L] (interpolação linear)."""
    L = x.shape[1]
    pos = pos.clamp(0, L - 1)
    i0 = pos.floor().long()
    i1 = (i0 + 1).clamp(max=L - 1)
    w = (pos - i0.float()).unsqueeze(-1)
    idx = lambda i: i.unsqueeze(-1).expand(-1, -1, x.shape[2])  # noqa: E731
    return x.gather(1, idx(i0)) * (1 - w) + x.gather(1, idx(i1)) * w


def jitter(x, y, s, scale_y):
    return x + s * torch.randn_like(x), y


def scaling(x, y, s, scale_y):
    f = 1 + s * torch.randn(x.shape[0], 1, 1, device=x.device)
    return x * f, (y * f[:, 0, 0] if scale_y else y)


def magwarp(x, y, s, scale_y):
    curve = _smooth_curve(x.shape[0], x.shape[1], s, x.device)
    return x * curve.unsqueeze(-1), (y * curve[:, -1] if scale_y else y)


def timewarp(x, y, s, scale_y):
    B, L, _ = x.shape
    speed = _smooth_curve(B, L, s, x.device).clamp(min=0.1)
    t = torch.cumsum(speed, dim=1)
    t = (t - t[:, :1]) / (t[:, -1:] - t[:, :1]) * (L - 1)  # começa em 0 e termina em L−1 (dia da decisão)
    return _resample(x, t), y


def permutation(x, y, n, scale_y):
    B, L, _ = x.shape
    n = max(1, min(int(n), L))
    seg = L // n
    perm = torch.argsort(torch.rand(B, n, device=x.device), dim=1)
    t = torch.arange(n * seg, device=x.device)
    new = perm.gather(1, (t // seg).expand(B, -1)) * seg + t % seg
    tail = torch.arange(n * seg, L, device=x.device).expand(B, -1)  # resto da divisão fica no fim
    idx = torch.cat([new, tail], dim=1).unsqueeze(-1).expand(-1, -1, x.shape[2])
    return x.gather(1, idx), y


def slicing(x, y, ratio, scale_y):
    B, L, _ = x.shape
    c = max(2, int(round(ratio * L)))
    start = torch.randint(0, L - c + 1, (B, 1), device=x.device).float()
    pos = start + torch.linspace(0, c - 1, L, device=x.device).unsqueeze(0)
    return _resample(x, pos), y


FUNCS = {"jitter": jitter, "scaling": scaling, "magwarp": magwarp, "timewarp": timewarp,
         "permutation": permutation, "slicing": slicing}


def apply(x, y, p):
    """Aumenta o batch (x[B, L, F], y[B]) conforme os hiperparâmetros `augment`, `aug_strength` e `aug_prob`."""
    methods = parse(p.get("augment", "none"))
    if not methods:
        return x, y
    strength = p.get("aug_strength", "fraca")
    scale_y = p.get("target", "log_return") == "log_return"
    xa, ya = x, y
    for m in methods:
        xa, ya = FUNCS[m](xa, ya, STRENGTH[m][strength], scale_y)
    mask = torch.rand(x.shape[0], device=x.device) < float(p.get("aug_prob", 0.5))
    return torch.where(mask[:, None, None], xa, x), torch.where(mask, ya, y)
