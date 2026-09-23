"""Constrói variações do modelo e faz um forward em dados aleatórios (sem treino)."""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import common  # noqa: E402
import models  # noqa: E402

base = common.default_params()
variants = [
    {},
    {"num_layers": 2, "rnn_dropout": 0.2, "bidirectional": True},
    {"pooling": "mean", "fc_neurons": [32], "layer_norm": True},
    {"pooling": "attention", "cell": "gru", "input_dropout": 0.1},
    {"model": "linear"},
]
for v in variants:
    p = {**base, **v}
    m = models.build_model(3, p)
    y = m(torch.randn(8, int(p["lookback"]), 3))
    assert y.shape == (8,), y.shape
    print(f"OK {v or 'padrão'}: {models.count_parameters(m):,} parâmetros")
# Célula própria: com tanh e os mesmos pesos, precisa reproduzir o nn.LSTM (valida a implementação)
torch.manual_seed(0)
ref = torch.nn.LSTM(3, 8, num_layers=2, batch_first=True)
own = models.CustomLSTM(3, 8, 2, 0.0, "tanh")
own.load_state_dict(ref.state_dict())
x = torch.randn(4, 15, 3)
diff = (ref(x)[0] - own(x)[0]).abs().max().item()
assert diff < 1e-5, f"CustomLSTM difere do nn.LSTM: {diff}"
print(f"OK célula própria = nn.LSTM (dif. máx. {diff:.1e})")
for act in ["sigmoid", "softsign", "relu"]:
    m = models.build_model(3, {**base, "lstm_activation": act, "num_layers": 2})
    assert m(torch.randn(8, int(base["lookback"]), 3)).shape == (8,)
    print(f"OK lstm_activation={act}")
for scheme in models.WEIGHT_INITS:
    m = models.build_model(3, {**base, "weight_init": scheme, "num_layers": 2})
    assert torch.isfinite(m(torch.randn(8, int(base["lookback"]), 3))).all()
    print(f"OK weight_init={scheme}")
for kind in sorted(models.NAIVE_MODELS):
    assert models.describe({**base, "model": kind}, 3)["num_parameters"] == 0
print("OK: modelos e referências ingênuas.")
