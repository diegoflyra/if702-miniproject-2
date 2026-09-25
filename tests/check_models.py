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
# arquiteturas da fase 5: pilha com tamanhos diferentes, residual, dropout recorrente, CNN-LSTM, CNN pura, bidirecional
arqs = [
    {"hidden_sizes": [100, 50]},
    {"hidden_sizes": [64, 64], "residual": True},
    {"num_layers": 2, "residual": True, "bidirectional": True},
    {"recurrent_dropout": 0.3, "num_layers": 2},
    {"conv_layers": 2, "conv_filters": 16, "conv_kernel": 5},
    {"conv_layers": 1, "hidden_sizes": [32, 16], "cell": "gru", "pooling": "attention"},
    {"model": "cnn1d", "conv_filters": 16},
]
for v in arqs:
    m = models.build_model(3, {**base, **v})
    m.train()
    y = m(torch.randn(8, int(base["lookback"]), 3))
    assert y.shape == (8,) and torch.isfinite(y).all(), v
    print(f"OK {v}: {models.count_parameters(m):,} parâmetros")
# sem hiperparâmetros novos, o modelo é exatamente o de antes (fases anteriores continuam reprodutíveis)
assert not models.uses_stack(base)
# dropout recorrente: em avaliação é identidade; em treino, muda a saída
own = models.CustomLSTM(3, 8, 1, 0.0, "tanh", recurrent_dropout=0.5)
x = torch.randn(4, 15, 3)
own.eval()
ref_out = own(x)[0]
own.recurrent_dropout = 0.0
assert torch.allclose(ref_out, own(x)[0]), "recurrent_dropout alterou a saída em modo avaliação"
own.recurrent_dropout = 0.5
own.train()
assert not torch.allclose(ref_out, own(x)[0]), "recurrent_dropout sem efeito no treino"
print("OK dropout recorrente (desligado na avaliação, ativo no treino)")
# funções de erro
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import train  # noqa: E402
pred, yv = torch.tensor([0.5, -0.2, 0.1]), torch.tensor([0.4, 0.3, 0.1])
for name in ["mse", "mae", "huber", "logcosh", "direcional"]:
    val = float(train.make_loss(name, base)(pred, yv))
    assert val >= 0 and float(train.make_loss(name, base)(yv, yv)) < 1e-6, name
    print(f"OK loss {name}: {val:.4f}")
assert float(train.make_loss("direcional", {"loss_lambda": 1.0})(pred, yv)) > float(train.make_loss("mse")(pred, yv))
for kind in sorted(models.NAIVE_MODELS):
    assert models.describe({**base, "model": kind}, 3)["num_parameters"] == 0
print("OK: modelos e referências ingênuas.")
