"""Modelos: LSTM configurável, uma referência linear e as referências ingênuas (sem treino).

Hiperparâmetros do modelo (ver config/estudo.json → padrao):
  num_layers, hidden_size   camadas LSTM ocultas e nós (unidades) por camada
  fc_neurons                unidades das camadas densas depois da LSTM (ex.: [10]; [] = direto para a saída)
  activation                ativação das camadas densas (relu, tanh, sigmoid, elu, gelu)
  lstm_activation           ativação da célula LSTM (candidata e saída; as portas são sempre sigmoid).
                            tanh usa o nn.LSTM (cuDNN); sigmoid, softsign e relu usam uma célula própria, mais lenta
  weight_init               padrao (PyTorch: U(±1/√h)), uniforme (U(±0,05)), normal (N(0; 0,05)),
                            xavier (Glorot uniforme), glorot_ortogonal (padrão do Keras: Glorot na entrada,
                            ortogonal na recorrência, bias da porta de esquecimento = 1), he (Kaiming uniforme)
  dropout                   depois da última LSTM e entre as camadas densas (nunca na saída)
  rnn_dropout               entre camadas LSTM empilhadas (só vale com num_layers > 1)
  input_dropout             na entrada da primeira LSTM
A camada de saída é linear: o alvo é contínuo (regressão), então sigmoid/softmax na saída não se aplicam.
"""
import math

import torch
from torch import nn

NAIVE_MODELS = {"naive_zero", "naive_mean", "naive_last"}
TRAINED_MODELS = {"lstm", "linear"}
ACTIVATIONS = {"relu": nn.ReLU, "tanh": nn.Tanh, "sigmoid": nn.Sigmoid, "elu": nn.ELU, "gelu": nn.GELU,
               "softsign": nn.Softsign}
WEIGHT_INITS = ["padrao", "uniforme", "normal", "xavier", "glorot_ortogonal", "he"]


class CustomLSTM(nn.Module):
    """LSTM empilhada com ativação configurável na célula: c = f·c + i·act(g); h = o·act(c).

    Mesma parametrização do nn.LSTM (pesos por porta na ordem i, f, g, o). A projeção da entrada é feita de uma
    vez para a janela inteira; só a recorrência percorre o tempo.
    """

    def __init__(self, input_size, hidden_size, num_layers, dropout, activation):
        super().__init__()
        self.hidden_size, self.num_layers = hidden_size, num_layers
        self.act = ACTIVATIONS[activation]()
        self.drop = nn.Dropout(dropout)
        for layer in range(num_layers):
            n_in = input_size if layer == 0 else hidden_size
            k = 1 / math.sqrt(hidden_size)
            self.register_parameter(f"weight_ih_l{layer}", nn.Parameter(torch.empty(4 * hidden_size, n_in).uniform_(-k, k)))
            self.register_parameter(f"weight_hh_l{layer}", nn.Parameter(torch.empty(4 * hidden_size, hidden_size).uniform_(-k, k)))
            self.register_parameter(f"bias_ih_l{layer}", nn.Parameter(torch.empty(4 * hidden_size).uniform_(-k, k)))
            self.register_parameter(f"bias_hh_l{layer}", nn.Parameter(torch.empty(4 * hidden_size).uniform_(-k, k)))

    def forward(self, x):
        B, L, _ = x.shape
        H = self.hidden_size
        seq = x
        for layer in range(self.num_layers):
            if layer > 0:
                seq = self.drop(seq)
            w_ih, w_hh = getattr(self, f"weight_ih_l{layer}"), getattr(self, f"weight_hh_l{layer}")
            xproj = seq @ w_ih.T + getattr(self, f"bias_ih_l{layer}") + getattr(self, f"bias_hh_l{layer}")
            h = x.new_zeros(B, H)
            c = x.new_zeros(B, H)
            outs = []
            for t in range(L):
                gates = xproj[:, t] + h @ w_hh.T
                i, f, g, o = gates.chunk(4, dim=1)
                c = torch.sigmoid(f) * c + torch.sigmoid(i) * self.act(g)
                h = torch.sigmoid(o) * self.act(c)
                outs.append(h)
            seq = torch.stack(outs, dim=1)
        return seq, None


class RecurrentRegressor(nn.Module):
    """Janela [B, L, F] → dropout de entrada → LSTM/GRU empilhada → resumo da sequência → dropout → densas → saída linear."""

    def __init__(self, n_features, p):
        super().__init__()
        hidden, layers = int(p["hidden_size"]), int(p["num_layers"])
        rnn_drop = float(p["rnn_dropout"]) if layers > 1 else 0.0
        self.pooling = p["pooling"]
        self.input_dropout = nn.Dropout(p["input_dropout"])
        if p["cell"] == "lstm" and p.get("lstm_activation", "tanh") != "tanh":
            if p["bidirectional"]:
                raise ValueError("lstm_activation ≠ tanh não suporta bidirectional")
            self.rnn = CustomLSTM(n_features, hidden, layers, rnn_drop, p["lstm_activation"])
        else:
            rnn_cls = {"lstm": nn.LSTM, "gru": nn.GRU}[p["cell"]]
            self.rnn = rnn_cls(n_features, hidden, num_layers=layers, batch_first=True, dropout=rnn_drop,
                               bidirectional=bool(p["bidirectional"]))
        out = hidden * (2 if p["bidirectional"] else 1)
        self.attn = nn.Linear(out, 1) if self.pooling == "attention" else None
        self.norm = nn.LayerNorm(out) if p["layer_norm"] else nn.Identity()
        head = []
        for width in p["fc_neurons"]:
            head += [nn.Dropout(p["dropout"]), nn.Linear(out, int(width)), ACTIVATIONS[p["activation"]]()]
            out = int(width)
        head += [nn.Dropout(p["dropout"]), nn.Linear(out, 1)]
        self.head = nn.Sequential(*head)
        init_weights(self, p.get("weight_init", "padrao"), p.get("activation", "relu"))

    def forward(self, x):
        seq, _ = self.rnn(self.input_dropout(x))
        if self.pooling == "last":
            z = seq[:, -1]
        elif self.pooling == "mean":
            z = seq.mean(1)
        elif self.pooling == "attention":
            z = (torch.softmax(self.attn(seq), dim=1) * seq).sum(1)
        else:
            raise ValueError(f"pooling desconhecido: {self.pooling}")
        return self.head(self.norm(z)).squeeze(-1)


class LinearRegressor(nn.Module):
    """Referência não recorrente: regressão linear sobre a janela achatada (mesmas entradas do LSTM)."""

    def __init__(self, n_features, p):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Dropout(p["dropout"]), nn.Linear(n_features * int(p["lookback"]), 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


@torch.no_grad()
def init_weights(model, scheme, dense_activation="relu"):
    """Aplica o esquema de inicialização às matrizes recorrentes e densas; `padrao` mantém o do PyTorch."""
    if scheme == "padrao":
        return
    if scheme not in WEIGHT_INITS:
        raise ValueError(f"weight_init desconhecido: {scheme} (opções: {WEIGHT_INITS})")
    is_lstm = isinstance(getattr(model, "rnn", None), (nn.LSTM, CustomLSTM))
    gain_dense = nn.init.calculate_gain(dense_activation if dense_activation in ("relu", "tanh", "sigmoid") else "relu")
    for name, w in model.named_parameters():
        is_rnn = name.startswith("rnn.")
        if name.endswith("bias") or "bias_" in name:
            nn.init.zeros_(w)
            if scheme == "glorot_ortogonal" and is_lstm and is_rnn and "bias_ih" in name:
                h = w.numel() // 4
                w[h:2 * h] = 1.0  # porta de esquecimento começa aberta (unit_forget_bias do Keras)
            continue
        if w.dim() < 2:
            continue  # LayerNorm
        if scheme == "uniforme":
            nn.init.uniform_(w, -0.05, 0.05)
        elif scheme == "normal":
            nn.init.normal_(w, 0.0, 0.05)
        elif scheme == "xavier":
            nn.init.xavier_uniform_(w, gain=1.0 if is_rnn else gain_dense)
        elif scheme == "he":
            nn.init.kaiming_uniform_(w, nonlinearity="relu")
        elif scheme == "glorot_ortogonal":
            if is_rnn and "weight_hh" in name:
                for chunk in w.chunk(w.shape[0] // w.shape[1], dim=0):  # uma matriz ortogonal por porta
                    nn.init.orthogonal_(chunk)
            else:
                nn.init.xavier_uniform_(w)


def build_model(n_features, p):
    if p["model"] == "lstm":
        return RecurrentRegressor(n_features, p)
    if p["model"] == "linear":
        return LinearRegressor(n_features, p)
    raise ValueError(f"modelo treinável desconhecido: {p['model']}")


def count_parameters(model):
    return sum(t.numel() for t in model.parameters() if t.requires_grad)


def describe(p, n_features):
    """Arquitetura resolvida (vai para parametros.json) e número de parâmetros, sem precisar de dados."""
    if p["model"] in NAIVE_MODELS:
        return {"tipo": p["model"], "num_parameters": 0}
    model = build_model(n_features, p)
    return {"tipo": p["model"], "num_parameters": count_parameters(model), "modulos": str(model)}
