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
  recurrent_dropout         dropout no estado oculto entre passos de tempo (o `recurrent_dropout` do Keras; máscara
                            fixa por sequência, "variacional"); usa a célula própria (também bidirecional)
  hidden_sizes              pilha com tamanhos por camada (ex.: [100, 50], como no paper de Wu et al.); substitui
                            hidden_size × num_layers
  residual                  soma a entrada de cada camada recorrente à sua saída (projeção linear se os tamanhos diferem)
  conv_layers, conv_filters, conv_kernel
                            convoluções 1D antes da LSTM (CNN-LSTM); 0 = sem convolução
Modelos: lstm (acima), cnn1d (só convoluções + resumo no tempo, referência do paper), linear, ingênuos e, em train.py,
os de scikit-learn/XGBoost (svr, random_forest, xgboost), treinados sobre a janela achatada.
A camada de saída é linear: o alvo é contínuo (regressão), então sigmoid/softmax na saída não se aplicam.
"""
import math

import torch
from torch import nn

NAIVE_MODELS = {"naive_zero", "naive_mean", "naive_last"}
TRAINED_MODELS = {"lstm", "linear", "cnn1d"}
SKLEARN_MODELS = {"svr", "random_forest", "xgboost"}
ACTIVATIONS = {"relu": nn.ReLU, "tanh": nn.Tanh, "sigmoid": nn.Sigmoid, "elu": nn.ELU, "gelu": nn.GELU,
               "softsign": nn.Softsign}
WEIGHT_INITS = ["padrao", "uniforme", "normal", "xavier", "glorot_ortogonal", "he"]


class CustomLSTM(nn.Module):
    """LSTM empilhada com ativação configurável na célula: c = f·c + i·act(g); h = o·act(c).

    Mesma parametrização do nn.LSTM (pesos por porta na ordem i, f, g, o). A projeção da entrada é feita de uma
    vez para a janela inteira; só a recorrência percorre o tempo.
    """

    def __init__(self, input_size, hidden_size, num_layers, dropout, activation, recurrent_dropout=0.0):
        super().__init__()
        self.hidden_size, self.num_layers = hidden_size, num_layers
        self.recurrent_dropout = float(recurrent_dropout)
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
            mask = None
            if self.training and self.recurrent_dropout > 0:  # mesma máscara em todos os passos (dropout variacional)
                keep = 1 - self.recurrent_dropout
                mask = torch.bernoulli(x.new_full((B, H), keep)) / keep
            outs = []
            for t in range(L):
                gates = xproj[:, t] + (h if mask is None else h * mask) @ w_hh.T
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
        self.conv = _conv_front(n_features, p)
        n_in = int(p.get("conv_filters", 32)) if self.conv is not None else n_features
        self.stacked = uses_stack(p)
        if self.stacked:
            self.rnn, out = _build_stack(n_in, p)
        elif p["cell"] == "lstm" and p.get("lstm_activation", "tanh") != "tanh":
            self.rnn = CustomLSTM(n_in, hidden, layers, rnn_drop, p["lstm_activation"])
        else:
            rnn_cls = {"lstm": nn.LSTM, "gru": nn.GRU}[p["cell"]]
            self.rnn = rnn_cls(n_in, hidden, num_layers=layers, batch_first=True, dropout=rnn_drop,
                               bidirectional=bool(p["bidirectional"]))
        if not self.stacked:
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
        x = self.input_dropout(x)
        if self.conv is not None:
            x = self.conv(x.transpose(1, 2)).transpose(1, 2)
        seq = self.rnn(x) if self.stacked else self.rnn(x)[0]
        if self.pooling == "last":
            z = seq[:, -1]
        elif self.pooling == "mean":
            z = seq.mean(1)
        elif self.pooling == "attention":
            z = (torch.softmax(self.attn(seq), dim=1) * seq).sum(1)
        else:
            raise ValueError(f"pooling desconhecido: {self.pooling}")
        return self.head(self.norm(z)).squeeze(-1)


def uses_stack(p):
    """O caminho novo (camada a camada) só é usado quando algum hiperparâmetro novo está ativo, ou quando a célula
    própria precisa ser bidirecional; o resto constrói exatamente o mesmo modelo de antes (fases anteriores
    continuam reprodutíveis)."""
    custom_bidir = (p.get("cell", "lstm") == "lstm" and bool(p.get("bidirectional"))
                    and p.get("lstm_activation", "tanh") != "tanh")
    return (bool(p.get("hidden_sizes")) or bool(p.get("residual")) or float(p.get("recurrent_dropout", 0.0)) > 0
            or custom_bidir)


class BiCustomLSTM(nn.Module):
    """Célula própria bidirecional: uma LSTM lê a janela em ordem, outra de trás para frente (pesos separados);
    as saídas de cada passo são concatenadas, como no nn.LSTM(bidirectional=True)."""

    def __init__(self, input_size, hidden_size, activation, recurrent_dropout):
        super().__init__()
        self.fwd = CustomLSTM(input_size, hidden_size, 1, 0.0, activation, recurrent_dropout)
        self.bwd = CustomLSTM(input_size, hidden_size, 1, 0.0, activation, recurrent_dropout)

    def forward(self, x):
        out_f = self.fwd(x)[0]
        out_b = self.bwd(torch.flip(x, dims=[1]))[0]
        return torch.cat([out_f, torch.flip(out_b, dims=[1])], dim=2), None


def _conv_front(n_features, p):
    n = int(p.get("conv_layers", 0))
    if n <= 0:
        return None
    k, f = int(p.get("conv_kernel", 3)), int(p.get("conv_filters", 32))
    mods, c_in = [], n_features
    for _ in range(n):
        mods += [nn.Conv1d(c_in, f, k, padding=k // 2), nn.ReLU()]
        c_in = f
    return nn.Sequential(*mods)


class _Stack(nn.Module):
    """Camadas recorrentes empilhadas uma a uma: tamanhos por camada, dropout entre elas e conexões residuais."""

    def __init__(self, layers, projections, dropout, residual):
        super().__init__()
        self.layers, self.proj = nn.ModuleList(layers), nn.ModuleList(projections)
        self.drop, self.residual = nn.Dropout(dropout), residual

    def forward(self, x):
        for i, (layer, proj) in enumerate(zip(self.layers, self.proj)):
            inp = x if i == 0 else self.drop(x)
            out = layer(inp)[0]
            x = out + proj(inp) if self.residual else out
        return x


def _build_stack(n_in, p):
    sizes = [int(h) for h in p.get("hidden_sizes") or [p["hidden_size"]] * int(p["num_layers"])]
    bidir = bool(p["bidirectional"])
    custom = p["cell"] == "lstm" and (p.get("lstm_activation", "tanh") != "tanh" or float(p.get("recurrent_dropout", 0)) > 0)
    layers, projs, d = [], [], n_in
    for h in sizes:
        if custom and bidir:
            layers.append(BiCustomLSTM(d, h, p.get("lstm_activation", "tanh"), p.get("recurrent_dropout", 0.0)))
        elif custom:
            layers.append(CustomLSTM(d, h, 1, 0.0, p.get("lstm_activation", "tanh"), p.get("recurrent_dropout", 0.0)))
        else:
            layers.append({"lstm": nn.LSTM, "gru": nn.GRU}[p["cell"]](d, h, batch_first=True, bidirectional=bidir))
        out = h * (2 if bidir else 1)
        projs.append(nn.Identity() if out == d else nn.Linear(d, out, bias=False))
        d = out
    return _Stack(layers, projs, float(p["rnn_dropout"]) if len(sizes) > 1 else 0.0, bool(p.get("residual"))), d


class CNNRegressor(nn.Module):
    """Referência convolucional (paper): convoluções 1D → resumo no tempo (média) → densas → saída linear."""

    def __init__(self, n_features, p):
        super().__init__()
        q = {**p, "conv_layers": max(1, int(p.get("conv_layers", 0)) or 2)}
        self.conv = _conv_front(n_features, q)
        out, head = int(q.get("conv_filters", 32)), []
        for width in p["fc_neurons"]:
            head += [nn.Dropout(p["dropout"]), nn.Linear(out, int(width)), ACTIVATIONS[p["activation"]]()]
            out = int(width)
        head += [nn.Dropout(p["dropout"]), nn.Linear(out, 1)]
        self.head = nn.Sequential(*head)

    def forward(self, x):
        return self.head(self.conv(x.transpose(1, 2)).mean(2)).squeeze(-1)


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
    is_lstm = isinstance(getattr(model, "rnn", None), (nn.LSTM, CustomLSTM)) or (
        isinstance(getattr(model, "rnn", None), _Stack)
        and isinstance(model.rnn.layers[0], (nn.LSTM, CustomLSTM, BiCustomLSTM)))
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
    if p["model"] == "cnn1d":
        return CNNRegressor(n_features, p)
    raise ValueError(f"modelo treinável desconhecido: {p['model']}")


def count_parameters(model):
    return sum(t.numel() for t in model.parameters() if t.requires_grad)


def describe(p, n_features):
    """Arquitetura resolvida (vai para parametros.json) e número de parâmetros, sem precisar de dados."""
    if p["model"] in NAIVE_MODELS or p["model"] in SKLEARN_MODELS:
        return {"tipo": p["model"], "num_parameters": 0}
    model = build_model(n_features, p)
    return {"tipo": p["model"], "num_parameters": count_parameters(model), "modulos": str(model)}
