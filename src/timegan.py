"""TimeGAN (Yoon, Jarrett & van der Schaar, NeurIPS 2019) para gerar janelas sintéticas de treino.

Treinado DENTRO de cada fold, só com as janelas de treino daquele fold: nada da validação ou do teste entra no gerador.
Cada sequência real é a janela de entrada do LSTM com o alvo junto, como um canal a mais:
    S_t = [features(t−L+1 … t), y(t−L+1 … t)]      (L passos, F + 1 canais, no espaço normalizado do fold)
O gerador aprende a distribuição conjunta dessas sequências; cada sequência sintética vira um exemplo de treino
(janela = canais de features; alvo = canal y no último passo).

Componentes (GRUs): embedder e recovery (autoencoder no espaço latente), generator e supervisor (dinâmica temporal
no espaço latente) e discriminator. Treino em três etapas: autoencoder, supervisão e treino conjunto adversarial.

Diagnósticos (a pergunta é se o gerador aprende as propriedades do BTC ou só o ruído):
  discriminativo   um classificador GRU tenta separar janelas reais de sintéticas; score = |acurácia − 0,5|
                   (0 = indistinguíveis; 0,5 = trivialmente distinguíveis)
  fatos estilizados, no canal de retorno: curtose (caudas pesadas), autocorrelação do retorno (≈ 0 em mercado líquido)
                   e autocorrelação do |retorno| (aglomerados de volatilidade), real × sintético
  TSTR             "train on synthetic, test on real": configuração com timegan = "so_sintetico" (treino só com
                   sintético), avaliada na validação real como qualquer outra
"""
import hashlib
import json
import os
import time

import numpy as np
import torch
from torch import nn

import common


class _Net(nn.Module):
    def __init__(self, n_in, hidden, n_out, layers, act=True):
        super().__init__()
        self.rnn = nn.GRU(n_in, hidden, num_layers=layers, batch_first=True)
        self.fc = nn.Linear(hidden, n_out)
        self.act = act

    def forward(self, x):
        out = self.fc(self.rnn(x)[0])
        return torch.sigmoid(out) if self.act else out


class TimeGAN(nn.Module):
    def __init__(self, n_feat, hidden=24, layers=3):
        super().__init__()
        self.embedder = _Net(n_feat, hidden, hidden, layers)
        self.recovery = _Net(hidden, hidden, n_feat, layers)
        self.generator = _Net(n_feat, hidden, hidden, layers)
        self.supervisor = _Net(hidden, hidden, hidden, max(1, layers - 1))
        self.discriminator = _Net(hidden, hidden, 1, layers, act=False)
        self.n_feat = n_feat

    def generate(self, n, T, device, batch=1024):
        self.eval()
        out = []
        with torch.no_grad():
            for i in range(0, n, batch):
                z = torch.rand(min(batch, n - i), T, self.n_feat, device=device)
                out.append(self.recovery(self.supervisor(self.generator(z))))
        return torch.cat(out)


def train_timegan(real, iters=1000, hidden=24, layers=3, batch=128, lr=1e-3, gamma=1.0, seed=0, log_every=0):
    """real: tensor [N, T, C] em [0, 1]. Devolve o modelo treinado."""
    torch.manual_seed(seed)
    device = real.device
    N, T, C = real.shape
    m = TimeGAN(C, hidden, layers).to(device)
    bce, mse = nn.BCEWithLogitsLoss(), nn.MSELoss()
    opt_e = torch.optim.Adam(list(m.embedder.parameters()) + list(m.recovery.parameters()), lr=lr)
    opt_g = torch.optim.Adam(list(m.generator.parameters()) + list(m.supervisor.parameters()), lr=lr)
    opt_d = torch.optim.Adam(m.discriminator.parameters(), lr=lr)
    sample = lambda: real[torch.randint(0, N, (min(batch, N),), device=device)]  # noqa: E731
    noise = lambda k: torch.rand(k, T, C, device=device)  # noqa: E731
    m.train()
    # 1) autoencoder: X → H → X̂
    for _ in range(iters):
        x = sample()
        loss = mse(m.recovery(m.embedder(x)), x)
        opt_e.zero_grad(); loss.backward(); opt_e.step()
    # 2) supervisão: o supervisor prevê o próximo estado latente (aprende a dinâmica temporal)
    for _ in range(iters):
        h = m.embedder(sample()).detach()
        loss = mse(m.supervisor(h)[:, :-1], h[:, 1:])
        opt_g.zero_grad(); loss.backward(); opt_g.step()
    # 3) treino conjunto
    for it in range(iters):
        for _ in range(2):  # gerador (2 passos por passo do discriminador, como no artigo)
            x = sample()
            z = noise(len(x))
            e_hat = m.generator(z)
            h_hat = m.supervisor(e_hat)
            x_hat = m.recovery(h_hat)
            h = m.embedder(x)
            y_fake, y_fake_e = m.discriminator(h_hat), m.discriminator(e_hat)
            g_adv = bce(y_fake, torch.ones_like(y_fake)) + gamma * bce(y_fake_e, torch.ones_like(y_fake_e))
            g_sup = mse(m.supervisor(h)[:, :-1], h[:, 1:])
            g_mom = (torch.abs(x_hat.std(0) - x.std(0)).mean() + torch.abs(x_hat.mean(0) - x.mean(0)).mean())
            loss_g = g_adv + 100 * torch.sqrt(g_sup + 1e-8) + 100 * g_mom
            opt_g.zero_grad(); loss_g.backward(); opt_g.step()
            # embedder/recovery continuam aprendendo a reconstruir (com um pouco de supervisão)
            h = m.embedder(x)
            loss_e = 10 * torch.sqrt(mse(m.recovery(h), x) + 1e-8) + 0.1 * mse(m.supervisor(h)[:, :-1], h[:, 1:])
            opt_e.zero_grad(); loss_e.backward(); opt_e.step()
        x = sample()
        z = noise(len(x))
        with torch.no_grad():
            h, e_hat = m.embedder(x), m.generator(z)
            h_hat = m.supervisor(e_hat)
        y_real, y_fake, y_fake_e = m.discriminator(h), m.discriminator(h_hat), m.discriminator(e_hat)
        loss_d = (bce(y_real, torch.ones_like(y_real)) + bce(y_fake, torch.zeros_like(y_fake))
                  + gamma * bce(y_fake_e, torch.zeros_like(y_fake_e)))
        if loss_d.item() > 0.15:  # só treina o discriminador quando ele não está vencendo com folga
            opt_d.zero_grad(); loss_d.backward(); opt_d.step()
        if log_every and it % log_every == 0:
            print(f"    timegan it {it}: D {loss_d.item():.3f} | G {loss_g.item():.3f} | E {loss_e.item():.3f}", flush=True)
    return m


def _acf(x, lag=1):
    x = x - x.mean()
    d = (x * x).sum()
    return float((x[lag:] * x[:-lag]).sum() / d) if d > 0 else 0.0


def stylized_facts(seqs):
    """seqs [N, T] (canal de retorno): fatos estilizados médios por sequência."""
    s = np.asarray(seqs, float)
    z = (s - s.mean(1, keepdims=True)) / (s.std(1, keepdims=True) + 1e-9)
    return {"curtose": float(np.mean((z ** 4).mean(1) - 3)),
            "acf_retorno": float(np.mean([_acf(r) for r in s])),
            "acf_abs_retorno": float(np.mean([_acf(np.abs(r)) for r in s])),
            "desvio": float(s.std())}


def discriminative_score(real, synth, epochs=5, seed=0):
    """|acurácia − 0,5| de um classificador GRU pequeno (real × sintético), em dados separados do treino dele."""
    torch.manual_seed(seed)
    device = real.device
    n = min(len(real), len(synth))
    x = torch.cat([real[torch.randperm(len(real), device=device)[:n]], synth[:n]])
    y = torch.cat([torch.ones(n, device=device), torch.zeros(n, device=device)])
    perm = torch.randperm(2 * n, device=device)
    x, y = x[perm], y[perm]
    cut = int(0.8 * len(x))
    clf = _Net(x.shape[2], 16, 1, 1, act=False).to(device)
    opt = torch.optim.Adam(clf.parameters(), lr=1e-3)
    bce = nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        for i in range(0, cut, 128):
            logit = clf(x[i:min(i + 128, cut)])[:, -1, 0]
            loss = bce(logit, y[i:min(i + 128, cut)])
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        acc = ((clf(x[cut:])[:, -1, 0] > 0).float() == y[cut:]).float().mean().item()
    return abs(acc - 0.5), acc


def synthetic_for_fold(fd, p, fold):
    """Janelas sintéticas (X [Ns, L, F], y [Ns]) do fold, com cache em disco (reaproveitado pelas configurações que
    só mudam a proporção) e os diagnósticos do gerador."""
    iters, hidden = int(p.get("timegan_iter", 1000)), int(p.get("timegan_hidden", 24))
    key_src = json.dumps({"estudo": common.load_study().get("id"), "fold": fold, "lookback": fd.lookback,
                          "features": fd.features, "tickers": fd.tickers, "iters": iters, "hidden": hidden,
                          "scaler": p.get("scaler"), "seed": int(p.get("seed", 42)),
                          "treino_inicio": p.get("treino_inicio"), "alvo_vol": p.get("alvo_vol")}, sort_keys=True)
    key = hashlib.md5(key_src.encode()).hexdigest()[:16]
    cache_dir = os.path.join(common.output_dir(), "_timegan")
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"{key}.pt")
    if os.path.isfile(path):
        blob = torch.load(path, map_location=fd.device)
        return blob["X"], blob["y"], blob["diag"]

    t0 = time.time()
    rows = fd.idx["train"]
    real = torch.cat([fd.windows(rows), fd.y[rows[:, None] + fd._offsets].unsqueeze(-1)], dim=2)  # [N, L, F+1]
    lo, hi = real.amin((0, 1), keepdim=True), real.amax((0, 1), keepdim=True)
    span = torch.where(hi - lo > 1e-9, hi - lo, torch.ones_like(hi))
    real01 = (real - lo) / span
    gan = train_timegan(real01, iters=iters, hidden=hidden, seed=int(p.get("seed", 42)))
    n_syn = 2 * len(real)
    syn01 = gan.generate(n_syn, real.shape[1], fd.device)
    syn = syn01 * span + lo
    ds, acc = discriminative_score(real01, syn01)
    ret_ch = fd.features.index("log_return") if "log_return" in fd.features else fd.n_features  # senão, o alvo
    diag = {"discriminativo": ds, "acuracia_classificador": acc, "segundos": time.time() - t0, "n_real": len(real),
            "n_sintetico": n_syn, "canal_fatos": fd.features[ret_ch] if ret_ch < fd.n_features else "alvo",
            "real": stylized_facts(real[:, :, ret_ch].cpu().numpy()),
            "sintetico": stylized_facts(syn[:, :, ret_ch].cpu().numpy())}
    X, y = syn[:, :, :fd.n_features].contiguous(), syn[:, -1, fd.n_features].contiguous()
    torch.save({"X": X.cpu(), "y": y.cpu(), "diag": diag}, path + ".tmp")
    os.replace(path + ".tmp", path)
    print(f"  TimeGAN fold {fold}: {len(real)} janelas reais → {n_syn} sintéticas em {diag['segundos']:.0f}s | "
          f"discriminativo {ds:.3f} (acurácia {acc:.2f})", flush=True)
    return X.to(fd.device), y.to(fd.device), diag
