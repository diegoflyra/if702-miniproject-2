"""Espelho opcional dos resultados no GitHub (o disco local continua sendo a cópia principal).

Os arquivos de outputs/ vão para a pasta `{RUN_NAME}/outputs/` do branch `RESULTS_BRANCH` de `RESULTS_REPO_URL`
(o branch é criado órfão se não existir, assim o histórico de código não se mistura com o de resultados).
Pesos (.pth) só vão com --pesos; arquivos acima de 50 MB nunca vão (limite prático do GitHub).

Variáveis de ambiente: GITHUB_TOKEN (com permissão de escrita), RESULTS_REPO_URL, RESULTS_BRANCH (padrão
"resultados"), RUN_NAME. Sem token ou sem URL, não faz nada e avisa: o treino nunca depende disso.

Uso:  python src/github_sync.py push --outputs /kaggle/working/outputs [--pesos] [-m "mensagem"]
      python src/github_sync.py pull --dest /content/outputs          (retomar uma execução anterior)
"""
import argparse
import os
import shutil
import subprocess
import time

MAX_FILE = 50 * 1024 * 1024
CLONE = os.environ.get("RESULTS_CLONE_DIR", "/tmp/lstm-resultados")


def _cfg():
    token = os.environ.get("GITHUB_TOKEN", "").strip()
    url = os.environ.get("RESULTS_REPO_URL", "").strip()
    branch = os.environ.get("RESULTS_BRANCH", "resultados").strip()
    run = os.environ.get("RUN_NAME", "").strip()
    return token, url, branch, run


def _auth_url(url, token):
    return url.replace("https://", f"https://x-access-token:{token}@", 1) if token else url


def _git(*args, check=True, token=""):
    out = subprocess.run(["git", "-C", CLONE, *args], capture_output=True, text=True)
    if check and out.returncode != 0:
        msg = (out.stderr or out.stdout).replace(token, "***") if token else (out.stderr or out.stdout)
        raise RuntimeError(f"git {' '.join(args[:2])} falhou: {msg.strip()[-500:]}")
    return out


def _ensure_clone(url, branch, token):
    auth = _auth_url(url, token)
    if os.path.isdir(os.path.join(CLONE, ".git")):
        _git("fetch", "-q", "origin", branch, check=False, token=token)
        if _git("rev-parse", "--verify", f"origin/{branch}", check=False).returncode == 0:
            _git("reset", "-q", "--hard", f"origin/{branch}", token=token)
        return
    shutil.rmtree(CLONE, ignore_errors=True)
    out = subprocess.run(["git", "clone", "-q", "--depth", "1", "--branch", branch, auth, CLONE], capture_output=True, text=True)
    if out.returncode == 0:
        return
    # branch ainda não existe: cria um branch órfão só de resultados
    shutil.rmtree(CLONE, ignore_errors=True)
    out = subprocess.run(["git", "clone", "-q", "--depth", "1", auth, CLONE], capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError("clone falhou (URL/token?): " + out.stderr.replace(token, "***")[-500:])
    _git("checkout", "-q", "--orphan", branch)
    _git("rm", "-rq", "--cached", ".", check=False)
    for name in os.listdir(CLONE):
        if name != ".git":
            path = os.path.join(CLONE, name)
            shutil.rmtree(path) if os.path.isdir(path) else os.remove(path)
    with open(os.path.join(CLONE, "README.md"), "w", encoding="utf-8") as f:
        f.write("# Resultados do estudo LSTM\n\nCada pasta é uma execução (RUN_NAME) com o conteúdo de `outputs/`.\n")


def _copy(src, dst, weights):
    copied = skipped = 0
    for root, _, files in os.walk(src):
        rel = os.path.relpath(root, src)
        for name in files:
            path = os.path.join(root, name)
            if name.endswith(".tmp") or (name.endswith(".pth") and not weights) or os.path.getsize(path) > MAX_FILE:
                skipped += 1
                continue
            target = os.path.join(dst, rel, name)
            os.makedirs(os.path.dirname(target), exist_ok=True)
            if not os.path.isfile(target) or os.path.getmtime(target) < os.path.getmtime(path) or os.path.getsize(target) != os.path.getsize(path):
                shutil.copy2(path, target)
                copied += 1
    return copied, skipped


def push(outputs, weights=False, message=None):
    token, url, branch, run = _cfg()
    if not (token and url and run):
        print("GitHub: sync desativado (defina GITHUB_TOKEN, RESULTS_REPO_URL e RUN_NAME). Resultados só em disco.")
        return False
    try:
        _ensure_clone(url, branch, token)
        copied, skipped = _copy(outputs, os.path.join(CLONE, run, "outputs"), weights)
        _git("add", "-A")
        if _git("diff", "--cached", "--quiet", check=False).returncode == 0:
            print("GitHub: nada novo para enviar.")
            return True
        msg = message or f"{run}: resultados {time.strftime('%Y-%m-%d %H:%M')}"
        _git("-c", "user.name=lstm-experimentos", "-c", "user.email=lstm-experimentos@users.noreply.github.com",
             "commit", "-q", "-m", msg)
        for attempt in range(3):
            out = _git("push", "-q", "origin", f"HEAD:{branch}", check=False, token=token)
            if out.returncode == 0:
                print(f"GitHub: {copied} arquivos enviados para {branch}/{run}/outputs ({skipped} ignorados: .pth/grandes).")
                return True
            _git("pull", "-q", "--rebase", "origin", branch, check=False, token=token)
            time.sleep(2 * (attempt + 1))
        raise RuntimeError(out.stderr.replace(token, "***")[-500:])
    except Exception as e:  # noqa: BLE001 — o espelho nunca derruba o estudo
        print(f"GitHub: falha no envio ({e}). Os resultados continuam em {outputs}.")
        return False


def pull(dest):
    token, url, branch, run = _cfg()
    if not (url and run):
        print("GitHub: defina RESULTS_REPO_URL e RUN_NAME para retomar.")
        return False
    try:
        _ensure_clone(url, branch, token)
    except Exception as e:  # noqa: BLE001
        print(f"GitHub: falha ao buscar resultados ({e}).")
        return False
    src = os.path.join(CLONE, run, "outputs")
    if not os.path.isdir(src):
        print(f"GitHub: {branch}/{run}/outputs não existe; começando do zero.")
        return False
    shutil.copytree(src, dest, dirs_exist_ok=True)
    print(f"GitHub: resultados de {branch}/{run} copiados para {dest}; o que já foi concluído será pulado.")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("push")
    p.add_argument("--outputs", required=True)
    p.add_argument("--pesos", action="store_true")
    p.add_argument("-m", "--message", default=None)
    q = sub.add_parser("pull")
    q.add_argument("--dest", required=True)
    args = parser.parse_args()
    if args.cmd == "push":
        push(args.outputs, args.pesos, args.message)
    else:
        pull(args.dest)


if __name__ == "__main__":
    main()
