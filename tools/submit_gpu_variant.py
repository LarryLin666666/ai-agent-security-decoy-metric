"""Build + push + CLI-submit one GPU attack variant (SDK v3.1.2 engine).
Usage: python submit_gpu_variant.py <vid> '<json_overrides>' "<message>" [--no-submit]

<vid> must be a FRESH kernel id slug (reused ids that hit quota go bad-state).
<json_overrides> maps knob -> value; each is injected as a module-global override
(e.g. {"slow_multipost_n":3,"replay_safe_frac":0.96}). Empty {} = engine defaults.
"""
import os, sys, json, time, base64, subprocess
# Kaggle credentials are read from the environment (KAGGLE_USERNAME / KAGGLE_KEY)
# or ~/.kaggle/kaggle.json. Never hard-code them.
assert os.environ.get("KAGGLE_KEY"), "Set KAGGLE_USERNAME and KAGGLE_KEY in your environment."

os.environ["PYTHONUTF8"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"
ROOT = os.path.dirname(os.path.abspath(__file__))
COMP = "ai-agent-security-multi-step-tool-attacks"
KAGGLE = "kaggle"


def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True)


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": text.splitlines(keepends=True)}


def build_nb(overrides: dict):
    src = open(f"{ROOT}/attack.py", encoding="utf-8").read()
    if overrides:
        lines = ["", "# --- variant overrides (injected) ---"]
        for k, v in overrides.items():
            lines.append(f"{k.upper()} = {v!r}")
        src = src + "\n".join(lines) + "\n"
    b64 = base64.b64encode(src.encode("utf-8")).decode("ascii")
    b64w = "\n".join(b64[i:i + 100] for i in range(0, len(b64), 100))

    setup = (
        "import sys, glob\n"
        "from pathlib import Path\n"
        "sys.argv = [sys.argv[0]]\n"
        "for c in glob.glob('/kaggle/input/**/kaggle_evaluation', recursive=True):\n"
        "    r = str(Path(c).parent)\n"
        "    if r not in sys.path: sys.path.insert(0, r)\n"
        "    print('Dataset root:', r); break\n"
        "print('setup done | IS_RERUN:', bool(__import__('os').getenv('KAGGLE_IS_COMPETITION_RERUN')))"
    )
    write = (
        "import base64\n"
        "from pathlib import Path\n"
        f'ATTACK_B64 = """\n{b64w}\n"""\n'
        "src = base64.b64decode(ATTACK_B64.encode('ascii'))\n"
        "Path('/kaggle/working/attack.py').write_bytes(src)\n"
        "assert b'class AttackAlgorithm' in src\n"
        "print('attack.py written', len(src), 'bytes')"
    )
    # Env-free smoke test: proves the build compiles and the gateway's env-free
    # path returns a valid non-empty list. Fast (no models, no env) -> safe before serve().
    smoke = (
        "import importlib.util, py_compile\n"
        "py_compile.compile('/kaggle/working/attack.py', doraise=True)\n"
        "_s = importlib.util.spec_from_file_location('attack', '/kaggle/working/attack.py')\n"
        "_m = importlib.util.module_from_spec(_s); _s.loader.exec_module(_m)\n"
        "_fb = _m.AttackAlgorithm().run(None, None)\n"
        "assert _fb and _fb[0].user_messages and _fb[0].user_messages[-1].strip(), 'smoke failed'\n"
        "print('smoke OK: env-free fallback', len(_fb), 'candidates; last msg:', _fb[0].user_messages[-1][:60])"
    )
    serve = (
        "import os, csv\n"
        "if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):\n"
        "    import kaggle_evaluation.jed_attack_134815.jed_attack_inference_server as server\n"
        "    server.JEDAttackInferenceServer().serve()\n"
        "else:\n"
        "    with open('/kaggle/working/submission.csv', 'w', newline='') as fh:\n"
        "        w = csv.writer(fh); w.writerow(['Id', 'Score'])\n"
        "        w.writerows([['gpt_oss_public', 0.0], ['gpt_oss_private', 0.0],\n"
        "                     ['gemma_public', 0.0], ['gemma_private', 0.0]])\n"
        "    print('placeholder submission.csv written (commit).')"
    )
    return {"cells": [code(setup), code(write), code(smoke), code(serve)],
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                         "language_info": {"name": "python", "version": "3.11"}},
            "nbformat": 4, "nbformat_minor": 5}


def main():
    vid = sys.argv[1]
    overrides = json.loads(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2] else {}
    msg = sys.argv[3] if len(sys.argv) > 3 else "SECRET_MARKER validation-fill"
    no_submit = "--no-submit" in sys.argv
    kernel = f"{os.environ['KAGGLE_USERNAME']}/{vid}"
    vdir = f"{ROOT}/sub_new/{vid}"
    os.makedirs(vdir, exist_ok=True)
    with open(f"{vdir}/nb.ipynb", "w", encoding="utf-8") as f:
        json.dump(build_nb(overrides), f, indent=1)
    meta = {"id": kernel, "title": vid.replace("-", " "), "code_file": "nb.ipynb",
            "language": "python", "kernel_type": "notebook", "is_private": True,
            "enable_gpu": True, "enable_tpu": False, "enable_internet": False,
            "machine_shape": "NvidiaTeslaT4", "dataset_sources": [],
            "competition_sources": [COMP], "kernel_sources": []}
    with open(f"{vdir}/kernel-metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f)
    print(f"== push {kernel} overrides={overrides} ==", flush=True)
    r = sh(f'{KAGGLE} kernels push -p "{vdir}"')
    print(r.stdout, r.stderr)
    blob = (r.stdout + r.stderr).lower()
    if "quota" in blob:
        print("ABORT: GPU quota exhausted."); return
    if "successfully pushed" not in blob:
        print("ABORT: push failed."); return
    print("== poll commit ==", flush=True)
    done = False
    for i in range(200):
        s = sh(f"{KAGGLE} kernels status {kernel}")
        m = (s.stdout + s.stderr).strip()
        if i % 4 == 0:
            print(f"[{i}] {m}", flush=True)
        low = m.lower()
        if "complete" in low:
            done = True; break
        if "error" in low or "cancel" in low:
            print("ABORT: commit failed:", m); return
        time.sleep(30)
    if not done:
        print("ABORT: commit poll timed out."); return
    out = f"{vdir}/_out"
    sh(f"rm -rf {out} && mkdir -p {out}")
    sh(f"{KAGGLE} kernels output {kernel} -p {out}")
    if not os.path.exists(f"{out}/submission.csv"):
        print("ABORT: no submission.csv in commit output (build likely errored)."); return
    print("commit produced submission.csv (placeholder).")
    if no_submit:
        print("--no-submit set; stopping before competition submit."); return
    print("== submit ==", flush=True)
    r = sh(f'{KAGGLE} competitions submit {COMP} -k {kernel} -v 1 -f submission.csv -m "{msg}"')
    print(r.stdout, r.stderr)
    time.sleep(8)
    s = sh(f"{KAGGLE} competitions submissions {COMP}")
    print("\n".join(s.stdout.splitlines()[:5]))


if __name__ == "__main__":
    main()