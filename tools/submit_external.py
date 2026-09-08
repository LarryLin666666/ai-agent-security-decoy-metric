"""Build + push + CLI-submit an EXTERNAL attack.py (e.g. a top team's open-sourced
solution) to reproduce their private score firsthand. Clearly a reproduction, attributed
in the working note — not our own method.

Usage: python submit_external.py <vid> <path/to/attack.py> "<message>"
"""
import os, sys, base64, subprocess, time
# Kaggle credentials are read from the environment (KAGGLE_USERNAME / KAGGLE_KEY)
# or ~/.kaggle/kaggle.json. Never hard-code them.
assert os.environ.get("KAGGLE_KEY"), "Set KAGGLE_USERNAME and KAGGLE_KEY in your environment."

os.environ["PYTHONUTF8"] = "1"; os.environ["PYTHONIOENCODING"] = "utf-8"
ROOT = os.path.dirname(os.path.abspath(__file__))
COMP = "ai-agent-security-multi-step-tool-attacks"


def sh(c): return subprocess.run(c, shell=True, capture_output=True, text=True)
def code(t): return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": t.splitlines(keepends=True)}


def build_nb(attack_path):
    src = open(attack_path, "rb").read()
    b64 = base64.b64encode(src).decode("ascii")
    b64w = "\n".join(b64[i:i+100] for i in range(0, len(b64), 100))
    setup = (
        "import sys, glob\nfrom pathlib import Path\nsys.argv=[sys.argv[0]]\n"
        "for c in glob.glob('/kaggle/input/**/kaggle_evaluation', recursive=True):\n"
        "    r=str(Path(c).parent)\n    if r not in sys.path: sys.path.insert(0,r)\n    print('root',r); break\n"
        "print('IS_RERUN', bool(__import__('os').getenv('KAGGLE_IS_COMPETITION_RERUN')))")
    write = ("import base64\nfrom pathlib import Path\n"
        f'A="""\n{b64w}\n"""\n'
        "src=base64.b64decode(A.encode('ascii'))\nPath('/kaggle/working/attack.py').write_bytes(src)\n"
        "assert b'class AttackAlgorithm' in src\nprint('attack.py written',len(src))")
    # smoke: register in sys.modules BEFORE exec so frozen dataclasses resolve their module.
    smoke = ("import importlib.util, py_compile, sys\n"
        "py_compile.compile('/kaggle/working/attack.py', doraise=True)\n"
        "_s=importlib.util.spec_from_file_location('attack','/kaggle/working/attack.py')\n"
        "_m=importlib.util.module_from_spec(_s); sys.modules['attack']=_m; _s.loader.exec_module(_m)\n"
        "_fb=_m.AttackAlgorithm().run(None,None)\n"
        "assert _fb and _fb[0].user_messages and _fb[0].user_messages[-1].strip(), 'smoke failed'\n"
        "print('smoke OK', len(_fb),'cands')")
    serve = ("import os, csv\n"
        "if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):\n"
        "    import kaggle_evaluation.jed_attack_134815.jed_attack_inference_server as server\n"
        "    server.JEDAttackInferenceServer().serve()\n"
        "else:\n"
        "    with open('/kaggle/working/submission.csv','w',newline='') as fh:\n"
        "        w=csv.writer(fh); w.writerow(['Id','Score'])\n"
        "        w.writerows([['gpt_oss_public',0.0],['gpt_oss_private',0.0],['gemma_public',0.0],['gemma_private',0.0]])\n"
        "    print('placeholder submission.csv')")
    return {"cells": [code(setup), code(write), code(smoke), code(serve)],
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                         "language_info": {"name": "python", "version": "3.11"}}, "nbformat": 4, "nbformat_minor": 5}


def main():
    import json
    vid, apath = sys.argv[1], sys.argv[2]
    msg = sys.argv[3] if len(sys.argv) > 3 else "reproduction"
    kernel = f"{os.environ['KAGGLE_USERNAME']}/{vid}"; vdir = f"{ROOT}/repro/{vid}"; os.makedirs(vdir, exist_ok=True)
    json.dump(build_nb(apath), open(f"{vdir}/nb.ipynb", "w", encoding="utf-8"), indent=1)
    meta = {"id": kernel, "title": vid.replace("-", " "), "code_file": "nb.ipynb", "language": "python",
            "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_tpu": False,
            "enable_internet": False, "machine_shape": "NvidiaTeslaT4", "dataset_sources": [],
            "competition_sources": [COMP], "kernel_sources": []}
    json.dump(meta, open(f"{vdir}/kernel-metadata.json", "w"))
    print(f"== push {kernel} ==", flush=True)
    r = sh(f'kaggle kernels push -p "{vdir}"'); print(r.stdout, r.stderr)
    if "successfully pushed" not in (r.stdout + r.stderr).lower():
        print("ABORT: push failed"); return
    for i in range(200):
        s = sh(f"kaggle kernels status {kernel}"); m = (s.stdout + s.stderr).strip()
        if i % 4 == 0: print(f"[{i}] {m}", flush=True)
        low = m.lower()
        if "complete" in low: break
        if "error" in low or "cancel" in low: print("ABORT commit:", m); return
        time.sleep(30)
    out = f"{vdir}/_out"; sh(f"rm -rf {out} && mkdir -p {out}"); sh(f"kaggle kernels output {kernel} -p {out}")
    if not os.path.exists(f"{out}/submission.csv"): print("ABORT: no submission.csv (build errored)"); return
    print("== submit ==", flush=True)
    r = sh(f'kaggle competitions submit {COMP} -k {kernel} -v 1 -f submission.csv -m "{msg}"')
    print(r.stdout, r.stderr)


if __name__ == "__main__":
    main()