"""Run the portable Q4/FNO example; default: original smooth 120,626,288-DOF beam."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def matlab_executable(selection):
    if selection:
        candidates = [Path(selection)]
    else:
        folder = Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "MATLAB"
        candidates = [folder / "R2025a/bin/win64/MATLAB.exe"]
        candidates += sorted(folder.glob("*/bin/win64/MATLAB.exe"), reverse=True)
        if shutil.which("matlab"):
            candidates.append(Path(shutil.which("matlab")))
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError("MATLAB not found. Set --matlab to your MATLAB executable.")


def quote(path):
    return str(path).replace("\\", "/").replace("'", "''")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", choices=["small", "10m", "120m"], default="120m")
    parser.add_argument("--case", choices=["smooth", "iid"], default="smooth",
                        help="smooth: original 120m benchmark (default); iid: previous independent random material")
    parser.add_argument("--edge", choices=["fno", "exact"], default="fno")
    parser.add_argument("--matlab", default=os.environ.get("MATLAB_EXE"))
    parser.add_argument("--save-fields", action="store_true")
    parser.add_argument("--skip-reference", action="store_true")
    parser.add_argument("--build", action="store_true", help="Rebuild all five CUDA MEX files before running")
    args = parser.parse_args()
    env = os.environ.copy()
    env.pop("PYTHONHOME", None)
    env.pop("PYTHONPATH", None)
    env.pop("MATLABPATH", None)
    for key in ("LC_ALL", "LANG"):
        env.pop(key, None)
    result = subprocess.run([sys.executable, "-I", "-c", "import torch; print(torch.__version__); print(torch.__file__)"],
                            check=True, capture_output=True, text=True, env=env)
    version, module = result.stdout.strip().splitlines()[-2:]
    if version != "2.5.1+cu121":
        raise RuntimeError("This MEX build uses torch 2.5.1+cu121. Run SETUP.cmd.")
    env["V9_PORTABLE_PYTHON"] = sys.executable
    env["V9_PORTABLE_TORCH"] = str(Path(module).parent)
    torch_lib = Path(module).parent / "lib"
    env["PATH"] = str(torch_lib) + os.pathsep + env.get("PATH", "")
    out = ROOT / "results" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out.mkdir(parents=True)
    expression = f"restoredefaultpath;cd('{quote(ROOT)}');"
    if args.build:
        expression += "build_mex;"
    expression += (f"run_random_density('{args.size}','{args.edge}','{quote(out)}',"
                   f"{str(not args.skip_reference).lower()},{str(args.save_fields).lower()},'{args.case}');")
    log = out / "matlab.log"
    print(f"MATLAB log: {log}", flush=True)
    with log.open("wb") as stream:
        proc = subprocess.run([str(matlab_executable(args.matlab)), "-batch", expression], cwd=ROOT, env=env,
                              stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                              creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if proc.returncode:
        sys.stderr.write(log.read_text(encoding="utf-8", errors="replace")[-5000:])
        raise SystemExit(proc.returncode)
    receipt = out / "result.json"
    data = json.loads(receipt.read_text(encoding="utf-8-sig"))
    print(json.dumps(data, ensure_ascii=False, indent=2))
    print(f"Results: {receipt}")


if __name__ == "__main__":
    main()
