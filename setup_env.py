"""Install the tested CUDA runtime in this repository's local virtual environment."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def python310(selection):
    candidates = [[selection]] if selection else [[sys.executable], ["py", "-3.10"], ["python3.10"]]
    for command in candidates:
        try:
            result = subprocess.run(command + ["-I", "-c", "import sys,json;print(json.dumps([list(sys.version_info[:2]),sys.executable]))"],
                                    check=True, capture_output=True, text=True, timeout=30)
            version, exe = json.loads(result.stdout.strip())
            if version == [3, 10]:
                return exe
        except (OSError, subprocess.SubprocessError, ValueError):
            continue
    raise RuntimeError("Install 64-bit Python 3.10, or run setup_env.py --python PATH_TO_PYTHON_3.10.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", help="Python 3.10 executable used to create .venv")
    args = parser.parse_args()
    if os.name != "nt":
        raise RuntimeError("This release contains the verified Windows MATLAB GPU build.")
    env = os.environ.copy()
    env.pop("PYTHONHOME", None)
    env.pop("PYTHONPATH", None)
    exe = ROOT / ".venv/Scripts/python.exe"
    if not exe.is_file():
        subprocess.run([python310(args.python), "-m", "venv", str(ROOT / ".venv")], check=True, env=env)
    subprocess.run([str(exe), "-m", "pip", "install", "-r", str(ROOT / "requirements.txt")], check=True, env=env)
    probe = "import sys,torch,numpy,scipy; assert sys.version_info[:2]==(3,10); assert torch.__version__=='2.5.1+cu121'; assert torch.cuda.is_available(), 'CUDA GPU unavailable'; print('Ready:',torch.cuda.get_device_name(),numpy.__version__,scipy.__version__)"
    subprocess.run([str(exe), "-I", "-c", probe], check=True, env=env)
    print("Environment ready. Run RUN.cmd or .venv\\Scripts\\python.exe run.py.")


if __name__ == "__main__":
    main()
