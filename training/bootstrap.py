#!/usr/bin/env python3
"""Install only inside a dedicated environment; never alter the notebook kernel."""
from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
import sys
import venv

ROOT = Path(__file__).resolve().parents[1]
ENV = ROOT / '.venv-faust'
PY = ENV / 'bin/python'
REQ = ROOT / 'training/requirements.txt'

def run(*args):
    subprocess.run([str(x) for x in args], check=True)


def main():
    if os.name != 'posix' or sys.platform == 'darwin':
        raise SystemExit('此入口用于 Linux/Colab/WSL2 NVIDIA CUDA；Mac 或原生 Windows 请使用 Colab。')
    if not (3, 10) <= sys.version_info[:2] <= (3, 13):
        raise SystemExit('需要 Python 3.10–3.13；建议全新 Python 3.11/3.12 环境。')
    if shutil.which('nvidia-smi') is None:
        raise SystemExit('未找到 NVIDIA 驱动。Colab 请先切换为 GPU 运行时。')
    lock = hashlib.sha256(REQ.read_bytes() + b'torch==2.9.1+cu128').hexdigest()
    marker = ENV / 'faust_environment.json'
    if marker.exists() and json.loads(marker.read_text()).get('lock') == lock and PY.exists():
        run(PY, '-c', 'import torch, transformers, peft, bitsandbytes; assert torch.cuda.is_available(), "CUDA unavailable"')
        print(PY)
        return
    if shutil.disk_usage(ROOT).free < 35 * 2**30:
        raise SystemExit('建议至少 35 GiB 可用磁盘用于环境、8B底模缓存及检查点；请释放空间再安装。')
    if not PY.exists():
        venv.EnvBuilder(with_pip=True, system_site_packages=False).create(ENV)
    run(PY, '-m', 'pip', 'install', '--upgrade', 'pip')
    run(PY, '-m', 'pip', 'install', 'torch==2.9.1', '--index-url', 'https://download.pytorch.org/whl/cu128')
    run(PY, '-m', 'pip', 'install', '-r', REQ)
    run(PY, '-m', 'pip', 'check')
    run(PY, '-c', 'import torch, transformers, peft, bitsandbytes; assert torch.cuda.is_available(), "CUDA unavailable: check GPU/driver"')
    marker.write_text(json.dumps({'lock': lock, 'torch': '2.9.1+cu128'}) + '\n')
    print(PY)

if __name__ == '__main__':
    main()
