"""One local command to install the isolated runtime and resume official weights."""
import os
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PYTHON = ROOT / '.venv-qwen2b' / 'Scripts' / 'python.exe'
LOG = ROOT / 'runs' / 'qwen2b_setup'


def run(args):
    print('\nRUN:', subprocess.list2cmdline([str(a) for a in args]), flush=True)
    with (LOG / 'setup.log').open('a', encoding='utf-8') as log:
        log.write('\nRUN: ' + subprocess.list2cmdline([str(a) for a in args]) + '\n')
        with subprocess.Popen([str(a) for a in args], cwd=ROOT, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace') as proc:
            for line in proc.stdout:
                print(line, end='', flush=True)
                log.write(line)
                log.flush()
            if proc.wait():
                raise RuntimeError('Step failed. See runs/qwen2b_setup/setup.log; later steps were not run.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check-only', action='store_true', help='Check installed GPU runtime without downloads')
    args = parser.parse_args()
    LOG.mkdir(parents=True, exist_ok=True)
    if args.check_only:
        run([PYTHON, '-u', ROOT / 'scripts' / 'qwen2b' / 'check_runtime.py'])
        return
    uv = shutil.which('uv')
    if not uv:
        raise RuntimeError('uv not found in PATH.')
    os.environ['UV_HTTP_TIMEOUT'] = '300'
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    cache = ROOT / '.cache' / 'uv'
    if not PYTHON.exists():
        run([uv, 'venv', ROOT / '.venv-qwen2b', '--python', sys.executable, '--cache-dir', cache])
    run([uv, 'pip', 'install', '--python', PYTHON, '--cache-dir', cache,
         'torch==2.10.0', '--index-url', 'https://download.pytorch.org/whl/cu128'])
    run([uv, 'pip', 'install', '--python', PYTHON, '--cache-dir', cache,
         '-r', ROOT / 'scripts' / 'qwen2b' / 'requirements.txt'])
    run([uv, 'pip', 'check', '--python', PYTHON])
    # A real GPU operation and NF4 layer operation, not just an import check.
    run([PYTHON, '-u', ROOT / 'scripts' / 'qwen2b' / 'check_runtime.py'])
    frozen = subprocess.check_output([uv, 'pip', 'freeze', '--python', str(PYTHON)], cwd=ROOT, text=True)
    (LOG / 'installed.freeze.txt').write_text(frozen, encoding='utf-8')
    run([PYTHON, '-u', ROOT / 'scripts' / 'qwen2b' / 'download.py'])
    print('\nSETUP_COMPLETE. Send runs/qwen2b_setup/setup.log for review. Model inference has not run.', flush=True)


if __name__ == '__main__':
    main()
