#!/usr/bin/env python3
"""
Скачивает официальный репозиторий FaceFusion в ./facefusion (корень этого проекта).

Бенчмарк не включает код FaceFusion в пакет — он ожидает каталог facefusion/ рядом с deepfake_benchmark/.
Новый пользователь один раз клонирует репозиторий этим скриптом или вручную.

Использование:
  poetry run python scripts/setup_facefusion.py
  poetry run python scripts/setup_facefusion.py --ref 3.0.0
  poetry run python scripts/setup_facefusion.py --install-deps --python C:/path/to/conda/envs/facefusion/python.exe
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

DEFAULT_REPO = "https://github.com/facefusion/facefusion.git"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clone FaceFusion into ./facefusion (required for the benchmark)."
    )
    parser.add_argument(
        "--repo",
        default=DEFAULT_REPO,
        help=f"Git URL (default: {DEFAULT_REPO})",
    )
    parser.add_argument(
        "--target",
        type=Path,
        default=None,
        help="Target directory (default: <repo_root>/facefusion)",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=1,
        help="Shallow clone depth (1 = latest commit only; 0 = full history)",
    )
    parser.add_argument(
        "--ref",
        default=None,
        help="After clone, git checkout this tag or branch (optional, for reproducibility)",
    )
    parser.add_argument(
        "--install-deps",
        action="store_true",
        help="After clone, run pip install -r (use with --python pointing to conda/venv)",
    )
    parser.add_argument(
        "--python",
        type=Path,
        default=None,
        help="Python executable for --install-deps (e.g. conda env .../python.exe)",
    )
    parser.add_argument(
        "--requirements-file",
        type=Path,
        default=None,
        help=(
            "Requirements file for --install-deps. "
            "Default: facefusion/requirements.txt (upstream). "
            "Alternative minimal headless: "
            "deepfake_benchmark/core/generators/facefusion/requirements_facefusion.txt"
        ),
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    target = (args.target or (repo_root / "facefusion")).resolve()

    if target.exists() and any(target.iterdir()):
        print(f"[setup_facefusion] Directory already exists and is not empty: {target}")
        print("[setup_facefusion] Remove it or use --target to clone elsewhere.")
        sys.exit(1)

    cmd = ["git", "clone"]
    if args.depth and args.depth > 0:
        cmd.extend(["--depth", str(args.depth)])
    if args.ref:
        cmd.extend(["--branch", args.ref])
    cmd.extend([args.repo, str(target)])

    print(f"[setup_facefusion] Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    if args.ref:
        print(f"[setup_facefusion] Cloned at ref/branch: {args.ref}")

    print(f"[setup_facefusion] Done. FaceFusion is at: {target}")

    req_default = target / "requirements.txt"
    req_file = args.requirements_file
    if req_file is None:
        req_file = req_default if req_default.exists() else None

    if args.install_deps:
        py = args.python or Path(sys.executable)
        if not py.exists():
            print(f"[setup_facefusion] Python not found: {py}", file=sys.stderr)
            sys.exit(1)
        if req_file is None or not req_file.is_file():
            print(
                "[setup_facefusion] No requirements file found. "
                "Pass --requirements-file explicitly.",
                file=sys.stderr,
            )
            sys.exit(1)
        pip_cmd = [str(py), "-m", "pip", "install", "-r", str(req_file.resolve())]
        print(f"[setup_facefusion] Installing deps: {' '.join(pip_cmd)}")
        subprocess.run(pip_cmd, check=True)
        print("[setup_facefusion] Dependencies installed.")
    else:
        print(
            "[setup_facefusion] Next (manual): create conda/venv, then e.g.\n"
            f"  {sys.executable} -m pip install -r {req_default}\n"
            "  (upstream full stack), or for minimal headless pins:\n"
            f"  <conda_python> -m pip install -r "
            f"{repo_root / 'deepfake_benchmark/core/generators/facefusion/requirements_facefusion.txt'}\n"
            "Then set facefusion_python in configs/*.yaml and run the CLI."
        )


if __name__ == "__main__":
    main()
