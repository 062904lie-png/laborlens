"""
run.py  — ALWAYS use this to start LaborLens on Windows.
Place in backend/ and run:  python run.py

This sets PYTHONPATH in the OS environment BEFORE the first import,
so the multiprocessing spawn on Windows inherits the correct path.
"""
import subprocess
import sys
import os

backend_dir = os.path.dirname(os.path.abspath(__file__))

env = os.environ.copy()
env["PYTHONPATH"] = backend_dir

subprocess.run(
    [
        sys.executable, "-m", "uvicorn",
        "main:app",
        "--host", os.environ.get("LABORLENS_HOST", "0.0.0.0"),
        "--port", "8000",
        "--reload",
    ],
    cwd=backend_dir,
    env=env,
)
