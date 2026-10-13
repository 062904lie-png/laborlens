"""
sitecustomize.py
Place this file in:  backend/

Python automatically imports this file on startup for EVERY process,
including the child processes uvicorn spawns during --reload.
This guarantees 'services', 'routers', and 'config' are always findable.
"""
import sys
import os

backend_dir = os.path.dirname(os.path.abspath(__file__))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)
