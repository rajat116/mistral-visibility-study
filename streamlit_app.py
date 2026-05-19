# Root entry point for Streamlit Community Cloud deployment.
# Streamlit Cloud looks for this file at the repo root.
import runpy, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
runpy.run_path(str(Path(__file__).parent / "src" / "dashboard" / "app.py"), run_name="__main__")
