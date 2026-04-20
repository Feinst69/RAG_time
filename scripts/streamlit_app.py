"""Launcher for the Streamlit application."""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    from streamlit.web.cli import main as streamlit_main

    app_path = Path(__file__).resolve().parents[1] / "app.py"
    sys.argv = ["streamlit", "run", str(app_path)]
    raise SystemExit(streamlit_main())


if __name__ == "__main__":
    main()
