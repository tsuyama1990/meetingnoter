import re

with open("tutorials/UAT_AND_TUTORIAL.py", "r") as f:
    content = f.read()

# Insert sys.path modification at the top cell
sys_path_code = """
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src" if "__file__" in globals() else Path().resolve() / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent if "__file__" in globals() else Path().resolve()))
"""
if "sys.path.insert" not in content:
    content = content.replace('def cell_imports() -> tuple[Any]:\n    import marimo as mo', f'def cell_imports() -> tuple[Any]:\n    import marimo as mo\n{sys_path_code}')

with open("tutorials/UAT_AND_TUTORIAL.py", "w") as f:
    f.write(content)
