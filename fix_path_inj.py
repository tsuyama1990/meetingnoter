with open("tutorials/UAT_AND_TUTORIAL.py", "r") as f:
    c = f.read()

# Remove the global sys.path stuff entirely
global_path_stuff = """import sys
from pathlib import Path
try:
    _base_dir = Path(__file__).parent.parent
except NameError:
    _base_dir = Path().resolve()

_src_dir = _base_dir / "src"
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))
if str(_base_dir) not in sys.path:
    sys.path.insert(0, str(_base_dir))

"""
c = c.replace(global_path_stuff, '')

with open("tutorials/UAT_AND_TUTORIAL.py", "w") as f:
    f.write(c)
