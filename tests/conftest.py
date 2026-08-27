"""让 pytest 能从项目根目录 import agent / llm / tools / config。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
