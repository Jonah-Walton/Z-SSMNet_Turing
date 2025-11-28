
"""
Minimal subset of DIAGNijmegen/msk-tiger/io.py, as required for the nnunet wrapper.
"""

import json
from pathlib import Path
from typing import Any

class JSONEncoder(json.JSONEncoder):
    def default(self, o: Any) -> Any:
        if isinstance(o, Path):
            return o.as_posix()
        else:
            return super().default(o)