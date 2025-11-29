import re, json, random
from pathlib import Path
random.seed(42)

def to_filename(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", s).strip("_")

def write_text(p: Path, txt: str):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(txt, encoding="utf-8")

def html_escape(t: str) -> str:
    return (t.replace("&","&amp;").replace("<","&lt;")
              .replace(">","&gt;").replace('"',"&quot;"))

def safe_json(s: str):
    try: return json.loads(s)
    except Exception: return None