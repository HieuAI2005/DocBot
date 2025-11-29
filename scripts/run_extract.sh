set -e
PY=${PY:-python}

$PY - <<'PYCODE'
from pathlib import Path
from src.extract.pdf2md_ai import extract_pdf
from src.config import INPUT_DIR
for pdf in sorted(Path(INPUT_DIR).glob("*.pdf")):
    print("Extract:", pdf)
    extract_pdf(pdf)
PYCODE

# $PY - <<'PYCODE'
# from src.rag.build_index import build_index
# build_index()
# PYCODE
# echo "[OK] Extract + Index built under outputs/"