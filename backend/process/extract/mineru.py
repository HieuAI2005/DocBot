import os, re, shutil, subprocess, cv2, hashlib
from pathlib import Path
from typing import Tuple, List, Dict, Set

from ..config import OUTPUT_DIR, FALLBACK_COPY_ALL_IMAGES
from ..utils import write_text

IMG_DIRNAME = "images"      # thư mục ảnh đích
RAW_DIRNAME = "_mineru_raw" # thư mục tạm MinerU

# ==== ENV tuỳ chỉnh nhanh (đã set default để khớp main.md mẫu) ====
# -> images/image1.png, image2.png, ...
IMG_PREFIX  = os.getenv("IMG_PREFIX", "image")   # "image" -> image1.png
IMG_START   = int(os.getenv("IMG_START", "1"))   # 1 (image1.png) hoặc 0 (image0.png)
IMG_EXT     = os.getenv("IMG_EXT", ".png")       # ép phần mở rộng đầu ra (".png")
KEEP_DEBUG_RAW = os.getenv("KEEP_MINERU_RAW", "0") == "1"  # 1: giữ _mineru_raw để debug

# các phần mở rộng ảnh mà MinerU có thể sinh
IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tiff", ".svg"}

# --- helpers chạy CLI ---
def _run(cmd: List[str]) -> Tuple[int, str, str]:
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out, err = p.communicate()
    return p.returncode, out, err

def _find_cli() -> str:
    for name in ("mineru",):
        rc, _, _ = _run([name, "--help"])
        if rc == 0:
            return name
    possible = [
        str(Path(os.environ.get("VIRTUAL_ENV",""))/"bin/mineru"),
        str(Path(os.sys.executable).parent/"mineru"),
    ]
    for p in possible:
        if Path(p).exists():
            rc, _, _ = _run([p, "--help"])
            if rc == 0:
                return p
    raise RuntimeError("Không tìm thấy CLI 'mineru'. Hãy `pip install mineru` và chắc chắn gọi được `mineru --help`.")

# --- tìm md chính MinerU sinh ---
def _pick_markdown_file(root: Path) -> Path | None:
    mds = list(root.rglob("*.md"))
    if not mds: return None
    for p in mds:
        if p.name.lower() == "main.md":
            return p
    mds.sort(key=lambda p: p.stat().st_size, reverse=True)
    return mds[0]

# --- gom tất cả file ảnh MinerU sinh ---
def _collect_all_image_files(src_root: Path) -> List[Path]:
    imgs = []
    for p in src_root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            imgs.append(p)
    return imgs

# --- chuẩn hoá link ảnh về images/<basename> ---
MD_IMG_RE   = re.compile(r'!\[[^\]]*\]\(([^)]+)\)')
HTML_IMG_RE = re.compile(r'<img\s+[^>]*src=["\']([^"\']+)["\'][^>]*>', flags=re.I)

def _rewrite_md_image_links_to_basename(md_text: str) -> str:
    def repl_md(m):
        base = Path(m.group(1).strip()).name
        return f"![](images/{base})"
    md_text = MD_IMG_RE.sub(repl_md, md_text)

    def repl_html(m):
        base = Path(m.group(1).strip()).name
        return f"![](images/{base})"
    md_text = HTML_IMG_RE.sub(repl_html, md_text)
    return md_text

# --- lấy danh sách ảnh theo thứ tự xuất hiện trong MD (chỉ trong nội dung chính) ---
def _list_image_basenames_in_order(md_text: str) -> List[str]:
    """
    Lấy danh sách ảnh theo thứ tự xuất hiện, nhưng bỏ qua ảnh ở phần đầu (title/abstract).
    Chỉ lấy ảnh trong nội dung chính.
    """
    lines = md_text.splitlines()
    names = []
    
    # Tìm vị trí bắt đầu nội dung chính (sau title H1 và abstract italic)
    content_start_idx = 0
    found_h1 = False
    found_abstract_end = False
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        # Tìm H1 đầu tiên (title)
        if not found_h1 and stripped.startswith('# ') and not stripped.startswith('##'):
            found_h1 = True
            content_start_idx = i + 1
        # Tìm kết thúc abstract (đoạn italic đầu tiên kết thúc)
        if found_h1 and not found_abstract_end:
            if stripped.startswith('_') and stripped.endswith('_'):
                found_abstract_end = True
                content_start_idx = i + 1
            elif stripped.startswith('_') and not found_abstract_end:
                # Abstract có thể nhiều dòng
                continue
            elif found_abstract_end and stripped and not stripped.startswith('#'):
                content_start_idx = i
                break
    
    # Chỉ tìm ảnh sau content_start_idx
    content_text = '\n'.join(lines[content_start_idx:])
    for m in re.finditer(r'!\[[^\]]*\]\(\s*([^)]+)\)', content_text, flags=re.I):
        img_path = m.group(1).strip()
        # Bỏ qua nếu là link ảnh đã có prefix images/
        if img_path.startswith('images/'):
            img_path = img_path[7:]  # Remove 'images/' prefix
        names.append(img_path)
    
    # unique theo thứ tự, loại bỏ duplicate
    seen = set()
    ordered = []
    for n in names:
        basename = Path(n).name
        if basename not in seen:
            seen.add(basename)
            ordered.append(basename)
    
    return ordered

# --- dọn bảng header/watermark rác, bullets, heading, code, khoảng trắng ---
HEADER_TABLE_BAD_WORDS = ("VIETTEL AI RACE", "LẦN BAN HÀNH", "LAN BAN HANH", "NHAN DIEN", "NHẬN DIỆN", "TD0")
TABLE_RE   = re.compile(r'<table>.*?</table>', flags=re.S|re.I)
PAGE_NUM_RE= re.compile(r'^\s*\d+\s*/\s*\d+\s*$', flags=re.M)

def _strip_bad_tables(md: str) -> str:
    def repl(m):
        blk = m.group(0)
        up = blk.upper()
        if any(k in up for k in HEADER_TABLE_BAD_WORDS): return ""
        return blk
    return TABLE_RE.sub(repl, md)

def _fix_bullets(md: str) -> str:
    md = re.sub(r' ?• ?', '\n- ', md)            # tách dot • thành list
    md = re.sub(r'\n- +-', '\n- ', md)           # gộp “-  - ”
    return md

def _fix_headings(md: str) -> str:
    """
    Convert numbered lines to headings AND clean number prefixes from existing headings.
    Handles both cases:
    - Plain text with numbers: "1.2.3 Title" -> "### Title" (level based on dots)
    - Headings with numbers: "# 1.2.3 Title" -> "### Title" (level based on dots, NOT original #)
    """
    out = []
    for ln in md.splitlines():
        s = ln.strip()
        
        # Case 1: Heading with number prefix like "# 1.2.3 Title" or "## 1.2 Title"
        m1 = re.match(r'^(#{1,6})\s+(\d+(?:\.\d+)*\.?)\s+(.+)$', s)
        if m1:
            number = m1.group(2).rstrip('.')
            title = m1.group(3)
            # Calculate level based on number of dots, NOT original # count
            level = min(6, 1 + number.count('.'))
            out.append(f"{'#' * level} {title}")
            continue
        
        # Case 2: Already a heading without numbers - keep as is
        if s.startswith('#'):
            out.append(ln)
            continue
        
        # Case 3: Plain numbered text like "1.2.3 Title"
        m2 = re.match(r'^(\d+(?:\.\d+)*\.?)\s+([^.:;#<][^#<]{0,95})$', s)
        if m2:
            number = m2.group(1).rstrip('.')
            title = m2.group(2).strip()
            # Skip if the text is too long (likely a sentence, not a heading)
            if len(title) < 100 and not title.endswith('.'):
                level = min(6, 1 + number.count('.'))
                out.append(f"{'#' * level} {title}")
            else:
                out.append(ln)
        else:
            out.append(ln)
    
    return "\n".join(out)

def _wrap_code(md: str) -> str:
    """Wrap actual code in code blocks, but be very strict to avoid wrapping normal text."""
    lines = md.splitlines()
    out=[]; in_code=False
    
    def is_code(t: str) -> bool:
        """Detect if a line is actual code (very strict heuristic)."""
        t=t.strip()
        if not t: return False
        if t.startswith("#"): return False  # Markdown heading
        if t.startswith("<table>"): return False
        if t.startswith("![]("): return False
        if t.startswith("```"): return False
        
        # Very strict heuristics for code detection
        # Must have multiple code indicators
        code_indicators = [
            "def " in t,
            "import " in t,
            "from " in t and " import " in t,
            t.startswith("class "),
            "cv2." in t,
            "random." in t,
            "imutils." in t,
            "return " in t and "(" in t,
        ]
        
        # Count special characters typical in code
        special_count = sum([
            t.count("=") >= 2,
            t.count("{") >= 1,
            t.count("}") >= 1,
            t.count("[") >= 1,
            t.count("]") >= 1,
            t.count(";") >= 2,
        ])
        
        # Need at least 2 code indicators OR significant special characters
        return sum(code_indicators) >= 2 or special_count >= 3
    
    for ln in lines:
        if not in_code and is_code(ln):
            out.append("```")
            out.append(ln)
            in_code=True
        elif in_code and not is_code(ln):
            out.append("```")
            out.append(ln)
            in_code=False
        else:
            out.append(ln)
    if in_code: out.append("```")
    return "\n".join(out)

def _dedent(md: str) -> str:
    md = re.sub(r'[ \t]+', ' ', md)
    md = re.sub(r'\n{3,}', '\n\n', md)
    md = PAGE_NUM_RE.sub("", md)
    return md.strip()

def _clean_heading_numbers(md: str) -> str:
    """Remove number prefixes from headings while PRESERVING original level from MinerU."""
    lines = md.splitlines()
    result = []
    
    for line in lines:
        stripped = line.strip()
        # Match headings with number prefixes like "### 1.2.3 Title"
        m = re.match(r'^(#{1,6})\s+(\d+(?:\.\d+)*\.?)\s+(.+)$', stripped)
        if m:
            original_hashes = m.group(1)  # Keep original level from MinerU
            title = m.group(3)
            # Use original level - trust MinerU's heading detection
            result.append(f"{original_hashes} {title}")
        else:
            result.append(line)
    
    return '\n'.join(result)

def _wrap_abstract_italic(md: str) -> str:
    """Wrap the first substantial paragraph (likely abstract) in italic formatting."""
    lines = md.splitlines()
    result = []
    in_first_para = False
    first_para_lines = []
    found_first_para = False
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        
        # Skip initial headings and empty lines
        if not found_first_para:
            if not stripped or stripped.startswith('#'):
                result.append(line)
                continue
            
            # Found the start of first paragraph
            if len(stripped) > 50 and not stripped.startswith('```'):
                in_first_para = True
                found_first_para = True
                first_para_lines.append(line)
                continue
        
        # Collecting first paragraph
        if in_first_para:
            # End of paragraph
            if not stripped:
                # Wrap collected paragraph in italic
                combined = ' '.join(l.strip() for l in first_para_lines)
                result.append(f"_{combined}_")
                result.append(line)  # Add the empty line
                in_first_para = False
                continue
            else:
                first_para_lines.append(line)
                continue
        
        result.append(line)
    
    # If we ended while still in first paragraph
    if in_first_para and first_para_lines:
        combined = ' '.join(l.strip() for l in first_para_lines)
        result.append(f"_{combined}_")
    
    return '\n'.join(result)

# --- derive tiêu đề H1 theo mẫu "Public_048" từ stem (vd: "Public048") ---
def _derive_h1_from_stem(stem: str) -> str:
    # Chèn "_" giữa phần chữ và phần số nếu có
    m = re.match(r'^([A-Za-z]+)[ _-]?(\d+)$', stem)
    if m:
        return f"{m.group(1)}_{m.group(2)}"
    # Fallback: Title Case từ stem
    t = stem.replace('-', ' ').replace('_', ' ').strip()
    return re.sub(r'\s+', ' ', t).title()

# --- sao chép/đổi tên ảnh theo thứ tự xuất hiện ---
def _copy_and_renumber_images(
    md_text: str, raw_root: Path, img_out: Path,
    prefix: str, start_idx: int, target_ext: str
) -> Tuple[str, Dict[str,str]]:
    """
    Trả về (md_mới, mapping_old_basename->new_name).
    Chỉ copy những ảnh xuất hiện trong MD và đặt tên tuần tự như image1.png, image2.png, ...
    """
    ordered = _list_image_basenames_in_order(md_text)

    # gom ảnh nguồn (theo basename -> path)
    src_map: Dict[str, Path] = {}
    for p in _collect_all_image_files(raw_root):
        src_map.setdefault(p.name, p)

    # xoá ảnh cũ để đồng bộ
    if img_out.exists():
        for old in img_out.glob("*"):
            if old.is_file(): old.unlink()
    else:
        img_out.mkdir(parents=True, exist_ok=True)

    mapping={}
    idx = start_idx
    image_hashes: Set[str] = set()  # Để detect duplicate
    
    def _get_image_hash(img_path: Path) -> str:
        """Tính hash của ảnh để detect duplicate."""
        try:
            img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
            if img is not None:
                # Resize nhỏ để so sánh nhanh hơn
                small = cv2.resize(img, (32, 32))
                img_bytes = small.tobytes()
                return hashlib.md5(img_bytes).hexdigest()
        except Exception:
            pass
        # Fallback: hash của file size + first bytes
        try:
            with open(img_path, 'rb') as f:
                data = f.read(1024)
                return hashlib.md5(data).hexdigest()
        except Exception:
            return ""
    
    for base in ordered:
        src = src_map.get(base)
        if not src:
            # thử tìm theo basename nếu trong subdir
            cands = list(raw_root.rglob(Path(base).name))
            src = cands[0] if cands else None
        if not src:
            continue

        # Kiểm tra duplicate bằng hash
        img_hash = _get_image_hash(src)
        if img_hash and img_hash in image_hashes:
            # Ảnh duplicate, skip nhưng vẫn map để link đúng
            # Tìm ảnh đã copy trước đó
            for existing_base, existing_name in mapping.items():
                existing_src = src_map.get(existing_base)
                if existing_src and _get_image_hash(existing_src) == img_hash:
                    mapping[base] = existing_name
                    break
            continue
        
        if img_hash:
            image_hashes.add(img_hash)

        new_name = f"{prefix}{idx}{target_ext}"
        dst = img_out / new_name

        # cố gắng convert sang PNG; nếu đọc thất bại (svg, tiff đặc biệt) -> copy nguyên & đổi ext theo gốc
        ok_write = False
        try:
            img = cv2.imread(str(src), cv2.IMREAD_UNCHANGED)
            if img is not None:
                cv2.imwrite(str(dst), img)
                ok_write = True
        except Exception:
            ok_write = False
        if not ok_write:
            # giữ ext gốc (nhưng vẫn theo tên image{idx}.<ext gốc>)
            dst = img_out / f"{prefix}{idx}{src.suffix.lower()}"
            shutil.copy2(src, dst)
            new_name = dst.name

        mapping[base] = new_name
        idx += 1

    # Không copy tất cả ảnh từ raw_dir vì có thể có watermark/header/duplicate
    # Chỉ copy ảnh thực sự được reference trong MD content (đã xử lý ở trên)
    # Nếu không có ảnh nào trong MD, không copy gì cả (giữ nguyên behavior cũ)

    # thay link ảnh trong md theo mapping -> ![](images/imageN.png)
    def repl(m):
        base = Path(m.group(1).strip()).name
        new = mapping.get(base, base)
        return f"![](images/{new})"
    md_text = re.sub(r'!\[[^\]]*\]\(\s*images/([^)]+)\)', repl, md_text, flags=re.I)

    return md_text, mapping

# ================== MAIN ==================
def extract_one(pdf_path: Path) -> Path:
    """
    Chạy MinerU trên 1 PDF và hậu xử lý:
    - outputs/<stem>/main.md (H1 theo mẫu "Public_048" nếu có thể suy ra)
    - outputs/<stem>/images/image1.png, image2.png, ...
    """
    stem = pdf_path.stem
    out_dir = OUTPUT_DIR / stem
    raw_dir = out_dir / RAW_DIRNAME
    img_dir = out_dir / IMG_DIRNAME

    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    img_dir.mkdir(parents=True, exist_ok=True)

    cli = _find_cli()
    cmd = [cli, "-p", str(pdf_path), "-o", str(raw_dir), "-m", "auto", "--ocr"]
    rc, out, err = _run(cmd)
    if rc != 0:
        raise RuntimeError(
            "[mineru] CLI lỗi.\nCommand: {}\nrc={}\nstdout:\n{}\nstderr:\n{}".format(
                " ".join(cmd), rc, out, err
            )
        )

    md_src = _pick_markdown_file(raw_dir)
    if md_src is None:
        raise RuntimeError(f"[mineru] Không tìm thấy file .md trong {raw_dir}")

    # 1) Chuẩn hoá link ảnh về images/<basename>
    md_text = md_src.read_text(encoding="utf-8", errors="ignore")
    md_text = _rewrite_md_image_links_to_basename(md_text)

    # 2) Làm sạch Markdown (loại bảng header/watermark, bullets, heading số, code, khoảng trắng)
    md_text = _strip_bad_tables(md_text)
    md_text = _fix_bullets(md_text)
    md_text = _fix_headings(md_text)
    md_text = _wrap_code(md_text)
    md_text = _dedent(md_text)
    md_text = _clean_heading_numbers(md_text)  # Clean number prefixes from headings

    # 3) Đổi tên ảnh theo thứ tự xuất hiện -> imageN.png và thay link
    md_text, _ = _copy_and_renumber_images(
        md_text, raw_dir, img_dir, IMG_PREFIX, IMG_START, IMG_EXT
    )

    # 4) Wrap abstract/summary paragraph in italic BEFORE adding H1
    md_text = _wrap_abstract_italic(md_text)
    
    # 5) Always prepend H1 title from PDF stem (e.g., "# Public_001")
    h1 = _derive_h1_from_stem(stem)
    md_text = f"# {h1}\n\n{md_text}"

    write_text(out_dir / "main.md", md_text)

    # 5) Dọn _mineru_raw nếu không cần debug
    if not KEEP_DEBUG_RAW:
        if raw_dir.exists():
            for p in raw_dir.glob("*"):
                if p.is_file(): p.unlink()
                else: shutil.rmtree(p, ignore_errors=True)
            try: raw_dir.rmdir()
            except Exception: pass

    return out_dir

def extract_all(input_dir: Path):
    pdfs = sorted(Path(input_dir).glob("*.pdf"))
    if not pdfs:
        raise RuntimeError(f"Không tìm thấy PDF trong {input_dir}")
    for pdf in pdfs:
        print(f"[mineru] Extract: {pdf}")
        extract_one(pdf)
    print("[mineru] DONE")