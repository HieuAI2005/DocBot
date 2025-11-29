import os, re, json, torch
from typing import List, Dict

HEAD_NUM_RE = re.compile(r'^\s*(\d+(?:\.\d+)*)\s+(.+?)\s*$')
KW_HEAD = re.compile(r'^(Giới thiệu|Kết luận|Tổng quan|Dịch vụ|Các khía cạnh|Các thành phần|Ví dụ|Điện toán đám mây)\b', re.I)
CAPTION_RE = re.compile(r'^\s*(Hình|Hinh|Figure|Bảng)\s+\d+(?:\.\d+)?\s*[:\-–]\s*.+$', re.I)
CODE_KEYS = ("def ","class ","import ","from ","return","for ","while "," if ","cv2.","np.","torch.","random.","imutils.","fitz.","pdfplumber.")
CODE_CHARS = set("=()[]{};:.,#<>+-*/\\|\"'_%")

USE_LLM = os.environ.get("USE_LLM_TAGGER","0")=="1"

_llm = None
_tok = None

def _looks_code(text: str, font: str, size: float, size_med: float) -> bool:
    mono = any(k in (font or "").lower() for k in ("mono","consolas","courier","dejavu sans mono","fira code","source code"))
    ratio = sum(1 for ch in text if ch in CODE_CHARS)/max(1,len(text))
    return mono or any(k in text for k in CODE_KEYS) or ratio >= 0.12

def _looks_heading(text: str, size: float, size_med: float) -> bool:
    if HEAD_NUM_RE.match(text.strip()) and not text.strip().endswith(('.',':',';')): return True
    if KW_HEAD.match(text.strip()) and not text.strip().endswith(('.',':',';')) and len(text.strip())<=120: return True
    # bump theo size lớn hơn median
    return (size >= size_med + 1.2) and (not text.strip().endswith(('.',':',';'))) and len(text.strip())<=120

def _heuristic_tag(line_objs: List[Dict]) -> List[str]:
    tags=[]
    sizes=[o.get("size",0.0) for o in line_objs]; size_med = (sorted(sizes)[len(sizes)//2] if sizes else 0.0)
    for o in line_objs:
        t = (o.get("text") or "").strip()
        if not t:
            tags.append("Blank"); continue
        if CAPTION_RE.match(t): tags.append("Caption"); continue
        if _looks_heading(t, o.get("size",0.0), size_med): tags.append("Heading")
        elif _looks_code(t, o.get("font",""), o.get("size",0.0), size_med): tags.append("Code")
        else: tags.append("Paragraph")
    return tags

def _load_llm():
    global _llm, _tok
    if _llm is not None: return _llm, _tok
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_id="Qwen/Qwen2.5-1.5B-Instruct"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    _tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    _llm = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", low_cpu_mem_usage=True, dtype=dtype)
    _llm.eval()
    return _llm, _tok

def _llm_batch_tag(lines: List[str]) -> List[str]:
    if not lines: return []
    model, tok = _load_llm()
    sys = ("Bạn là bộ phân loại dòng. Với mỗi dòng, trả ra một nhãn duy nhất trong tập: "
           "Heading, Caption, Code, Watermark, Paragraph. Trả JSON list các chuỗi nhãn, không giải thích.")
    user = "Các dòng:\n" + "\n".join(f"- {s}" for s in lines) + "\nJSON:"
    enc = tok.apply_chat_template(
        [{"role":"system","content":sys},{"role":"user","content":user}],
        tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)
    out = model.generate(enc, max_new_tokens=256, do_sample=False, temperature=0.0, pad_token_id=tok.eos_token_id)
    txt = tok.decode(out[0], skip_special_tokens=True)
    m = re.search(r"\[(.*?)\]", txt, flags=re.S)
    if not m:
        return ["Paragraph"]*len(lines)
    try:
        arr = json.loads("["+m.group(1)+"]")
        arr = [str(x) for x in arr]
        if len(arr)!=len(lines): arr = (arr + ["Paragraph"]*len(lines))[:len(lines)]
        return arr
    except Exception:
        return ["Paragraph"]*len(lines)

def tag_lines(line_objs: List[Dict]) -> List[str]:
    """
    line_objs: list of {text, font, size, y, x}
    """
    tags = _heuristic_tag(line_objs)
    if not USE_LLM: return tags

    # gửi LLM cho các dòng mơ hồ & ngắn
    idxs=[i for i,(o,t) in enumerate(zip(line_objs,tags))
          if t in ("Paragraph","Heading") and len((o.get("text") or ""))<=120]
    if not idxs: return tags

    batch = [line_objs[i]["text"] for i in idxs]
    llm_out = _llm_batch_tag(batch)
    for j,i in enumerate(idxs):
        tags[i] = llm_out[j] or tags[i]
    return tags