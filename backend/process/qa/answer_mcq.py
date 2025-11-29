import os, json, re, math, pickle, pandas as pd, torch
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from ..config import (
    OUTPUT_DIR,
    INDEX_DIR,
    QUESTIONS_CSV,
    LLM,
    LLM_4BIT,
    MAX_CONTEXT_CHARS,
    THRESH_CHOOSE,
    THRESH_FALLBACK,
    THRESH_MULTI,
    THRESH_MULTI_GAP,
    MAX_MULTI_ANS,
    LOW_EVIDENCE_COUNT,
    LOW_EVIDENCE_SCORE,
    RELAX_CHOOSE_DELTA,
    RELAX_FALLBACK_DELTA,
    RELAX_MULTI_DELTA,
    MULTI_HINT_CHOOSE_DELTA,
    MULTI_HINT_MULTI_DELTA,
    SEED,
    USE_CHAIN_OF_THOUGHT,
    COT_MAX_REASONING_TOKENS,
    USE_SELF_CONSISTENCY,
    SELF_CONSISTENCY_SAMPLES,
    SELF_CONSISTENCY_TEMPERATURE,
    SAVE_REASONING_TRACES,
    REASONING_TRACE_DIR,
)
from ..rag.retrieve import Retriever
from ..utils import write_text, safe_json
from .few_shot_examples import get_few_shot_examples, detect_question_type
from .reasoning_parser import parse_cot_response, validate_and_fix_probs

torch.manual_seed(SEED)

def _load_llm():
    import torch, os
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # ưu tiên biến cấu hình nếu có
    model_id = LLM if (LLM and isinstance(LLM, str)) else "Qwen/Qwen2.5-1.5B-Instruct"

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)

    if torch.cuda.is_available():
        max_mem = {
            0: os.getenv("MAX_GPU_MEM", "3.7GiB"),   # key phải là số GPU
            "cpu": os.getenv("MAX_CPU_MEM", "16GiB"),
        }
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            low_cpu_mem_usage=True,
            dtype=dtype,                # dùng 'dtype' thay cho 'torch_dtype'
            max_memory=max_mem,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="cpu",
            low_cpu_mem_usage=True,
            dtype=dtype,
        )
    model.eval()
    return model, tok

def _build_system_prompt(question: str, use_cot: bool = True) -> str:
    """
    Build system prompt with appropriate few-shot examples.
    
    Args:
        question: The question text for type detection
        use_cot: Whether to use Chain-of-Thought reasoning
    
    Returns:
        System prompt string
    """
    if not use_cot:
        # Legacy simple prompt
        return """You are a QA assistant. Analyze evidence and select the best answer.
Return ONLY valid JSON: {"A": <prob>, "B": <prob>, "C": <prob>, "D": <prob>}
Probabilities must sum to 1.0."""
    
    # Chain-of-Thought prompt
    q_type = detect_question_type(question)
    examples = get_few_shot_examples(num_examples=2, question_type=q_type)
    
    return f"""You are a QA assistant. Use step-by-step reasoning to analyze evidence and select the best answer.

Follow this format:
1. Evidence Analysis: Summarize relevant facts from the evidence
2. Option Evaluation: Analyze each option and eliminate incorrect ones
3. Conclusion: State your final answer with justification
4. ANSWER: Provide probabilities as JSON

{examples}

Now analyze the following question using the same reasoning format."""

def _prompt(question: str, options: Dict[str,str], evid: List[Dict], use_cot: bool = True) -> str:
    """
    Build prompt for the LLM.
    
    Args:
        question: The question text
        options: Dictionary of answer options
        evid: List of evidence chunks
        use_cot: Whether to use Chain-of-Thought format
    
    Returns:
        Formatted prompt string
    """
    ctx=[]
    for i,ev in enumerate(evid,1):
        src = f"[{ev.get('doc_id', ev.get('pdf', ''))}]"
        text = ev.get("text","")
        if ev.get("is_table",False) and ev.get("text","")[:6]!="<table>":
            text = ev["text"]  # bảng đã chuyển sang text
        ctx.append(f"### Evidence {i} {src}\n{text.strip()}")
    
    ctx_text = "\n\n".join(ctx) if ctx else "No evidence found."
    if len(ctx_text) > MAX_CONTEXT_CHARS:
        ctx_text = ctx_text[:MAX_CONTEXT_CHARS] + "..."

    opt = "\n".join([f"{k}. {v}" for k,v in options.items()])
    
    if use_cot:
        user = f"""EVIDENCE:
{ctx_text}

QUESTION: {question}

OPTIONS:
{opt}

REASONING:"""
    else:
        user = f"""EVIDENCE:
{ctx_text}

QUESTION: {question}

OPTIONS:
{opt}

Provide answer as JSON only."""
    
    return user

def _gen_json(model, tok, system_prompt: str, user_prompt: str, temperature: float = 0.3, use_cot: bool = True) -> Tuple[Dict[str,float], str]:
    """
    Generate response from LLM with optional CoT reasoning.
    
    Args:
        model: The language model
        tok: The tokenizer
        system_prompt: System prompt
        user_prompt: User prompt
        temperature: Sampling temperature
        use_cot: Whether using Chain-of-Thought
    
    Returns:
        Tuple of (probabilities dict, raw response text)
    """
    enc = tok.apply_chat_template(
        [{"role":"system","content":system_prompt},{"role":"user","content":user_prompt}],
        tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)
    
    max_tokens = COT_MAX_REASONING_TOKENS if use_cot else 64
    
    out = model.generate(
        enc, max_new_tokens=max_tokens, do_sample=True, 
        temperature=temperature, top_p=0.95,
        pad_token_id=tok.eos_token_id
    )
    txt = tok.decode(out[0], skip_special_tokens=True)
    
    # Extract only assistant's response
    if "assistant" in txt.lower():
        parts = re.split(r'(?i)assistant', txt)
        txt = parts[-1].strip()
    
    if use_cot:
        # Use parser for CoT responses
        parsed = parse_cot_response(txt)
        probs = parsed.get("probabilities")
        probs = validate_and_fix_probs(probs)
        return probs, txt
    else:
        # Legacy JSON extraction
        m = re.search(r'\{[^}]+\}', txt)
        probs = safe_json(m.group(0) if m else "{}") or {"A":0.25,"B":0.25,"C":0.25,"D":0.25}
        return probs, txt

MULTI_HINT_RE = re.compile(r'(?:chọn|choose|select)\s*(?:tối đa\s*)?(\d+)', re.I)
MULTI_HINT_KEYWORDS = (
    "chọn tất cả", "chọn các đáp án đúng", "select all", "select two", "select three",
    "which of the following statements are true", "các đáp án đúng",
    "chọn 2 đáp án", "chọn 2 đáp án đúng", "chọn hai đáp án", "chọn ba đáp án",
    "những đáp án đúng", "những phát biểu đúng", "các câu đúng", "những câu đúng",
)

DEBUG_QA_DIR = OUTPUT_DIR / "debug" / "qa_low_conf"


def _detect_multi_hint(question: str) -> Dict[str, int | bool | None]:
    text = (question or "").lower()
    match = MULTI_HINT_RE.search(text)
    target = None
    if match:
        try:
            target = int(match.group(1))
        except Exception:
            target = None
    keyword_hit = any(kw in text for kw in MULTI_HINT_KEYWORDS)
    is_multi = bool(target and target >= 2) or keyword_hit
    if is_multi and target and target < 2:
        target = 2
    return {"is_multi": is_multi, "target": target}


def _clamp(val: float, lo: float = 0.05, hi: float = 0.95) -> float:
    return max(lo, min(hi, val))


def _derive_thresholds(hint: Dict, evid_quality: float, evid_count: int):
    choose_th = THRESH_CHOOSE
    fallback_th = THRESH_FALLBACK
    multi_th = THRESH_MULTI
    multi_gap = THRESH_MULTI_GAP
    max_multi = MAX_MULTI_ANS

    if hint.get("is_multi"):
        choose_th -= MULTI_HINT_CHOOSE_DELTA
        multi_th -= MULTI_HINT_MULTI_DELTA
        target = hint.get("target")
        if target:
            max_multi = max(2, min(MAX_MULTI_ANS, target))
        else:
            max_multi = max(2, min(MAX_MULTI_ANS, max_multi))

    if evid_count < LOW_EVIDENCE_COUNT or evid_quality < LOW_EVIDENCE_SCORE:
        choose_th -= RELAX_CHOOSE_DELTA
        fallback_th -= RELAX_FALLBACK_DELTA
        multi_th -= RELAX_MULTI_DELTA

    choose_th = _clamp(choose_th)
    fallback_th = _clamp(fallback_th)
    multi_th = _clamp(multi_th, lo=0.02)
    multi_gap = _clamp(multi_gap, hi=0.5)
    return choose_th, fallback_th, multi_th, multi_gap, max_multi


def _choose(
    p_map: Dict[str,float],
    choose_th: float,
    fallback_th: float,
    multi_th: float,
    multi_gap: float,
    max_multi: int,
) -> Tuple[List[str], Dict[str, Any]]:
    # Ensure dict has all required keys
    default_prob = 0.25
    for key in ['A', 'B', 'C', 'D']:
        if key not in p_map:
            p_map[key] = default_prob
    
    # Ensure all values are valid numbers
    clean_map = {}
    for k, v in p_map.items():
        if k not in ['A', 'B', 'C', 'D']:
            continue
        try:
            clean_map[k] = float(v) if v is not None else default_prob
        except (ValueError, TypeError):
            # If value is text or invalid, use default
            clean_map[k] = default_prob
    
    p_map = clean_map
    
    s = sum(p_map.values()) + 1e-9
    probs = {k: max(0.0, v)/s for k,v in p_map.items()}
    ordered = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)

    chosen = []
    forced_reason = None
    for key, prob in ordered:
        if prob >= choose_th and len(chosen) < max_multi:
            chosen.append(key)

    if not chosen:
        top_key, top_prob = ordered[0]
        if top_prob >= fallback_th:
            chosen = [top_key]
        else:
            forced_reason = "forced_low_conf_top"
            chosen = [top_key]

    top_key, top_prob = ordered[0]

    for key, prob in ordered:
        if key in chosen or len(chosen) >= max_multi:
            continue
        if prob < multi_th:
            continue
        if top_prob - prob <= multi_gap:
            chosen.append(key)

    result = sorted(set(chosen))
    debug = {
        "ordered": ordered,
        "forced_reason": forced_reason,
        "top_prob": top_prob,
    }
    return result, debug


def _log_low_confidence(idx: int, question: str, options: Dict[str,str],
                        evid: List[Dict], probs: Dict[str,float],
                        chosen: List[str], reason: str, ordered: List[Tuple[str,float]]):
    try:
        DEBUG_QA_DIR.mkdir(parents=True, exist_ok=True)
        fp = DEBUG_QA_DIR / f"q{idx:04d}.md"
        lines = [
            f"# Question {idx}",
            "",
            f"Reason: {reason}",
            "",
            "## Question",
            question,
            "",
            "## Options",
        ]
        for k,v in options.items():
            lines.append(f"- {k}: {v}")
        lines.extend(["", "## Probabilities"])
        for key, _ in ordered:
            lines.append(f"- {key}: {probs.get(key, 0.0):.3f}")
        lines.extend([
            "",
            f"Chosen: {', '.join(chosen) if chosen else '(none)'}",
            "",
            "## Evidence (top 3)",
        ])
        for idx_ev, ev in enumerate(evid[:3], 1):
            src = ev.get("pdf") or ev.get("source") or ""
            lines.append(f"### Evidence {idx_ev} {src}")
            lines.append(ev.get("text","")[:800])
            lines.append("")
        fp.write_text("\n".join(lines), encoding="utf-8")
    except Exception:
        pass

def _self_consistency(
    model, tok, system_prompt: str, user_prompt: str, 
    num_samples: int = 3, temperature: float = 0.7
) -> Tuple[Dict[str,float], List[Dict]]:
    """
    Generate multiple reasoning paths and vote for final answer.
    
    Args:
        model: The language model
        tok: The tokenizer
        system_prompt: System prompt
        user_prompt: User prompt
        num_samples: Number of reasoning paths to generate
        temperature: Sampling temperature
    
    Returns:
        Tuple of (aggregated probabilities, list of all samples)
    """
    samples = []
    all_probs = []
    
    for i in range(num_samples):
        probs, response = _gen_json(model, tok, system_prompt, user_prompt, 
                                     temperature=temperature, use_cot=True)
        samples.append({
            "sample_id": i + 1,
            "response": response,
            "probabilities": probs
        })
        all_probs.append(probs)
    
    # Aggregate probabilities through averaging
    aggregated = {"A": 0.0, "B": 0.0, "C": 0.0, "D": 0.0}
    for probs in all_probs:
        for key in aggregated:
            aggregated[key] += probs.get(key, 0.0)
    
    # Normalize
    for key in aggregated:
        aggregated[key] /= num_samples
    
    return aggregated, samples

def _save_reasoning_trace(
    idx: int, question: str, options: Dict[str,str],
    evid: List[Dict], response: str, probs: Dict[str,float],
    chosen: List[str], samples: Optional[List[Dict]] = None
):
    """
    Save comprehensive reasoning trace for review.
    
    Args:
        idx: Question index
        question: The question text
        options: Answer options
        evid: Evidence chunks
        response: LLM response with reasoning
        probs: Final probabilities
        chosen: Chosen answers
        samples: Self-consistency samples (if used)
    """
    if not SAVE_REASONING_TRACES:
        return
    
    try:
        REASONING_TRACE_DIR.mkdir(parents=True, exist_ok=True)
        fp = REASONING_TRACE_DIR / f"q{idx:04d}_reasoning.md"
        
        lines = [
            f"# Question {idx} - Reasoning Trace",
            "",
            "## Question",
            question,
            "",
            "## Options",
        ]
        for k, v in options.items():
            lines.append(f"- **{k}**: {v}")
        
        lines.extend(["", "## Evidence"])
        for i, ev in enumerate(evid, 1):
            src = ev.get("doc_id", ev.get("pdf", "unknown"))
            score = ev.get("score", 0.0)
            lines.append(f"### Evidence {i} [{src}] (score: {score:.3f})")
            lines.append(ev.get("text", "")[:600] + "...")
            lines.append("")
        
        if samples:
            lines.extend(["", "## Self-Consistency Samples", ""])
            for sample in samples:
                lines.append(f"### Sample {sample['sample_id']}")
                lines.append("```")
                lines.append(sample["response"][:800])
                lines.append("```")
                lines.append(f"**Probabilities**: {sample['probabilities']}")
                lines.append("")
            lines.extend(["", "## Aggregated Result"])
        else:
            lines.extend(["", "## Reasoning"])
            lines.append("```")
            lines.append(response)
            lines.append("```")
            lines.append("")
        
        lines.extend([
            "",
            "## Final Answer",
            f"**Probabilities**: {probs}",
            f"**Chosen**: {', '.join(chosen) if chosen else '(none)'}",
        ])
        
        fp.write_text("\n".join(lines), encoding="utf-8")
    except Exception as e:
        print(f"Warning: Failed to save reasoning trace for Q{idx}: {e}")

def answer_all():
    retr = Retriever()
    model, tok = _load_llm()

    df = pd.read_csv(QUESTIONS_CSV)
    records = []
    task_rows = []  # "num_correct,answers"
    
    print(f"Processing {len(df)} questions...")
    print(f"CoT enabled: {USE_CHAIN_OF_THOUGHT}")
    print(f"Self-consistency enabled: {USE_SELF_CONSISTENCY}")
    print(f"Reasoning traces: {SAVE_REASONING_TRACES}")
    print()

    for i, r in df.iterrows():
        q = str(r.iloc[0])
        opts = {"A":str(r.iloc[1]), "B":str(r.iloc[2]), "C":str(r.iloc[3]), "D":str(r.iloc[4])}
        
        # Enhanced retrieval with query expansion
        evid = retr.query(q, options=opts)
        best_score = max((ev.get("score", 0.0) for ev in evid), default=0.0)
        hint = _detect_multi_hint(q)
        choose_th, fallback_th, multi_th, multi_gap, max_multi = _derive_thresholds(
            hint, best_score, len(evid)
        )
        
        # Build prompts with CoT if enabled
        use_cot = USE_CHAIN_OF_THOUGHT
        system_prompt = _build_system_prompt(q, use_cot=use_cot)
        user_prompt = _prompt(q, opts, evid, use_cot=use_cot)
        
        # Generate answer with optional self-consistency
        samples = None
        if USE_SELF_CONSISTENCY and use_cot:
            probs, samples = _self_consistency(
                model, tok, system_prompt, user_prompt,
                num_samples=SELF_CONSISTENCY_SAMPLES,
                temperature=SELF_CONSISTENCY_TEMPERATURE
            )
            response = f"Self-consistency with {len(samples)} samples"
        else:
            probs, response = _gen_json(model, tok, system_prompt, user_prompt, 
                                        temperature=0.3, use_cot=use_cot)
        
        # Choose final answer
        chosen, choose_meta = _choose(probs, choose_th, fallback_th, multi_th, multi_gap, max_multi)
        
        # Save reasoning trace
        _save_reasoning_trace(i + 1, q, opts, evid, response, probs, chosen, samples)

        # Log low confidence cases to old format
        if choose_meta.get("forced_reason"):
            _log_low_confidence(
                i + 1, q, opts, evid, probs, chosen,
                choose_meta["forced_reason"],
                choose_meta.get("ordered", [])
            )
        elif not evid:
            _log_low_confidence(
                i + 1, q, opts, evid, probs, chosen,
                "no_evidence",
                choose_meta.get("ordered", [])
            )

        # Standardize answer format
        chosen_norm = sorted(list(dict.fromkeys([c.strip().upper() for c in chosen if c])))
        ans_join = ",".join(chosen_norm)
        ans_out = f"\"{ans_join}\"" if "," in ans_join else ans_join
        num_correct = len(chosen_norm)

        records.append({"id": i+1, "answer": ans_join})
        task_rows.append(f"{num_correct},{ans_out}")
        
        # Progress indicator
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(df)} questions...")

    # ----- Ghi answer.md -----
    parts = ["### TASK EXTRACT"]
    # (giữ hành vi cũ: dump các main.md để dễ review; có thể bỏ nếu muốn gọn)
    for pdf_dir in sorted(OUTPUT_DIR.glob("*/")):
        if (pdf_dir / "main.md").exists():
            parts.append(f"# {pdf_dir.name}\n")
            parts.append((pdf_dir / "main.md").read_text(encoding="utf-8"))

    # block kết quả theo format yêu cầu
    parts.append("\n\n### TASK QA  num_correct,answers\n")
    parts.extend(task_rows)

    write_text(OUTPUT_DIR / "answer.md", "\n".join(parts))
    print("Wrote", OUTPUT_DIR / "answer.md")

    # ----- Ghi thêm answers.csv -----
    (OUTPUT_DIR / "answers.csv").write_text(
        "num_correct,answers\n" + "\n".join(task_rows) + "\n",
        encoding="utf-8"
    )
    print("Wrote", OUTPUT_DIR / "answers.csv")

if __name__ == "__main__":
    answer_all()