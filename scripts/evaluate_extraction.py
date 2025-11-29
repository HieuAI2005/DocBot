#!/usr/bin/env python3
"""
Script để đánh giá chất lượng extraction bằng cách so sánh outputs với training data.
"""

import re
from pathlib import Path
from difflib import SequenceMatcher
from typing import Dict, List, Tuple
import json


def normalize_text(text: str) -> str:
    """Chuẩn hóa text để so sánh (loại bỏ whitespace thừa, lowercase)."""
    # Loại bỏ multiple spaces/newlines
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def calculate_text_similarity(text1: str, text2: str) -> float:
    """Tính độ tương đồng giữa 2 đoạn text (0.0 - 1.0)."""
    normalized1 = normalize_text(text1)
    normalized2 = normalize_text(text2)
    return SequenceMatcher(None, normalized1, normalized2).ratio()


def extract_headings(md_text: str) -> List[Tuple[int, str]]:
    """Trích xuất tất cả headings từ markdown (level, title)."""
    headings = []
    for line in md_text.splitlines():
        match = re.match(r'^(#{1,6})\s+(.+)$', line.strip())
        if match:
            level = len(match.group(1))
            title = match.group(2).strip()
            headings.append((level, title))
    return headings


def compare_headings(headings1: List[Tuple[int, str]], 
                     headings2: List[Tuple[int, str]]) -> Dict[str, float]:
    """So sánh cấu trúc headings giữa 2 files."""
    # Tính tỉ lệ heading khớp
    if not headings1 and not headings2:
        return {"count_match": 1.0, "structure_match": 1.0}
    
    if not headings1 or not headings2:
        return {"count_match": 0.0, "structure_match": 0.0}
    
    # Tỉ lệ số lượng heading
    count_ratio = min(len(headings1), len(headings2)) / max(len(headings1), len(headings2))
    
    # Tỉ lệ cấu trúc (level + title) khớp
    matches = 0
    for h1, h2 in zip(headings1, headings2):
        level1, title1 = h1
        level2, title2 = h2
        
        # Chuẩn hóa title để so sánh
        title1_norm = normalize_text(title1.lower())
        title2_norm = normalize_text(title2.lower())
        
        # Khớp nếu level giống và title có độ tương đồng > 0.8
        if level1 == level2 and SequenceMatcher(None, title1_norm, title2_norm).ratio() > 0.8:
            matches += 1
    
    structure_ratio = matches / max(len(headings1), len(headings2))
    
    return {
        "count_match": count_ratio,
        "structure_match": structure_ratio
    }


def count_images(md_text: str) -> int:
    """Đếm số lượng ảnh trong markdown."""
    return len(re.findall(r'!\[.*?\]\(.*?\)', md_text))


def count_images_in_dir(img_dir: Path) -> int:
    """Đếm số lượng file ảnh trong thư mục."""
    if not img_dir.exists():
        return 0
    return len(list(img_dir.glob("*.png")) + list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.jpeg")))


def evaluate_single_pdf(output_dir: Path, training_dir: Path) -> Dict:
    """Đánh giá 1 PDF extraction."""
    results = {
        "name": output_dir.name,
        "exists": False,
        "text_similarity": 0.0,
        "heading_count_match": 0.0,
        "heading_structure_match": 0.0,
        "image_count_match": 0.0,
        "overall_score": 0.0
    }
    
    # Check if both directories exist
    output_md = output_dir / "main.md"
    training_md = training_dir / "main.md"
    
    if not output_md.exists():
        results["error"] = "Output main.md not found"
        return results
    
    if not training_md.exists():
        results["error"] = "Training main.md not found"
        return results
    
    results["exists"] = True
    
    # Read markdown files
    output_text = output_md.read_text(encoding="utf-8", errors="ignore")
    training_text = training_md.read_text(encoding="utf-8", errors="ignore")
    
    # 1. Text similarity
    results["text_similarity"] = calculate_text_similarity(output_text, training_text)
    
    # 2. Heading comparison
    output_headings = extract_headings(output_text)
    training_headings = extract_headings(training_text)
    
    heading_metrics = compare_headings(output_headings, training_headings)
    results["heading_count_match"] = heading_metrics["count_match"]
    results["heading_structure_match"] = heading_metrics["structure_match"]
    
    # 3. Image count comparison
    output_img_count = count_images_in_dir(output_dir / "images")
    training_img_count = count_images_in_dir(training_dir / "images")
    
    if output_img_count == 0 and training_img_count == 0:
        results["image_count_match"] = 1.0
    elif output_img_count == 0 or training_img_count == 0:
        results["image_count_match"] = 0.0
    else:
        results["image_count_match"] = min(output_img_count, training_img_count) / max(output_img_count, training_img_count)
    
    # 4. Overall score (weighted average)
    results["overall_score"] = (
        results["text_similarity"] * 0.5 +
        results["heading_structure_match"] * 0.3 +
        results["image_count_match"] * 0.2
    )
    
    return results


def evaluate_all(output_base: Path, training_base: Path) -> Dict:
    """Đánh giá tất cả PDFs."""
    all_results = []
    
    # Lấy danh sách tất cả thư mục output
    output_dirs = sorted([d for d in output_base.iterdir() if d.is_dir() and d.name.startswith("Public")])
    
    print(f"Found {len(output_dirs)} output directories to evaluate\n")
    
    for output_dir in output_dirs:
        training_dir = training_base / output_dir.name
        result = evaluate_single_pdf(output_dir, training_dir)
        all_results.append(result)
        
        # Print progress
        if result["exists"]:
            print(f"{result['name']}: Overall={result['overall_score']:.2%} "
                  f"(Text={result['text_similarity']:.2%}, "
                  f"Heading={result['heading_structure_match']:.2%}, "
                  f"Image={result['image_count_match']:.2%})")
        else:
            print(f"{result['name']}: {result.get('error', 'Not evaluated')}")
    
    # Calculate summary statistics
    valid_results = [r for r in all_results if r["exists"]]
    
    if valid_results:
        summary = {
            "total_files": len(all_results),
            "evaluated_files": len(valid_results),
            "avg_text_similarity": sum(r["text_similarity"] for r in valid_results) / len(valid_results),
            "avg_heading_match": sum(r["heading_structure_match"] for r in valid_results) / len(valid_results),
            "avg_image_match": sum(r["image_count_match"] for r in valid_results) / len(valid_results),
            "avg_overall_score": sum(r["overall_score"] for r in valid_results) / len(valid_results),
        }
    else:
        summary = {
            "total_files": len(all_results),
            "evaluated_files": 0,
            "avg_text_similarity": 0.0,
            "avg_heading_match": 0.0,
            "avg_image_match": 0.0,
            "avg_overall_score": 0.0,
        }
    
    return {
        "summary": summary,
        "details": all_results
    }


def main():
    """Main function."""
    # Paths
    project_root = Path(__file__).parent.parent
    output_base = project_root / "outputs"
    training_base = project_root / "inputs" / "training_output"
    
    print("=" * 60)
    print("PDF EXTRACTION EVALUATION")
    print("=" * 60)
    print(f"Output directory: {output_base}")
    print(f"Training directory: {training_base}")
    print("=" * 60)
    print()
    
    # Run evaluation
    results = evaluate_all(output_base, training_base)
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    summary = results["summary"]
    print(f"Total files: {summary['total_files']}")
    print(f"Evaluated files: {summary['evaluated_files']}")
    print(f"Average text similarity: {summary['avg_text_similarity']:.2%}")
    print(f"Average heading match: {summary['avg_heading_match']:.2%}")
    print(f"Average image match: {summary['avg_image_match']:.2%}")
    print(f"Average overall score: {summary['avg_overall_score']:.2%}")
    print("=" * 60)
    
    # Save results to JSON
    output_file = project_root / "evaluation_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nDetailed results saved to: {output_file}")


if __name__ == "__main__":
    main()
