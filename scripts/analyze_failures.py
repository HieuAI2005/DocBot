#!/usr/bin/env python3
"""
Phân tích các file có điểm thấp trong evaluation để tìm pattern lỗi.
"""

import json
from pathlib import Path
from collections import defaultdict

def analyze_failures():
    """Phân tích evaluation results để tìm pattern lỗi."""
    project_root = Path(__file__).parent.parent
    eval_file = project_root / "evaluation_results.json"
    
    with open(eval_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    details = data["details"]
    
    # Phân loại theo điểm số
    very_low = []  # < 0.3
    low = []       # 0.3 - 0.5
    medium = []    # 0.5 - 0.7
    high = []      # > 0.7
    
    for item in details:
        if not item.get("exists", False):
            continue
        score = item["overall_score"]
        if score < 0.3:
            very_low.append(item)
        elif score < 0.5:
            low.append(item)
        elif score < 0.7:
            medium.append(item)
        else:
            high.append(item)
    
    print("=" * 80)
    print("PHÂN TÍCH CÁC FILE ĐIỂM THẤP")
    print("=" * 80)
    print(f"\nTổng số file: {len(details)}")
    print(f"Rất thấp (<30%): {len(very_low)}")
    print(f"Thấp (30-50%): {len(low)}")
    print(f"Trung bình (50-70%): {len(medium)}")
    print(f"Cao (>70%): {len(high)}")
    
    # Phân tích nguyên nhân chính
    print("\n" + "=" * 80)
    print("TOP 20 FILE ĐIỂM THẤP NHẤT")
    print("=" * 80)
    
    all_sorted = sorted([d for d in details if d.get("exists")], 
                       key=lambda x: x["overall_score"])
    
    for i, item in enumerate(all_sorted[:20], 1):
        print(f"\n{i}. {item['name']}: {item['overall_score']:.2%}")
        print(f"   Text: {item['text_similarity']:.2%} | "
              f"Heading: {item['heading_structure_match']:.2%} | "
              f"Image: {item['image_count_match']:.2%}")
        
        # Xác định vấn đề chính
        issues = []
        if item['text_similarity'] < 0.5:
            issues.append("Text similarity rất thấp")
        if item['heading_structure_match'] < 0.3:
            issues.append("Heading structure kém")
        if item['image_count_match'] < 0.5:
            issues.append("Thiếu ảnh")
        
        if issues:
            print(f"   Vấn đề: {', '.join(issues)}")
    
    # Phân tích pattern
    print("\n" + "=" * 80)
    print("PHÂN TÍCH PATTERN LỖI")
    print("=" * 80)
    
    text_low = sum(1 for d in details if d.get("exists") and d.get("text_similarity", 1) < 0.5)
    heading_low = sum(1 for d in details if d.get("exists") and d.get("heading_structure_match", 1) < 0.3)
    image_low = sum(1 for d in details if d.get("exists") and d.get("image_count_match", 1) < 0.5)
    
    print(f"\nSố file có text similarity < 50%: {text_low}")
    print(f"Số file có heading structure < 30%: {heading_low}")
    print(f"Số file có image match < 50%: {image_low}")
    
    # Tìm file có nhiều vấn đề cùng lúc
    print("\n" + "=" * 80)
    print("FILE CÓ NHIỀU VẤN ĐỀ CÙNG LÚC (cần ưu tiên fix)")
    print("=" * 80)
    
    critical = []
    for item in details:
        if not item.get("exists"):
            continue
        count = 0
        if item.get("text_similarity", 1) < 0.5:
            count += 1
        if item.get("heading_structure_match", 1) < 0.3:
            count += 1
        if item.get("image_count_match", 1) < 0.5:
            count += 1
        
        if count >= 2:
            critical.append((item, count))
    
    critical.sort(key=lambda x: (x[1], -x[0]["overall_score"]), reverse=True)
    
    for item, count in critical[:15]:
        print(f"\n{item['name']}: {item['overall_score']:.2%} ({count} vấn đề)")
        print(f"  Text: {item['text_similarity']:.2%}, "
              f"Heading: {item['heading_structure_match']:.2%}, "
              f"Image: {item['image_count_match']:.2%}")
    
    # Gợi ý cải thiện
    print("\n" + "=" * 80)
    print("GỢI Ý CẢI THIỆN")
    print("=" * 80)
    
    suggestions = []
    
    if text_low > len(details) * 0.2:
        suggestions.append("1. Text similarity thấp: Cần cải thiện OCR/PDF parsing")
        suggestions.append("   - Tăng IMG_DPI trong config")
        suggestions.append("   - Kiểm tra MinerU OCR settings")
        suggestions.append("   - Xử lý đặc biệt cho PDF scan/phức tạp")
    
    if heading_low > len(details) * 0.2:
        suggestions.append("\n2. Heading structure kém: Cần cải thiện heading detection")
        suggestions.append("   - Sử dụng font size/layout từ MinerU thay vì regex")
        suggestions.append("   - Bật USE_LLM_TAGGER=1 để dùng LLM phân loại")
        suggestions.append("   - Giảm aggressive heading normalization")
    
    if image_low > len(details) * 0.1:
        suggestions.append("\n3. Image extraction: Cần cải thiện image detection")
        suggestions.append("   - Kiểm tra _copy_and_renumber_images logic")
        suggestions.append("   - Fallback copy tất cả ảnh nếu không match")
        suggestions.append("   - Xử lý table-to-image conversion")
    
    for s in suggestions:
        print(s)
    
    # Lưu danh sách file cần fix
    critical_files = [item["name"] for item, _ in critical[:20]]
    output_file = project_root / "critical_files.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({"critical_files": critical_files, "count": len(critical_files)}, 
                 f, indent=2, ensure_ascii=False)
    
    print(f"\n\nDanh sách {len(critical_files)} file cần ưu tiên đã lưu vào: {output_file}")

if __name__ == "__main__":
    analyze_failures()

