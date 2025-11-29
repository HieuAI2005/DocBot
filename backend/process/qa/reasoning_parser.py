"""
Parser for extracting structured information from Chain-of-Thought reasoning responses.
Handles both Vietnamese and English responses with multiple fallback strategies.
"""

import re
import json
from typing import Dict, List, Optional, Tuple


def extract_answer_probs(text: str) -> Optional[Dict[str, float]]:
    """
    Extract answer probabilities from LLM response.
    Tries multiple patterns in order of specificity.
    
    Args:
        text: The full LLM response text
    
    Returns:
        Dictionary with probabilities for A, B, C, D or None if not found
    """
    # Pattern 1: JSON object with ANSWER prefix
    pattern1 = r'ANSWER:\s*({[^}]+})'
    match = re.search(pattern1, text, re.IGNORECASE)
    if match:
        try:
            return _parse_json_probs(match.group(1))
        except:
            pass
    
    # Pattern 2: Any JSON object with A, B, C, D keys
    pattern2 = r'\{\s*"A"\s*:\s*[\d.]+.*?"D"\s*:\s*[\d.]+\s*\}'
    match = re.search(pattern2, text, re.DOTALL)
    if match:
        try:
            return _parse_json_probs(match.group(0))
        except:
            pass
    
    # Pattern 3: Inline format like A: 0.8, B: 0.1, C: 0.05, D: 0.05
    pattern3 = r'A[:\s]+(\d*\.?\d+).*?B[:\s]+(\d*\.?\d+).*?C[:\s]+(\d*\.?\d+).*?D[:\s]+(\d*\.?\d+)'
    match = re.search(pattern3, text, re.IGNORECASE | re.DOTALL)
    if match:
        try:
            return {
                "A": float(match.group(1)),
                "B": float(match.group(2)),
                "C": float(match.group(3)),
                "D": float(match.group(4))
            }
        except:
            pass
    
    # Pattern 4: Extract from conclusion (e.g., "đáp án là B")
    conclusion_answer = extract_conclusion_answer(text)
    if conclusion_answer:
        return _create_probs_from_letters(conclusion_answer)
    
    return None


def _parse_json_probs(json_str: str) -> Dict[str, float]:
    """Parse JSON string to probability dict with validation."""
    # Clean up common issues
    json_str = json_str.strip()
    # Handle single quotes
    json_str = json_str.replace("'", '"')
    
    probs = json.loads(json_str)
    
    # Validate and normalize
    result = {}
    for key in ['A', 'B', 'C', 'D']:
        val = probs.get(key, 0.0)
        # Handle string values
        if isinstance(val, str):
            val = float(val) if val else 0.0
        result[key] = max(0.0, min(1.0, float(val)))
    
    # Normalize to sum to 1.0
    total = sum(result.values())
    if total > 0:
        result = {k: v/total for k, v in result.items()}
    else:
        # Fallback to uniform
        result = {k: 0.25 for k in ['A', 'B', 'C', 'D']}
    
    return result


def extract_conclusion_answer(text: str) -> Optional[List[str]]:
    """
    Extract answer letters from conclusion statement.
    Handles both single and multiple answers.
    
    Args:
        text: The full response text
    
    Returns:
        List of answer letters (e.g., ['B'] or ['A', 'C', 'D'])
    """
    # Vietnamese patterns
    patterns = [
        r'[Cc]onclusion:.*?đáp án là\s*([A-D](?:\s*,?\s*(?:và\s+)?[A-D])*)',
        r'[Cc]onclusion:.*?answer is\s*([A-D](?:\s*,?\s*(?:and\s+)?[A-D])*)',
        r'đáp án:?\s*([A-D](?:\s*,?\s*(?:và\s+)?[A-D])*)',
        r'[Tt]he answer is:?\s*([A-D](?:\s*,?\s*(?:and\s+)?[A-D])*)',
        r'[Cc]họn:?\s*([A-D](?:\s*,?\s*(?:và\s+)?[A-D])*)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            answer_str = match.group(1)
            # Extract all letters
            letters = re.findall(r'[A-D]', answer_str.upper())
            if letters:
                return list(set(letters))  # Remove duplicates
    
    return None


def _create_probs_from_letters(letters: List[str]) -> Dict[str, float]:
    """
    Create probability distribution from selected letters.
    Distributes probability equally among selected answers.
    
    Args:
        letters: List of selected answer letters
    
    Returns:
        Probability dictionary
    """
    if not letters:
        return {"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25}
    
    prob_per_answer = 1.0 / len(letters)
    return {
        "A": prob_per_answer if "A" in letters else 0.0,
        "B": prob_per_answer if "B" in letters else 0.0,
        "C": prob_per_answer if "C" in letters else 0.0,
        "D": prob_per_answer if "D" in letters else 0.0,
    }


def extract_reasoning_steps(text: str) -> Dict[str, str]:
    """
    Extract structured reasoning steps from CoT response.
    
    Args:
        text: The full response text
    
    Returns:
        Dictionary with reasoning components
    """
    result = {
        "evidence_analysis": "",
        "option_evaluation": "",
        "conclusion": "",
        "full_reasoning": ""
    }
    
    # Try to extract REASONING block
    reasoning_match = re.search(r'REASONING:(.*?)(?:ANSWER:|$)', text, re.DOTALL | re.IGNORECASE)
    if reasoning_match:
        reasoning_text = reasoning_match.group(1).strip()
        result["full_reasoning"] = reasoning_text
        
        # Extract evidence analysis (step 1)
        evidence_match = re.search(r'1\.\s*Evidence Analysis:?\s*(.*?)(?=\n\d+\.|\n\n|$)', 
                                   reasoning_text, re.DOTALL | re.IGNORECASE)
        if evidence_match:
            result["evidence_analysis"] = evidence_match.group(1).strip()
        
        # Extract option evaluation (step 2)
        option_match = re.search(r'2\.\s*(?:Option Evaluation|Calculation):?\s*(.*?)(?=\n\d+\.|\n\n|$)', 
                                 reasoning_text, re.DOTALL | re.IGNORECASE)
        if option_match:
            result["option_evaluation"] = option_match.group(1).strip()
        
        # Extract conclusion (last step)
        conclusion_match = re.search(r'\d+\.\s*Conclusion:?\s*(.*?)(?=\n\n|$)', 
                                     reasoning_text, re.DOTALL | re.IGNORECASE)
        if conclusion_match:
            result["conclusion"] = conclusion_match.group(1).strip()
    
    return result


def extract_eliminated_options(text: str) -> Dict[str, str]:
    """
    Extract elimination reasoning for each option.
    
    Args:
        text: The full response text
    
    Returns:
        Dictionary mapping option letter to elimination reason
    """
    eliminated = {}
    
    # Look for option evaluation patterns
    option_patterns = [
        r'Option ([A-D])[:\s]+([^-]*?)(?:loại bỏ|Sai|incorrect|wrong|eliminate)',
        r'-\s*([A-D])[:\s]+([^-]*?)(?:loại bỏ|Sai|incorrect|wrong)',
    ]
    
    for pattern in option_patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE | re.DOTALL):
            option = match.group(1).upper()
            reason = match.group(2).strip()
            if option not in eliminated and reason:
                eliminated[option] = reason[:200]  # Truncate long reasons
    
    return eliminated


def parse_cot_response(text: str) -> Dict:
    """
    Main parser function to extract all information from CoT response.
    
    Args:
        text: The full LLM response with CoT reasoning
    
    Returns:
        Dictionary with all extracted information
    """
    return {
        "probabilities": extract_answer_probs(text),
        "selected_answers": extract_conclusion_answer(text),
        "reasoning_steps": extract_reasoning_steps(text),
        "eliminated_options": extract_eliminated_options(text),
        "raw_response": text
    }


def validate_and_fix_probs(probs: Optional[Dict[str, float]]) -> Dict[str, float]:
    """
    Validate probability distribution and fix if needed.
    
    Args:
        probs: Probability dictionary or None
    
    Returns:
        Valid probability dictionary
    """
    if not probs:
        return {"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25}
    
    # Ensure all keys present
    for key in ['A', 'B', 'C', 'D']:
        if key not in probs:
            probs[key] = 0.0
    
    # Ensure all values are valid floats
    for key in ['A', 'B', 'C', 'D']:
        try:
            probs[key] = float(probs[key])
            if probs[key] < 0 or probs[key] > 1:
                probs[key] = max(0.0, min(1.0, probs[key]))
        except (ValueError, TypeError):
            probs[key] = 0.0
    
    # Normalize to sum to 1.0
    total = sum(probs.values())
    if total > 0:
        probs = {k: v/total for k, v in probs.items()}
    else:
        probs = {"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25}
    
    return probs
