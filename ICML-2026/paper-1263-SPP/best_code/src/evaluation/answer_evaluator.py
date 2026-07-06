"""
Answer evaluation module
Supports evaluation methods for 4 question types (multiple_choice, factual, code, reasoning)
Uses the EM Match method for answer matching
References the evaluation methods under the benchmark directory (but does not use an LLM judge)
Supports evaluating 1-3 answer sets (each scored with EM, then averaged)

Important:
- All accuracy evaluation is based on the EM match ratio (0-1)
- Match ratio = how many of the keywords in the expected answer (after removing common words) are matched by the predicted answer
- Match ratio = number of matched keywords / total number of keywords in the expected answer
- There is no accuracy evaluation method based on semantic similarity
"""

import re
import numpy as np
from typing import Dict, Optional, List

# Try to import the key content extraction module (if it exists)
try:
    from scripts.key_content_extractor import extract_key_content, calculate_key_content_similarity
except ImportError:
    # If the import fails, define empty functions
    def extract_key_content(result, expected):
        return {'result': {}, 'expected': {}}
    def calculate_key_content_similarity(key_content):
        return {'key_content_similarity': 0.0, 'final_answer_match': False}


def _text_to_number(text):
    """Convert a textual number into a numeric string"""
    text_lower = text.lower().strip()

    # Number mapping
    number_map = {
        'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
        'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9',
        'ten': '10', 'eleven': '11', 'twelve': '12', 'thirteen': '13',
        'fourteen': '14', 'fifteen': '15', 'sixteen': '16', 'seventeen': '17',
        'eighteen': '18', 'nineteen': '19', 'twenty': '20', 'thirty': '30',
        'forty': '40', 'fifty': '50', 'sixty': '60', 'seventy': '70',
        'eighty': '80', 'ninety': '90', 'hundred': '100', 'thousand': '1000'
    }
    
    # Direct match
    if text_lower in number_map:
        return number_map[text_lower]

    # Handle compound numbers (e.g. "three hundred")
    words = text_lower.split()
    if len(words) == 2:
        first = words[0]
        second = words[1]
        if first in number_map and second in ['hundred', 'thousand']:
            base = int(number_map[first])
            multiplier = 100 if second == 'hundred' else 1000
            return str(base * multiplier)
    
    return None


def extract_final_answer(text, return_all_candidates=False):
    """
    Improved final-answer extraction (supports numbers, textual numbers, and text answers)
    Handles a wide variety of output formats and precisely matches the key content
    """
    if not text:
        return [] if return_all_candidates else None
    
    text = text.strip()
    candidates = []
    
    # Method 1: Extract the content after #### (standard format, highest priority=10)
    match = re.search(r'####\s*([^\n]+)', text)
    if match:
        answer = match.group(1).strip()
        if ',' in answer:
            numbers = re.findall(r'\b(\d+\.?\d*)\b', answer)
            if numbers:
                for num in numbers:
                    candidates.append((num, 10, 'explicit_marker'))
                candidates.append((answer, 10, 'explicit_marker'))
            else:
                candidates.append((answer, 10, 'explicit_marker'))
        elif re.match(r'^[\d.]+$', answer):
            candidates.append((answer, 10, 'explicit_marker'))
        else:
            num = _text_to_number(answer.lower())
            if num:
                candidates.append((num, 10, 'explicit_marker'))
            else:
                num_match = re.search(r'\b(\d+\.?\d*)\b', answer)
                if num_match:
                    candidates.append((num_match.group(1), 10, 'explicit_marker'))
                else:
                    cleaned = re.sub(r'^(is|are|was|were|the|a|an)\s+', '', answer, flags=re.IGNORECASE).strip()
                    if cleaned and len(cleaned) < 50:
                        candidates.append((cleaned, 10, 'explicit_marker'))
    
    # Method 2: Extract the \boxed{content} format (LaTeX format, priority=9)
    match = re.search(r'\\boxed\{([^}]+)\}', text)
    if match:
        answer = match.group(1).strip()
        if re.match(r'^[\d.]+$', answer):
            candidates.append((answer, 9, 'latex_boxed'))
        else:
            num = _text_to_number(answer.lower())
            if num:
                candidates.append((num, 9, 'latex_boxed'))
            else:
                candidates.append((answer, 9, 'latex_boxed'))
    
    # Method 3: Extract the "answer: content" format (priority=8)
    patterns = [
        r'answer[:\s]+([^\n.]+?)(?:[.,;!?]|$)',
        r'the answer is[:\s]+([^\n.]+?)(?:[.,;!?]|$)',
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            answer = match.group(1).strip()
            answer = re.sub(r'^(is|are|was|were)\s+', '', answer, flags=re.IGNORECASE).strip()
            num_match = re.search(r'([\d.]+)', answer)
            if num_match:
                candidates.append((num_match.group(1), 8, 'answer_marker'))
            else:
                num = _text_to_number(answer.lower())
                if num:
                    candidates.append((num, 8, 'answer_marker'))
                else:
                    answer = re.sub(r'[.,;!?]+$', '', answer).strip()
                    if answer:
                        candidates.append((answer, 8, 'answer_marker'))
            break
    
    # Method 4: Extract the answer from the last few paragraphs (the answer is usually at the end, priority=6)
    paragraphs = text.split('\n\n')
    if len(paragraphs) > 0:
        for para in paragraphs[-3:]:
            if any(marker in para.lower() for marker in ['answer', 'result', 'thus', 'therefore']):
                numbers = re.findall(r'\b(\d+\.?\d*)\b', para)
                if numbers:
                    filtered = [n for n in numbers if not (1900 <= float(n) <= 2100) and float(n) <= 1000000]
                    if filtered:
                        candidates.append((filtered[-1], 6, 'paragraph_marker'))
                break
    
    # Method 5: Extract the last number in the entire text (fallback option, priority=5)
    numbers = re.findall(r'\b(\d+\.?\d*)\b', text)
    if numbers:
        filtered_numbers = []
        for num in numbers:
            try:
                num_float = float(num)
                if 1900 <= num_float <= 2100:
                    continue
                if num_float > 1000000:
                    continue
                filtered_numbers.append(num)
            except:
                filtered_numbers.append(num)
        
        if filtered_numbers:
            candidates.append((filtered_numbers[-1], 5, 'last_number'))
        elif numbers:
            candidates.append((numbers[-1], 5, 'last_number'))
    
    if return_all_candidates:
        answer_votes = {}
        for answer, priority, source in candidates:
            normalized_answer = str(answer).lower().strip()
            if normalized_answer not in answer_votes:
                answer_votes[normalized_answer] = {
                    'answer': answer,
                    'votes': 0,
                    'max_priority': 0,
                    'sources': []
                }
            answer_votes[normalized_answer]['votes'] += 1
            answer_votes[normalized_answer]['max_priority'] = max(
                answer_votes[normalized_answer]['max_priority'], priority
            )
            answer_votes[normalized_answer]['sources'].append((priority, source))
        
        result_candidates = []
        for normalized, info in answer_votes.items():
            result_candidates.append((
                info['answer'],
                info['max_priority'],
                ', '.join([s[1] for s in sorted(info['sources'], reverse=True)]),
                info['votes']
            ))
        
        result_candidates.sort(key=lambda x: (x[3], x[1]), reverse=True)
        return result_candidates
    
    if candidates:
        answer_votes = {}
        for answer, priority, source in candidates:
            normalized_answer = str(answer).lower().strip()
            if normalized_answer not in answer_votes:
                answer_votes[normalized_answer] = {
                    'answer': answer,
                    'votes': 0,
                    'max_priority': 0,
                    'total_priority': 0
                }
            answer_votes[normalized_answer]['votes'] += 1
            answer_votes[normalized_answer]['max_priority'] = max(
                answer_votes[normalized_answer]['max_priority'], priority
            )
            answer_votes[normalized_answer]['total_priority'] += priority
        
        best_answer = None
        best_score = -1
        
        for normalized, info in answer_votes.items():
            score = info['votes'] * 2 + info['max_priority'] + info['total_priority'] / 10
            if score > best_score:
                best_score = score
                best_answer = info['answer']
        
        return best_answer
    
    return None


def _calculate_em_match_ratio(predicted: str, expected: str) -> float:
    """
    EM Match method: compute the match ratio (0-1)

    Computes how many of the keywords in the expected answer (after removing common words) are matched by the predicted answer
    Returns match ratio = number of matched keywords / total number of keywords

    Args:
        predicted: the answer predicted by the model
        expected: the expected answer

    Returns:
        Match ratio (0-1), where 1.0 means an exact match
    """
    if not predicted or not expected:
        return 0.0

    # Try numeric comparison
    try:
        num1 = float(str(predicted))
        num2 = float(str(expected))
        if abs(num1 - num2) < 0.01:
            return 1.0
        else:
            return 0.0
    except (ValueError, TypeError):
        pass

    # Text comparison (case-insensitive)
    pred_lower = str(predicted).lower().strip()
    exp_lower = str(expected).lower().strip()

    # Exact match
    if pred_lower == exp_lower:
        return 1.0

    # Compare after removing punctuation
    pred_clean = re.sub(r'[^\w\s]', '', pred_lower)
    exp_clean = re.sub(r'[^\w\s]', '', exp_lower)

    if pred_clean == exp_clean:
        return 1.0

    # EM Match: compute the keyword match ratio
    # Extract the keywords in the expected answer (removing stopwords)
    stopwords = {
        'the', 'a', 'an', 'is', 'are', 'was', 'were', 'of', 'in', 'on', 'at', 'to', 'for', 
        'with', 'by', 'and', 'or', 'but', 'that', 'this', 'these', 'those', 'from', 'as',
        'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should',
        'can', 'could', 'may', 'might', 'must', 'shall', 'it', 'its', 'they', 'them',
        'their', 'there', 'here', 'where', 'when', 'what', 'which', 'who', 'how', 'why',
        'answer', 'answers'  # "answer" is also a stopword
    }

    # Extract all tokens in the expected answer (including digits and letters)
    all_exp_tokens = re.findall(r'\b\w+\b', exp_clean)

    # Separate digits/letters from words
    # Digits and single letters (option letters) should be kept
    exp_tokens = set()
    for w in all_exp_tokens:
        w_lower = w.lower()
        # If it is a number (pure digits), keep it
        if w.isdigit():
            exp_tokens.add(w)
        # If it is a single letter (possibly an option letter), keep it
        elif len(w) == 1 and w.isalpha():
            exp_tokens.add(w_lower)
        # If it is a word, remove stopwords and words of length <= 2
        elif w_lower not in stopwords and len(w) > 2:
            exp_tokens.add(w_lower)

    # If there are no keywords, check for an exact match
    if not exp_tokens:
        # If the expected answer is very short (possibly a single word or number), compare directly
        if len(exp_clean) <= 3:
            if exp_clean in pred_clean or pred_clean in exp_clean:
                return 1.0
            else:
                return 0.0
        else:
            # If the expected answer is long but has no keywords, it may have too many stopwords; use partial matching
            if exp_clean in pred_clean:
                return 1.0
            elif pred_clean in exp_clean:
                return 0.5  # predicted is a subset of expected
            else:
                return 0.0

    # Extract the keywords in the predicted answer (using the same logic)
    all_pred_tokens = re.findall(r'\b\w+\b', pred_clean)
    pred_tokens = set()
    for w in all_pred_tokens:
        w_lower = w.lower()
        # If it is a number (pure digits), keep it
        if w.isdigit():
            pred_tokens.add(w)
        # If it is a single letter (possibly an option letter), keep it
        elif len(w) == 1 and w.isalpha():
            pred_tokens.add(w_lower)
        # If it is a word, remove stopwords and words of length <= 2
        elif w_lower not in stopwords and len(w) > 2:
            pred_tokens.add(w_lower)

    # Count the number of matched keywords
    common_tokens = pred_tokens & exp_tokens

    # Match ratio = number of matched keywords / total number of keywords in the expected answer
    match_ratio = len(common_tokens) / len(exp_tokens) if len(exp_tokens) > 0 else 0.0

    return match_ratio


def _compare_final_answers_em(answer1: str, answer2: str) -> bool:
    """
    EM Match method: compare whether two answers match (returns bool, kept for backward compatibility)

    Note: new code should use _calculate_em_match_ratio to obtain the match ratio
    """
    match_ratio = _calculate_em_match_ratio(answer1, answer2)
    # If the match ratio is >= 0.8, consider it a match
    return match_ratio >= 0.8


def _extract_multiple_choice_option(text: str) -> Optional[str]:
    """
    Extract a multiple-choice option (letter or number) from text

    Supports multiple formats, including:
    - "answer is A" / "The answer is B" (highest priority)
    - A standalone letter: A, B, C, D (only when the text is very short)
    - Letter + period: A., B., C., D.
    - Letter + parenthesis: A), B), C), D)
    - A letter in brackets: (A), (B), [A], [B]
    - Numbers: 1, 2, 3, 4 (converted to letters)
    - Extracted from the reasoning process

    Important: prioritize matching the "answer is X" pattern to avoid mistakenly extracting the leading letter A from "Answer is B"

    Args:
        text: the input text (may contain reasoning process, choice content, etc.)

    Returns:
        The extracted option (letter), or None if not found
    """
    if not text:
        return None

    text = text.strip()
    text_upper = text.upper()

    # Method 1: Prioritize matching the "answer is A" / "The answer is B" format (highest priority)
    # This avoids mistakenly extracting the leading letter A from "Answer is B"
    answer_patterns = [
        r'(?:^|\s)(?:the\s+)?answer[:\s]+is[:\s]+([A-Z])\b',  # "answer is A" or "The answer is B"
        r'(?:^|\s)option[:\s]+([A-Z])\b',  # "option A"
        r'(?:^|\s)correct\s+answer[:\s]+(?:is\s+)?([A-Z])\b',  # "correct answer is A"
        r'(?:^|\s)my\s+answer[:\s]+is[:\s]+([A-Z])\b',  # "my answer is A"
        r'(?:^|\s)final\s+answer[:\s]+is[:\s]+([A-Z])\b',  # "final answer is A"
        r'(?:^|\s)choose[:\s]+([A-Z])\b',  # "choose A"
    ]
    for pattern in answer_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            option = match.group(1).upper()
            if 'A' <= option <= 'Z':
                return option
    
    # Method 2: Match a single letter at the start of the text (if followed by a newline, space, or end)
    # This is the most common format: D\n\n... or D, etc.
    # Note: do not match cases where another letter immediately follows (e.g. BHumanitarian), as it is prone to misclassification
    match = re.search(r'^([A-Z])(?:\s|\n|$)', text_upper)
    if match:
        option = match.group(1)
        if 'A' <= option <= 'Z':
            return option
    
    # Method 3: Match a letter in brackets, e.g. (A), (B), [A], [B]
    match = re.search(r'[\(\[]([A-Z])[\)\]]', text_upper)
    if match:
        option = match.group(1)
        if 'A' <= option <= 'Z':
            return option
    
    # Method 3: If the text is very short (<=10 characters), it may be a direct answer; extract a single letter
    # But multi-line text (e.g. "A\nB\nC\nD") needs to be excluded
    text_stripped = text.strip()
    if len(text_stripped) <= 10:
        # Check whether it is multi-line text (contains a newline)
        if '\n' in text_stripped:
            # Multi-line text; extract the first line
            first_line = text_stripped.split('\n')[0].strip()
            match = re.match(r'^([A-Z])\b', first_line.upper())
            if match:
                option = match.group(1)
                if 'A' <= option <= 'Z':
                    return option
        else:
            # Single-line text; extract directly
            match = re.match(r'^([A-Z])\b', text_upper)
            if match:
                option = match.group(1)
                if 'A' <= option <= 'Z':
                    return option
    
    # Method 4: Extract the "A.", "B.", "C.", "D." format at the start of the text
    match = re.match(r'^([A-Z])\.', text_upper)
    if match:
        return match.group(1)

    # Method 5: Extract the "A)", "B)", "C)", "D)" format at the start of the text
    match = re.match(r'^([A-Z])\)', text_upper)
    if match:
        return match.group(1)

    # Method 6: Match the "A.", "B.", "C.", "D." format (not at the start)
    match = re.search(r'\b([A-Z])\.', text_upper)
    if match:
        option = match.group(1)
        if 'A' <= option <= 'Z':
            return option
    
    # Method 7: Match the "A)", "B)", "C)", "D)" format (not at the start)
    match = re.search(r'\b([A-Z])\)', text_upper)
    if match:
        option = match.group(1)
        if 'A' <= option <= 'Z':
            return option

    # Method 8: Match numeric options (1, 2, 3, 4, etc.)
    # Numeric options for multiple-choice questions are usually between 1 and 10
    match = re.search(r'\b([1-9]|10)\b', text)
    if match:
        num = int(match.group(1))
        if 1 <= num <= 10:
            # Convert to a letter (1->A, 2->B, ...)
            return chr(ord('A') + num - 1)

    # Method 9: Extract from the result of extract_final_answer
    final_answer = extract_final_answer(text)
    if final_answer:
        final_str = str(final_answer).strip().upper()
        # Directly match a single letter
        match = re.match(r'^([A-Z])\b', final_str)
        if match:
            option = match.group(1)
            if 'A' <= option <= 'Z':
                return option
    
    # Method 10: Search the last few lines of the text (possibly the answer after reasoning)
    lines = text.split('\n')
    # Check the last 3 lines
    for line in lines[-3:]:
        line_upper = line.strip().upper()
        # Check whether it contains an option letter (excluding common words)
        common_words = {'A', 'I', 'T', 'AN', 'AS', 'AT', 'BE', 'DO', 'GO', 'HE', 'IF', 'IN', 'IS', 'IT', 'ME', 'MY', 'NO', 'OF', 'ON', 'OR', 'SO', 'TO', 'UP', 'WE'}
        match = re.search(r'\b([A-Z])\b', line_upper)
        if match:
            option = match.group(1)
            if 'A' <= option <= 'Z' and option not in common_words:
                # Check whether this line is very short (possibly the answer)
                if len(line.strip()) <= 5:
                    return option

    # Method 11: Final fallback; search the entire text for a single letter (but exclude common words)
    # Exclude letters that start common words (e.g. A, I, T appearing alone may be words)
    common_words = {'A', 'I', 'T', 'AN', 'AS', 'AT', 'BE', 'DO', 'GO', 'HE', 'IF', 'IN', 'IS', 'IT', 'ME', 'MY', 'NO', 'OF', 'ON', 'OR', 'SO', 'TO', 'UP', 'WE'}
    # Only match a single letter that is not in the common words
    match = re.search(r'\b([A-Z])\b', text_upper)
    if match:
        option = match.group(1)
        if 'A' <= option <= 'Z' and option not in common_words:
            return option
    
    return None


def _evaluate_multiple_choice(predicted: str, expected: str, prompt: str = "") -> Dict:
    """
    Evaluate a multiple-choice answer (0-1 match)

    Supports two matching methods:
    1. Option-letter match: if predicted contains the option letter of expected (A, B, C, D, etc.), return 1.0
    2. Option-content match: if predicted contains the content corresponding to the expected option, return 1.0

    Note: multiple-choice questions have only two accuracy values, 1 and 0, not a match ratio
    """
    # Extract the option letter of the expected answer
    exp_letter = _extract_multiple_choice_option(expected)
    if not exp_letter:
        exp_letter_match = re.search(r'\b([A-Z])\b', expected.upper())
        if exp_letter_match:
            exp_letter = exp_letter_match.group(1)

    if not exp_letter:
        # If the option of the expected answer cannot be extracted, return 0 (a data issue)
        return {
            'accuracy': 0.0,
            'exact_match': False,
            'final_answer_match': False,
            'quality_score': 0.0,
            'predicted_final': None,
            'expected_final': None
        }
    
    # Method 1: Check whether predicted contains the option letter of expected
    pred_letter = _extract_multiple_choice_option(predicted)
    if pred_letter and pred_letter == exp_letter:
        return {
            'accuracy': 1.0,
            'exact_match': True,
            'final_answer_match': True,
            'quality_score': 1.0,
            'predicted_final': pred_letter,
            'expected_final': exp_letter
        }
    
    # Method 2: Extract the option contents from the prompt and check whether predicted contains the content of the expected option
    if prompt:
        # Extract all options: A. content1 B. content2 C. content3 D. content4
        option_pattern = r'([A-Z])\.\s*([^\n]+?)(?=\s+[A-Z]\.|$)'
        options = {}
        for match in re.finditer(option_pattern, prompt, re.IGNORECASE):
            letter = match.group(1).upper()
            content = match.group(2).strip()
            # Clean the content (remove trailing punctuation)
            content = re.sub(r'[.,;!?]+$', '', content).strip()
            if letter and content:
                options[letter] = content

        # If the content of the expected option was found
        if exp_letter in options:
            exp_content = options[exp_letter].lower()
            pred_lower = predicted.lower()

            # Check whether predicted contains the content of the expected option (at least the key part)
            # Extract the keywords of the option content
            exp_words = set(re.findall(r'\b\w+\b', exp_content))
            # Remove stopwords
            stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'of', 'in', 'on', 'at', 'to', 'for',
                        'with', 'by', 'and', 'or', 'but', 'that', 'this', 'these', 'those', 'from', 'as'}
            exp_keywords = {w for w in exp_words if w not in stopwords and len(w) > 2}

            if exp_keywords:
                # Check whether predicted contains enough keywords (at least 50%)
                pred_words = set(re.findall(r'\b\w+\b', pred_lower))
                matched_keywords = exp_keywords & pred_words
                if len(matched_keywords) >= len(exp_keywords) * 0.5:
                    return {
                        'accuracy': 1.0,
                        'exact_match': True,
                        'final_answer_match': True,
                        'quality_score': 1.0,
                        'predicted_final': predicted[:50],  # Take the first 50 characters
                        'expected_final': exp_letter
                    }
            else:
                # If the option content is very short, directly check for containment
                if exp_content in pred_lower or any(word in pred_lower for word in exp_content.split() if len(word) > 3):
                    return {
                        'accuracy': 1.0,
                        'exact_match': True,
                        'final_answer_match': True,
                        'quality_score': 1.0,
                        'predicted_final': predicted[:50],
                        'expected_final': exp_letter
                    }
    
    # If nothing matches, return 0
    return {
        'accuracy': 0.0,
        'exact_match': False,
        'final_answer_match': False,
        'quality_score': 0.0,
        'predicted_final': pred_letter if pred_letter else predicted[:50],
        'expected_final': exp_letter
    }


def _evaluate_factual(predicted: str, expected: str) -> Dict:
    """
    Evaluate a factual question answer (uses EM Match, returns the match ratio)

    accuracy = how many of the keywords in the expected answer are matched by the predicted answer
    """
    # Exact match
    exact_match = predicted.lower().strip() == expected.lower().strip()

    # Compute the EM Match ratio
    match_ratio = _calculate_em_match_ratio(predicted, expected)

    # Accuracy = match ratio (0-1)
    accuracy = match_ratio

    # quality_score = accuracy (kept consistent)
    quality_score = accuracy

    return {
        'accuracy': accuracy,
        'exact_match': exact_match,
        'final_answer_match': match_ratio >= 0.8,  # Kept for backward compatibility
        'quality_score': quality_score,
        'predicted_final': predicted,
        'expected_final': expected
    }


def _evaluate_code(predicted: str, expected: str) -> Dict:
    """
    Evaluate a code answer (uses EM Match, returns the match ratio)

    accuracy = how many of the keywords in the expected answer are matched by the predicted answer
    """
    # Remove markdown code block markers
    pred_clean = re.sub(r'```[\w]*\n?', '', predicted)
    pred_clean = re.sub(r'```', '', pred_clean)
    exp_clean = re.sub(r'```[\w]*\n?', '', expected)
    exp_clean = re.sub(r'```', '', exp_clean)

    # Remove comments
    pred_clean = re.sub(r'#.*$', '', pred_clean, flags=re.MULTILINE)
    exp_clean = re.sub(r'#.*$', '', exp_clean, flags=re.MULTILINE)

    # Remove extra whitespace
    pred_clean = re.sub(r'\s+', ' ', pred_clean).strip()
    exp_clean = re.sub(r'\s+', ' ', exp_clean).strip()

    # Exact match
    exact_match = pred_clean == exp_clean

    # Compute the EM Match ratio (using the cleaned code)
    match_ratio = _calculate_em_match_ratio(pred_clean, exp_clean)

    # Accuracy = match ratio (0-1)
    accuracy = match_ratio

    # quality_score = accuracy (kept consistent)
    quality_score = accuracy

    return {
        'accuracy': accuracy,
        'exact_match': exact_match,
        'final_answer_match': match_ratio >= 0.8,  # Kept for backward compatibility
        'quality_score': quality_score,
        'predicted_final': pred_clean,
        'expected_final': exp_clean
    }


def _evaluate_reasoning(predicted: str, expected: str) -> Dict:
    """
    Evaluate a reasoning question answer (uses the EM Match method, returns the match ratio)

    accuracy = how many of the keywords in the expected answer are matched by the predicted answer
    """
    # Extract the final answer
    pred_final = extract_final_answer(predicted)
    exp_final = extract_final_answer(expected)

    # Exact match
    exact_match = predicted.lower().strip() == expected.lower().strip()

    # Compute the EM Match ratio
    # Prefer comparing the extracted final answers
    if pred_final and exp_final:
        match_ratio = _calculate_em_match_ratio(pred_final, exp_final)
    else:
        # If no final answer was extracted, use the full text
        match_ratio = _calculate_em_match_ratio(predicted, expected)

    # Accuracy = match ratio (0-1)
    accuracy = match_ratio

    # quality_score = accuracy (kept consistent)
    quality_score = accuracy

    return {
        'accuracy': accuracy,
        'exact_match': exact_match,
        'final_answer_match': match_ratio >= 0.8,  # Kept for backward compatibility
        'quality_score': quality_score,
        'predicted_final': pred_final if pred_final else predicted,
        'expected_final': exp_final if exp_final else expected
    }


def _evaluate_default(predicted: str, expected: str) -> Dict:
    """
    Default evaluation method (compatible with legacy code, used when task_type is not specified)
    Uses the EM Match method and returns the match ratio
    """
    # Exact match
    exact_match = predicted.lower() == expected.lower()

    # Extract and compare the final answers
    pred_final = extract_final_answer(predicted)
    exp_final = extract_final_answer(expected)

    # Compute the EM Match ratio
    # Prefer comparing the extracted final answers
    if pred_final and exp_final:
        match_ratio = _calculate_em_match_ratio(pred_final, exp_final)
    else:
        # If no final answer was extracted, use the full text
        match_ratio = _calculate_em_match_ratio(predicted, expected)

    # Accuracy = match ratio (0-1)
    accuracy = match_ratio

    # quality_score = accuracy (kept consistent)
    quality_score = accuracy

    return {
        'accuracy': accuracy,
        'exact_match': exact_match,
        'final_answer_match': match_ratio >= 0.8,  # Kept for backward compatibility
        'quality_score': quality_score,
        'predicted_final': pred_final,
        'expected_final': exp_final
    }


def evaluate_answer_accuracy(
    predicted_answer: str,
    expected_answer: str,
    prompt: str = "",
    category: Optional[str] = None,
    task_type: Optional[str] = None
) -> Dict:
    """
    Evaluate answer accuracy (supports 4 question types)

    References the evaluation methods under the benchmark directory, using EM Match for answer matching
    but does not use an LLM judge (to stay lightweight)

    Args:
        predicted_answer: the answer predicted by the model
        expected_answer: the expected answer
        prompt: the input prompt (optional)
        category: the question category (optional, deprecated, use task_type)
        task_type: the question type (multiple_choice, factual, code, reasoning)

    Returns:
        An evaluation result dictionary containing:
        - accuracy: accuracy (0-1, the EM match ratio)
        - exact_match: whether it is an exact match
        - final_answer_match: whether the final answer matches (match ratio >= 0.8)
        - quality_score: quality score (equals accuracy)

    Note:
    - accuracy is based on the EM match ratio: how many of the keywords in the expected answer are matched by the predicted answer
    - match ratio = number of matched keywords / total number of keywords in the expected answer (after removing common words)
    - there is no evaluation based on semantic similarity
    """
    if not predicted_answer or not expected_answer:
        return {
            'accuracy': 0.0,
            'exact_match': False,
            'final_answer_match': False,
            'quality_score': 0.0
        }
    
    # Handle the case where expected_answer may be a list (supporting multiple answers)
    if isinstance(expected_answer, list):
        if len(expected_answer) == 0:
            return {
                'accuracy': 0.0,
                'exact_match': False,
                'final_answer_match': False,
                'quality_score': 0.0
            }
        # If there are multiple answers, use the first one (or could average them)
        expected_answer = expected_answer[0]

    predicted = str(predicted_answer).strip()
    expected = str(expected_answer).strip()

    # Legacy compatibility: if category is provided but task_type is not, try to infer task_type from category
    if task_type is None and category:
        task_type = category

    # Use a different evaluation strategy depending on task_type
    if task_type == 'multiple_choice':
        return _evaluate_multiple_choice(predicted, expected, prompt)
    elif task_type == 'factual':
        return _evaluate_factual(predicted, expected)
    elif task_type == 'code':
        return _evaluate_code(predicted, expected)
    elif task_type == 'reasoning':
        return _evaluate_reasoning(predicted, expected)
    else:
        # Default evaluation method (compatible with legacy code)
        return _evaluate_default(predicted, expected)


def evaluate_answer_multi_reference(
    predicted_answers: List[str],  # Multiple predicted answers (1-3 answer sets)
    expected_answer: str,
    prompt: str = "",
    task_type: Optional[str] = None,
    remove_outliers: bool = True,
    outlier_threshold: float = 2.0
) -> Dict:
    """
    Evaluate multiple answers (supports 1-3 answer sets, each scored with EM, then averaged)

    References the evaluate_answer_multi_reference implementation under the benchmark directory
    For non-multiple-choice questions, supports generating 1-3 answer sets, each scored with EM, then averaged

    Args:
        predicted_answers: list of multiple predicted answers (at least 1, at most 3)
        expected_answer: the expected answer
        prompt: the input prompt (optional)
        task_type: the question type (if None, it must be auto-detected from the prompt)
        remove_outliers: whether to remove outliers (when there are 3 answers)
        outlier_threshold: outlier threshold (number of standard deviations)

    Returns:
        An evaluation result dictionary containing:
        - accuracy: average accuracy (0-1)
        - num_answers: number of answers
        - num_valid_answers: number of valid answers (after removing outliers)
        - individual_scores: list of accuracies for each answer
        - individual_results: detailed evaluation result for each answer
        - outliers_removed: list of indices of removed outliers
        - task_type: the detected question type
    """
    if not predicted_answers:
        raise ValueError("At least 1 predicted answer must be provided")

    if len(predicted_answers) > 3:
        raise ValueError(f"At most 3 answer sets are supported, but {len(predicted_answers)} were provided")

    # Evaluate each answer
    evaluation_results = []
    for idx, pred_answer in enumerate(predicted_answers):
        if not pred_answer or not pred_answer.strip():
            continue  # Skip empty answers

        result = evaluate_answer_accuracy(
            predicted_answer=pred_answer,
            expected_answer=expected_answer,
            prompt=prompt,
            task_type=task_type
        )
        result['answer_idx'] = idx
        evaluation_results.append(result)
    
    if not evaluation_results:
        return {
            'accuracy': 0.0,
            'num_answers': len(predicted_answers),
            'num_valid_answers': 0,
            'individual_scores': [],
            'individual_results': [],
            'outliers_removed': [],
            'task_type': task_type
        }
    
    # Extract the accuracy scores
    accuracy_scores = [r.get('accuracy', 0.0) for r in evaluation_results]

    # Remove outliers (if enabled and there are more than 2 answers)
    outliers_removed = []
    if remove_outliers and len(accuracy_scores) >= 3:
        scores_array = np.array(accuracy_scores)
        mean_score = np.mean(scores_array)
        std_score = np.std(scores_array)
        
        if std_score > 0:
            z_scores = np.abs((scores_array - mean_score) / std_score)
            outlier_mask = z_scores > outlier_threshold
            
            if np.any(outlier_mask):
                outlier_indices = np.where(outlier_mask)[0]
                outliers_removed = [int(i) for i in outlier_indices]

                # Remove the outliers
                accuracy_scores = [score for i, score in enumerate(accuracy_scores) if not outlier_mask[i]]
                evaluation_results = [r for i, r in enumerate(evaluation_results) if not outlier_mask[i]]

    # Compute the average accuracy
    avg_accuracy = float(np.mean(accuracy_scores)) if accuracy_scores else 0.0

    # Build the result
    result = {
        'accuracy': avg_accuracy,
        'num_answers': len(predicted_answers),
        'num_valid_answers': len(accuracy_scores),
        'individual_scores': [float(s) for s in accuracy_scores],
        'individual_results': evaluation_results,
        'outliers_removed': outliers_removed,
        'task_type': task_type
    }
    
    return result
