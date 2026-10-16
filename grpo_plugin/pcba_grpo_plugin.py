import re
from typing import Any, Dict, List, Optional
import math
from swift.rewards import ORM, orms


# ============================================================
# PCBA GRPO rewards, balanced partial-credit version
#
# Route:
#   non-think SFT -> non-think GRPO
#
# Expected output:
#   Explanation: exact selected option text
#
#   Answer: one uppercase option letter from the provided options
#
# Choice/binary reward:
#   - Answer must be a direct uppercase letter, e.g. A/B/.../H.
#   - Answer text such as "QFN" is NOT mapped back to "A".
#   - Reward is graded:
#       Answer correct                              +0.70
#       Explanation equals correct option text      +0.20
#       Explanation matches the selected Answer     +0.10
#     So:
#       Answer correct + Explanation correct        = 1.00
#       Answer correct + Explanation wrong          = 0.70
#       Answer wrong + Explanation correct          = 0.20
#       Answer wrong + Explanation matches wrong option = 0.10
#
# Integer/count reward:
#   - exact match: 1.0
#   - absolute error 1: 0.7
#   - absolute error 2: 0.4
#   - larger error: decreasing score, eventually 0.0
# ============================================================


def completion_to_text(completion: Any) -> str:
    """
    Make the reward function tolerant to different completion structures.
    Some trainers pass a plain string; some pass a list/dict with content.
    """
    if isinstance(completion, str):
        return completion

    if isinstance(completion, dict):
        return str(completion.get("content", completion))

    if isinstance(completion, list):
        # Common structure: [{"role": "assistant", "content": "..."}]
        for item in reversed(completion):
            if isinstance(item, dict) and "content" in item:
                return str(item["content"])
        return str(completion)

    return str(completion)


def strip_think(text: str) -> str:
    text = str(text)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL)
    return text.strip()


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip())


# def normalize_for_match(text: str) -> str:
#     # Case-insensitive text match for Explanation/option text.
#     return normalize_space(text).lower()

def normalize_for_match(text: str) -> str:
    return normalize_space(text)


def extract_answer_field(text: str) -> str:
    """
    Extract the content after the last Answer: line.
    """
    text = strip_think(text)

    matches = re.findall(
        r"(?im)^\s*(?:Answer|答案)\s*[:：]\s*(.+?)\s*$",
        text,
    )
    if matches:
        return matches[-1].strip()

    fallback = text.strip()
    if re.fullmatch(r"[A-Z]", fallback) or re.fullmatch(r"-?\d+", fallback):
        return fallback

    return ""


def extract_explanation_field(text: str) -> str:
    """
    Extract Explanation: ... before Answer:.
    """
    text = strip_think(text)
    m = re.search(
        r"(?is)(?:^|\n)\s*(?:Explanation|解释)\s*[:：]\s*(.*?)(?=\n\s*(?:Answer|答案)\s*[:：]|\Z)",
        text,
    )
    return m.group(1).strip() if m else ""


def extract_direct_uppercase_letter(raw_answer: str) -> str:
    """
    Strictly require Answer to be exactly one uppercase letter.

    Accepted:
        A
        B
        H

    Rejected:
        a
        A.
        (A)
        Option A
        QFN
    """
    raw_answer = str(raw_answer).strip()
    if re.fullmatch(r"[A-Z]", raw_answer):
        return raw_answer
    return ""


def get_choice_text(choices: Any, letter: str) -> str:
    if not choices or not letter:
        return ""

    if isinstance(choices, dict):
        return str(choices.get(letter, "")).strip()

    return ""


def normalize_int_text(text: str) -> Optional[int]:
    m = re.search(r"-?\d+", str(text))
    if not m:
        return None
    try:
        return int(m.group(0))
    except Exception:
        return None


# def count_reward(pred_int: int, gold_int: int) -> float:
#     diff = abs(pred_int - gold_int)

#     # 精确或非常接近时，给明确分档
#     if diff == 0:
#         return 1.0
#     if diff == 1:
#         return 0.7
#     if diff == 2:
#         return 0.4

#     # gold = 0 时，只能按绝对误差判断
#     # 0 -> 1 给 0.7，0 -> 2 给 0.4，0 -> 更大快速衰减
#     if gold_int == 0:
#         reward = 0.4 * math.exp(-(diff - 2) / 2.0)
#         return round(max(0.0, min(0.4, reward)), 4)

#     relative_error = diff / abs(gold_int)

#     # 极端离谱的答案仍然给 0
#     # 例如 gold=120, pred=0，relative_error=1.0，给 0
#     if relative_error >= 0.9:
#         return 0.0

#     # 绝对误差尺度：gold 越大，允许的绝对误差稍微大一点
#     scale = max(3.0, 0.25 * abs(gold_int))

#     abs_score = 0.4 * math.exp(-(diff - 2) / scale)

#     # 相对误差尺度：relative_error 越大，分数越低
#     rel_score = 0.4 * (1.0 - relative_error / 0.9) ** 2

#     # 同时受绝对误差和相对误差约束，避免小数字计数过度给分
#     reward = min(abs_score, rel_score)

#     return round(max(0.0, min(0.4, reward)), 4)

def count_reward(pred_int: int, gold_int: int) -> float:
    diff = abs(pred_int - gold_int)

    if diff == 0:
        return 1.0
    if diff == 1:
        return 0.8
    if diff == 2:
        return 0.6

    if gold_int == 0:
        reward = 0.6 * math.exp(-(diff - 2) / 2.0)
        return round(max(0.0, min(0.6, reward)), 4)

    relative_error = diff / abs(gold_int)

    # 极端错误不给分
    if relative_error >= 0.9:
        return 0.0

    # 继续从 0.6 往下衰减
    scale = max(3.0, 0.25 * abs(gold_int))

    abs_score = 0.6 * math.exp(-(diff - 2) / scale)
    rel_score = 0.6 * (1.0 - relative_error / 0.9) ** 2

    reward = min(abs_score, rel_score)

    return round(max(0.0, min(0.6, reward)), 4)


def is_choice_like(answer_type: str) -> bool:
    return str(answer_type).lower() in {
        "choice",
        "binary",
        "option",
        "multiple_choice",
        "yes_no",
    }


def choice_partial_reward(
    raw_answer: str,
    explanation: str,
    gold_letter: str,
    choices: Any,
    solution_text: str = "",
) -> float:
    """
    Graded reward for choice/binary tasks.

    Answer correctness is the main part.
    Explanation correctness is secondary.
    Answer-Explanation consistency is a small auxiliary part.
    """
    pred_letter = extract_direct_uppercase_letter(raw_answer)
    gold_letter = extract_direct_uppercase_letter(gold_letter)

    expected_gold_text = get_choice_text(choices, gold_letter) or str(solution_text).strip()
    selected_text = get_choice_text(choices, pred_letter)

    explanation_norm = normalize_for_match(explanation)
    expected_gold_norm = normalize_for_match(expected_gold_text)
    selected_norm = normalize_for_match(selected_text)

    reward = 0.0

    # 1) Main signal: final Answer letter is correct.
    if pred_letter and gold_letter and pred_letter == gold_letter:
        reward += 0.70

    # 2) Secondary signal: Explanation identifies the correct option text.
    if explanation_norm and expected_gold_norm and explanation_norm == expected_gold_norm:
        reward += 0.20

    # 3) Consistency signal: Explanation matches what the selected letter means.
    # If Answer is correct and Explanation is correct, this adds the last 0.10.
    # If Answer is wrong but internally consistent with a wrong option, it only gets 0.10.
    if pred_letter and explanation_norm and selected_norm and explanation_norm == selected_norm:
        reward += 0.10

    # Safety cap.
    return min(1.0, reward)


class PCBAAccuracyORM(ORM):
    """
    Main reward with partial credit.

    For choice/binary:
        Answer correct                              +0.70
        Explanation equals correct option text      +0.20
        Explanation matches selected Answer option  +0.10

    For integer/count:
        exact match gives 1.0;
        off-by-1 gives 0.7;
        off-by-2 gives 0.4;
        larger errors decay toward 0.
    """

    def __call__(
        self,
        completions: List[Any],
        solution: List[str],
        answer_type: Optional[List[str]] = None,
        choices: Optional[List[Dict[str, str]]] = None,
        solution_text: Optional[List[str]] = None,
        **kwargs,
    ) -> List[float]:
        rewards: List[float] = []

        answer_type = answer_type or ["choice"] * len(completions)
        choices = choices or [None] * len(completions)
        solution_text = solution_text or [""] * len(completions)

        for completion, sol, a_type, ch, sol_text in zip(
            completions, solution, answer_type, choices, solution_text
        ):
            content = completion_to_text(completion)
            raw_answer = extract_answer_field(content)
            explanation = extract_explanation_field(content)
            sol = str(sol).strip()

            if str(a_type).lower() == "integer":
                pred_int = normalize_int_text(raw_answer)
                gold_int = normalize_int_text(sol)

                if pred_int is None or gold_int is None:
                    rewards.append(0.0)
                else:
                    rewards.append(count_reward(pred_int, gold_int))
                continue

            if is_choice_like(a_type):
                rewards.append(
                    choice_partial_reward(
                        raw_answer=raw_answer,
                        explanation=explanation,
                        gold_letter=sol,
                        choices=ch,
                        solution_text=sol_text,
                    )
                )
                continue

            # Text-type fallback: exact match only.
            if normalize_for_match(raw_answer) == normalize_for_match(sol):
                rewards.append(1.0)
            else:
                rewards.append(0.0)

        return rewards


class PCBAFormatORM(ORM):
    """
    Auxiliary format reward.

    It checks whether the model follows the required plain format:
        Explanation: ...
        Answer: ...

    For choice/binary tasks:
        Answer must be one direct uppercase option letter
        and the letter should exist in choices when choices are available.

    For integer/count tasks:
        Answer must contain a valid integer.

    Keep the weight small, e.g. 0.05 or 0.1.
    """

    def __call__(
        self,
        completions: List[Any],
        answer_type: Optional[List[str]] = None,
        choices: Optional[List[Dict[str, str]]] = None,
        **kwargs,
    ) -> List[float]:
        rewards: List[float] = []

        answer_type = answer_type or ["choice"] * len(completions)
        choices = choices or [None] * len(completions)

        for completion, a_type, ch in zip(completions, answer_type, choices):
            content = completion_to_text(completion)
            explanation = extract_explanation_field(content)
            raw_answer = extract_answer_field(content)

            has_explanation = bool(explanation)

            if str(a_type).lower() == "integer":
                has_valid_answer = normalize_int_text(raw_answer) is not None

            elif is_choice_like(a_type):
                pred_letter = extract_direct_uppercase_letter(raw_answer)

                if isinstance(ch, dict) and ch:
                    has_valid_answer = bool(pred_letter and pred_letter in ch)
                else:
                    has_valid_answer = bool(pred_letter)

            else:
                has_valid_answer = bool(raw_answer)

            if has_explanation and has_valid_answer:
                rewards.append(1.0)
            elif has_valid_answer:
                rewards.append(0.5)
            elif has_explanation:
                rewards.append(0.2)
            else:
                rewards.append(0.0)

        return rewards

class PCBAStrictAccuracyORM(ORM):
    """
    Optional strict diagnostic reward.

    It gives 1.0 only when:
      - Answer is the correct direct uppercase letter;
      - Explanation equals the correct option text.

    Recommended as a logging/diagnostic reward with low or zero weight,
    not as the only main reward in early GRPO.
    """

    def __call__(
        self,
        completions: List[Any],
        solution: List[str],
        answer_type: Optional[List[str]] = None,
        choices: Optional[List[Dict[str, str]]] = None,
        solution_text: Optional[List[str]] = None,
        **kwargs,
    ) -> List[float]:
        rewards: List[float] = []

        answer_type = answer_type or ["choice"] * len(completions)
        choices = choices or [None] * len(completions)
        solution_text = solution_text or [""] * len(completions)

        for completion, sol, a_type, ch, sol_text in zip(
            completions, solution, answer_type, choices, solution_text
        ):
            if not is_choice_like(a_type):
                rewards.append(0.0)
                continue

            content = completion_to_text(completion)
            raw_answer = extract_answer_field(content)
            explanation = extract_explanation_field(content)

            pred_letter = extract_direct_uppercase_letter(raw_answer)
            gold_letter = extract_direct_uppercase_letter(str(sol).strip())
            expected_text = get_choice_text(ch, gold_letter) or str(sol_text).strip()

            ok = (
                pred_letter
                and gold_letter
                and pred_letter == gold_letter
                and explanation
                and expected_text
                and normalize_for_match(explanation) == normalize_for_match(expected_text)
            )
            rewards.append(1.0 if ok else 0.0)

        return rewards


orms["pcba_accuracy"] = PCBAAccuracyORM
orms["pcba_format"] = PCBAFormatORM
orms["pcba_strict_accuracy"] = PCBAStrictAccuracyORM
