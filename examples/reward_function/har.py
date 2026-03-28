import re
from typing import Any


# Metadata
REWARD_NAME = "math"
REWARD_TYPE = "batch"


# ---- 修改①：格式匹配改成 <reasoning>...</reasoning><answer>...</answer> ----
def format_reward(response: str) -> float:
    pattern = re.compile(
        r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>",
        re.DOTALL
    )
    format_match = re.fullmatch(pattern, response)
    return 1.0 if format_match else 0.0


# ---- 修改②：答案抽取改成解析 <answer>CLASS</answer> ----
def extract_answer_tag(response: str) -> str:
    m = re.search(r"<answer>(.*?)</answer>", response)
    return m.group(1).strip() if m else ""


def accuracy_reward(response: str, ground_truth: str) -> float:
    answer = extract_answer_tag(response)
    return 1.0 if answer == ground_truth else 0.0


# ---- 下面保持不变 ----
def compute_score(reward_inputs: list[dict[str, Any]], format_weight: float = 0.1) -> list[dict[str, float]]:
    scores = []
    for reward_input in reward_inputs:
        response = re.sub(r"\s*(<|>|/)\s*", r"\1", reward_input["response"])  # handle qwen2.5vl-32b format
        format_score = format_reward(response)
        accuracy_score = accuracy_reward(response, reward_input["ground_truth"])
        scores.append(
            {
                "overall": (1 - format_weight) * accuracy_score + format_weight * format_score,
                "format": format_score,
                "accuracy": accuracy_score,
            }
        )

    return scores

