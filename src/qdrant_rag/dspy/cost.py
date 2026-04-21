# Demander à Xavier quels fixs il a appliqué pour avoir les chiffres correct au lieu de ce qui est ici.

def get_tokens(lm_history: list) -> tuple[int, int]:
    record = lm_history[0]
    usage = record.get("usage", {})
    prompt_tokens = usage.get("prompt_tokens") or record.get("input_tokens") or 0
    completion_tokens = usage.get("completion_tokens") or record.get("output_tokens") or 0
    return int(prompt_tokens), int(completion_tokens)


def get_cost(lm_history: list) -> float:
    return float(lm_history[0].get("cost") or 0.0)

def get_info(lm_history: list) -> list[int, int, float]:
    ptokens, ctokens = get_tokens(lm_history) if len(lm_history) > 0 else (0,0)
    cost = get_cost(lm_history)
    return [ptokens, ctokens, cost] 
