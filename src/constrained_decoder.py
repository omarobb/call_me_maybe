from enum import Enum
from .models import (is_name_token_allowed,
                     is_valid_integer_continuation,
                     is_valid_string_continuation,
                     is_valid_number_continuation)
from math import inf


class GenState(Enum):
    IN_FUNCTION_NAME = "in_function_name"
    IN_PARAMETER_VALUE_STRING = "in_parameter_value_string"
    IN_PARAMETER_VALUE_NUMBER = "in_parameter_value_number"
    IN_PARAMETER_VALUE_FLOAT = "in_parameter_value_float"


def mask_logits(logits: list[float], typed: str, state: GenState,
                valid_name: list[str],
                token_lookup: dict[int, str]) -> list[float]:
    masked = logits.copy()

    for token_ids, score in enumerate(logits):
        condidate_string = token_lookup[token_ids]

        if state == GenState.IN_FUNCTION_NAME:
            allowed = is_name_token_allowed(condidate_string, typed,
                                            valid_name)
        elif state == GenState.IN_PARAMETER_VALUE_STRING:
            allowed = is_valid_string_continuation(condidate_string, typed)
        elif state == GenState.IN_PARAMETER_VALUE_NUMBER:
            allowed = is_valid_integer_continuation(condidate_string, typed)
        elif state == GenState.IN_PARAMETER_VALUE_FLOAT:
            allowed = is_valid_number_continuation(condidate_string, typed)

        if not allowed:
            masked[token_ids] = -inf

    return masked
