from llm_sdk import Small_LLM_Model as LLM
from pydantic import BaseModel, ValidationError, TypeAdapter
from typing import Any
import sys
import json


class ParameterInfo(BaseModel):
    type: str


class Prompt(BaseModel):
    prompt: str


class FunctionEntry(BaseModel):
    name: str
    description: str
    parameters: dict[str, ParameterInfo]
    returns: ParameterInfo


class FunctionCallResult(BaseModel):
    prompt: str
    name: str
    parameters: dict[str, (int | str | float)]


def is_complete_name(typed: str, valid: list[str]) -> bool:
    return typed in valid


def is_valid(s: str, typed: str, valid: list[str]) -> bool:
    return any(d for d in valid if d.startswith(typed + s))


def is_name_token_allowed(candidate_token: str,
                          typed: str, valid: list[str]) -> bool:
    if candidate_token == '"':
        return is_complete_name(typed, valid)
    else:
        return is_valid(candidate_token, typed, valid)


def count_trailing_backslashes(text: str) -> int:
    count = 0
    i = len(text) - 1
    while (i >= 0 and text[i] == '\\'):
        count += 1
        i -= 1
    return count


def is_valid_string_continuation(s: str, typed: str) -> bool:
    condidate = typed + s
    if '"' in s:
        if not s[-1] == '"':
            return False
        if s.count('"') > 1:
            return False
        before_quote = condidate[:-1]
        if count_trailing_backslashes(before_quote) % 2 != 0:
            return False
    return True


def is_valid_integer_continuation(s: str, typed: str) -> bool:
    if s == '-' or s == '':
        if not typed:
            return True
    if typed:
        if typed[0] == '-':
            return (typed+s)[1:].isdigit()
    return (typed+s).isdigit()


def is_valid_number_continuation(s: str, typed: str) -> bool:
    condidate = typed + s
    if condidate == '' or condidate == '-':
        return True
    if condidate.startswith('-'):
        condidate = condidate[1:]
        if '-' in condidate:
            return False
    if condidate.count('.') > 1:
        return False
    without_dot = condidate.replace('.', '', 1)
    return without_dot.isdigit() or without_dot == ''


def build_parameter_schema(fn_name: str,
                           fn_defintion: list[FunctionEntry])\
                           -> dict[str, ParameterInfo]:
    s = ParameterInfo(type="empty")
    unknown = {"parameters": s}
    for fun in fn_defintion:
        if fun.name == fn_name:
            return fun.parameters
    return unknown


def load_function_definitions(path: str) -> Any:
    try:
        with open('./unknown.json', 'r', encoding='utf-8') as f:
            un = json.load(f)
        if un and isinstance(un, list) and un[0].get('name'):
            un[0]['name'] = 'Unknown'
        with open(path, 'r', encoding='utf-8') as p:
            ls = json.load(p)
            ls = ls + un
            validation = TypeAdapter(list[FunctionEntry])
            return validation.validate_python(ls)
    except (json.JSONDecodeError, FileNotFoundError,
            TypeError, ValidationError) as e:
        print(f"ERROR in function_definitions: {e}")
        sys.exit(1)


def load_function_name(path: str) -> list[str]:
    try:
        with open(path, 'r', encoding='utf-8') as p:
            names = []
            ls = json.load(p)
            for fn in ls:
                names.append(fn['name'])
            if 'Unknown' not in names:
                names.append('Unknown')
            return names
    except (json.JSONDecodeError, FileNotFoundError,
            TypeError, ValidationError) as e:
        print(f"ERROR in function_definitions: {e}")
        sys.exit(1)


def load_prompt_definitions(path: str) -> list[Prompt] | Any:
    try:
        with open(path, 'r', encoding='utf-8') as p:
            ls = json.load(p)
            validation = TypeAdapter(list[Prompt])
            return validation.validate_python(ls)
    except (json.JSONDecodeError, FileNotFoundError,
            TypeError, ValidationError) as e:
        print(f"ERROR in prompt_definitions: {e}")
        sys.exit(1)


def build_token_loockup(sdk: LLM) -> dict[int, str]:
    ids = sdk.encode('a')
    ids = ids.tolist()[0]
    log = sdk.get_logits_from_input_ids(ids)
    vocab_size = len(log)
    lookup = {}
    for token_id in range(vocab_size):
        lookup[token_id] = sdk.decode([token_id])
    return lookup


def build_priming_text(prompt_text: str,
                       function_defs: list[FunctionEntry]) -> str:
    instruction = (
        "You are a function-calling assistant. Choose exactly one "
        "function from the list that matches the user's request, and "
        "extract its parameters.\n"
        "Copy every value exactly as written in the request, character "
        "for character. Do not change, add, remove, or guess anything.\n"
        "Each parameter value must contain ONLY that parameter's own "
        "value — never another parameter's name or value.\n"
        "\n"
        "For a regex parameter, output ONLY the raw pattern — no prose, "
        "no labels, nothing else. Match these examples exactly:\n"
        "- 'replace all numbers' -> regex: ([0-9]+)\n"
        "- 'replace all vowels' -> regex: [aeiouAEIOU]\n"
        "- 'substitute the word cat with dog' -> regex: [cat] \n"
        "\n"
        "Return only one JSON object with keys 'name' and 'parameters'. "
        "Nothing before it, nothing after it."
    )

    function_json = TypeAdapter(list[FunctionEntry]).dump_json(function_defs)

    return (
        f"{instruction!r}\n\n"
        f"Functions:\n{function_json!r}\n\n"
        f"Request:\n<<<{prompt_text}>>>\n\n"
    )


def has_repeating_tail(field_typed: str, block_size: int) -> bool:
    if len(field_typed) < block_size * 2:
        return False

    last_block = field_typed[-block_size:]
    previous_block = field_typed[-block_size*2: -block_size]
    return last_block == previous_block
