from llm_sdk import Small_LLM_Model as LLM
from .constrained_decoder import GenState, mask_logits
from typing import Any
from .models import (build_priming_text, FunctionEntry,
                     build_parameter_schema,
                     FunctionCallResult,
                     has_repeating_tail,
                     count_trailing_backslashes)
import re


def generate_field(sdk: LLM, current_ids: list[int],
                   typed: str, state: GenState,
                   valid_name: list[str],
                   token_lookup: dict[int, str],
                   int_value: list[Any]) -> tuple[list[int] | list[float],
                                                  str]:
    field_typed = ""
    go = True
    MAX_LENGTH = 100
    while go:
        if len(field_typed) >= MAX_LENGTH:
            if state != GenState.IN_PARAMETER_VALUE_NUMBER\
               or state != GenState.IN_PARAMETER_VALUE_FLOAT:
                typed = typed[:-1] + '"'
            else:
                typed = typed[:-1]
            break
        logits = sdk.get_logits_from_input_ids(current_ids)
        masked = mask_logits(logits, field_typed,
                             state, valid_name, token_lookup)
        h_token_id = masked.index(max(masked))
        best_token_str = token_lookup[h_token_id]
        current_ids.append(h_token_id)
        typed = typed+best_token_str
        print(typed)
        if field_typed and field_typed.isdigit():
            if any(v for v in int_value
                   if field_typed == v):
                if len(int_value) > 1:
                    best_token_str = ','
                else:
                    best_token_str = '}'
        if state == GenState.IN_PARAMETER_VALUE_FLOAT:
            if field_typed and field_typed.isdigit():
                if any(v for v in int_value
                        if v.startswith(field_typed + '.')):
                    best_token_str = '.'
            if any(v for v in int_value
                    if field_typed == v):
                if len(int_value) > 1:
                    best_token_str = ','
                else:
                    best_token_str = '}'
        field_typed += best_token_str
        if state == GenState.IN_PARAMETER_VALUE_STRING:
            for block_size in [3, 5, 6, 7, 8,
                               9, 10, 11, 15, 17, 18, 19, 20]:
                if has_repeating_tail(field_typed, block_size):
                    typed = typed[:-block_size-2] + '"'
                    go = False
                    break
        if state == GenState.IN_FUNCTION_NAME and best_token_str == '"':
            typed = typed[0:-1]
            break
        if state == GenState.IN_PARAMETER_VALUE_STRING \
                and best_token_str.endswith('"')\
                and count_trailing_backslashes(typed) % 2 == 0:
            break
        if state == GenState.IN_PARAMETER_VALUE_NUMBER\
                and best_token_str in (',', '}'):
            current_ids.pop()
            typed = typed[0:-1]
            field_typed = field_typed[0:-1]
            break
        if state == GenState.IN_PARAMETER_VALUE_FLOAT and best_token_str\
                in (',', '}'):
            break
    return (current_ids, typed)


def generate_one_call(sdk: LLM, prompt_txt: Any,
                      function_defs: list[FunctionEntry],
                      valid_names: list[str],
                      token_lookup: dict[int, str]) -> FunctionCallResult:
    int_value = re.findall(r'\d+.\d+', prompt_txt)
    for i in re.findall(r'\d+', prompt_txt):
        if any(int(float(f)) == int(i) for f in int_value):
            continue
        else:
            int_value.append(i)

    priming_txt = build_priming_text(prompt_txt, function_defs)
    typed = '{"name": "'
    current_ids = sdk.encode(priming_txt + typed).tolist()[0]
    current_ids, typed = generate_field(sdk, current_ids, typed,
                                        GenState.IN_FUNCTION_NAME,
                                        valid_names, token_lookup, int_value)

    function_name = typed.split('"')[3]
    schema = build_parameter_schema(function_name, function_defs)
    typed = typed + '", "parameters": {'
    current_ids = sdk.encode(priming_txt + typed).tolist()[0]
    parameters: dict[Any, Any] = {}
    for i, (key, value) in enumerate(schema.items()):
        typed = typed + '"' + key + '": '
        if value.type == 'string':
            typed += '"'
        current_ids = sdk.encode(priming_txt+typed).tolist()[0]
        state = GenState.IN_PARAMETER_VALUE_STRING
        if value.type == 'string':
            state = GenState.IN_PARAMETER_VALUE_STRING
        elif value.type == 'integer':
            state = GenState.IN_PARAMETER_VALUE_NUMBER
        elif value.type == 'number':
            state = GenState.IN_PARAMETER_VALUE_FLOAT
        typed_b = typed
        current_ids, typed = generate_field(sdk, current_ids, typed,
                                            state, valid_names,
                                            token_lookup, int_value)
        r_value = typed[len(typed_b):]
        try:
            if value.type == 'string':
                parameters[key] = r_value.strip()\
                                  .rstrip('"').replace('\\\\', '\\')
            elif value.type == 'integer':
                parameters[key] = int(r_value)
            elif value.type == 'number':
                parameters[key] = float(r_value)
            elif value.type == 'boolean':
                parameters[key] = bool(r_value)
        except ValueError:
            break

        if i < len(schema)-1:
            typed += ', '
            current_ids = sdk.encode(priming_txt + typed)\
                             .tolist()[0]
    typed += '}}'

    return FunctionCallResult(prompt=prompt_txt,
                              name=function_name, parameters=parameters)
