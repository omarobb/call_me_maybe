import argparse
from .models import (load_function_definitions,
                     load_prompt_definitions, FunctionEntry,
                     load_function_name,
                     Prompt)


def loader() -> tuple[list[FunctionEntry],
                      list[Prompt], str, list[str]]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--functions_definition",
                        default="data/input/functions_definition.json")
    parser.add_argument("--input",
                        default="data/input/function_calling_tests.json")
    parser.add_argument("--output",
                        default="data/output/function_calls.json")
    args = parser.parse_args()
    fun_def = load_function_definitions(args.functions_definition)
    fun_prompt = load_prompt_definitions(args.input)
    function_name = load_function_name(args.functions_definition)
    return (fun_def, fun_prompt, args.output, function_name)
