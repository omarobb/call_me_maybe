from .models import build_token_loockup
from llm_sdk import Small_LLM_Model as LLM
from .cli import loader
from .generator import generate_one_call
from json import dump
from typing import Any
if __name__ == "__main__":
    sdk = LLM()
    OKBLUE = '\033[95m'
    GREEN = "\033[32m"
    RESET = "\033[0m"
    (function_definitions, prompt, out, function_name) = loader()
    final = []
    loockup = build_token_loockup(sdk)
    for i, pro in enumerate(prompt):
        print(f"{OKBLUE}Test {i+1}: {pro}{RESET}")
        print("------------------------------------"
              "------------------------------------")
        if str(pro) == "prompt=''":
            print("Empty prompt")
            continue
        data: dict[str, Any] = generate_one_call(sdk, pro.prompt,
                                                 function_definitions,
                                                 function_name,
                                                 loockup).model_dump()
        final.append(data)
        print(f"{GREEN}{data}{RESET}")
        print("------------------------------------"
              "------------------------------------")
    with open(out, 'w') as f:
        dump(final, f, indent=4)
