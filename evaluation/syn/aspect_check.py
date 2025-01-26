from openai import OpenAI
import os
import argparse
import shutil
import json
import json
import sys
from pathlib import Path
utils_dir = Path(__file__).parent.parent / 'utils'
sys.path.append(str(utils_dir))
from openai_utils import *


def prepare_messages(error_type, limitation):
    if error_type in ["data", "inappropriate"]:
        aspect = "methodology"
    elif error_type in ["baseline", "dataset", "replace", "ablation"]:
        aspect = "experimental design"
    elif error_type in ["metric","analysis"]:
        aspect = "result analysis"    
    elif error_type in ["citation", "review", "description"]:
        aspect = "literature review"   
    else:
        print("invalid subtype")
        return None
    SYSTEM_INPUT = f"Please check whether the following limitation of a scientific paper is related to the {aspect}.\n\nOutput only \"yes\" or \"no\"."
    
    # print(user_prompt)
    return prepare_message(SYSTEM_INPUT, limitation)

# Example usage
if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--openai_key",type=str,default="xxxx")
    parser.add_argument("--openai_base",type=str,default="xxxx")
    parser.add_argument("--error_type",type=str,default="baseline")
    parser.add_argument("--model_name",type=str,default="gpt-4o-mini")
    parser.add_argument("--label",type=str,default="non-retrieval")
    args=parser.parse_args()
    error_type, model_name, label = args.error_type, args.model_name, args.label


    os.environ["OPENAI_BASE_URL"] = args.openai_base
    os.environ["OPENAI_API_KEY"] = args.openai_key # Set your OpenAI API key here

    input_path = "" # Set your input path
    output_path = f"./evaluation/{model_name}/{label}/{error_type}/checked_limitation.json"



    os.makedirs(f"./evaluation/{model_name}/{label}/{error_type}", exist_ok=True)

    client = AsyncOpenAI()

    messages = []

    with open(input_path, 'r', encoding='utf-8') as file:
        limit_data = json.load(file)

    result_dict = {}

    for paper, limitations in limit_data.items():
        result_dict[paper] = []

        for limitation in limitations["limitation"]:
            message = prepare_messages(error_type, limitation)
            if message:
                messages.append(message)
                result_dict[paper].append({"limitation": limitation, "skip": False})
            else:
                result_dict[paper].append({"limitation": limitation, "skip": True})
                print("Failed to generate messages:", paper, "\nThe limitation is", limitation)


    responses = asyncio.run(
        generate_from_openai_chat_completion(
            client,
            messages=messages, 
            engine_name="gpt-4o", # gpt-3.5-turbo
            max_tokens=1000, # 32
            requests_per_minute = 90,
            # response_format={"type":"json_object"},
        )
    )

    idx = 0
    paper_list = result_dict.keys()
    for paper in paper_list:
        for limitation in result_dict[paper]:
            if limitation["skip"]:
                continue
            response = responses[idx]
            if response.lower().startswith("yes"):
                response = "yes"
            elif response.lower().startswith("no"):
                response = "no"
            else:
                print("unexpected answer", response)

            limitation["answer"] = response
            idx += 1



    with open(output_path, "w") as file:
        json.dump(result_dict, file, indent=4)