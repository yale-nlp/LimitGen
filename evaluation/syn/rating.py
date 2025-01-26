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

def prepare_messages(prompt_path, gen_limitation, gt_limitation):
    with open(prompt_path, 'r', encoding='utf-8') as file:
        SYSTEM_INPUT = file.read()

    USER_INPUT = f"Ground truth limitation: \n{gt_limitation}\n\nGenerated limitation: \n{gen_limitation}\n\nProvide a brief explanation, then assign a rating (1-5)."
    # print(USER_INPUT)
    return prepare_message(SYSTEM_INPUT, USER_INPUT)


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--openai_key",type=str,default="xxxx")
    parser.add_argument("--openai_base",type=str,default="xxxx")
    parser.add_argument("--section_dir",type=str,default="../data/syn/papers/sections") 
    parser.add_argument("--prompt_path",type=str, default="./prompts/rating.txt")
    parser.add_argument("--error_type",type=str,default="metric")
    parser.add_argument("--model_name",type=str,default="gpt-4o-mini")
    parser.add_argument("--label",type=str,default="non-retrieval")
    args=parser.parse_args()
    section_dir, prompt_path, error_type = args.section_dir, args.prompt_path, args.error_type
    model_name, label = args.model_name, args.label


    os.environ["OPENAI_BASE_URL"] = args.openai_base
    os.environ["OPENAI_API_KEY"] = args.openai_key # Set your OpenAI API key here
    client = AsyncOpenAI()

    input_path = f"./evaluation/{model_name}/{label}/{error_type}/coarse_limitation.json"
    output_path = f"./evaluation/{model_name}/{label}/{error_type}/fine_limitation.json"
    gt_path = os.path.join(section_dir, error_type+".json")

    messages = []
    result_dict = {}
    with open(input_path, 'r', encoding='utf-8') as file:
        limit_data = json.load(file)

    with open(gt_path, 'r', encoding='utf-8') as file:
        gt_data = json.load(file)

    for paper, limitations in limit_data.items():
        gt_limitation = gt_data[paper]["ground_truth"]
        result_dict[paper] = []
        for limitation in limitations:
            if limitation["coarse"]:
                messages.append(prepare_messages(prompt_path, limitation["limitation"], gt_limitation))
            else:
                result_dict[paper].append({"limitation": limitation["limitation"], "rating": 0})

    responses = asyncio.run(
        generate_from_openai_chat_completion(
            client,
            messages=messages, 
            engine_name="gpt-4o", # gpt-3.5-turbo
            max_tokens=1000, # 32
            requests_per_minute = 90,
            response_format={"type":"json_object"},
        )
    )

    idx = 0
    for paper, limitations in limit_data.items():
        for limitation in limitations:
            if limitation["coarse"]:
                try:
                    response_data = responses[idx]
                    response = json.loads(response_data)
                    explanation = response["explanation"]
                    rating = response["rating"]
                    result_dict[paper].append({"limitation": limitation["limitation"], "explanation": explanation, "rating": rating})
                except:
                    print(f"Parsing error: {paper}")
                    print(f"Response: {response_data}")
                    result_dict[paper].append({"limitation": limitation["limitation"], "response": response_data})
                idx += 1
            else:
                continue


    with open(output_path, "w") as file:
        json.dump(result_dict, file, indent=4)