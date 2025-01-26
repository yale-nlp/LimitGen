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
    parser.add_argument("--prompt_path",type=str, default="./prompts/rating.txt")
    parser.add_argument("--model_name",type=str,default="gpt-4o")
    parser.add_argument("--label",type=str,default="retrieval")
    args=parser.parse_args()
    prompt_path = args.prompt_path
    model_name, label = args.model_name, args.label


    os.environ["OPENAI_BASE_URL"] = args.openai_base
    os.environ["OPENAI_API_KEY"] = args.openai_key # Set your OpenAI API key here
    client = AsyncOpenAI()

    input_path = f"./evaluation/{label}/{model_name}/match.json"
    output_path = f"./evaluation/{label}/{model_name}/rating.json"

    messages = []
    with open(input_path, 'r', encoding='utf-8') as file:
        limit_data = json.load(file)

    paper_cnt = 10
    for doc_id, content in limit_data.items():
        paper_cnt -= 1
        if paper_cnt < 0:
            break
        for pair in content:
            if pair["relatedness"] in ["medium", "high"]:
                message = prepare_messages(prompt_path, pair["pair"]["gen"], pair["pair"]["ref"])
                messages.append(message)
            else:
                pair["rating"] = 0
    paper_cnt = 10
    print(message)
    responses = asyncio.run(
        generate_from_openai_chat_completion(
            client,
            messages=messages, 
            engine_name="gpt-4o", # gpt-3.5-turbo
            max_tokens=200, # 32
            requests_per_minute = 80,
            response_format={"type":"json_object"},
        )
    )
    idx = 0
    for doc_id, content in limit_data.items():
        paper_cnt -= 1
        if paper_cnt < 0:
            break
        for pair in content:
            if pair["relatedness"] in ["medium", "high"]:
                response = responses[idx]
                idx += 1
                try:
                    response_data = json.loads(response)
                    pair["rating"] = response_data["rating"]
                    pair["explanation"] = response_data["explanation"]

                
                except:
                    print(doc_id, idx, response)
                    pair["response"] = response

        
            
    with open(output_path, "w") as file:
        json.dump(limit_data, file, indent=4)