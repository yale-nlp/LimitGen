from openai import OpenAI
import os
import argparse
import shutil
import json
import json
import sys
from pathlib import Path
from tqdm import tqdm
utils_dir = Path(__file__).parent.parent / 'utils'
sys.path.append(str(utils_dir))
from openai_utils import *


def prepare_messages(pairs):
    prompt_path = "./prompts/overlap.txt"
    with open(prompt_path, 'r') as file:
        prompt = file.read()

    messages = []
    for pair in pairs:
        user_prompt = prompt.format(ground_truth = pair["ref"], generated_limitation = pair["gen"])
        message = [
            {"role": "system", "content": "You are an AI language model tasked with analyzing academic texts."},
            {"role": "user", "content": user_prompt},
            ]
        messages.append(message)
    
    # print(user_prompt)
    return messages

def get_limitation_pairs(doc_id, content):
    ref_path = "../data/human/classified_limitations.json"
    with open(ref_path, 'r', encoding='utf-8') as file:
        ref_data = json.load(file)

    pairs = []

    for limitation in content["methodology"]:
        for reference in ref_data[doc_id]["limitations"]["methodology"]:
            pair = {
                "aspect": "methodology",
                "gen": limitation,
                "ref": reference
            }
            pairs.append(pair)

    for limitation in content["experiment"]:
        for reference in ref_data[doc_id]["limitations"]["experimental design"]:
            pair = {
                "aspect": "experiment",
                "gen": limitation,
                "ref": reference
            }
            pairs.append(pair)

    for limitation in content["result"]:
        for reference in ref_data[doc_id]["limitations"]["result analysis"]:
            pair = {
                "aspect": "result",
                "gen": limitation,
                "ref": reference
            }
            pairs.append(pair)
    
    for limitation in content["literature"]:
        for reference in ref_data[doc_id]["limitations"]["literature review"]:
            pair = {
                "aspect": "literature",
                "gen": limitation,
                "ref": reference
            }
            pairs.append(pair)


    
    return pairs

def prepare_limit(limit_data):
    limitation_dict = {}
    for doc_id in limit_data.keys():
        limitation_dict[doc_id] = limit_data[doc_id]["limitation"]


    return limitation_dict

# Example usage
if __name__ == "__main__":
    os.environ["OPENAI_BASE_URL"] = "xxxx"
    os.environ["OPENAI_API_KEY"] = "xxxx" # Set your OpenAI API key here
    
    client = AsyncOpenAI()
    AsyncOpenAI.api_key = os.getenv('OPENAI_API_KEY')

    parser = argparse.ArgumentParser(description='evaluate llm generated limitation')
    parser.add_argument('--model_name', type=str, default='marg', help='Name of the model')
    parser.add_argument("--retrieval",type=bool,default=True)
    args = parser.parse_args()
    
    model_name = args.model_name

    if args.retrieval:
        dir_name = "retrieval"
    else:
        dir_name = "non-retrieval"
    limit_path = f"" # set your input path
    

    with open(limit_path, 'r', encoding='utf-8') as file:
        limit_data = json.load(file)
        limitation_dict = prepare_limit(limit_data)

    result_list = {}
    for doc_id, content in tqdm(limitation_dict.items()):
        results = []
        pairs = get_limitation_pairs(doc_id, content)
        messages = prepare_messages(pairs)
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
        for response in responses:
            try:
                response_data = json.loads(response)
                result = {
                    "pair": pairs[idx],
                    "relatedness": response_data["relatedness"],
                    "specificity": response_data["specificity"]
                }
                idx += 1
            except:
                print(doc_id, idx, response)
                result = {
                    "pair": pairs[idx],
                    "response": response
                }
                idx += 1

            results.append(result)

        result_list[doc_id] = results
        
            
    with open(f"./evaluation/{dir_name}/{model_name}/match.json", "w") as file:
        json.dump(result_list, file, indent=4)