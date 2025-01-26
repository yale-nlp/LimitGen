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

def prepare_messages(aspect, sections):
    if aspect == "result":
        SYSTEM_INPUT = """Concatenate all the content from the result analysis sections of a paper.
Remove sentences that are irrelevant to the result analysis of the experiments, and keep details about the metrics, case study and how the paper presents the results.
Organize the result in JSON format as follows:

{
    "result_text": str, not dict, not a summary
}
"""
    elif aspect == "methodology":
        SYSTEM_INPUT = """Concatenate all the content from the methodology sections of a paper.
Remove sentences that are irrelevant to the proposed methodology or models, and keep details about key components and innovations.
Organize the result in JSON format as follows:

{
    "methodology_text": str, not dict, not a summary
}
"""
    elif aspect == "experiment":
        SYSTEM_INPUT = """Concatenate all the content from the experimental design sections of a paper.
Remove sentences that are irrelevant to the experiment setup, and keep details about the datasets, baselines, and main experimental, ablation studies.
Organize the result in JSON format as follows:

{
    "experiment_text": str, not dict, not a summary
}
"""
    elif aspect == "literature":
        SYSTEM_INPUT = """Concatenate all the content from the literature review sections of a paper.
Remove sentences that are irrelevant to the literature review, and keep details about the related works.
Organize the result in JSON format as follows:

{
    "literature_text": str, not dict, not a summary
}
"""

    messages = []
    for section in sections:
        USER_INPUT = section
        cur_message=prepare_message(SYSTEM_INPUT, USER_INPUT)
        messages.append(cur_message)

    return messages

# Example usage
if __name__ == "__main__":
    os.environ["OPENAI_BASE_URL"] = "xxxx"
    os.environ["OPENAI_API_KEY"] = "xxxx"
    client = AsyncOpenAI()
    todo_aspect = "methodology" # select methodology, experiment, result, literature
    base_dir = './papers' # set your path to retrieved papers
    sections = []
    for root, dirs, files in os.walk(base_dir):
        if root == base_dir: 
            continue
        if f'{todo_aspect}.txt' not in files or f'{todo_aspect}_final.txt' in files:
            continue
        file_path = os.path.join(root, f'{todo_aspect}.txt')
        # print("Processing:", file_path)
        with open(file_path, 'r', encoding='utf-8') as f:
            sections.append(f.read())  

    messages = prepare_messages(todo_aspect,sections)

    
    responses = asyncio.run(
        generate_from_openai_chat_completion(
            client,
            messages=messages, 
            engine_name="gpt-4o", # gpt-3.5-turbo
            max_tokens=8192, # 32
            requests_per_minute = 40,
            response_format={"type":"json_object"},
        )
    )

    resp_idx = 0
    for root, dirs, files in os.walk(base_dir):
        if root == base_dir:
            continue
        if 'abstract.txt' in files or f'{todo_aspect}.txt' not in files:
            continue
        if f'{todo_aspect}_final.txt' in files:
            continue
        response = responses[resp_idx]
        resp_idx += 1
        try:
            response_data = json.loads(response)
            processed_paper = response_data[f"{todo_aspect}_text"]
            output_file_path = os.path.join(root, f'{todo_aspect}_final.txt')
            with open(output_file_path, 'w', encoding='utf-8') as output_file:
                output_file.write(processed_paper)
        except:
            print("Failed: ", root)
            print(response)
        

