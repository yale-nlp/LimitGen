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
        SYSTEM_INPUT = "Read the following section from a scientific paper. If the section is related to the paper's result analysis, output 'yes'; otherwise, output 'no'."
    elif aspect == "literature":
        SYSTEM_INPUT = "Read the following section from a scientific paper. If the section is related to the paper's literature review (introduction, related work), output 'yes'; otherwise, output 'no'."
    elif aspect == "experimental":
        SYSTEM_INPUT = "Read the following section from a scientific paper. If the section is related to the paper's experimental design, output 'yes'; otherwise, output 'no'."
    elif aspect == "methodology":
        SYSTEM_INPUT = "Read the following section from a scientific paper. If the section is related to the paper's methodology, output 'yes'; otherwise, output 'no'."

    messages = []
    for section in sections:
        USER_INPUT = section
        cur_message=prepare_message(SYSTEM_INPUT, USER_INPUT)
        messages.append(cur_message)

    return messages

def prepare_paper(sections, responses):
    paper_data = ""
    for section, response in zip(sections, responses):
        if response.strip().lower().startswith("yes"):
            paper_data += section + "\n"
        elif response.strip().lower().startswith("no"):
            pass
        else:
            print("Invalid response:", response)
            print("Section:", section)

    return paper_data


def prepare_sections(paper_path):
    sections = []
    is_start = 1
    with open(paper_path, 'r', encoding='utf-8') as f:
        for line in f:
            record = json.loads(line)
            if record['type'] == 'Section' or is_start:
                sections.append(record['text'])
                is_start = 0
            else:
                sections[-1] += record['text']

    return sections

# Example usage
if __name__ == "__main__":
    os.environ["OPENAI_BASE_URL"] = "xxxx"
    os.environ["OPENAI_API_KEY"] = "xxxx"
    client = AsyncOpenAI()
    todo_aspect = "methodology" # select methodology, experiment, result, literature
    base_dir = './papers'  # set your path to retrieved papers
    for root, dirs, files in os.walk(base_dir):
        if root == base_dir:
            continue
        if f'{todo_aspect}.txt' in files:
            continue
        file_path = os.path.join(root, 'sections.jsonl')
        
        print("Processing:", file_path)
        sections = prepare_sections(file_path)
        messages = prepare_messages(todo_aspect, sections)

        responses = asyncio.run(
            generate_from_openai_chat_completion(
                client,
                messages=messages, 
                engine_name="gpt-4o-mini", # gpt-3.5-turbo
                max_tokens=1000, # 32
                requests_per_minute = 100,
                # response_format={"type":"json_object"},
            )
        )

        processed_paper = prepare_paper(sections, responses)
        output_file_path = os.path.join(root, f'{todo_aspect}.txt')
        with open(output_file_path, 'w', encoding='utf-8') as output_file:
            output_file.write(processed_paper)

