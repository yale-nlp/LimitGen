from openai import OpenAI
import os
import argparse
import shutil
import json
import sys
from pathlib import Path
utils_dir = Path(__file__).parent.parent / 'utils'
sys.path.append(str(utils_dir))
from openai_utils import *

def prepare_messages(paper_contents):
    SYSTEM_INPUT = "Generate a TLDR in 5 words of the following text. Do not use any proposed model names or dataset names from the text. Output only the 5 words without punctuation."
    messages = []
    for paper_content in paper_contents:
        USER_INPUT = paper_content
        cur_message=prepare_message(SYSTEM_INPUT, USER_INPUT)
        messages.append(cur_message)

    
    # print(user_prompt)
    return messages

# Example usage
if __name__ == "__main__":

    parser=argparse.ArgumentParser()
    parser.add_argument("--openai_base",type=str,default="xxxx")
    parser.add_argument("--openai_key",type=str,default="xxxx")
    parser.add_argument("--paper_path",type=str,default="../data/human/classified_limitations.json") 
    parser.add_argument("--output_dir",type=str,default="./query") 
    args=parser.parse_args()
    paper_path, output_dir = args.paper_path, args.output_dir

    os.environ["OPENAI_BASE_URL"] = ars.openai_base
    os.environ["OPENAI_API_KEY"] = args.openai_key 
    client = AsyncOpenAI()

    with open(paper_path, 'r') as json_file:
        paper_data = json.load(json_file)

    paper_list = []
    paper_contents = []
    titles = []

    results = {}

    for doc_id, paper in paper_data.items():
        if "papers" in paper.keys():
            results[doc_id] = paper
        else:

            title = paper["title"]
            content = paper["abstract"]
            paper_list.append(doc_id)
            paper_contents.append(content)
            titles.append(title)
        
    messages = prepare_messages(paper_contents)

    responses = asyncio.run(
        generate_from_openai_chat_completion(
            client,
            messages=messages, 
            engine_name="gpt-4o", # gpt-3.5-turbo
            max_tokens=1000, # 32
            requests_per_minute = 150,
            # response_format={"type":"json_object"},
        )
    )

    

    for doc_id, response, title in zip(paper_list, responses, titles):
        if response:
            # print(doc_id)
            results[doc_id] = {"title": title, "query": response}
        else:
            print("Failed: ", doc_id)
        # print(f"Response: {response}")


    output_path = os.path.join(output_dir, "tldr.json")
    with open(output_path, "w") as file:
        json.dump(results, file, indent=4)