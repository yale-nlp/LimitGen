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

def prepare_messages(paper_contents, ans_lens):


    

    messages = []
    for paper_content, ans_len in zip(paper_contents,ans_lens):
        if ans_len < 5:
            SYSTEM_INPUT = f"Given the abstracts of {ans_len} papers and the abstract of a reference paper, rank the papers in order of relevance to the reference paper. Output the top {ans_len} as a list of integers in JSON format: {{'ranking': [1, 10, 4, 2, 8]}}."
        else:
            SYSTEM_INPUT = f"Given the abstracts of {ans_len} papers and the abstract of a reference paper, rank the papers in order of relevance to the reference paper. Output the top 5 as a list of integers in JSON format: {{'ranking': [1, 10, 4, 2, 8]}}."
        USER_INPUT = paper_content
        cur_message=prepare_message(SYSTEM_INPUT, USER_INPUT)
        messages.append(cur_message)

    
    # print(user_prompt)
    return messages

def prepare_paper(paper):
    content = ""
    cnt = 1

    for rec_paper in paper["papers"]:
        content += f"Paper {cnt}: {rec_paper['title']}\n{rec_paper['abstract']}\n\n"
        cnt += 1

    content += f"Reference Paper: {paper['title']}\n"
    content += f"Abstract: {paper['abstract']}\n"

    return content, len(paper["papers"])

# Example usage
if __name__ == "__main__":

    parser=argparse.ArgumentParser()
    parser.add_argument("--openai_base",type=str,default="xxxx")
    parser.add_argument("--openai_key",type=str,default="xxxx")
    parser.add_argument("--paper_path",type=str,default="./query/recommendation.json") 
    parser.add_argument("--output_path",type=str,default="./query/final_recommendation.json") 
    args=parser.parse_args()
    paper_path, output_path = args.paper_path, args.output_path

    os.environ["OPENAI_BASE_URL"] = ars.openai_base
    os.environ["OPENAI_API_KEY"] = args.openai_key 
    
    client = AsyncOpenAI()

    
    paper_contents = []
    paper_list = []
    ans_lens = []
    with open(paper_path, 'r', encoding='utf-8') as file:
        paper_data = json.load(file)

    for doc_id, paper in paper_data.items():
        content, ans_len = prepare_paper(paper)
        paper_contents.append(content)
        paper_list.append(doc_id)
        ans_lens.append(ans_len)
        
    messages = prepare_messages(paper_contents, ans_lens)

    responses = asyncio.run(
        generate_from_openai_chat_completion(
            client,
            messages=messages, 
            engine_name="gpt-4o", # gpt-3.5-turbo
            max_tokens=1000, # 32
            requests_per_minute = 105,
            response_format={"type":"json_object"},
        )
    )

    results = {}

    for paper, response in zip(paper_list, responses):
        results[paper] = {"title": paper_data[paper]["title"], "papers": paper_data[paper]["papers"]}
        try:
            response_data = json.loads(response)
            results[paper]["ranking"] = response_data["ranking"]
            if len(response_data["ranking"]) < 5:
                print("invalid answer: ", paper)
                rec_papers = []
                for num in response_data["ranking"][:5]:
                    num = int(num)
                    rec_papers.append(paper_data[paper]["papers"][num-1])
                    results[paper]["final_papers"] = rec_papers
            elif len(response_data["ranking"]) == 5:
                rec_papers = []
                for num in response_data["ranking"][:5]:
                    num = int(num)
                    rec_papers.append(paper_data[paper]["papers"][num-1])
                    results[paper]["final_papers"] = rec_papers
            else:
                print("invalid answer: ", paper)



        except:
            print("Failed: ", paper)
            results[paper]["response"] = response
            print(f"Response: {response}")


    with open(output_path, "w") as file:
        json.dump(results, file, indent=4)