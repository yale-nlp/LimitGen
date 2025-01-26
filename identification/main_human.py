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
import base64

def prepare_messages(aspect, paper_content, is_retrieval):
    if aspect == "experiment":
        hint = "experimental design"
    elif aspect == "methodology":
        hint = "methodology"
    elif "result":
        hint = "result analysis"
    elif "literature":
        hint = "literature review"
    SYSTEM_INPUT = f"Read the following scientific paper and generate 3 major limitations in this paper about its {hint}. Do not include any limitation mentioned in the paper itself. Return only the limitations in the following JSON format: {{\"limitations\": <a list of 3 limitations>}}."
    if is_retrieval:
        SYSTEM_INPUT = f"Read the following content from several papers to gain knowledge in the relevant field. Using this knowledge, review a new scientific paper in this field. Based on existing research, identify the limitations of the 'Paper to Review'. Generate three major limitations related to its {hint} in this paper. Do not include any limitation mentioned in the paper itself. Return only the limitations in the following JSON format: {{\"limitations\": <a list of 3 limitations>}}."
    # print(user_prompt)
    return prepare_message(SYSTEM_INPUT, paper_content)


def prepare_paper(paper_path, title):

    content = f"Paper to review: \nTitle: {title}\n"
    with open(paper_path, 'r', encoding='utf-8') as f:
        for line in f:
            record = json.loads(line)
            content += record["text"]

    return content


def paper_retrieval(doc_id, retrieval_path, retrieval_dir, aspect):
    with open(retrieval_path, 'r', encoding='utf-8') as file:
        paper_data = json.load(file)

    retrieval_content = ""
    papers = paper_data[doc_id]["final_papers"]
    cnt = 1
    for paper in papers:
        retrieval_content += f"Relevant Paper {str(cnt)}:\n"
        cnt += 1
        retrieval_content += f"Title: {paper['title']}\n"
        file_path = os.path.join(retrieval_dir, paper["paperId"], f'{aspect}_final.txt')
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                retrieval_content += f.read() + "\n\n"  # 读取整个文件内容
        except:
            retrieval_content += paper["abstract"] + "\n\n"
    return retrieval_content + "\n\n"


# Example usage
if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--openai_key",type=str,default="xxxx")
    parser.add_argument("--openai_base",type=str,default="xxxx")
    parser.add_argument("--paper_dir",type=str,default="../data/human/papers") 
    parser.add_argument("--output_dir",type=str,default="./limitations") 
    parser.add_argument("--id_path",type=str,default="../data/human/classified_limitations.json") 
    parser.add_argument("--retrieval",type=bool,default=False)
    parser.add_argument("--retrieval_path",type=str,default="../retrieval/query/final_recommendation.json")
    parser.add_argument("--retrieval_dir",type=str,default="../retrieval/papers")
    parser.add_argument("--model_name",type=str,default="gpt-4o")


    args=parser.parse_args()
    paper_dir, output_dir, retrieval_path, retrieval_dir, model_name = args.paper_dir, args.output_dir, args.retrieval_path, args.retrieval_dir, args.model_name
    id_path = args.id_path
    
    os.environ["OPENAI_BASE_URL"] = args.openai_base
    os.environ["OPENAI_API_KEY"] = args.openai_key



    if args.retrieval:
        dir_name = "retrieval"
    else:
        dir_name = "non-retrieval"
    
    output_path = f"./limitations/{dir_name}/{model_name}/generated_limitation.json"
    os.makedirs(f"./limitations/{dir_name}/{model_name}", exist_ok=True)

    client = AsyncOpenAI()

    messages = []
    with open(id_path, 'r', encoding='utf-8') as file:
        paper_list = json.load(file)


    error_list = []

    for doc_id, paper_data in paper_list.items():
        paper_path = os.path.join(paper_dir, doc_id, "sections.jsonl")
        if os.path.exists(paper_path):
            paper_content = prepare_paper(paper_path, paper_data["title"])
        else:
            paper_path = os.path.join(paper_dir, doc_id[:-1], "sections.jsonl")
            if os.path.exists(paper_path):
                paper_content = prepare_paper(paper_path, paper_data["title"])
            else:
                paper_path = os.path.join(paper_dir, doc_id[:-2], "sections.jsonl")
                paper_content = prepare_paper(paper_path, paper_data["title"])


        for aspect in ["experiment", "methodology", "result", "literature"]:


            if args.retrieval:
                paper_content = paper_retrieval(doc_id, retrieval_path, retrieval_dir, aspect) + paper_content

            message = prepare_messages(aspect, paper_content, args.retrieval)
            if message:
                messages.append(message)
            else:
                error_list.append(paper)
                print("failed to generate messages:", paper, aspect)


    responses = asyncio.run(
        generate_from_openai_chat_completion(
            client,
            messages=messages, 
            engine_name=model_name, # gpt-3.5-turbo
            max_tokens=500, # 32
            requests_per_minute = 15,
            top_p = 1,
            response_format={"type":"json_object"},
        )
    )

    results = {}
    idx = 0

    for doc_id, paper_data in paper_list.items():


        limitations = {}
        for aspect in ["experiment", "methodology", "result", "literature"]:

            response = responses[idx]

            if response == "Invalid Message":
                idx += 1
                print("faied to generate json format response", doc_id, aspect, response)
                continue
            try: 
                response_data = json.loads(response)
                limitations[aspect] = response_data["limitations"]
                if not isinstance(response_data["limitations"][0], str):
                    print(response)
                # print(results[doc_id])
            except:
                try:
                    response_data = response[response.index("{") :].strip().strip("`")
                    response_data = json.loads(response_data)
                    limitations[aspect] = response_data["limitations"]
                        
                except:
                    limitations[f"{aspect}_response"] = response
                    print("faied to generate json format response", doc_id, aspect, response)

            idx += 1
        results[doc_id] = {
            "limitation": limitations
        }

    output_path = f"./limitations/{dir_name}/{model_name}/generated_limitation.json"
    with open(output_path, "w") as file:
        json.dump(results, file, indent=4)