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

def prepare_messages(error_type, paper_content, figures, retrieval):
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
    SYSTEM_INPUT = f"Read the following scientific paper and generate 3 major limitations in this paper about its {aspect}. Do not include any limitation explicitly mentioned in the paper itself. Return the limitations in the following JSON format: {{\"limitations\": <a list of 3 limitations>}}."
    if retrieval:
        SYSTEM_INPUT = f"Read the following content from several papers to gain knowledge in the relevant field. Using this knowledge, review a new scientific paper in this field. Based on existing research, identify the limitations of the 'Paper to Review'. Generate three major limitations related to its {aspect} in this paper. Do not include any limitation explicitly mentioned in the paper itself. Return the limitations in the following JSON format: {{\"limitations\": <a list of 3 limitations>}}."
    # print(user_prompt)
    if figures == []:
        return prepare_message(SYSTEM_INPUT, paper_content)
    else:
        USER_INPUT = [{"type": "text", "text": paper_content}] + figures
        return prepare_message(SYSTEM_INPUT, USER_INPUT)

def prepare_paper(table, paper_path):
    with open(paper_path, 'r', encoding='utf-8') as file:
        paper_data = json.load(file)

    content = f"Paper to review: \nTitle: {paper_data['title']}\n"
    content += f"Abstract: {paper_data['abstract']}\n"
    for section in paper_data["sections"]:
        content += section["section_id"] + " " + section["section_name"] + ':\n'+ section["text"] + '\n\n'

    return content





def paper_retrieval(error_type, doc_id, retrieval_path, retrieval_dir):
    if error_type in ["data", "inappropriate"]:
        aspect = "methodology"
    elif error_type in ["baseline", "dataset", "replace", "ablation"]:
        aspect = "experiment"
    elif error_type in ["metric","analysis"]:
        aspect = "result"    
    elif error_type in ["citation", "review", "description"]:
        aspect = "literature"   
    else:
        print("invalid subtype")
        return None

    with open(retrieval_path, 'r', encoding='utf-8') as file:
        paper_data = json.load(file)

    retrieval_content = ""
    papers = paper_data[doc_id]["final_papers"]
    print(len(papers))
    cnt = 1
    for paper in papers:
        retrieval_content += f"Relevant Paper {str(cnt)}:\n"
        cnt += 1
        retrieval_content += f"Title: {paper['title']}\n"
        file_path = os.path.join(retrieval_dir, paper["paperId"], f'{aspect}_final.txt')
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                retrieval_content += f.read() + "\n\n"
        except:
            retrieval_content += paper["abstract"] + "\n\n"
    return retrieval_content + "\n\n"


# Example usage
if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--openai_key",type=str,default="xxxx")
    parser.add_argument("--openai_base",type=str,default="xxxx")
    parser.add_argument("--base_dir",type=str,default="../data/syn/papers") 
    parser.add_argument("--output_dir",type=str,default="./limitations") 
    parser.add_argument("--error_type",type=str,default="replace")
    parser.add_argument("--retrieval",type=bool,default=False)
    parser.add_argument("--model_name",type=str,default="gpt-4o")
    parser.add_argument("--retrieval_path",type=str,default="../retrieval/query/final_recommendation.json")
    parser.add_argument("--retrieval_dir",type=str,default="../retrieval/papers")

    args=parser.parse_args()
    base_dir, output_dir, error_type, retrieval_path, retrieval_dir, model_name = args.base_dir, args.output_dir, args.error_type, args.retrieval_path, args.retrieval_dir, args.model_name


    if args.retrieval:
        dir_name = "retrieval"
    else:
        dir_name = "non-retrieval"
    os.environ["OPENAI_BASE_URL"] = args.openai_base
    os.environ["OPENAI_API_KEY"] = args.openai_key 
    output_path = f"./limitations/{model_name}/{dir_name}/{error_type}/generated_limitation.json"
    os.makedirs(f"./limitations/{model_name}/{dir_name}/{error_type}", exist_ok=True)

    client = AsyncOpenAI()

    messages = []

    paper_dir = os.path.join(base_dir, error_type)

    paper_list = os.listdir(paper_dir)

    for paper in paper_list:
        paper_path = os.path.join(paper_dir, paper)
        paper_content = prepare_paper(args.table, paper_path)
        figures = []

        if args.figure:
            figures = prepare_figure(paper_path)

        if args.retrieval:
            paper_content = paper_retrieval(error_type, paper[:-5], retrieval_path, retrieval_dir) + paper_content

        message = prepare_messages(error_type, paper_content, figures, args.retrieval)
        if message:
            messages.append(message)

    print(message)
    responses = asyncio.run(
        generate_from_openai_chat_completion(
            client,
            messages=messages, 
            engine_name=model_name, 
            max_tokens=500, # 32
            requests_per_minute = 80,
            top_p = 1, 
            response_format={"type":"json_object"},
        )
    )

    results = {}
    idx = 0
    for paper, response in zip(paper_list,responses):
        if response == "Invalid Message":
            print("faied to generate json format response", paper[:-5], response)
            continue
        try: 
            response_data = json.loads(response)
            doc_id = paper[:-5]
            results[doc_id] = {
                "limitation": response_data["limitations"]
            }
        except:
            
            doc_id = paper[:-5]
            results[doc_id] = {
                "response": response
            }
            print("faied to generate json format response", doc_id, response)



    with open(output_path, "w") as file:
        json.dump(results, file, indent=4)
