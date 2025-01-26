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
        prompt = """Please classify the following limitation of a scientific paper into one of the following subtypes:
1. Low Data Quality - The data collection method is unreliable, potentially introducing bias and lacking adequate preprocessing.
2. Inappropriate Method - Some methods in the paper are unsuitable for addressing this research question and may lead to errors or oversimplifications.
3. Lack of Novelty - The work fails to enhance established techniques from prior research, remaining largely unchanged. This limited novelty may result in missed opportunities to improve model effectiveness and applicability.
4. Limited Performance - The method's performance is insufficiently impressive, often lacking robustness and generalization across various datasets or tasks.
5. Others

Output only the corresponding number."""
    elif error_type in ["baseline", "dataset", "replace", "ablation"]:
        aspect = "experimental design"
        prompt = """Please classify the following limitation of a scientific paper into one of the following subtypes:
1. Insufficient baseline models/methods - Fail to evaluate the proposed approach against a broad range of well-established methods.
2. Limited datasets - Rely on limited datasets, which may hinder the generalizability and robustness of the proposed approach.
3. Inappropriate datasets - Use of inappropriate datasets, which may not accurately reflect the target task or real-world scenarios.
4. Lack of an ablation study - Fail to perform an ablation study or account for a specific module, leaving the contribution of a certain component to the research unclear.
5. Others

Output only the corresponding number."""
    elif error_type in ["metric","analysis"]:
        aspect = "result analysis"  
        prompt = """Please classify the following limitation of a scientific paper into one of the following subtypes:
1. Insufficient evaluation metrics - Rely on insufficient evaluation metrics, which may provide an incomplete assessment of the model's overall performance.
2. Limited analysis - Offer insufficient insights into the model's behavior and failure cases.
3. Misalignment between text and tables/figure - Show discrepancies between the text and accompanying tables or figures, such as conflicting numerical values or inconsistent comparison outcomes.
4. Exaggerated or misleading conclusions - Draw exaggerated or misleading conclusions that may overstate the model's effectiveness or applicability beyond the presented evidence.
5. Others

Output only the corresponding number."""  
    elif error_type in ["citation", "review", "description"]:
        aspect = "literature review" 
        prompt = """Please classify the following limitation of a scientific paper into one of the following subtypes:
1. Limited Scope of the Review - The review may focus on a very specific subset of literature or methods, leaving out important studies or novel perspectives.
2. Irrelevant Citations - Include irrelevant references or outdated methods, which distracts from the main points and undermines the strength of conclusions.
3. Inaccurate Description of Existing Methods - Provide an inaccurate description of existing methods, which can hinder readers' understanding of the context and relevance of the proposed approach.
4. Others

Output only the corresponding number."""  
    else:
        print("invalid subtype")
        return None
    
    # print(user_prompt)
    return prepare_message(prompt, limitation)


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--openai_key",type=str,default="xxxx")
    parser.add_argument("--openai_base",type=str,default="xxxx")
    parser.add_argument("--error_type",type=str,default="metric")
    parser.add_argument("--model_name",type=str,default="gpt-4o-mini")
    parser.add_argument("--label",type=str,default="non-retrieval")
    args=parser.parse_args()
    error_type, model_name, label = args.error_type, args.model_name, args.label


    os.environ["OPENAI_BASE_URL"] = args.openai_base
    os.environ["OPENAI_API_KEY"] = args.openai_key # Set your OpenAI API key here

    input_path = f"./evaluation/{model_name}/{label}/{error_type}/checked_limitation.json"
    output_path = f"./evaluation/{model_name}/{label}/{error_type}/coarse_limitation.json"


    # os.makedirs(f"./evaluation/marg/{error_type}", exist_ok=True)

    client = AsyncOpenAI()

    messages = []

    with open(input_path, 'r', encoding='utf-8') as file:
        limit_data = json.load(file)

    result_dict = {}

    if error_type in ["data", "inappropriate"]:
        aspects = ["low data quality", "inappropriate method", "lack of novelty", "limited performance", "others"]
    elif error_type in ["baseline", "dataset", "replace", "ablation"]:
        aspects = ["insufficient baseline", "limited datasets", "inappropriate datasets", "ablation study", "others"]
    elif error_type in ["metric", "analysis"]:
        aspects = ["insufficient metric", "limited analysis", "result misalignment", "misleading conclusion", "others"]
    elif error_type in ["citation", "review", "description"]:
        aspects = ["limited scope", "irrelevant citations", "inadequate description", "others"]
    else:
        print("invalid subtype")
        exit 
        
        

    for paper, limitations in limit_data.items():
        result_dict[paper] = []

        for limitation in limitations:
            if limitation.get("skip") or limitation.get("answer") != "yes":
                continue
            message = prepare_messages(error_type, limitation["limitation"])
            if message:
                messages.append(message)
            else:
                limitation["skip"] = True
                print("Failed to generate messages:", paper, "\nThe limitation is", limitation)


    responses = asyncio.run(
        generate_from_openai_chat_completion(
            client,
            messages=messages, 
            engine_name="gpt-4o", # gpt-3.5-turbo
            max_tokens=1000, # 32
            requests_per_minute = 80,
            # response_format={"type":"json_object"},
        )
    )

    idx = 0
    paper_list = limit_data.keys()
    for paper in paper_list:
        for limitation in limit_data[paper]:
            if limitation.get("skip") or limitation.get("answer") != "yes":
                continue
            response = responses[idx]

            try:
                aspect = response
                limitation["subtype"] = aspects[int(aspect.strip()[0])-1]
            except:
                limitation["subtype"] = response       
                print("unexpected answer", response)

            idx += 1



    with open(output_path, "w") as file:
        json.dump(limit_data, file, indent=4)