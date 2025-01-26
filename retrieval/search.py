import os
import json
import requests
from tqdm import tqdm
import time
import sys

SEMANTIC_SCHOLAR_API_URL = "https://api.semanticscholar.org/graph/v1/paper/"


def get_paper(month, query):
    # query = query.replace(" ","+")
    # &publicationDateOrYear=:2024-01-01&
    if month == -1:
        url = f"{SEMANTIC_SCHOLAR_API_URL}search?query={query}&limit=3&fields=url,title,abstract&fieldsOfStudy=Computer Science"
    elif month == 1:
        url = f"{SEMANTIC_SCHOLAR_API_URL}search?query={query}&limit=3&fields=url,title,abstract&fieldsOfStudy=Computer Science&publicationDateOrYear=:2023-12-15"
    else:
        url = f"{SEMANTIC_SCHOLAR_API_URL}search?query={query}&limit=3&fields=url,title,abstract,publicationDate&fieldsOfStudy=Computer Science&publicationDateOrYear=:2024-0{str(month-1)}-01"
    response = requests.get(url)
    while response.status_code != 200:
        time.sleep(1)
        print(response)
        response = requests.get(url)

    return response.json()



if __name__ == "__main__":
    query = "tldr" # need to select tldr or title

    query_path = f"./query/tldr.json"  
    output_path = "./query/search_result.json"  

    with open(query_path, 'r', encoding='utf-8') as file:
        paper_data = json.load(file)

    results = {}
    for paper, data in tqdm(paper_data.items()):
        if "papers" in data.keys():
            results[paper] = data
            continue
        if query == "tldr":
            query = data["query"]
        else query == "title":
            query = data["title"]

        # month = int(paper[3:4]) # if from arxiv
        month = -1
        retrieval = get_paper(month, query)
        try:
            results[paper] = data
            results[paper]["papers"] = retrieval["data"]
        except:
            print(paper, retrieval)

    with open(output_path, "w") as file:
        json.dump(results, file, indent=4)