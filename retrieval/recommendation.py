import os
import json
import requests
from tqdm import tqdm
import time
import sys
import random

SEMANTIC_SCHOLAR_API_URL = "https://api.semanticscholar.org/recommendations/v1/papers/forpaper/"


# Extract date from arxiv_id, e.g., "2403.12432" extracts "2024-03"
def get_paper_date_from_arxiv_id(arxiv_id):
    year = int("20" + arxiv_id[:2])  # Extract year, "24" -> 2024
    month = int(arxiv_id[2:4])  # Extract month, "03" -> 3
    return year, month


# Check if the recommended paper's publication date is earlier than the target paper's date
def is_paper_earlier(publication_date, target_year, target_month):
    pub_year, pub_month, _ = map(int, publication_date.split("-"))
    # Return True if the recommended paper was published earlier than the target paper
    return (pub_year < target_year) or (pub_year == target_year and pub_month < target_month)


# Fetch recommended papers with a limit of 500 papers per request
def get_paper(s2_id, limit=500):
    url = f"{SEMANTIC_SCHOLAR_API_URL}{s2_id}?limit={limit}&fields=url,title,abstract,publicationDate,isOpenAccess"

    # print(url)
    response = requests.get(url)
    while response.status_code != 200:
        time.sleep(1)
        print(response)
        response = requests.get(url)

    return response.json()


if __name__ == "__main__":
    id_path = "./query/search_result.json" 
    output_path = "./query/recommendation.json"  

    with open(id_path, 'r', encoding='utf-8') as file:
        paper_data = json.load(file)

    results = {}
    insufficient_papers = []  # List to store arxiv ids with less than 10 recommended papers

    for doc_id, paper in tqdm(paper_data.items()):
        # target_year, target_month = get_paper_date_from_arxiv_id(doc_id)  # Extract target paper's publication date if from arxiv


        
        recommended_paper_list = []
        if "papers" not in paper.keys():
            continue
        for retrieved_paper in paper["papers"]:
            s2_id = retrieved_paper["paperId"]

            retrieval = get_paper(s2_id, limit=100)
            recommended_papers = []
            for recommended_paper in retrieval["recommendedPapers"]:
                if recommended_paper["abstract"] is None or recommended_paper["publicationDate"] is None:
                    continue

                # Check if the recommended paper's publication date is earlier than the target paper, if from arxiv
                # if is_paper_earlier(recommended_paper["publicationDate"], target_year, target_month) and recommended_paper["isOpenAccess"] == True:
                #     recommended_papers.append(recommended_paper)

                if recommended_paper["isOpenAccess"] == True:
                    recommended_papers.append(recommended_paper)

                if len(recommended_papers) >= 5:
                    break

            recommended_paper_list.extend(recommended_papers)


        with open(f"../data/human/classified_limitations.json", 'r', encoding='utf-8') as file: 
            paper_abstract = json.load(file)

        results[doc_id] = {
            "title": paper["title"],
            "abstract": paper_abstract[doc_id]["abstract"],
            "papers_1": [retrieved_paper for retrieved_paper in paper["papers"] if retrieved_paper["abstract"]],
            "papers_2": recommended_paper_list
        }

        combined_list = {item['paperId']: item for item in results[doc_id]["papers_1"] + recommended_paper_list}.values()
        combined_list = list(combined_list)
        combined_list = [item for item in combined_list if item['title'].lower() != paper['title'].lower()]
        random.shuffle(combined_list)
        if len(combined_list) <= 5:
            insufficient_papers.append((doc_id, len(combined_list)))

        results[doc_id]["papers"] = combined_list
        print(len(combined_list))

    # Save results to the output file
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=4, ensure_ascii=False)

    # Output paper ids with less than 5 valid recommended papers
    if insufficient_papers:
        print("The following paper IDs have fewer than 5 recommended papers:")
        for doc_id in insufficient_papers:
            print(doc_id)
