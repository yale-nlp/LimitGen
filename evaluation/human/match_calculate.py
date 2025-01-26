import json

def measure_overlap_for_all_papers(papers, gen_path):
    
    paper_count = {}

    ref_path = "../data/human/classified_limitations.json"
    with open(ref_path, 'r', encoding='utf-8') as file:
        ref_data = json.load(file)

    with open(gen_path, 'r', encoding='utf-8') as file:
        gen_data = json.load(file)

    total_recall, total_precision, total_jaccard = {}, {}, {}

    for aspect in ["methodology", "experiment", "result", "literature"]:
        total_recall[aspect], total_precision[aspect], total_jaccard[aspect] = 0.0, 0.0, 0.0
        paper_count[aspect] = 0

    for doc_id, pairs in papers.items():
        matched_gt, matched_gen = {}, {}

        ref_len = {}
        refs = ref_data[doc_id]["limitations"]
        ref_len["methodology"] = len(refs["methodology"])
        ref_len["experiment"] = len(refs["experimental design"])
        ref_len["result"] = len(refs["result analysis"])
        ref_len["literature"] = len(refs["literature review"])


        gen_len = {}
        gens = gen_data[doc_id]["limitation"]
        for aspect in gens.keys():
            gen_len[aspect] = len(gens[aspect])
            matched_gen[aspect] = set()
            matched_gt[aspect] = set()

        for pair in pairs:
            gen, gt, aspect = pair['pair']['gen'], pair['pair']['ref'], pair['pair']['aspect']
            if "relatedness" not in pair.keys():
                print(doc_id, pair)
            relatedness, specificity = pair['relatedness'], pair['specificity']
            if relatedness in ["high", "medium"]:
                matched_gt[aspect].add(gt)
                matched_gen[aspect].add(gen)
        

        for aspect in gens.keys():
            if ref_len[aspect] == 0:
                continue
            recall = (len(matched_gt[aspect])) / (ref_len[aspect])
            precision = (len(matched_gen[aspect])) / (gen_len[aspect])
            intersection = (len(matched_gt[aspect]) + len(matched_gen[aspect])) / 2
            jaccard = intersection / (ref_len[aspect] + gen_len[aspect] - intersection)

            total_recall[aspect] += recall
            # print(recall)
            total_precision[aspect] += precision
            total_jaccard[aspect] += jaccard
            paper_count[aspect] += 1

    for aspect in ["methodology", "experiment", "result", "literature"]:
        total_recall[aspect] = total_recall[aspect] / paper_count[aspect]
        total_precision[aspect] = total_precision[aspect] / paper_count[aspect]
        total_jaccard[aspect] = total_jaccard[aspect] / paper_count[aspect]

    return total_recall, total_precision, total_jaccard


model_names = ["gpt-4o", "gpt-4o-mini"]
retrievals = ["retrieval", "non-retrieval"]
for model_name in model_names:
    for retrieval in retrievals:
        match_path = f"./evaluation/{retrieval}/{model_name}/match.json"
        with open(match_path, 'r', encoding='utf-8') as file:
            papers = json.load(file)

        gen_path = f"" # set your input path
        recall, precision, jaccard = measure_overlap_for_all_papers(papers, gen_path)
        if retrieval == "non-retrieval":
            retrieval_label = "no retrieval"
        else:
            retrieval_label = retrieval
        print(model_name, retrieval_label)
        for aspect in ["methodology", "experiment", "result", "literature"]:
            print(aspect)
            print("recall:", recall[aspect], "precision:", precision[aspect], "jaccard:", jaccard[aspect])
    
