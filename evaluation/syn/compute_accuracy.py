import os
import json
import argparse

def find_ground_truth(error_type):
    if error_type == "ablation":
        return "ablation study"
    elif error_type == "data":
        return "low data quality"
    elif error_type == "inappropriate":
        return "inappropriate method"
    elif error_type == "baseline":
        return "insufficient baseline"
    elif error_type == "dataset":
        return "limited datasets"
    elif error_type == "replace":
        return "inappropriate datasets"
    elif error_type == "review":
        return "limited scope"
    elif error_type == "citation":
        return "irrelevant citations"
    elif error_type == "description":
        return "inaccurate description"
    elif error_type == "metric":
        return "insufficient metric"
    elif error_type == "analysis":
        return "limited analysis"


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--model_name",type=str,default="marg")
    parser.add_argument("--label",type=str,default="retrieval")
    args=parser.parse_args()
    model_name, label = args.model_name, args.label
    root_dir = f"./evaluation/{model_name}/{label}/"

    for root, dirs, files in os.walk(root_dir):
        for dir_name in dirs:
            result_path = os.path.join(root, dir_name, "coarse_limitation.json")
            print(dir_name)
            with open(result_path, 'r', encoding='utf-8') as file:
                limit_data = json.load(file)
            ground_truth = find_ground_truth(dir_name)

            correct_cnt = 0

            for paper, limitations in limit_data.items():
                
                for limitation in limitations:
                    if limitation.get("skip") or limitation.get("answer") != "yes":
                        continue
                    elif limitation["subtype"] == ground_truth:
                        correct_cnt += 1
                        break

            print(f"{ground_truth} has {correct_cnt}/{len(limit_data)} correct answers.")
            print(f"The acc is {correct_cnt/len(limit_data):.3f}\n\n")

    for root, dirs, files in os.walk(root_dir):
        for dir_name in dirs:
            result_path = os.path.join(root, dir_name, "coarse_limitation.json")
            with open(result_path, 'r', encoding='utf-8') as file:
                limit_data = json.load(file)
            ground_truth = find_ground_truth(dir_name)

            for paper, limitations in limit_data.items():
                
                for limitation in limitations:
                    if limitation.get("skip") or limitation.get("answer") != "yes":
                        limitation["coarse"] = False
                    elif limitation["subtype"] == ground_truth:
                        limitation["coarse"] = True
                    else: 
                        limitation["coarse"] = False

            with open(result_path, "w") as file:
                json.dump(limit_data, file, indent=4)