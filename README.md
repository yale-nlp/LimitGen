<p align="center">
  <h1 style="display: inline;">
    Can LLMs Identify Critical Limitations within Scientific Research? A Systematic Evaluation on AI Research Papers
  </h1>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/">📖 Paper</a> •
  <a href="https://huggingface.co/datasets/yale-nlp/">🤗 Data</a>
</p>


## 📰 News
- **2025-xx-xx**: We are excited to release the LimitGen paper, dataset, and code!

## LimitGen Benchmark

While LLMs show promise in various scientific tasks, their potential to assist with peer review, particularly in identifying paper limitations, remains understudied. LimitGen, the first comprehensive benchmark for evaluating LLMs' capability to support early-stage feedback and complement human peer review. Our benchmark consists of two subsets: LimitGen-Syn, a synthetic dataset carefully created through controlled perturbations of papers, and LimitGen-Human, a collection of real human-written limitations.

## 🚀 Quickstart
### 1. Setup
Install the required packages
```bash
pip install -r requirements.txt
```


### 2. Limitation Generation
As detailed in Section 5.1 of our paper, we evaluate the performance of four frontier LLMs on our benchmark. To run the experiments on different subsets, navigate to the `identification` directory and execute:

```bash
cd identification
python main_human.py    # for the human subset
# or
python main_syn.py --error_type <error_type>     # for the synthetic subset
```
Make sure to modify the data paths, specify the model name and API key, and choose whether to enable RAG before running.

### 3. RAG
1. Run the following scripts in order to obtain a list of relevant papers:
```
   cd retrieval
   python query_gen.py
   python search.py
   python recommendation.py
   python rerank.py
```

   This process will generate a list of recommended papers. You will need to download the PDFs.

2. (Optional) Use MMDA to preprocess the downloaded PDFs.

3. After downloading and preprocessing the papers, run:

   ```
   python section_locate.py
   python rewrite.py
   ```

   These scripts will generate reference content based on the selected papers.


### 4. Evaluation


## ✍️ Citation
If you use our work and are inspired by our work, please consider cite us (available soon):
```

```
