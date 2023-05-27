from transformers import pipeline
import pandas as pd
import os
import torch

from core_metrics.core_metrics import *


def make_example(model):
    test_q = 'Why is model conversion important?'
    test_context = 'The option to convert models between FARM and transformers gives freedom to the user and let people easily switch between frameworks.'
    QA_input = {
        'question': test_q,
        'context': test_context
    }
    print(QA_input)
    res = model(QA_input)
    print(res)

class HF_Wrapper:
    def __init__(self, hf_model):
        self.model = hf_model

    def predict(self, query:str, context:str) -> dict:
        input = {'question':query,
                 'context':context}
        return self.model(input)
    
    def eval_on_data(self, data_file:str) -> pd.DataFrame:
        '''
        Assumes input data from the msmarco-fira-21.qrels.qa-tuples.tsv data
        according to the assignment text:
        queryid documentid relevance-grade query-text document-text text-selection (multiple answers possible, split with tab)
        '''

        if not os.path.isfile(data_file):
            raise FileExistsError(f"{data_file} is not a file that exists!")
        
        result = []
        with open(data_file, "r") as f:
            for n,l in enumerate(f.readlines()):
                if n%100 == 0:
                    print(n)
                parts = [x for x in l.strip().split("\t") if len(x) > 0]
                qid = parts[0]
                did = parts[1]
                query = parts[3]
                context = parts[4]
                answers = parts[5:]

                pred = self.predict(query, context)["answer"]

                exact = 0
                f1 = 0
                for a in answers:
                    exact = max(compute_exact(a, pred), exact)
                    f1 = max(compute_f1(a, pred), f1)
                
                result.append(
                    {"queryid":qid,
                     "documentid":did,
                     "answer":pred,
                     "exact":exact,
                     "f1":f1})
        return pd.DataFrame(result)
    
    def summarize_results(self, results:pd.DataFrame) -> pd.DataFrame:
        return results[["exact", "f1"]].describe()
    
if __name__ == "__main__":

    model_name = "deepset/roberta-base-squad2"
    device = 0 if torch.cuda.is_available() else -1
    hfm = pipeline('question-answering', 
                   model=model_name, tokenizer=model_name, device=device)
    hfm = hfm
    model = HF_Wrapper(hfm)
    output = model.eval_on_data(
        "data/Part-3/msmarco-fira-21.qrels.qa-tuples.tsv")
    output.to_csv("hf_extractive_qa_output.csv", sep=",", index=False)
    print(model.summarize_results(output))