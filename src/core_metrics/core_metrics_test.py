import unittest
from .core_metrics import load_qrels, calculate_metrics_plain
import numpy as np

if __name__ == '__main__':
    unittest.main()

class CoreMetricsTests(unittest.TestCase):
    def setUp(self):
        self.qrel_dict = {
            k:v for k, v in list(load_qrels("./data/Part-2/msmarco_qrels.txt").items())[0:20]
        }
        self.ideal_ranking = {}
        for query_id in self.qrel_dict.keys():
            self.ideal_ranking[query_id] = list(self.qrel_dict[query_id].keys())
        
        all_doc_ids = list(set([int(doc_ids[0]) for doc_ids in self.ideal_ranking.values()]))
        print(all_doc_ids)
        self.any_ranking = {}
        for query_id, doc_ids in self.ideal_ranking.items():
            self.any_ranking[query_id] = np.random.permutation(all_doc_ids)

    def test_calculate_metrics_keys(self):
        metrics = calculate_metrics_plain(self.ideal_ranking, self.qrel_dict)
        
        self.assertCountEqual(metrics.keys(), [
            'MRR@10', 
            'Recall@10', 
            'QueriesWithNoRelevant@10', 
            'QueriesWithRelevant@10', 
            'AverageRankGoldLabel@10', 
            'MedianRankGoldLabel@10', 
            'MRR@20', 
            'Recall@20', 
            'QueriesWithNoRelevant@20', 
            'QueriesWithRelevant@20', 
            'AverageRankGoldLabel@20', 
            'MedianRankGoldLabel@20', 
            'MRR@1000', 
            'Recall@1000', 
            'QueriesWithNoRelevant@1000', 
            'QueriesWithRelevant@1000', 
            'AverageRankGoldLabel@1000', 
            'MedianRankGoldLabel@1000', 
            'nDCG@3', 
            'nDCG@5', 
            'nDCG@10', 
            'nDCG@20', 
            'nDCG@1000', 
            'QueriesRanked', 
            'MAP@1000'
        ])
    
    def test_calculate_metrics_keys_any(self):
        metrics = calculate_metrics_plain(self.any_ranking, self.qrel_dict)
        
        print(metrics)
        print(self.any_ranking)