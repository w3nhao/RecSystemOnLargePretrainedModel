from abc import ABC, abstractmethod
from torchmetrics import Metric
import torch

def get_topk_ranks(pred_scores, target, topk):
    """ get topk ranks of the target in the pred_scores
    example:
        import torch
        
        pred_scores, topk_idx = torch.randn((2048, 30)).topk(20, dim=1)
        target = torch.randint(0, 20, (2048, 1))

        hit_rank_arr = (target == topk_idx).nonzero()
        hit_rank_arr[:, 1:2] = hit_rank_arr[:, 1:2] + 1
        hit_preds = hit_rank_arr[:, :1]
        hit_rank = hit_rank_arr[:, 1:2]

        all_rank = torch.zeros_like(target)
        
        all_rank[:] = torch.iinfo(torch.int64).max
        all_rank.scatter_(0, hit_preds, hit_rank)

        for k in [5, 10, 20]:
            mrr_k = torch.sum(1.0 / all_rank[all_rank <= k]) / 2048
            print(f"mrr@{k}: {mrr_k.item()}")

        for k in [5, 10, 20]:
            ndcg_k = torch.sum(1.0 / torch.log2(all_rank[all_rank <= k] + 1)) / 2048
            print(f"ndcg@{k}: {ndcg_k.item()}")
            
        for k in [5, 10, 20]:
            hit_k = (all_rank <= k).sum()  / 2048
            print(f"hit@{k}: {hit_k.item()}")
    """
    assert target.shape[0] == pred_scores.shape[0]
    assert pred_scores.shape[1] >= topk
    
    if target.ndim == 1:
        target = target.unsqueeze(1)
        
    _, topk_idx = pred_scores.topk(topk, dim=1)
    
    hit_rank_arr = (target == topk_idx).nonzero()
    
    hit_rank_arr[:, 1:2] = hit_rank_arr[:, 1:2] + 1
    hit_query= hit_rank_arr[:, :1]
    hit_rank = hit_rank_arr[:, 1:2]
    
    
    all_rank = torch.zeros_like(target)

    # set all rank maximum value of int64
    # all_rank[:] = torch.iinfo(torch.int64).max

    # set all rank to topk + 1
    all_rank[:] = topk + 1

    all_rank.scatter_(0, hit_query, hit_rank)
    return all_rank

class RecRetrivalMetric(Metric, ABC):
    """RecRetrivalMetric
    """
    higher_is_better = True
    full_state_update = False
    is_differentiable = False
    
    def __init__(self, k):

        super().__init__()
        assert isinstance(k, int)
        assert k > 0
        self.k = k
        
        self.add_state(f"accumulate_metric", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("accumulate_count", default=torch.tensor(0), dist_reduce_fx="sum")
        
    def update(self, topk_rank, batch_size):
        self.accumulate_count += batch_size
        self.accumulate_metric += self._metric(topk_rank[topk_rank <= self.k])
    
    def compute(self):
        return self.accumulate_metric / self.accumulate_count
    
    @abstractmethod
    def _metric(self, topk_rank):
        raise NotImplementedError
    
class MRR(RecRetrivalMetric):
    higher_is_better = True
    full_state_update = False
    is_differentiable = False

    def __init__(self, k):
        super().__init__(k)
        
    def _metric(self, topk_rank):
        return torch.sum(1.0 / topk_rank)
    
class NDCG(RecRetrivalMetric):
    higher_is_better = True
    full_state_update = False
    is_differentiable = False

    def __init__(self, k):
        super().__init__(k)
    
    def _metric(self, topk_rank):
        return torch.sum(1.0 / torch.log2(topk_rank + 1))
    
class HR(RecRetrivalMetric):
    higher_is_better = True
    full_state_update = False
    is_differentiable = False

    def __init__(self, k):
        super().__init__(k)
    
    def _metric(self, topk_rank):
        return topk_rank.numel()