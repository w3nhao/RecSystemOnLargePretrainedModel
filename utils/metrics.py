from abc import ABC, abstractmethod
from torchmetrics import Metric
import torch


def get_topk_ranks(pred_scores, target, topk):
    """ get topk ranks of the target in the pred_scores
    """
    assert target.shape[0] == pred_scores.shape[0]
    assert pred_scores.shape[1] >= topk

    if target.ndim == 1:
        target = target.unsqueeze(1)

    _, topk_idx = pred_scores.topk(topk, dim=1)

    # get hit index and rank, e.g. hit_rank_arr = [[hit_query, rank], ...]
    hit_rank_arr = (target == topk_idx).nonzero()
    hit_query = hit_rank_arr[:, :1]
    hit_rank = hit_rank_arr[:, 1:2] + 1

    all_rank = torch.zeros_like(target)

    # first set all rank to maximum value of int64 or set all rank to topk + 1
    # all_rank[:] = torch.iinfo(torch.int64).max
    all_rank[:] = topk + 1

    # then scatter the exact rank to the hit query
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

        self.add_state(f"accumulate_metric",
                       default=torch.tensor(0.0),
                       dist_reduce_fx="sum")
        self.add_state("accumulate_count",
                       default=torch.tensor(0),
                       dist_reduce_fx="sum")

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


if __name__ == "__main__":
        # get all ranks example:
        #     import torch

        #     pred_scores, topk_idx = torch.randn((2048, 30)).topk(20, dim=1)
        #     target = torch.randint(0, 20, (2048, 1))

        #     hit_rank_arr = (target == topk_idx).nonzero()
        #     hit_preds = hit_rank_arr[:, :1]
        #     hit_rank = hit_rank_arr[:, 1:2] + 1

        #     all_rank = torch.zeros_like(target)

        #     all_rank[:] = torch.iinfo(torch.int64).max
        #     all_rank.scatter_(0, hit_preds, hit_rank)

        #     for k in [5, 10, 20]:
        #         mrr_k = torch.sum(1.0 / all_rank[all_rank <= k]) / 2048
        #         print(f"mrr@{k}: {mrr_k.item()}")

        #     for k in [5, 10, 20]:
        #         ndcg_k = torch.sum(1.0 / torch.log2(all_rank[all_rank <= k] + 1)) / 2048
        #         print(f"ndcg@{k}: {ndcg_k.item()}")

        #     for k in [5, 10, 20]:
        #         hit_k = (all_rank <= k).sum()  / 2048
        #         print(f"hit@{k}: {hit_k.item()}")
        
        # TODO: add test case
        import torch
        from torchmetrics import RetrievalHitRate, RetrievalMRR, RetrievalNormalizedDCG

        pred_scores, topk_idx = torch.randn((2048, 30)).topk(20, dim=1)
        indexes = torch.arange(0, 2048).unsqueeze(1).expand_as(pred_scores)
        target = torch.randint(0, 20, (2048, 1))
        targets = torch.split(target, 2048 // 10, dim=0)
        pred_scores = torch.split(pred_scores, 2048 // 10, dim=0)
        topk_idxes = torch.split(topk_idx, 2048 // 10, dim=0)
        indexes_list = torch.split(indexes, 2048 // 10, dim=0)

        hr = HR(k=20)
        ndcg = NDCG(k=20)
        mrr = MRR(k=20)

        hr1 = RetrievalHitRate(k=20)
        ndcg1 = RetrievalNormalizedDCG(k=20)
        mrr1 = RetrievalMRR()

        hit_rank_arrs = []
        hit_preds = []
        hit_ranks = []
        all_ranks = []
        new_targets = []

        for target, pred_score, topk_idx, indexes in zip(targets, pred_scores, topk_idxes, indexes_list):
            hit_rank_arr = (target == topk_idx).nonzero()
            hit_rank_arr[:, 1:2] = hit_rank_arr[:, 1:2] + 1
            hit_pred = hit_rank_arr[:, :1]
            hit_rank = hit_rank_arr[:, 1:2]
            hit_rank_arrs.append(hit_rank_arr)
            hit_preds.append(hit_pred)
            hit_ranks.append(hit_rank)
            all_rank = torch.zeros_like(target)
            all_rank[:] = torch.iinfo(torch.int64).max
            all_rank.scatter_(0, hit_pred, hit_rank)
            all_ranks.append(all_rank)
            hr.update(all_rank, len(target))
            ndcg.update(all_rank, len(target))
            mrr.update(all_rank, len(target))

            new_target = target == topk_idx
            
            new_targets.append(new_target)
            hr1.update(pred_score, new_target, indexes)
            ndcg1.update(pred_score, new_target, indexes)
            mrr1.update(pred_score, new_target, indexes)

        assert hr.compute() == hr1.compute()
        assert ndcg.compute() == ndcg1.compute()
        assert mrr.compute().float() == mrr1.compute().float()