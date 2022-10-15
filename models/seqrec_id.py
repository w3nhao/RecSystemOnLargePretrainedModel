import torch
from utils.pylogger import get_pylogger
from utils.metrics import get_topk_ranks
from utils.schedule_functions import get_lr_scheduler_function
from models.abstract_recommender import SeqRec, METRIC_LIST
from models.configs import SeqRecConfig
from models.utils import gather_indexes

log = get_pylogger(__name__)


class IDSeqRec(SeqRec):

    def __init__(self, config: SeqRecConfig):
        self.save_hyperparameters()
        super().__init__(self.hparams.config)

        # parameters initialization
        self.apply(self._init_weights)

    def _set_feature_extractor(self, config):
        self.item_embedding = torch.nn.Embedding(config.item_token_num,
                                                 config.sasrec_hidden_size,
                                                 padding_idx=0)

    def _feature_extract(self, item_id_seq):
        item_embs = self.item_embedding(item_id_seq)
        return item_embs
    
    def _val_test_step(self, batch, batch_idx, stage):
        target_seq, input_seq, seq_mask, _, _ = batch
        
        seq_emb = self.forward(input_seq) # (B, L, N_items)
        last_item_idx = torch.sum(seq_mask, dim=-1) - 1 # (B)
        seq_last_emb = gather_indexes(seq_emb, last_item_idx) # (B, N_items)
        last_id = target_seq.gather(1, last_item_idx.view(-1, 1)) # (B, 1)

        topk_list = self.hparams.config.topk_list
        pred_scores = seq_last_emb.softmax(dim=-1)
        all_ranks = get_topk_ranks(pred_scores=pred_scores,
                                   target=last_id,
                                   topk=max(topk_list))

        for k in topk_list:
            for metric_name in METRIC_LIST:
                metric = self.topk_metric[f"{metric_name}@{k}"]
                metric.update(all_ranks, last_id.numel())
        
    def forward(self, item_id_seq, item_seq_mask):
        item_embs = self._feature_extract(item_id_seq)
        output = self.sasrec(item_embs, item_seq_mask)  # (B, L_sas, H_sas)
        output = self.classification_head(output)
        return output  # (B, L, N_items)
        
    def training_step(self, batch, batch_idx):
        target_seq, input_seq, seq_mask, _, _ = batch
        seq_emb = self.forward(input_seq, seq_mask)  # (B, L, N_items)
        loss = self.loss_fct(seq_emb.reshape(-1, seq_emb.size(-1)),
                             target_seq.reshape(-1))
        return loss

    def configure_optimizers(self):
        lr = self.hparams.config.lr
        wd = self.hparams.config.weight_decay
        optimizer = torch.optim.AdamW(self.parameters(),
                                      lr=lr,
                                      weight_decay=wd)
        return optimizer

    @classmethod
    def build_model_config(cls, args, item_token_num):
        config = SeqRecConfig(item_token_num=item_token_num)
        config = super(IDSeqRec, cls).build_model_config(args, config)
        return config

    