import torch.nn as nn
import torch.nn.functional as F
import torch

import numpy as np
import math, os
from copy import deepcopy

from transformers import BertPreTrainedModel, BertModel, BertForMaskedLM
from methylbert.function import FocalLoss 
from methylbert.config import MethylBERTConfig

METHYLBERT_PRETRAINED_MODEL_ARCHIVE_MAP = {
    "hanyangii/methylbert_hg19_12l": "https://huggingface.co/hanyangii/methylbert_hg19_12l/resolve/main/pytorch_model.bin",
    "hanyangii/methylbert_hg19_8l": "https://huggingface.co/hanyangii/methylbert_hg19_8l/resolve/main/pytorch_model.bin",
    "hanyangii/methylbert_hg19_6l": "https://huggingface.co/hanyangii/methylbert_hg19_6l/resolve/main/pytorch_model.bin",
    "hanyangii/methylbert_hg19_4l": "https://huggingface.co/hanyangii/methylbert_hg19_4l/resolve/main/pytorch_model.bin",
    "hanyangii/methylbert_hg19_2l": "https://huggingface.co/hanyangii/methylbert_hg19_2l/resolve/main/pytorch_model.bin",
}

class MethylBertEmbeddedDMR(BertPreTrainedModel):
    pretrained_model_archive_map = METHYLBERT_PRETRAINED_MODEL_ARCHIVE_MAP
    config_class = MethylBERTConfig
    base_model_prefix = "methylbert"

    def __init__(self, config, seq_len=150):
        # from pretrained - calls the init
        super().__init__(config)
        self.num_labels = config.num_labels

        if config.loss not in ["bce", "focal_bce"]:
            raise ValueError(f"loss must be bce or focal_bce. {config.loss} is given.")

        self.loss = config.loss
        self.classification_loss_fct = self._setup_loss(self.loss)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.read_classifier = nn.Sequential(
            nn.Linear((config.hidden_size+1)*(seq_len+1), seq_len+1),
            nn.Dropout(0.05),#config.hidden_dropout_prob),
            nn.ReLU(),
            nn.LayerNorm(seq_len+1, eps=config.layer_norm_eps),
            nn.Linear(seq_len+1, 2)
        )

        self.seq_len = seq_len

        self.dmr_encoder = nn.Sequential(
            nn.Embedding(num_embeddings=self.num_labels, embedding_dim = seq_len+1),
        )

        self.init_weights()

    def _setup_loss(self, loss):
        if loss == "bce":
            print("Cross entropy loss assigned")
            return nn.CrossEntropyLoss() # this function requires unnormalised logits
        elif loss == "focal_bce":
            print("Focal loss assigned")
            return FocalLoss()


    def check_model_status(self):
        print("Bert model training mode : %s"%(self.bert.training))
        print("Dropout training mode : %s"%(self.dropout.training))
        print("Read classifier training mode : %s"%(self.read_classifier.training))
        
    def from_pretrained_read_classifier(self, pretrained_model_name_or_path, device="cpu"):
        self.read_classifier.load_state_dict(torch.load(pretrained_model_name_or_path, map_location=device))
        
    def from_pretrained_dmr_encoder(self, pretrained_model_name_or_path, device="cpu"):
        self.dmr_encoder.load_state_dict(torch.load(pretrained_model_name_or_path, map_location=device))
        
    def forward(
        self,
        step,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        ctype_label=None
    ):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)

        #DMR info 
        encoded_dmr = self.dmr_encoder(labels.view(-1))

        sequence_output =  torch.cat((sequence_output, encoded_dmr.unsqueeze(-1)), axis=-1)

        ctype_logits = self.read_classifier(sequence_output.view(-1,(self.seq_len+1)*769))
        
        loss = self.classification_loss_fct(ctype_logits.view(-1, 2), 
                                            F.one_hot(ctype_label, num_classes=2).to(torch.float32).view(-1, 2))
        ctype_logits = ctype_logits.softmax(dim=1)

        outputs = {"loss": loss,
                    "dmr_logits":sequence_output,
                   "classification_logits": ctype_logits}
        
        return outputs  # (loss), logits, (hidden_states), (attentions)


