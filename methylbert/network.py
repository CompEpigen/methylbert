import torch.nn as nn
import torch.nn.functional as F
import torch

import numpy as np
import math, os

from transformers import BertPreTrainedModel, BertModel

# https://github.com/kaidic/LDAM-DRW/blob/master/losses.py
class LDAMLoss(nn.Module):
    
    def __init__(self, cls_num_list, max_m=0.5, weight=None, s=30):
        super(LDAMLoss, self).__init__()

        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list)) # C = max_m / np.max(m_list)
        m_list = torch.cuda.FloatTensor(m_list)
        self.m_list = m_list
        assert s > 0
        self.s = s
        self.weight = weight

    def forward(self, x, target):
        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)
        
        index_float = index.type(torch.cuda.FloatTensor)
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0,1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m
    
        output = torch.where(index, x_m, x)
        return F.cross_entropy(self.s*output, target, weight=self.weight)

class CNNClassification(nn.Module):
    def __init__(self, seq_len, n_hidden, label_count = None, n_seq=150):
        super(CNNClassification, self).__init__()
        self.seq_len = seq_len
        self.model = nn.Sequential(
            nn.Linear(in_features=769, out_features=n_hidden*2, bias=True),
            nn.Dropout(p=0.1),
            nn.Linear(in_features=n_hidden*2, out_features=n_hidden, bias=True),
            nn.ReLU(),
            nn.Dropout(p=0.1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=n_hidden*(n_seq+1), out_features=2), # binary classification,
            nn.Sigmoid() #nn.Softmax()
        )

        # Weight initialisation
        self.init_weights()

    def init_weights(self):
        """ Initialize and prunes weights if needed. """
        # Initialize weights
        self.apply(self._init_weights)


    def _init_weights(self, module):
        # From hugging face
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, bert_features, methyl_seqs):
        # Methyl seq feature learning

        methyl_seqs = methyl_seqs[:, :, None]

        x =  torch.cat((bert_features.to(torch.float32), methyl_seqs), axis=-1)

        features = self.model(x)
        logits  = self.classifier(torch.flatten(features, 1))

        
        outputs = (logits,) + (features,)
        
        return outputs

class MethylseqFeature(CNNClassification):
    def __init__(self, seq_len, bert_feature_shape, n_hidden):
        self.seq_len = seq_len
        self.bert_feature_shape = bert_feature_shape

        # Methylation sequence embedding
        self.embedding = nn.Sequential(
            nn.conv2D(in_channels = 1, out_channels=self.bert_feature_shape, )
        )


class MethylBertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config, seq_len=150):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)

        self.read_classifier =  CNNClassification(seq_len=seq_len,
                                                   n_hidden=100)

        
        self.init_weights()

    def set_nclass(self, n_classes):
        self.n_classes=n_classes
        print(self.n_classes)

    def from_pretrained_read_classifier(self, pretrained_model_name_or_path, device="cpu"):
        self.read_classifier.load_state_dict(torch.load(pretrained_model_name_or_path, map_location=device))
        
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
        methyl_seqs=None,
        ctype_label=None
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
            Labels for computing the sequence classification/regression loss.
            Indices should be in :obj:`[0, ..., config.num_labels - 1]`.
            If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).

    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.BertConfig`) and inputs:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`label` is provided):
            Classification (or regression if config.num_labels==1) loss.
        logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, config.num_labels)`):
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.

    Examples::

        from transformers import BertTokenizer, BertForSequenceClassification
        import torch

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)

        loss, logits = outputs[:2]

        """
    
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        
        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        classification_output = self.read_classifier(bert_features=outputs[1][-1],
                                                     methyl_seqs = methyl_seqs)
        
        if type(self.n_classes) == list :
            classification_loss_fct = LDAMLoss(cls_num_list = self.n_classes)
        else:
            classification_loss_fct = nn.BCELoss()

        dmr_loss_fct = nn.CrossEntropyLoss()
        dmr_loss = dmr_loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        binary_loss = classification_loss_fct(classification_output[0].view(-1, 2), 
            nn.functional.one_hot(ctype_label, num_classes=2).to(torch.float32).view(-1, 2))

        loss = dmr_loss + binary_loss

        outputs = (loss,) + outputs

        outputs = {"bert_logits": outputs[1],
                   "dmr_logits": logits,
                   "loss": loss,
                   "classification_logits": classification_output[0],
                   "attentions": outputs[-1]}

        return outputs  # (loss), logits, (hidden_states), (attentions)

