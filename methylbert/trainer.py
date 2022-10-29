import torch
import torch.nn as nn
from torch.optim import Adam, AdamW
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from torch.cuda.amp import autocast, GradScaler

from sklearn.metrics import roc_curve, auc, accuracy_score
from copy import deepcopy
from collections import OrderedDict

from methylbert.network import CNNClassification, MethylBertForSequenceClassification
from transformers import BertForSequenceClassification, BertConfig, BertForMaskedLM

import tqdm, time
import numpy as np
import os



def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, decrease_steps=100000, last_epoch=-1):
    # From DNABERT code
    """ Create a schedule with a learning rate that decreases linearly after
    linearly increasing during a warmup period.
    """

    def lr_lambda(current_step):
        if current_step <= num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        elif current_step >= decrease_steps:
            return max(
                0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - decrease_steps))
            )
        return 1

    return LambdaLR(optimizer, lr_lambda, last_epoch)



class Config(object):
    def __init__(self, config_dict):
        for k, v in config_dict.items():
            setattr(self, k, v)

def get_config(**kwargs):
    config = OrderedDict(
          [
            ('lr', 1e-4),
            ('beta', (0.9, 0.999)),
            ('weight_decay', 0.01),
            ('warmup_step', 10000),
            ('eps', 1e-6),
            ('with_cuda', True),
            ('log_freq', 10),
            ('eval_freq', 10),
            ('n_hidden', None),
            ("decrease_steps", 1500),
            ('eval', False),
            ('amp', False),
            ("gradient_accumulation_steps", 1), 
            ("max_grad_norm", 1.0),
            ("eval", False)
          ]
        )

    if kwargs is not None:
        for key in config.keys():
            if key in kwargs.keys():
                config[key] = kwargs.pop(key)

    return Config(config)



class MethylBertTrainer(object):
    def __init__(self, vocab_size: int, save_path: str = "", 
                 train_dataloader: DataLoader = None, test_dataloader: DataLoader = None,
                 **kwargs):

        # Setup config
        self._config = get_config(**kwargs)
        self.train_data = train_dataloader
            
        # Setup cuda device for BERT training, argument -c, --cuda should be true
        cuda_condition = torch.cuda.is_available() and self._config.with_cuda
        self.device = torch.device("cuda:0" if cuda_condition else "cpu")

        # This BERT model will be saved every epoch
        #self.bert = bert
        
        self.test_data = test_dataloader

        self.min_loss = np.inf
        self.save_path = save_path
        if save_path and not os.path.exists(save_path):
            os.mkdir(save_path)

    def save(self, file_path="output/bert_trained.model"):
        """
        Saving the current BERT model on file_path

        :param file_path: model output path which gonna be file_path+"ep%d" % epoch
        :return: final_output_path
        """
        self.bert.to("cpu")
        self.bert.save_pretrained(file_path)
        self.bert.to(self.device)
        print("Step:%d Model Saved on:" % self.step, file_path)

    def _setup_model(self):
        self.model = self.bert.to(self.device)
        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))
        if not self._config.eval:
            # Setting the Adam optimizer with hyper-param
            self.optim = AdamW(self.model.parameters(), 
                               lr=self._config.lr, betas=self._config.beta, eps=self._config.eps, weight_decay=self._config.weight_decay)

        # Distributed GPU training if CUDA can detect more than 1 GPU

        if self._config.with_cuda and torch.cuda.device_count() > 1:
            print("Using %d GPUS for BERT" % torch.cuda.device_count())
            self.model = nn.DataParallel(self.model)

    def train(self, steps=0, warm_up=0):
        return self._iteration(steps, self.train_data)

    def test(self, test_dataloader):
        # Implement test functions -> returning logit/weights and so on
        #return self.test_iteration(test_dataloader)
        pass

    def load(self, file_path):
        print("Restore the pretrained model")
        self.bert = BertForMaskedLM.from_pretrained(file_path, #BertForSequenceClassification.from_pretrained(args.pretrain, 
            num_labels=self.train_data.dataset.num_dmrs(), 
            output_attentions=True, 
            output_hidden_states=True, 
            hidden_dropout_prob=0.01, 
            vocab_size = len(self.train_data.dataset.vocab))
        # Initialize the BERT Language Model, with BERT model
        self._setup_model()

    def _create_new_model(self):
        print('Create a new bert model with a new configuration file')
        config = BertConfig(vocab_size = len(self.train_data.dataset.vocab),
                        hidden_size = self.config.hidden, 
                        num_hidden_layers = self.config.layers,
                        num_attention_heads = self.config.attn_heads,
                        hidden_act = "gelu", 
                        hidden_dropout_prob=0.1,
                        attention_probs_dropout_prob=0.1)
    
        self.bert = BertForMaskedLM(config)



    def create_model(self, config_file=None):

        if config_file:
            print("Create a bert model from the configuration file %s"%(config_file))
            config = BertConfig.from_pretrained(config_file, num_labels=self.train_data.dataset.num_dmrs(), 
                output_attentions=True, 
                output_hidden_states=True, 
                hidden_dropout_prob=0.01, 
                vocab_size = len(self.train_data.dataset.vocab))
        
            self.bert = BertForSequenceClassification(config=config)

        else:
            self._create_new_model()
    
        self._setup_model()

    def _acc(self, pred, label):
        """
        Calculate accruacy between the predicted tokens and the ground-truth

        :param pred: predicted tokens
        :param label: ground-truth tokens
        """

        # Calculate accuracy only on the masked tokens
        if type(pred).__module__ != np.__name__:
            pred = pred.numpy()
        if type(label).__module__ != np.__name__:
            label = label.numpy()

        return accuracy_score(y_true=label, y_pred=pred)#auc(fpr, tpr) #np.sum(gt==pred)/len(gt)




class MethylBertPretrainTrainer(MethylBertTrainer):
    """
    BERTTrainer make the pretrained BERT model with two LM training method.

        1. Masked Language Model : 3.3.1 Task #1: Masked LM
        2. Next Sentence prediction : 3.3.2 Task #2: Next Sentence Prediction

    please check the details on README.md with simple example.

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        pass

    def _eval_iteration(self, data_loader):
        """
        loop over the data_loader for evaluation 

        :param data_loader: torch.utils.data.DataLoader for test
        :return: DataFrame, 
        """
        
        # Setting the tqdm progress bar
        
        predict_res = {"prediction": [], "input": [], "label": [], "mask": []}

        mean_loss = 0 
        self.model.eval()
        
        for i, batch in enumerate(data_loader):
            
            data = {key: value.to(self.device) for key, value in batch.items()}

            with torch.no_grad():
                if self._config.amp:
                    with autocast():
                        mask_lm_output = self.model.forward(input_ids = data["input"],
                                                        masked_lm_labels = data["label"])
                else:
                    mask_lm_output = self.model.forward(input_ids = data["input"],
                                                        masked_lm_labels = data["label"])
                    
                mean_loss += mask_lm_output[0].mean().item()/len(data_loader)
                predict_res["prediction"].append(np.argmax(mask_lm_output[1].cpu().detach(), axis=-1))
                predict_res["input"].append(data["input"].cpu().detach())
                predict_res["label"].append(data["label"].cpu().detach())
                predict_res["mask"].append(data["mask"].cpu().detach())
                    
            if self._config.eval:
                print("Batch %d/%d is done...."%(i, len(data_loader)))
            
            del mask_lm_output
            del data

        # Integrate all results 
        predict_res["prediction"] = np.concatenate(predict_res["prediction"], axis=0)
        predict_res["input"] = np.concatenate(predict_res["input"], axis=0)
        predict_res["label"] = np.concatenate(predict_res["label"],  axis=0)
        predict_res["mask"] = np.concatenate(predict_res["mask"],  axis=0)

        self.model.train()
        return predict_res, np.mean(mean_loss)


    def _iteration(self, steps, data_loader):
        """
        loop over the data_loader for training or testing
        if on train status, backward operation is activated
        and also auto save the model every peoch

        :param steps: total steps to train
        :param data_loader: torch.utils.data.DataLoader for training
        :param warm_up: number of steps for warming up the learning rate
        :return: None
        """
        predict_res = {"prediction": [], "input": [], "label": []}
        self.step = 0

        if os.path.exists(self.save_path.split(".")[0]+"_train.csv"):
            os.remove(self.save_path.split(".")[0]+"_train.csv")

        with open(self.save_path.split(".")[0]+"_train.csv", "a") as f_perform:
            f_perform.write("step\tloss\tacc\tlr\n")

        if os.path.exists(self.save_path.split(".")[0]+"_eval.csv"):
            os.remove(self.save_path.split(".")[0]+"_eval.csv")

        with open(self.save_path.split(".")[0]+"_eval.csv", "a") as f_perform:
            f_perform.write("step\ttest_acc\ttest_loss\n")


        # Set up a learning rate scheduler
        self.scheduler = get_linear_schedule_with_warmup(self.optim,  
                                                         num_warmup_steps=self._config.warmup_step, 
                                                         num_training_steps=steps,
                                                         decrease_steps=self._config.decrease_steps)

        # Set up configuration for train iteration 
        global_step_loss = 0
        local_step = 0

        epochs = steps // (len(data_loader) // self._config.gradient_accumulation_steps) + 1
        
        self.model.zero_grad()
        train_prediction_res = {"prediction":[], "label":[]}
        
        scaler = GradScaler() if self._config.amp else None

        duration = 0
        for epoch in range(epochs):
            for i, batch in enumerate(data_loader):
                # 0. batch_data will be sent into the device(GPU or cpu)
                data = {key: value.to(self.device) for key, value in batch.items()}

                start = time.time()

                if self._config.amp:
                    with autocast():
                        mask_lm_output = self.model.forward(input_ids = data["bert_input"],
                                                masked_lm_labels = data["bert_label"])
                else:
                    mask_lm_output = self.model.forward(input_ids = data["bert_input"],
                                                masked_lm_labels = data["bert_label"])

                loss = mask_lm_output[0]

                # Concatenate predicted sequences for the evaluation 
                train_prediction_res["prediction"].append(np.argmax(mask_lm_output[1].cpu().detach(), axis=-1))
                train_prediction_res["label"].append(data["bert_label"].cpu().detach())

                # Calculate loss and back-propagation
                if "cuda" in self.device.type:
                    loss = loss.mean()
                loss = loss/self._config.gradient_accumulation_steps
                scaler.scale(loss).backward() if self._config.amp else loss.backward()

                global_step_loss += loss.item()
                duration += time.time() - start

                # Gradient accumulation 
                if (local_step+1) % self._config.gradient_accumulation_steps == 0:

                    if self._config.amp:
                        scaler.unscale_(self.optim)
                        nn.utils.clip_grad_norm_(self.model.parameters(), self._config.max_grad_norm)
                        scaler.step(self.optim)
                        scaler.update()
                    else: 
                        nn.utils.clip_grad_norm_(self.model.parameters(), self._config.max_grad_norm)
                        self.optim.step()
                        
                    self.scheduler.step()
                    self.model.zero_grad()
                    
                    # Evaluation with both train and testdata
                    if self.test_data is not None and self.step % self._config.eval_freq == 0 and self.step > 0:
                        
                        test_pred, test_loss = self._eval_iteration(self.test_data)
                        test_pred_acc = self._acc(test_pred["prediction"], test_pred["label"])

                        with open(self.save_path.split(".")[0]+"_eval.csv", "a") as f_perform:
                            f_perform.write("\t".join([str(self.step), str(test_pred_acc), str(test_loss)]) +"\n")

                        del test_pred

                    if self.step % self._config.log_freq == 0:
                        print("\nTrain Step %d iter - loss : %f / lr : %f"%(self.step, global_step_loss, self.optim.param_groups[0]["lr"]))
                        print(f"Running time for iter = {duration}")
                    
                    if self.min_loss > global_step_loss:
                        print("Step %d loss (%f) is lower than the current min loss (%f). Save the model at %s"%(self.step, global_step_loss, self.min_loss, self.save_path))
                        self.save(self.save_path)
                        self.min_loss = global_step_loss

                    # Save the step info (step, loss, lr, acc)
                    with open(self.save_path.split(".")[0]+"_train.csv", "a") as f_perform:

                        train_prediction_res["prediction"] = np.concatenate(train_prediction_res["prediction"], axis=0)
                        train_prediction_res["label"] = np.concatenate(train_prediction_res["label"],  axis=0)
                        train_pred_acc = self._acc(train_prediction_res["prediction"], train_prediction_res["label"])
                        
                        f_perform.write("\t".join([str(self.step), str(global_step_loss), str(train_pred_acc), str(self.optim.param_groups[0]["lr"])])+"\n")

                    self.step += 1

                    duration=0
                    global_step_loss = 0
                    del train_prediction_res
                    train_prediction_res = {"prediction":[], "label":[]}
     
                if steps == self.step:
                    break
                local_step+=1

            if steps == self.step:
                break



class MethylBertFinetuneTrainer(MethylBertTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def summary(self):

        print(self.model)
        print(self.classification_model)

    def create_model(self, config_file=None):
        config = BertConfig.from_pretrained(config_file, num_labels=self.train_data.dataset.num_dmrs(), 
            output_attentions=True, 
            output_hidden_states=True, 
            hidden_dropout_prob=0.01, 
            vocab_size = len(self.train_data.dataset.vocab))

        self.bert = MethylBertForSequenceClassification(config=config, 
                                                        nclasses=self.train_data.dataset.ctype_label_count)
        #self._creaete_classification_model()
        # Initialize the BERT Language Model, with BERT model
        self._setup_model()

    def _creaete_classification_model(self):
        self.classification_model = CNNClassification(seq_len=768*(self.train_data.dataset.seq_len+1)+self.train_data.dataset.seq_len,
            label_count=self.train_data.dataset.ctype_label_count, 
                                                      n_hidden=100).to(self.device)

    def _eval_iteration(self, data_loader, dmr_loss=True):
        """
        loop over the data_loader for testing 

        :param data_loader: torch.utils.data.DataLoader for test
        :return: DataFrame, 
        """
        
        predict_res = {"pred_dmr_label":[], "dmr_label":[], "pred_ctype_label":[], "ctype_label":[]}
        mean_loss = 0 
        self.model.eval()
        
        for i, batch in enumerate(data_loader):
            
            # 0. batch_data will be sent into the device(GPU or cpu)
            data = {key: value.to(self.device) for key, value in batch.items()}
            with torch.no_grad():

                if self._config.amp:
                    with autocast():
                        mask_lm_output = self.model.forward(step=self.step,
                                                input_ids = data["dna_seq"],
                                                labels = data["dmr_label"],
                                                methyl_seqs=data["methyl_seq"],
                                                ctype_label=data["ctype_label"]) if dmr_loss else self.model.forward(input_ids = data["dna_seq"])
                else:
                    mask_lm_output = self.model.forward(step=self.step,
                                                input_ids = data["dna_seq"],
                                                labels = data["dmr_label"],
                                                methyl_seqs=data["methyl_seq"],
                                                ctype_label=data["ctype_label"]) if dmr_loss else self.model.forward(input_ids = data["dna_seq"])

                if dmr_loss:
                    #mean_loss += mask_lm_output[0].mean().item()/len(data_loader) 
                    mean_loss += mask_lm_output["loss"].mean().item()/len(data_loader) 
                #bert_features = mask_lm_output[2][-1]
                #bert_features= torch.flatten(mask_lm_output[2][-1], 1) if dmr_loss else torch.flatten(mask_lm_output[1][-1], 1) #mask_lm_output[1]

                # Concatenate predicted sequences for the evaluation 
                predict_res["pred_dmr_label"].append(np.argmax(mask_lm_output["dmr_logits"].cpu().detach(), axis=-1) if dmr_loss else "NA")
                predict_res["dmr_label"].append(data["dmr_label"].cpu().detach())
                
                # Cell-type classification 

                '''
                rclass_output = self.classification_model.forward(bert_features=bert_features, 
                                                   methyl_seqs = data["methyl_seq"], 
                                                   labels = data["ctype_label"])
                
                mean_loss += rclass_output[0].mean().item()/len(data_loader)
                '''
                predict_res["pred_ctype_label"].append(np.argmax(mask_lm_output["classification_logits"].cpu().detach(), axis=-1))
                predict_res["ctype_label"].append(data["ctype_label"].cpu().detach())

            
            del mask_lm_output
            del data

        predict_res["dmr_label"] = np.concatenate(predict_res["dmr_label"],  axis=0)
        predict_res["ctype_label"] = np.concatenate(predict_res["ctype_label"],  axis=0)
        predict_res["pred_dmr_label"] = np.concatenate(predict_res["pred_dmr_label"], axis=0) if dmr_loss else np.zeros(predict_res["dmr_label"].shape)
        predict_res["pred_ctype_label"] = np.concatenate(predict_res["pred_ctype_label"], axis=0) 

        self.model.train()
        return predict_res, np.mean(mean_loss)

    def _iteration(self, steps, data_loader, dmr_loss = True):
        """
        loop over the data_loader for training or testing
        if on train status, backward operation is activated
        and also auto save the model every peoch

        :param steps: total steps to train
        :param data_loader: torch.utils.data.DataLoader for training
        :param warm_up: number of steps for warming up the learning rate
        :return: None
        """
        
        self.step = 0

        if os.path.exists(self.save_path.split(".")[0]+"_train.csv"):
            os.remove(self.save_path.split(".")[0]+"_train.csv")

        with open(self.save_path.split(".")[0]+"_train.csv", "a") as f_perform:
            f_perform.write("step\tloss\tdmr_acc\tctype_acc\tlr\n")

        if os.path.exists(self.save_path.split(".")[0]+"_eval.csv"):
            os.remove(self.save_path.split(".")[0]+"_eval.csv")

        with open(self.save_path.split(".")[0]+"_eval.csv", "a") as f_perform:
            f_perform.write("step\tloss\tdmr_acc\tctype_acc\n")


        # Set up a learning rate scheduler
        self.scheduler = get_linear_schedule_with_warmup(self.optim,  
                                                         num_warmup_steps=self._config.warmup_step, 
                                                         num_training_steps=steps,
                                                         decrease_steps=self._config.decrease_steps)
        global_step_loss = 0
        local_step = 0

        epochs = steps // (len(data_loader) // self._config.gradient_accumulation_steps) + 1
        
        self.model.zero_grad()
        #self.classification_model.zero_grad()
        train_prediction_res = {"pred_dmr_label":[], "dmr_label":[], "pred_ctype_label":[], "ctype_label":[]}
        
        if self._config.amp:
            scaler = GradScaler()

        duration = 0
        for epoch in range(epochs):
            for i, batch in enumerate(data_loader):
                # 0. batch_data will be sent into the device(GPU or cpu)
                data = {key: value.to(self.device) for key, value in batch.items()}

                start = time.time()

                if self._config.amp:
                    with autocast():
                        mask_lm_output = self.model.forward(step=self.step,
                                                input_ids = data["dna_seq"],
                                                labels = data["dmr_label"],
                                                methyl_seqs=data["methyl_seq"],
                                                ctype_label=data["ctype_label"]) if dmr_loss else self.model.forward(input_ids = data["dna_seq"])
                else:
                    mask_lm_output = self.model.forward(step=self.step,
                                                input_ids = data["dna_seq"],
                                                labels = data["dmr_label"],
                                                methyl_seqs=data["methyl_seq"],
                                                ctype_label=data["ctype_label"]) if dmr_loss else self.model.forward(input_ids = data["dna_seq"])

                #print("forward : ", GPUtil.showUtilization())
                loss = mask_lm_output["loss"]#mask_lm_output.loss 

                #bert_features=torch.flatten(mask_lm_output[2][-1], 1) if dmr_loss else torch.flatten(mask_lm_output[1][-1], 1)
                #bert_features = mask_lm_output[2][-1]
                # Concatenate predicted sequences for the evaluation 
                train_prediction_res["pred_dmr_label"].append(np.argmax(mask_lm_output["dmr_logits"].cpu().detach(), axis=-1) if dmr_loss else "NA")
                train_prediction_res["dmr_label"].append(data["dmr_label"].cpu().detach())


                # Cell-type classification 
                '''
                rclass_output = self.classification_model.forward(bert_features=bert_features, 
                                                   methyl_seqs = data["methyl_seq"], 
                                                   labels = data["ctype_label"])
                rclass_loss = rclass_output[0]
                '''
                train_prediction_res["pred_ctype_label"].append(np.argmax(mask_lm_output["classification_logits"].cpu().detach(), axis=-1))
                train_prediction_res["ctype_label"].append(data["ctype_label"].cpu().detach())


                # Calculate loss and back-propagation
                if dmr_loss:
                    loss = mask_lm_output["loss"].mean() if "cuda" in self.device.type else mask_lm_output["loss"] 
                    loss = loss/self._config.gradient_accumulation_steps
                    scaler.scale(loss).backward(retain_graph=True) if self._config.amp else loss.backward(retain_graph=True)

                '''
                if "cuda" in self.device.type:
                    rclass_loss = rclass_loss.mean()
                
                rclass_loss = rclass_loss/self._config.gradient_accumulation_steps
                scaler.scale(rclass_loss).backward(retain_graph=True) if self._config.amp else rclass_loss.backward(retain_graph=True)
                '''
                global_step_loss += loss.item()
                duration += time.time() - start

                # Gradient accumulation 
                if (local_step+1) % self._config.gradient_accumulation_steps == 0:
                    #if torch.cuda.device_count() > 1:
                    #    nn.utils.clip_grad_norm_(amp.master_params(self.optim), self.max_grad_norm)
                    #else:

                    if self._config.amp:
                        scaler.unscale_(self.optim)
                        nn.utils.clip_grad_norm_(self.model.parameters(), self._config.max_grad_norm)
                        scaler.step(self.optim)
                        scaler.update()
                    else: 
                        nn.utils.clip_grad_norm_(self.model.parameters(), self._config.max_grad_norm)
                        self.optim.step()
                        
                    self.scheduler.step()
                    self.model.zero_grad()
                    #self.classification_model.zero_grad()
                    
                    # Evaluation with both train and testdata
                                        
                    if self.step % self._config.eval_freq == 0 and self.step > 0:
                        
                        test_pred, test_loss = self._eval_iteration(self.test_data, dmr_loss=dmr_loss)
                        test_dmr_acc = self._acc(test_pred["pred_dmr_label"], test_pred["dmr_label"]) if dmr_loss else 0.0
                        test_ctype_acc = self._acc(test_pred["pred_ctype_label"], test_pred["ctype_label"])

                        with open(self.save_path.split(".")[0]+"_eval.csv", "a") as f_perform:
                            f_perform.write("\t".join([str(self.step), str(test_loss), str(test_dmr_acc), str(test_ctype_acc)]) +"\n")

                        del test_pred
                    
                    
                    if self.step % self._config.log_freq == 0:
                        print("\nTrain Step %d iter - loss : %f / lr : %f"%(self.step, global_step_loss, self.optim.param_groups[0]["lr"]))
                        print(f"Running time for iter = {duration}")
                    
                    if self.min_loss > global_step_loss:
                        print("Step %d loss (%f) is lower than the current min loss (%f). Save the model at %s"%(self.step, global_step_loss, self.min_loss, self.save_path))
                        self.save(self.save_path)
                        self.min_loss = global_step_loss

                    # Save the step info (step, loss, lr, acc)
                    with open(self.save_path.split(".")[0]+"_train.csv", "a") as f_perform:

                        
                        train_prediction_res["dmr_label"] = np.concatenate(train_prediction_res["dmr_label"],  axis=0)
                        train_prediction_res["pred_dmr_label"] = np.concatenate(train_prediction_res["pred_dmr_label"], axis=0) if dmr_loss else np.zeros(train_prediction_res["dmr_label"].shape)
                        train_dmr_acc = self._acc(train_prediction_res["pred_dmr_label"], train_prediction_res["dmr_label"]) if dmr_loss else 0.0

                        train_prediction_res["pred_ctype_label"] = np.concatenate(train_prediction_res["pred_ctype_label"], axis=0)
                        train_prediction_res["ctype_label"] = np.concatenate(train_prediction_res["ctype_label"],  axis=0)
                        train_ctype_acc = self._acc(train_prediction_res["pred_ctype_label"], train_prediction_res["ctype_label"])

                        f_perform.write("\t".join([str(self.step), str(global_step_loss), str(train_dmr_acc), str(train_ctype_acc),  str(self.optim.param_groups[0]["lr"])])+"\n")

                    self.step += 1
                    duration=0
                    global_step_loss = 0
                    del train_prediction_res
                    train_prediction_res =  {"pred_dmr_label":[], "dmr_label":[], "pred_ctype_label":[], "ctype_label":[]}
             
                if steps == self.step:
                    break
                local_step+=1

            if steps == self.step:
                break

    def save(self, file_path="output/bert_trained.model"):
        #super().save(file_path)
        self.bert.to("cpu")
        self.bert.save_pretrained(file_path)
        torch.save(self.bert.read_classifier.state_dict(), os.path.dirname(file_path)+"/read_classification_cnn.pickle")

        self.bert.to(self.device)
        print("Step:%d Model Saved on:" % self.step, file_path)
        
    def load(self, file_path: str, n_dmrs=None):
        print("Restore the pretrained model")
        self.bert = MethylBertForSequenceClassification.from_pretrained(file_path, #BertForSequenceClassification.from_pretrained(args.pretrain, 
            num_labels=self.train_data.dataset.num_dmrs() if not n_dmrs else n_dmrs, 
            output_attentions=True, 
            output_hidden_states=True, 
            hidden_dropout_prob=0.01, 
            vocab_size = len(self.train_data.dataset.vocab))

        '''
        self._creaete_classification_model()
        '''
        if os.path.exists(os.path.dirname(file_path)+"/read_classification_cnn.pickle"):
            print("Restore read classification CNN model from %s"%(os.path.dirname(file_path)+"/read_classification_cnn.pickle"))
            self.bert.from_pretrained_read_classifier(os.path.dirname(file_path)+"/read_classification_cnn.pickle", self.device)
        self.bert.set_nclass(self.train_data.dataset.ctype_label_count)
        self._setup_model()