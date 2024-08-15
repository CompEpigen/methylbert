import torch
import torch.nn as nn
from torch.optim import Adam, AdamW
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from torch.cuda.amp import GradScaler

from sklearn.metrics import roc_curve, auc, accuracy_score

from methylbert.network import MethylBertEmbeddedDMR
from methylbert.data.vocab import MethylVocab
from methylbert.utils import get_dna_seq
from methylbert.config import get_config, MethylBERTConfig
from transformers import BertForSequenceClassification, BertConfig, BertForMaskedLM

from tqdm import tqdm
import numpy as np
import os, warnings, time
import pandas as pd


def learning_rate_scheduler(optimizer, num_warmup_steps: int, num_training_steps: int, decrease_steps: int):
    """ 
    Modified version of get_linear_schedule_with_warmup from transformers
    Learning rate scheduler including warm-up, retaining and decrease 

    optimizer: torch.optim.Optimizer
        Optimizer 
    num_warmup_steps: int
        Initial steps for linear warm-up
    num_training_steps: int
        Total training steps
    decrease_steps:
        Steps when the learning rate decrease starts
    """

    def lr_lambda(current_step):
        if current_step <= num_warmup_steps: # warm-up
            return float(current_step) / float(max(1, num_warmup_steps))
        elif current_step >= decrease_steps: # decrease 
            return max(
                0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - decrease_steps))
            )
        return 1 # Otherwise, keep the current learning rate 

    return LambdaLR(optimizer, lr_lambda, last_epoch = -1)



class MethylBertTrainer(object):
    def __init__(self, vocab_size: int, save_path: str = "", 
                 train_dataloader: DataLoader = None, test_dataloader: DataLoader = None,
                 **kwargs):

        # Setup config
        self._config = get_config(**kwargs)
        self.train_data = train_dataloader
            
        # Setup cuda device for BERT training, argument -c, --cuda should be true
        self._config.amp = torch.cuda.is_available() and self._config.with_cuda
        if self._config.with_cuda and torch.cuda.device_count() < 1:
            print("No detected GPU device. Load the model on CPU")
            self._config.with_cuda = False
        print("The model is loaded on %s"%("GPU" if self._config.with_cuda else "CPU"))
        self.device = torch.device("cuda:0" if self._config.with_cuda else "cpu")

        self.test_data = test_dataloader

        # To save the best model
        self.min_loss = np.inf
        self.save_path = save_path
        if save_path and not os.path.exists(save_path):
            os.mkdir(save_path)
        self.f_train = os.path.join(self.save_path, "train.csv")
        self.f_eval = os.path.join(self.save_path, "eval.csv")


    def save(self, file_path="output/bert_trained.model"):
        '''
        Saving the current BERT model on file_path

        file_path: str
            model output path which gonna be file_path+"ep%d" % epoch
        '''
        self.bert.to("cpu")
        self.bert.save_pretrained(file_path)
        self.bert.to(self.device)
        print("Step:%d Model Saved on:" % self.step, file_path)

    def _setup_model(self):
        '''        
        Load the model to the designated device (CPU or GPU) and create an optimiser

        '''
        self.model = self.bert.to(self.device)
        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))
        
        # Distributed GPU training if CUDA can detect more than 1 GPU
        if self._config.with_cuda and torch.cuda.device_count() > 1:
            print("Using %d GPUs for BERT" % torch.cuda.device_count())
            self.model = nn.DataParallel(self.model)

        if not self._config.eval:
            # Setting the AdamW optimizer with hyper-param
            self.optim = AdamW(self.model.parameters(), 
                               lr=self._config.lr, betas=self._config.beta, eps=self._config.eps, weight_decay=self._config.weight_decay)


    def train(self, steps: int = 0):
        '''
        Train MethylBERT over given steps

        steps: int
            number of steps to train the model
        '''
        return self._iteration(steps, self.train_data)

    def test(self, test_dataloader: DataLoader):
        '''
        Test/Evaluation of MethylBERT model with given data

        test_dataloader: DataLoader
            Data loader for test data
        '''
        pass

    def load(self, file_path: str):
        '''        
        Restore the BERT model store in the given path
        '''
        print(f"Restore the pretrained model from {file_path}")
        self.bert = BertForMaskedLM.from_pretrained(file_path, 
            num_labels=self.train_data.dataset.num_dmrs(), 
            output_attentions=True, 
            output_hidden_states=True, 
            hidden_dropout_prob=0.01, 
            vocab_size = len(self.train_data.dataset.vocab))

        # Initialize the model
        self._setup_model()

    def create_model(self, config_file=None):
        """
        Create a new BERT MLM model from the configuration
        :param config_file: path to the configuration file
        """
        pass

    def _acc(self, pred, label):
        """
        Calculate accruacy between the predicted and the ground-truth values

        :param pred: predicted values
        :param label: ground-truth values
        """

        if type(pred).__module__ != np.__name__:
            pred = pred.numpy()
        if type(label).__module__ != np.__name__:
            label = label.numpy()

        if len(pred.shape) > 1:
            pred = pred.flatten()
        if len(label.shape) > 1:
            label = label.flatten()

        return accuracy_score(y_true=label, y_pred=pred)


class MethylBertPretrainTrainer(MethylBertTrainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        pass

    def create_model(self, *args, **kwargs):
        config = BertConfig(vocab_size = len(self.train_data.dataset.vocab), *args, **kwargs)
        self.bert = BertForMaskedLM(config)
        self._setup_model()

    def _eval_iteration(self, data_loader):
        """
        loop over the data_loader for evaluation 

        :param data_loader: torch.utils.data.DataLoader for test
        :return: DataFrame, 
        """
        
        predict_res = {"prediction": [], "input": [], "label": [], "mask": []}

        mean_loss = 0 
        self.model.eval()
        
        for i, batch in enumerate(data_loader):
            
            data = {key: value.to(self.device) for key, value in batch.items()}

            with torch.no_grad():
                with torch.autocast(device_type="cuda" if self._config.with_cuda else "cpu", 
                                    enabled=self._config.amp):
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

        if os.path.exists(self.f_train):
            os.remove(self.f_train)

        with open(self.f_train, "a") as f_perform:
            f_perform.write("step\tloss\tacc\tlr\n")

        if os.path.exists(self.f_eval):
            os.remove(self.f_eval)

        with open(self.f_eval, "a") as f_perform:
            f_perform.write("step\ttest_acc\ttest_loss\n")


        # Set up a learning rate scheduler
        self.scheduler = learning_rate_scheduler(self.optim,  
                                                 num_warmup_steps=self._config.warmup_step, 
                                                 num_training_steps=steps,
                                                 decrease_steps=self._config.decrease_steps)

        # Set up configuration for train iteration 
        global_step_loss = 0
        local_step = 0

        epochs = steps // (len(data_loader) // self._config.gradient_accumulation_steps) + 1
        
        self.model.zero_grad()
        self.model.train()
        train_prediction_res = {"prediction":[], "label":[]}
        
        scaler = GradScaler() if self._config.amp else None

        duration = 0
        for epoch in range(epochs):
            for i, batch in enumerate(data_loader):
                # 0. batch_data will be sent into the device(GPU or cpu)
                data = {key: value.to(self.device) for key, value in batch.items()}

                start = time.time()

                with torch.autocast(device_type="cuda" if self._config.with_cuda else "cpu", 
                                    enabled=self._config.amp):
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
                        idces = np.where(test_pred["label"]>=0)
                        test_pred_acc = self._acc(test_pred["prediction"][idces[0], idces[1]], 
                                                  test_pred["label"][idces[0], idces[1]])

                        with open(self.f_eval, "a") as f_perform:
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
                    with open(self.f_train, "a") as f_perform:

                        train_prediction_res["prediction"] = np.concatenate(train_prediction_res["prediction"], axis=0)
                        train_prediction_res["label"] = np.concatenate(train_prediction_res["label"],  axis=0)

                        idces = np.where(train_prediction_res["label"]>=0)
                        train_pred_acc = self._acc(train_prediction_res["prediction"][idces[0], idces[1]], 
                            train_prediction_res["label"][idces[0], idces[1]])
                        
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
        '''
        Print the summary of the MethylBERT model
        '''

        print(self.model)
        print(self.classification_model)

    def create_model(self, config_file: str = None):
        '''
        Create a new MethylBERT model from the configuration
        '''

        config = MethylBERTConfig.from_pretrained(config_file, 
            num_labels=self.train_data.dataset.num_dmrs(), 
            output_attentions=True, 
            output_hidden_states=True, 
            hidden_dropout_prob=0.01, 
            vocab_size = len(self.train_data.dataset.vocab),
            loss=self._config.loss)
        
        self.bert = MethylBertEmbeddedDMR(config=config, 
                                          seq_len=self.train_data.dataset.seq_len)
        
        # Initialize the BERT Language Model, with BERT model
        self._setup_model()

    def _eval_iteration(self, data_loader: DataLoader, return_logits: bool = False):
        """
        loop over the data_loader for eval/test 

        :param data_loader: torch.utils.data.DataLoader for test
        :return: DataFrame, 
        """
        
        predict_res = {"dmr_label":[], "pred_ctype_label":[], "ctype_label":[]}
        logits = list()

        mean_loss = 0 
        self.model.eval()
        
        for i, batch in enumerate(data_loader):
            
            # 0. batch_data will be sent into the device(GPU or cpu)
            data = {key: value.to(self.device) for key, value in batch.items() if type(value) != list}
            with torch.no_grad():
                with torch.autocast(device_type="cuda" if self._config.with_cuda else "cpu", 
                                    enabled=self._config.amp):
                    mask_lm_output = self.model.forward(step=self.step,
                                            input_ids = data["dna_seq"],
                                            token_type_ids=data["methyl_seq"],
                                            labels = data["dmr_label"],
                                            ctype_label=data["ctype_label"]) 

                loss = mask_lm_output["loss"].mean().item() if "cuda" in self.device.type else mask_lm_output["loss"].item()
                mean_loss += loss/len(data_loader)

                predict_res["dmr_label"].append(data["dmr_label"].cpu().detach())
                predict_res["pred_ctype_label"].append(np.argmax(mask_lm_output["classification_logits"].cpu().detach(), axis=-1))
                predict_res["ctype_label"].append(data["ctype_label"].cpu().detach())

                if return_logits:
                    logits.append(mask_lm_output["classification_logits"].cpu().detach().numpy())

            
            del mask_lm_output
            del data


        predict_res["dmr_label"] = np.concatenate(predict_res["dmr_label"],  axis=0)
        predict_res["ctype_label"] = np.concatenate(predict_res["ctype_label"],  axis=0)
        predict_res["pred_ctype_label"] = np.concatenate(predict_res["pred_ctype_label"], axis=0) 

        self.model.train()

        if not return_logits:
            return predict_res, mean_loss
        else:
            return predict_res, mean_loss, np.concatenate(return_logits, axis=0)

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
        
        self.step = 0
        
        if os.path.exists(self.f_train):
            os.remove(self.f_train)

        with open(self.f_train, "w") as f_perform:
            f_perform.write("step\tloss\tctype_acc\tlr\n")

        if os.path.exists(self.f_eval):
            os.remove(self.f_eval)

        with open(self.f_eval, "w") as f_perform:
            f_perform.write("step\tloss\tctype_acc\n")


        # Set up a learning rate scheduler
        self.scheduler = learning_rate_scheduler(self.optim,  
                                                         num_warmup_steps=self._config.warmup_step, 
                                                         num_training_steps=steps,
                                                         decrease_steps=self._config.decrease_steps)
        global_step_loss = 0
        local_step = 0

        epochs = steps // (len(data_loader) // self._config.gradient_accumulation_steps) + 1
        
        self.model.zero_grad()
        #print(self.model.training)
        self.model.train()
        train_prediction_res = {"dmr_label":[], "pred_ctype_label":[], "ctype_label":[]}
        
        scaler = GradScaler() if self._config.amp else None

        duration = 0
        for epoch in range(epochs):
            for i, batch in enumerate(data_loader):
                # 0. batch_data will be sent into the device(GPU or cpu)
                data = {key: value.to(self.device) for key, value in batch.items() if type(value) != list}

                start = time.time()
                with torch.autocast(device_type="cuda" if self._config.with_cuda else "cpu", 
                                    enabled=self._config.amp):
                    mask_lm_output = self.model.forward(step=self.step,
                                            input_ids = data["dna_seq"],
                                            token_type_ids=data["methyl_seq"],
                                            labels = data["dmr_label"],
                                            ctype_label=data["ctype_label"]) 
                loss = mask_lm_output["loss"]

                # Concatenate predicted sequences for the evaluation 
                train_prediction_res["dmr_label"].append(data["dmr_label"].cpu().detach())


                # Cell-type classification 
                train_prediction_res["pred_ctype_label"].append(np.argmax(mask_lm_output["classification_logits"].cpu().detach(), axis=-1))
                train_prediction_res["ctype_label"].append(data["ctype_label"].cpu().detach())


                # Calculate loss and back-propagation
                loss = mask_lm_output["loss"].mean() if "cuda" in self.device.type else mask_lm_output["loss"] 
                loss = loss/self._config.gradient_accumulation_steps
                scaler.scale(loss).backward(retain_graph=True) if self._config.amp else loss.backward(retain_graph=True)

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
                    
                    # Evaluation 
                         
                    eval_pred, eval_loss = self._eval_iteration(self.test_data)
                    eval_acc = self._acc(eval_pred["pred_ctype_label"], eval_pred["ctype_label"])

                    with open(self.f_eval, "a") as f_perform:
                        f_perform.write("\t".join([str(self.step), str(eval_loss), str(eval_acc)]) +"\n")

                    del eval_pred
                    
                    
                    if self.step % self._config.log_freq == 0:
                        print("\nTrain Step %d iter - loss : %f / lr : %f"%(self.step, global_step_loss, self.optim.param_groups[0]["lr"]))
                        print(f"Running time for iter = {duration}")
                    
                    if self.min_loss > eval_loss:
                        print("Step %d loss (%f) is lower than the current min loss (%f). Save the model at %s"%(self.step, eval_loss, self.min_loss, self.save_path))
                        self.save(self.save_path)
                        self.min_loss = eval_loss
                    
                    # For saving an interim model to track the training
                    if ( type(self._config.save_freq) == int ) and (self.step % self._config.save_freq == 0):
                        step_save_dir=self.save_path.replace("bert.model", "bert.model_step%d"%(self.step))
                        print("Step %d: Save an interim model at %s"%(self.step, step_save_dir))
                        if not os.path.exists(step_save_dir):
                            os.mkdir(step_save_dir)
                        self.save(step_save_dir)
                                        
                    # Save the step info (step, loss, lr, acc)
                    with open(self.f_train, "a") as f_perform:

                        train_prediction_res["dmr_label"] = np.concatenate(train_prediction_res["dmr_label"],  axis=0)
                        train_prediction_res["pred_ctype_label"] = np.concatenate(train_prediction_res["pred_ctype_label"], axis=0)
                        train_prediction_res["ctype_label"] = np.concatenate(train_prediction_res["ctype_label"],  axis=0)
                        train_ctype_acc = self._acc(train_prediction_res["pred_ctype_label"], train_prediction_res["ctype_label"])

                        f_perform.write("\t".join([str(self.step), str(global_step_loss), str(train_ctype_acc),  str(self.optim.param_groups[0]["lr"])])+"\n")

                    self.step += 1
                    duration=0
                    global_step_loss = 0

                    # Reset prediction result
                    del train_prediction_res
                    train_prediction_res =  {"dmr_label":[], "pred_ctype_label":[], "ctype_label":[]}
             
                if steps == self.step:
                    break
                local_step+=1

            if steps == self.step:
                break

    def save(self, file_path: str="output/bert_trained.model"):
        '''
        Save the MethylBERT model in the given path
        '''
        self.bert.to("cpu")
        self.bert.save_pretrained(file_path)

        if hasattr(self.bert, "read_classifier"):
            torch.save(self.bert.read_classifier.state_dict(), os.path.dirname(file_path)+"/read_classification_model.pickle")

        if hasattr(self.bert, "dmr_encoder"):
            torch.save(self.bert.dmr_encoder.state_dict(), os.path.dirname(file_path)+"/dmr_encoder.pickle")

        self.bert.to(self.device)
        print("Step:%d Model Saved on:" % self.step, file_path)
        
    def load(self, dir_path: str, n_dmrs: int=None, load_fine_tune: bool=False):
        '''
        Load pre-trained / fine-tuned MethylBERT model 
        dir_path: str
            Directory to the saved bert model. It must contain "config.json" and "pytorch_model.bin" files
        n_dmrs: int (default: None)
            Number of DMRs to reconstruct the MethylBERT model. If the number is not given, the trainer auto-calculates the number from the same data
        load_fine_tune: bool (default: False)
            Whether the loaded model is a fine-tuned model including num_dmrs or a pre-trained model without num_dmrs
        '''
        print(f"Restore the pretrained model {dir_path}")

        if load_fine_tune:
            if n_dmrs is not None:
                raise ValueError("You cannot give a new number of DMRs for loading a fine-tuned model. The model should contains one. Please set either n_dmrs=None or load_fine_tune=False")
            self.bert = MethylBertEmbeddedDMR.from_pretrained(dir_path, 
                output_attentions=True, 
                output_hidden_states=True, 
                seq_len = self.train_data.dataset.seq_len,
                loss=self._config.loss
                )
            
            try:
                self.bert.from_pretrained_dmr_encoder(os.path.dirname(dir_path)+"/dmr_encoder.pickle", self.device)
                print("Restore DMR encoder from %s"%(os.path.dirname(dir_path)+"/dmr_encoder.pickle"))
            except FileNotFoundError:
                print(os.path.dirname(dir_path)+"/dmr_encoder.pickle is not found.")

            try:
                self.bert.from_pretrained_read_classifier(os.path.dirname(dir_path)+"/read_classification_model.pickle", self.device)
                print("Restore read classification FCN model from %s"%(os.path.dirname(dir_path)+"/read_classification_model.pickle"))
            except FileNotFoundError:
                print(os.path.dirname(dir_path)+"/read_classification_model.pickle is not found.")
        else:
            self.bert = MethylBertEmbeddedDMR.from_pretrained(dir_path, 
                num_labels=self.train_data.dataset.num_dmrs() if not n_dmrs else n_dmrs, 
                output_attentions=True, 
                output_hidden_states=True, 
                seq_len = self.train_data.dataset.seq_len,
                loss=self._config.loss
                )

        self._setup_model()

    def read_classification(self, data_loader: DataLoader = None, tokenizer: MethylVocab = None, logit: bool = False):
        '''
        Classify sequencing reads into cell types

        data_loader: torch.utils.data.DataLoader
            DataLoader containing reads to classify. If nothing is given, the trainer tries to assign 'test_data'
        output_dir: str
            Directory to save the result. If nothing is given, the results is saved in 'save_path'
        save_logit: bool (default: False)
            Whether save the calculated classification logits or not
        '''

        if data_loader is None:
            if self.test_data is None:
                ValueError("There is no test_data assigned to the trainer. Please give a DataLoader as an input.")
            else:
                data_loader = self.test_data

        # classification
        res = dict()
        logits = list()
        self.model.eval()

        pbar = tqdm(total=len(data_loader))
        for i, batch in enumerate(data_loader):
            
            # 0. batch_data will be sent into the device(GPU or cpu)
            data = dict()

            for k, v in batch.items():
                if type(v) != list:
                    data[k] = v.to(self.device)
                if k not in res.keys():
                    res[k] = v.numpy() if type(v) == torch.Tensor else v 
                else:
                    res[k] = np.concatenate([res[k], v.numpy() if type(v) == torch.Tensor else v], axis=0)

            with torch.no_grad():
                with torch.autocast(device_type="cuda" if self._config.with_cuda else "cpu", 
                                    enabled=self._config.amp):
                    mask_lm_output = self.model.forward(step=0,
                                            input_ids = data["dna_seq"],
                                            token_type_ids=data["methyl_seq"],
                                            labels = data["dmr_label"],
                                            ctype_label=data["ctype_label"]) 
                
                if "pred" in res.keys():
                    res["pred"] = np.concatenate([res["pred"], np.argmax(mask_lm_output["classification_logits"].cpu().detach(), axis=-1)], axis=0)
                else:
                    res["pred"] = np.argmax(mask_lm_output["classification_logits"].cpu().detach(), axis=-1)
                
                if logit:
                    logits.append(mask_lm_output["classification_logits"].cpu().detach().numpy())
            
            del mask_lm_output
            del data

            pbar.update(1)
        pbar.close()

        if logit:
            logits = np.concatenate(logits, axis=0)
        res["dna_seq"]=[get_dna_seq(s, tokenizer) for s in res["dna_seq"]]
        res["methyl_seq"]=["".join([str(mm) for mm in m]) for m in res["methyl_seq"]]

        res = pd.DataFrame(res)

        return res if not logit else res, logits
