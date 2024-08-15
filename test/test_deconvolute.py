from methylbert.data.vocab import MethylVocab
from methylbert.data.dataset import MethylBertFinetuneDataset
from torch.utils.data import DataLoader
from methylbert.deconvolute import deconvolute
from methylbert.trainer import MethylBertFinetuneTrainer

import pandas as pd
import os

def test_adjustment(trainer, tokenizer, data_loader, output_path, df_train):
	deconvolute(trainer = trainer,
	            tokenizer = tokenizer,                                    
	            data_loader = data_loader,
	            output_path = output_path,
	            df_train = df_train,
	            adjustment = True)

	assert pd.read_csv(os.path.join(output_path, "FI.csv"), sep="\t").shape[0] == len(df_train["dmr_label"].unique())

def test_multi_cell_type(trainer, tokenizer, data_loader, output_path, df_train):
	deconvolute(trainer = trainer,
	            tokenizer = tokenizer,                                    
	            data_loader = data_loader,
	            output_path = output_path,
	            df_train = df_train,
	            adjustment = False)

	assert pd.read_csv(os.path.join(output_path, "deconvolution.csv"), sep="\t").shape[0] == 3

if __name__=="__main__":
	f_bulk = "data/processed/test_seq.csv"
	f_train = "data/processed/train_seq.csv"
	model_dir = "res/bert.model/"
	out_dir = "res/deconvolution/"

	tokenizer = MethylVocab(k=3)
	
	dataset = MethylBertFinetuneDataset(f_bulk,
	                                    tokenizer, 
	                                    seq_len=150)
	data_loader = DataLoader(dataset, batch_size=50, num_workers=20)
	df_train = pd.read_csv(f_train, sep="\t")

	trainer = MethylBertFinetuneTrainer(len(tokenizer), 
	                                    train_dataloader=data_loader, 
	                                    test_dataloader=data_loader,
	                                    )
	trainer.load(model_dir)

	test_adjustment(trainer, tokenizer, data_loader, out_dir, df_train)
		# multiple cell type
	model_dir = "data/multi_cell_type/res/bert.model/"
	f_bulk = "data/multi_cell_type/test_seq.csv"
	f_train = "data/multi_cell_type/train_seq.csv"
	dataset = MethylBertFinetuneDataset(f_bulk,
	                                    tokenizer, 
	                                    seq_len=150)
	data_loader = DataLoader(dataset, batch_size=50, num_workers=20)
	df_train = pd.read_csv(f_train, sep="\t")

	trainer = MethylBertFinetuneTrainer(len(tokenizer), 
	                                    train_dataloader=data_loader, 
	                                    test_dataloader=data_loader,
	                                    )
	trainer.load(model_dir)
	test_multi_cell_type(trainer, tokenizer, data_loader, out_dir, df_train)

	print("Everything passed!")
