from methylbert.data import finetune_data_generate as fdg
from methylbert.data.dataset import MethylBertFinetuneDataset
from methylbert.data.vocab import MethylVocab
from methylbert.trainer import MethylBertFinetuneTrainer

from torch.utils.data import DataLoader
import pandas as pd
import os, shutil, json

def load_data(train_dataset: str, test_dataset: str, batch_size: int = 64, num_workers: int = 40):
	tokenizer=MethylVocab(k=3)

	# Load data sets
	train_dataset = MethylBertFinetuneDataset(train_dataset, tokenizer, seq_len=150)
	test_dataset = MethylBertFinetuneDataset(test_dataset, tokenizer, seq_len=150) 

	if len(test_dataset) > 500:
		test_dataset.subset_data(500)

	# Create a data loader
	print("Creating Dataloader")
	local_step_batch_size = int(batch_size/4)
	print("Local step batch size : ", local_step_batch_size)
	
	train_data_loader = DataLoader(train_dataset, batch_size=local_step_batch_size, 
								   num_workers=num_workers, pin_memory=False, shuffle=True)

	test_data_loader = DataLoader(test_dataset, batch_size=local_step_batch_size, 
								  num_workers=num_workers, pin_memory=True, 
								  shuffle=False) if test_dataset is not None else None

	return tokenizer, train_data_loader, test_data_loader

def test_finetune_no_pretrain(tokenizer : MethylVocab, 
				  save_path : str, 
				  train_data_loader : DataLoader, 
				  test_data_loader : DataLoader, 
				  pretrain_model : str, 
				  steps : int=10):

	trainer = MethylBertFinetuneTrainer(vocab_size = len(tokenizer), 
						  save_path=save_path, 
						  train_dataloader=train_data_loader, 
						  test_dataloader=test_data_loader,
						  with_cuda=False)

	trainer.create_model(config_file=os.path.join(pretrain_model, "config.json"))

	trainer.train(steps)

	assert os.path.exists(os.path.join(save_path, "config.json"))
	assert os.path.exists(os.path.join(save_path, "dmr_encoder.pickle"))
	assert os.path.exists(os.path.join(save_path, "pytorch_model.bin"))
	assert os.path.exists(os.path.join(save_path, "read_classification_model.pickle"))
	assert os.path.exists(os.path.join(save_path, "eval.csv"))
	assert os.path.exists(os.path.join(save_path, "train.csv"))
	assert steps == pd.read_csv(os.path.join(save_path, "train.csv")).shape[0]

def test_finetune_no_pretrain_focal(tokenizer : MethylVocab, 
				  save_path : str, 
				  train_data_loader : DataLoader, 
				  test_data_loader : DataLoader, 
				  pretrain_model : str, 
				  steps : int=10):

	trainer = MethylBertFinetuneTrainer(vocab_size = len(tokenizer), 
						  save_path=save_path, 
						  train_dataloader=train_data_loader, 
						  test_dataloader=test_data_loader,
						  with_cuda=False,
						  loss="focal_bce")

	trainer.create_model(config_file=os.path.join(pretrain_model, "config.json"))

	trainer.train(steps)
	assert os.path.exists(os.path.exists(os.path.join(save_path, "config.json")))
	
	with open(os.path.join(save_path, "config.json")) as fp:
		config = json.load(fp)
		assert config["loss"] == "focal_bce"

def test_finetune_no_pretrain_focal(tokenizer : MethylVocab, 
				  save_path : str, 
				  train_data_loader : DataLoader, 
				  test_data_loader : DataLoader, 
				  pretrain_model : str, 
				  steps : int=10):

	trainer = MethylBertFinetuneTrainer(vocab_size = len(tokenizer), 
						  save_path=save_path, 
						  train_dataloader=train_data_loader, 
						  test_dataloader=test_data_loader,
						  with_cuda=False,
						  loss="focal_bce")

	trainer.create_model(config_file=os.path.join(pretrain_model, "config.json"))

	trainer.train(steps)
	assert os.path.exists(os.path.exists(os.path.join(save_path, "config.json")))
	
	with open(os.path.join(save_path, "config.json")) as fp:
		config = json.load(fp)
		assert config["loss"] == "focal_bce"


def test_finetune_focal_multicelltype(tokenizer : MethylVocab, 
									  save_path : str, 
									  train_data_loader : DataLoader, 
									  test_data_loader : DataLoader, 
									  pretrain_model : str, 
									  steps : int=10):

	trainer = MethylBertFinetuneTrainer(vocab_size = len(tokenizer), 
						  save_path=save_path+"bert.model/", 
						  train_dataloader=train_data_loader, 
						  test_dataloader=test_data_loader,
						  with_cuda=False,
						  loss="focal_bce")
	trainer.load(pretrain_model)
	trainer.train(steps)

	assert os.path.exists(os.path.join(save_path, "bert.model/config.json"))
	assert os.path.exists(os.path.join(save_path, "bert.model/dmr_encoder.pickle"))
	assert os.path.exists(os.path.join(save_path, "bert.model/pytorch_model.bin"))
	assert os.path.exists(os.path.join(save_path, "bert.model/read_classification_model.pickle"))

def test_finetune(tokenizer : MethylVocab, 
				  save_path : str, 
				  train_data_loader : DataLoader, 
				  test_data_loader : DataLoader, 
				  pretrain_model : str, 
				  steps : int=10):

	trainer = MethylBertFinetuneTrainer(vocab_size = len(tokenizer), 
						  save_path=save_path+"bert.model/", 
						  train_dataloader=train_data_loader, 
						  test_dataloader=test_data_loader,
						  with_cuda=False)
	trainer.load(pretrain_model)
	trainer.train(steps)

	assert os.path.exists(os.path.join(save_path, "bert.model/config.json"))
	assert os.path.exists(os.path.join(save_path, "bert.model/dmr_encoder.pickle"))
	assert os.path.exists(os.path.join(save_path, "bert.model/pytorch_model.bin"))
	assert os.path.exists(os.path.join(save_path, "bert.model/read_classification_model.pickle"))

def reset_dir(dirname):
	if os.path.exists(dirname):
		shutil.rmtree(dirname)
	os.mkdir(dirname)

if __name__=="__main__":
	# For data processing
	f_bam_list = "data/bam_list.txt"
	f_dmr = "data/dmrs.csv"
	f_ref = "data/genome.fa"
	out_dir = "data/processed/"
	
	# Process data for fine-tuning
	fdg.finetune_data_generate(
		sc_dataset = f_bam_list,
		f_dmr = f_dmr,
		f_ref = f_ref,
		output_dir=out_dir,
		split_ratio = 0.7,
		n_cores=10,
		n_dmrs=10
	)

	tokenizer, train_data_loader, test_data_loader = \
	load_data(train_dataset = os.path.join(out_dir, "train_seq.csv"),
			  test_dataset = os.path.join(out_dir, "test_seq.csv"))

	# For fine-tuning
	model_dir="data/pretrained_model/"
	save_path="res/"
	train_step=3

	# Test
	# amp warning issue
	# https://github.com/pytorch/pytorch/issues/67598

	reset_dir(save_path)
	test_finetune(tokenizer, save_path, train_data_loader, test_data_loader, model_dir, train_step)
	
	reset_dir(save_path)
	test_finetune_no_pretrain(tokenizer, save_path, train_data_loader, test_data_loader, model_dir, train_step) 
	
	reset_dir(save_path)
	test_finetune_no_pretrain_focal(tokenizer, save_path, train_data_loader, test_data_loader, model_dir, train_step) 

	reset_dir(save_path)
	test_finetune_savefreq(tokenizer, save_path, train_data_loader, test_data_loader, model_dir, train_step, save_freq=1)
	
	# Multiple cell type
	out_dir="data/multi_cell_type/"
	tokenizer, train_data_loader, test_data_loader = \
	load_data(train_dataset = os.path.join(out_dir, "train_seq.csv"),
			  test_dataset = os.path.join(out_dir, "test_seq.csv"))
	
	# For fine-tuning
	model_dir="data/pretrained_model/"
	save_path="data/multi_cell_type/res/"
	
	reset_dir(save_path)
	test_finetune_focal_multicelltype(tokenizer, save_path, train_data_loader, test_data_loader, model_dir) 

	print("Everything passed!")
