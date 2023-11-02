import argparse, sys, os 

from methylbert.data.finetune_data_generate import finetune_data_generate

from methylbert.trainer import MethylBertFinetuneTrainer
from methylbert.data.vocab import MethylVocab
from methylbert.data.dataset import MethylBertFinetuneDataset
from methylbert.utils import set_seed
from methylbert import __version__

import torch
from torch.utils.data import DataLoader

def deconvolute_arg_parser(subparsers):
	parser = subparsers.add_parser('deconvolute', help='Run MethylBERT tumour deconvolution')

	parser.add_argument("-i", "--input_data", required=True, type=str, help="Bulk data to deconvolve")
	parser.add_argument("-m", "--model_dir", required=True, type=str, help="Trained methylbert model")
	parser.add_argument("-o", "--output_path", type=str, default="./", help="Directory to save deconvolution results. (default: ./)")

	# Running parametesr
	parser.add_argument("-b", "--batch_size", type=int, default=64, help="Batch size. Please decrease the number if you do not have enough memory to run the software (default: 64)")
	parser.add_argument("--save_logit",  default=False, action="store_true", help="Save logits from the model (default: False)")
	parser.add_argument("--adjustment",  default=False, action="store_true", help="Adjust the estimated tumour purity (default: False)")


def finetune_arg_parser(subparsers):
	parser = subparsers.add_parser('finetune', help='Run MethylBERT fine-tuning')

	parser.add_argument("-c", "--train_dataset", required=True, type=str, help="train dataset for train bert")
	parser.add_argument("-t", "--test_dataset", type=str, default=None, help="test set for evaluate train set")
	parser.add_argument("-o", "--output_path", required=True, type=str, help="ex)output/bert.model")
	parser.add_argument("-p", "--pretrain", type=str, default=None, help="a saved pretrained model to restore")
	parser.add_argument("-nm", "--n_mers", type=int, default=3, help="n-mers (default: 3)")
	
	parser.add_argument("-s", "--seq_len", type=int, default=150, help="maximum sequence len (default: 150)")
	parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm (default: 1.0)")
	parser.add_argument(
		"--gradient_accumulation_steps",
		type=int,
		default=1,
		help="Number of updates steps to accumulate before performing a backward/update pass. (default: 1)",
	)
	parser.add_argument("-b", "--batch_size", type=int, default=50, help="number of batch_size (default: 50)")
	parser.add_argument("--valid_batch", type=int, default=-1, help="number of batch_size in valid set. If it's not given, valid_set batch size is set same as the train_set batch size")
	parser.add_argument("-e", "--steps", type=int, default=600, help="number of training steps (default: 600)")
	parser.add_argument("--save_freq", type=int, default=None, help="Steps to save the interim model")
	parser.add_argument("-w", "--num_workers", type=int, default=20, help="dataloader worker size (default: 20)")

	parser.add_argument("--with_cuda", type=bool, default=True, help="training with CUDA: true, or false (default: True)")
	parser.add_argument("--log_freq", type=int, default=100, help="Frequency (steps) to print the loss values (default: 100)")
	parser.add_argument("--eval_freq", type=int, default=10, help="Evaluate the model every n iter (default: 10)")
	parser.add_argument("--corpus_lines", type=int, default=None, help="total number of lines in corpus")
	#parser.add_argument("--cuda_devices", type=int, nargs='+', default=None, help="CUDA device ids")
	#parser.add_argument("--on_memory", type=bool, default=True, help="Loading on memory: true or false")

	parser.add_argument("--lr", type=float, default=4e-4, help="learning rate of adamW (default: 4e-4)")
	parser.add_argument("--adam_weight_decay", type=float, default=0.01, help="weight_decay of adamW (default: 0.01)")
	parser.add_argument("--adam_beta1", type=float, default=0.9, help="adamW first beta value (default: 0.9)")
	parser.add_argument("--adam_beta2", type=float, default=0.98, help="adamW second beta value (default: 0.98)")
	parser.add_argument("--warm_up", type=int, default=100, help="steps for warm-up (default: 100)")
	parser.add_argument("--seed", type=int, default=950410, help="seed number (default: 950410)")
	parser.add_argument("--decrease_steps", type=int, default=200, help="step to decrease the learning rate (default: 200)")

def preprocess_finetune_arg_parser(subparsers):

	parser = subparsers.add_parser('preprocess_finetune', help='Preprocess .bam files for finetuning')

	parser.add_argument("-s", "--sc_dataset", required=False, default=None, type=str, help="a file all single-cell bam files are listed up. The first and second columns must indicate file names and cell types if cell types are given. Otherwise, each line must have one file path.")
	parser.add_argument("-f", "--input_file", required=False, default=None, type=str, help=".bam file to be processed")
	parser.add_argument("-d", "--f_dmr", required=True, type=str, help=".bed or .csv file DMRs information is contained")
	parser.add_argument("-o", "--output_dir", required=True, type=str, help="a directory where all generated results will be saved")
	parser.add_argument("-r", "--f_ref", required=True, type=str, help=".fasta file containing reference genome")

	parser.add_argument("-nm", "--n_mers", type=int, default=3, help="K for K-mer sequences (default: 3)")
	parser.add_argument("-p", "--split_ratio", type=float, default=0.8, help="the ratio between train and test dataset (default: 0.8)")
	parser.add_argument("-nd", "--n_dmrs", type=int, default=-1, help="Number of DMRs to take from the dmr file. If the value is not given, all DMRs will be used")
	parser.add_argument("-c", "--n_cores", type=int, default=1, help="number of cores for the multiprocessing (default: 1)")
	parser.add_argument("--seed", type=int, default=950410, help="random seed number (default: 950410)")
	parser.add_argument("--ignore_sex_chromo", type=bool, default=True, help="Whether DMRs at sex chromosomes (chrX and chrY) will be ignored (default: True)")

def finetune(args):
	if not os.path.exists(args.output_path):
		os.mkdir(args.output_path)

	with open(args.output_path+"/train_param.txt", "w") as f_param:
		dict_args = vars(args)
		for key in dict_args:
			f_param.write(key+"\t"+str(dict_args[key])+"\n")

	# Set seed
	set_seed(args.seed)

	#print("On memory: ", args.on_memory)

	# Create a tokenizer
	print("Create a tokenizer for %d-mers"%(args.n_mers))
	
	tokenizer=MethylVocab(k=args.n_mers)
	print("Vocab Size: ", len(tokenizer))

	torch.set_num_threads(40)
	print("CPU info:", torch.get_num_threads(), torch.get_num_interop_threads())


	# Load data sets
	print("Loading Train Dataset:", args.train_dataset)
	train_dataset = MethylBertFinetuneDataset(args.train_dataset, tokenizer, 
											  seq_len=args.seq_len)

	print("%d seqs with %d labels "%(len(train_dataset), train_dataset.num_dmrs()))
	print("Loading Test Dataset:", args.test_dataset)

	if args.test_dataset is not None:
		test_dataset = MethylBertFinetuneDataset(args.test_dataset, tokenizer, 
								   				 seq_len=args.seq_len) 

	# Create a data loader
	print("Creating Dataloader")
	local_step_batch_size = int(args.batch_size/args.gradient_accumulation_steps)
	print("Local step batch size : ", local_step_batch_size)
	
	train_data_loader = DataLoader(train_dataset, batch_size=local_step_batch_size, num_workers= args.num_workers, pin_memory=False,  shuffle=True)

	if args.valid_batch < 0:
		args.valid_batch = args.batch_size

	test_data_loader = DataLoader(test_dataset, batch_size=args.valid_batch, num_workers=args.num_workers, pin_memory=True,  shuffle=False) if test_dataset is not None else None

	# BERT train

	print("Creating BERT Trainer")
	trainer = MethylBertFinetuneTrainer(len(tokenizer), save_path=args.output_path+"bert.model/", 
						  train_dataloader=train_data_loader, 
						  test_dataloader=test_data_loader,
						  lr=args.lr, beta=(args.adam_beta1, args.adam_beta2), 
						  weight_decay=args.adam_weight_decay,
						  with_cuda=args.with_cuda, 
						  log_freq=args.log_freq,
						  eval_freq=args.eval_freq,
						  gradient_accumulation_steps=args.gradient_accumulation_steps, 
						  max_grad_norm = args.max_grad_norm,
						  warmup_step=args.warm_up,
						  #loss=args.loss,
						  decrease_steps=args.decrease_steps,
						  #methyl_learning=args.methyl_learning,
						  save_freq=args.save_freq)

	if args.pretrain:

		trainer.load(args.pretrain)
	else:
		print("No pretrained model is given")
		trainer.create_model(config_file="/omics/groups/OE0219/internal/Yunhee/DL_project/MethylBERT/model/3mers_pretrain/DNABERT_mouse_genome/bert.model/config.json")

	print("Training Start")
	trainer.train(args.steps)

def run_deconvolute(args):
	# Reload training parameters 
	params = dict()
	try:
		os.path.exists(args.model_dir+"train_param.txt")
	except:
		FileNotFoundError(f"{args.model_dir}train_param.txt does not exist. Please check if MethylBERT is fine-tuned")
		exit()

	with open(args.model_dir+'train_param.txt', "r") as fp:
		for li in fp.readlines():
			li = li.strip().split('\t')
			params[li[0]] = li[1]
	print("Restored parameters: %s", params)

	# Create a result directory
	if not os.path.exists(args.output_path):
		os.mkdir(args.output_path)
		print("New directory %s is created", args.output_path)

	# Restore the model
	tokenizer=MethylVocab(k=int(params["n_mers"]))
	dataset = MethylBertFinetuneDataset(args.input_data, tokenizer, seq_len=int(params["seq_len"]))
	data_loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=40)
	print("Bulk data (%s) is loaded", args.input_data)

	restore_dir = os.path.join(args.model_dir, "bert.model/")
	trainer = MethylBertFinetuneTrainer(len(tokenizer), save_path='./test',
										train_dataloader=data_loader, 
										test_dataloader=data_loader,
										#methyl_learning=params["methyl_learning"] if "methyl_learning" in params.keys() else "cnn",
										#loss=params["loss"] if "loss" in params.keys() else "bce"
										)
	trainer.load(restore_dir, n_dmrs=100)
	print("Trained model (%s) is restored", restore_dir)

	# Read classification
	total_res, logits = trainer.read_classification(data_loader=data_loader,
												tokenizer=tokenizer,
												logit=True)

	# Save the classification results 
	total_res.to_csv(args.output_path+"/res.csv", sep="\t", header=True, index=False)
	total_res["n_cpg"]=[sum(np.array([int(mm) for mm in m])<2) for m in total_res["methyl_seq"]]
	if args.save_logit:
		with open(args.output_path+"/test_classification_logit.pk", "wb") as fp:
			pk.dump(logits, fp)
	
	# Calculate margins
	df_train = pd.read_csv(params["train_dataset"], sep="\t")
	
	# Deconvolution
	unique_ctypes = df_train["ctype"].unique()
	if total_res["ctype"][0] not in ["N", "T"]:
		total_res["ctype"] = ["N" if c == "noncancer" else "T" for c in total_res["ctype"]]

	print("UNIQUE ", unique_ctypes)
	
	# Tumour-normal deconvolution
	margins = df_train.value_counts("ctype", normalize=True)[["N","T"]].tolist()
	print("Margins : ", margins)
	print(total_res.head())
	if ("T" in total_res["ctype"]) and ("N" in total_res["ctype"]):
		print(total_res.value_counts("ctype", normalize=True)[["N","T"]].tolist(), margins)

	logits = logits[total_res["n_cpg"]>0,]
	total_res  = total_res[total_res["n_cpg"]>0]
	
	if args.adjustment:
		tumour_pred_ratio, fi = deconvolve(logits=logits, margins=margins, dmr_labels=total_res["dmr_label"])
	else:
		tumour_pred_ratio, fi = deconvolve(logits=logits, margins=margins)
	
	print("Deconvolution result: ", tumour_pred_ratio)
	pd.DataFrame.from_dict({"cell_type":["T", "N"],
							"pred":[tumour_pred_ratio, 1-tumour_pred_ratio]}).to_csv(args.output_path+"/deconvolution.csv", sep="\t", header=True, index=False)
	pd.DataFrame.from_dict({"fi":fi}).to_csv(args.output_path+"/FI.csv", sep="\t", header=True, index=False)

def main(args=None):
	print(f"MethylBERT v{__version__}")
	options = ["preprocess_finetune", "finetune", "deconvolute"]
	
	if len(sys.argv) == 1:
		print(f"One option must be given from {options}")
		exit()

	parser_init = argparse.ArgumentParser("methylbert")
	subparsers = parser_init.add_subparsers(help="Options for MethylBERT")
	selected_option =  sys.argv[1]
	
	if selected_option == "preprocess_finetune":
		preprocess_finetune_arg_parser(subparsers)
		args = parser_init.parse_args()
		finetune_data_generate(f_dmr=args.f_dmr,
			output_dir=args.output_dir,
			f_ref=args.f_ref,
			sc_dataset=args.sc_dataset,
			input_file=args.input_file, 
			n_mers=args.n_mers,
			split_ratio=args.split_ratio,
			n_dmrs=args.n_dmrs,
			n_cores=args.n_cores,
			seed=args.seed,
			ignore_sex_chromo=args.ignore_sex_chromo
		)
	elif selected_option == "finetune":
		finetune_arg_parser(subparsers)
		args = parser_init.parse_args()
		finetune(args)
	elif selected_option == "deconvolute":
		deconvolute_arg_parser(subparsers)
		args = parser_init.parse_args()
	else:
		print(f"The option must be chosen in {options}")
