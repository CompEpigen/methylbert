import argparse, os, logging
import pandas as pd
import numpy as np

from methylbert.trainer import MethylBertFinetuneTrainer
from methylbert.data.vocab import WordVocab
from methylbert.data.dataset import MethylBertFinetuneDataset

from torch.utils.data import DataLoader
from torch import Tensor
from torch.nn.functional import softmax

from tqdm import tqdm
import pickle as pk

def arg_parser():
	parser = argparse.ArgumentParser()

	parser.add_argument("-i", "--input_data", required=True, type=str, help="Bulk data to deconvolve")
	parser.add_argument("-m", "--model_dir", required=True, type=str, help="Trained methylbert model")
	parser.add_argument("-o", "--output_path", type=str, default="./res/", help="Directory to save deconvolution results. default: ./res/")

	# Running parametesr
	parser.add_argument("-b", "--batch_size", type=int, default=64, help="Batch size. Please increase the number if you do not have enough memory to run the software, default: 64")
	parser.add_argument("--save_logit",  default=False, action="store_true", help="Save logits from the model")

	return parser.parse_args()

def get_dna_seq(tokens):
	# Convert n-mers tokens into a DNA sequence
	seq = tokenizer.from_seq(tokens)
	seq = [s for s in seq if "<" not in s]
	
	seq = seq[0][0] + "".join([s[1] for s in seq]) + seq[-1][-1]
	
	return seq
	
def read_classification(data_loader, trainer, output_dir, save_logit):
	total_res = list()
	classification_logits = list()
	
	trainer.model.eval()
	pbar = tqdm(total=len(data_loader))
	for data in data_loader:
		# prediction results from the model 
		mask_lm_output = trainer.model.forward(step=2000,
												input_ids = data["dna_seq"],
												token_type_ids=data["is_mehthyl"] if trainer.bert.config.type_vocab_size == 2 else data["methyl_seq"],
												labels = data["dmr_label"],
												methyl_seqs=data["methyl_seq"],
												ctype_label=data["ctype_label"])

		# Read classfication based on logit values 
		if len(total_res) == 0:
			total_res = {k:v.numpy() if type(v) == Tensor else v for k, v in data.items()}
			total_res["pred_ctype_label"] = np.argmax(mask_lm_output["classification_logits"].cpu().detach(), axis=-1)
		else:
			for k, v in data.items():
				total_res[k] = np.concatenate([total_res[k], v.numpy() if type(v) == Tensor else v], axis=0)
			total_res["pred_ctype_label"] = np.concatenate([total_res["pred_ctype_label"],
													  np.argmax(mask_lm_output["classification_logits"].cpu().detach(), axis=-1)], axis=-1)
		
		# Merge logits
		classification_logits.append(mask_lm_output["classification_logits"].cpu().detach())

		del mask_lm_output
		del data	
		pbar.update(1)
	pbar.close()

	
	# Convert the result to a data frame

	total_res["beta"]=[np.mean(m[m!=2]) for m in total_res["methyl_seq"]]
	total_res["n_cpg"]=[sum(m<2) for m in total_res["methyl_seq"]]
	total_res["dna_seq"]=[get_dna_seq(s) for s in total_res["dna_seq"]]
	total_res["methyl_seq"]=["".join([str(mm) for mm in m]) for m in total_res["methyl_seq"]]

	total_res.pop("is_mehthyl", None)
	
	total_res = pd.DataFrame(total_res)

	logging.info("%s", total_res)
	total_res["gt"] = total_res["ctype_label"]
	total_res["pred"] = total_res["pred_ctype_label"]
	total_res = total_res.drop(columns=["ctype_label", "pred_ctype_label"])
	
	total_res["is_correct"] = (total_res["pred"] == total_res["gt"])
	confusion_dict = {"00":"TN", "11": "TP", "10":"FN", "01":"FP" }
	total_res["group"] = [confusion_dict[str(g)+str(p)] for g, p in zip(total_res["gt"], total_res["pred"])]

	if save_logit:
		with open(output_dir+"/test_classification_logit.pk", "wb") as fp:
			pk.dump(classification_logits, fp)

	return total_res, classification_logits

def deconvolve(logits, n_grid=10000):
	margins = [0.5592661950339853, 0.4407338049660147]
	grid = np.zeros([1, n_grid])
	
	logging.info("Grid search (n=%d) for deconvolution", n_grid)
	for m_theta in tqdm(range(0, n_grid)):
		theta = m_theta*(1/n_grid)
		prob = np.log(theta*(logits[:,1]/margins[1]) + (1-theta)*(logits[:,0]/margins[0])).tolist()
		grid[0, m_theta] = np.sum(prob)

	estimation = np.argmax(grid, axis=1)*(1/n_grid)

	return estimation[0]
																																																																																													   
if __name__=="__main__":
	args = arg_parser()
	logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG)

	# Reload training parameters 
	params = dict()
	with open(args.model_dir+'train_param.txt', "r") as fp:
		for li in fp.readlines():
			li = li.strip().split('\t')
			params[li[0]] = li[1]
	logging.info("Restored parameters: %s", params)

	# Create a result directory
	if not os.path.exists(args.output_path):
		os.mkdir(args.output_path)
		logging.info("New directory %s is created", args.output_path)

	# Restore the model
	tokenizer=WordVocab(k=int(params["n_mers"]), cpg= False, chg= False, chh=False)
	dataset = MethylBertFinetuneDataset(args.input_data, tokenizer, seq_len=int(params["seq_len"]))
	data_loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=20)
	logging.info("Bulk data (%s) is loaded", args.input_data)

	restore_dir = os.path.join(args.model_dir, "bert.model/")
	trainer = MethylBertFinetuneTrainer(len(tokenizer), save_path='./test',
										train_dataloader=data_loader, 
										test_dataloader=data_loader,
										methyl_learning=params["methyl_learning"] if "methyl_learning" in params.keys() else "cnn",
										loss=params["loss"] if "loss" in params.keys() else "bce")
	trainer.load(restore_dir, n_dmrs=100)
	logging.info("Trained model (%s) is restored", restore_dir)

	# Read classification
	total_res, logits = read_classification(data_loader=data_loader, 
											trainer=trainer, 
											output_dir=args.output_path, 
											save_logit=args.save_logit)

	# Save the classification results 
	total_res.to_csv(args.output_path+"/res.csv", sep="\t", header=True, index=False)

	# Deconvolution

	# convert the output of methylbert into np.array
	logits = [l.numpy() for l in logits]
	logits = np.concatenate(logits, axis=0) # merge

	#logits = 1/(1 + np.exp(-logits)) # sigmoid


	logits = logits[total_res["n_cpg"]>0,]

	tumour_pred_ratio = deconvolve(logits)
	print("Deconvolution result: ", tumour_pred_ratio)
	pd.DataFrame.from_dict({"cell_type":["T", "N"],
							"pred":[tumour_pred_ratio, 1 - tumour_pred_ratio]}).to_csv(args.output_path+"/deconvolution.csv", sep="\t", header=True, index=False)