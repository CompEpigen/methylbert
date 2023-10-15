import argparse, os, logging
import pandas as pd
import numpy as np

from methylbert.trainer import MethylBertFinetuneTrainer
from methylbert.data.vocab import WordVocab
from methylbert.data.dataset import MethylBertFinetuneDataset
from methylbert.utils import _moment

from torch.utils.data import DataLoader
from torch import Tensor
from torch.nn.functional import softmax

from tqdm import tqdm
import pickle as pk

from scipy.optimize import minimize

def arg_parser():
	parser = argparse.ArgumentParser()

	parser.add_argument("-i", "--input_data", required=True, type=str, help="Bulk data to deconvolve")
	parser.add_argument("-m", "--model_dir", required=True, type=str, help="Trained methylbert model")
	parser.add_argument("-o", "--output_path", type=str, default="./res/", help="Directory to save deconvolution results. default: ./res/")

	# Running parametesr
	parser.add_argument("-b", "--batch_size", type=int, default=64, help="Batch size. Please increase the number if you do not have enough memory to run the software, default: 64")
	parser.add_argument("--save_logit",  default=False, action="store_true", help="Save logits from the model")
	parser.add_argument("--adjustment",  default=False, action="store_true", help="Adjust the estimated value")

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

	if save_logit:
		with open(output_dir+"/test_classification_logit.pk", "wb") as fp:
			pk.dump(classification_logits, fp)
	
	# Convert the result to a data frame

	total_res["beta"]=[np.mean(m[m!=2]) for m in total_res["methyl_seq"]]
	total_res["n_cpg"]=[sum(m<2) for m in total_res["methyl_seq"]]
	total_res["dna_seq"]=[get_dna_seq(s) for s in total_res["dna_seq"]]
	total_res["methyl_seq"]=["".join([str(mm) for mm in m]) for m in total_res["methyl_seq"]]

	if "is_methyl" in total_res.keys():
		total_res.pop("is_methyl", None)
	
	total_res = pd.DataFrame(total_res)

	logging.info("%s", total_res)
	total_res["gt"] = total_res["ctype_label"]
	total_res["pred"] = total_res["pred_ctype_label"]
	total_res = total_res.drop(columns=["ctype_label", "pred_ctype_label"])
	
	total_res["is_correct"] = (total_res["pred"] == total_res["gt"])
	confusion_dict = {"00":"TN", "11": "TP", "10":"FN", "01":"FP" }
	total_res["group"] = [confusion_dict[str(g)+str(p)] for g, p in zip(total_res["gt"], total_res["pred"])]



	return total_res, classification_logits

def grid_search(logits, margins, n_grid, verbose=True):
	grid = np.zeros([1, n_grid])
	
	if verbose:
		logging.info("Grid search (n=%d) for deconvolution", n_grid)
		pbar = tqdm(total=n_grid)
	for m_theta in range(0, n_grid):
		theta = m_theta*(1/n_grid)
		prob = np.log(theta*(logits[:,1]/margins[1]) + (1-theta)*(logits[:,0]/margins[0])).tolist()
		grid[0, m_theta] = np.sum(prob)
		if verbose:
			pbar.update(1)
	if verbose:
		pbar.close()

	# Fisher info calculation
	fi = np.var([np.abs(grid[0, f+1] -  grid[0, f]) * n_grid for f in range(n_grid-1)])

	return np.argmax(grid, axis=1)*(1/n_grid), fi


def grid_search_regions(logits, margins, n_grid, regions):
	print(logits.shape, regions.shape)
	regions = pd.DataFrame({
		"logit1" : logits[:, 0],
		"logit2" : logits[:, 1],
		"region" : regions if isinstance(regions, list) else regions.tolist()
		})

	dmr_labels = regions["region"].unique()
	region_purity = np.zeros(len(dmr_labels))
	fi = np.zeros(len(dmr_labels))
	for idx, dmr_label in enumerate(dmr_labels):
		dmr_logits = regions[regions["region"] == dmr_label]
		region_purity[idx], fi[idx] =grid_search(np.array(dmr_logits.loc[:, ["logit1", "logit2"]]), margins, n_grid, verbose=False)

	weights = np.ones(len(dmr_labels))
	prev_mean = np.inf
	for iterration in tqdm(range(10)):
		estimates = np.mean(np.multiply(region_purity,weights))
 
		def skewness_test(x):
			a = np.multiply(np.array(region_purity), x)
			m2 = _moment(a, 2, axis=0, mean=estimates)
			m3 = _moment(a, 3, axis=0, mean=estimates)
			with np.errstate(all='ignore'):
				zero = (m2 <= (np.finfo(m2.dtype).resolution * estimates)**2)
				vals = np.where(zero, np.nan, m3 / m2**1.5)
			return (vals[()])**2 


		x = minimize(skewness_test, weights) 
		weights = x["x"]
		if abs(estimates - prev_mean) < 0.0001:
			break
		prev_mean = estimates

	return [estimates, 1-estimates], fi


def deconvolve(logits, margins, n_grid=10000, method="grid", dmr_labels=None):
	'''

	method (str): method to optimise theta, either "grid" or "gredient"
	'''

	if dmr_labels is not None:
		estimation, fi = grid_search_regions(logits, margins, n_grid, dmr_labels)
	elif method == "grid":
		estimation, fi = grid_search(logits, margins, n_grid)
		fi = np.array([fi])
	else:
		logging.error("Please choose \"grid\" or \"gradient\" for the optimisation method")
	
	return estimation[0], fi
																																																																																													   
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

	if os.path.exists(args.output_path+"/test_classification_logit.pk"):
		logging.info("Load saved logits  and results")
		with open(args.output_path+"/test_classification_logit.pk", "rb") as fp:
			logits = pk.load(fp)

		total_res = pd.read_csv(args.output_path+"/res.csv", sep="\t")
	else:
		# Restore the model
		tokenizer=WordVocab(k=int(params["n_mers"]), cpg=False, chg=False, chh=False)
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


	# convert the output of methylbert into np.array
	logits = [l.detach().numpy() for l in logits]
	logits = np.concatenate(logits, axis=0) # merge

	
	# Calculate margins
	df_train = pd.read_csv(params["train_dataset"], sep="\t")
	df_train.columns = ['seqname', 'flag', 'ref_name', 'ref_pos', 'map_quality', 'cigar', 'next_ref_name', 'next_ref_pos', 'length', 'seq', 'qual', 'MD', 'PG', 'XG', 'NM', 'XM', 'XR', 'dna_seq', 'methyl_seq', 'dmr_ctype', 'dmr_label', 'ctype']

	# Deconvolution
	unique_ctypes = df_train["ctype"].unique()
	if total_res["ctype"][0] not in ["N", "T"]:
		total_res["ctype"] = ["N" if c == "noncancer" else "T" for c in total_res["ctype"]]

	print("UNIQUE ", unique_ctypes)
	if len(unique_ctypes) == 2:
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
	else:
		# Multiple cancer type deconvolution
		# load DMRs
		df_dmrs = pd.read_csv(os.path.dirname(params["train_dataset"])+"/dmrs.csv",
						   sep="\t")
		margins = {c:r for c, r in zip(unique_ctypes,
									   df_train.value_counts("ctype", normalize=True)[unique_ctypes].tolist())}
		estimated_tumour_proportions = dict()
		for dmr_ctype in df_dmrs["ctype"].unique():
			# Select reads only from cell-type DMRs
			ctype_dmr_ids = df_dmrs.loc[df_dmrs["ctype"]==dmr_ctype, "dmr_id"]
			ctype_reads = total_res.loc[total_res["dmr_label"].isin(ctype_dmr_ids),]
			print(ctype_reads.head())
			ctype_logits = logits[list(ctype_reads.index),]
			ctype_logits = ctype_logits[ctype_reads["n_cpg"]>0,]

			print(logits[0,:], margins)
			tumour_pred_ratio = deconvolve(logits=ctype_logits, margins=[margins["N"], margins[dmr_ctype]])
			estimated_tumour_proportions[dmr_ctype] = tumour_pred_ratio

		pd.DataFrame.from_dict({"cell_type":list(estimated_tumour_proportions.keys()),
								"pred":list(estimated_tumour_proportions.values())}).to_csv(args.output_path+"/deconvolution.csv", sep="\t", header=True, index=False)
