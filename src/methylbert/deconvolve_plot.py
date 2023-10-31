import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import minimize, fmin_bfgs
from tqdm import tqdm
import pickle as pk
import argparse, os, logging

from methylbert.trainer import MethylBertFinetuneTrainer
from methylbert.data.vocab import WordVocab
from methylbert.data.dataset import MethylBertFinetuneDataset
from methylbert.utils import _moment

from torch.utils.data import DataLoader
from torch import Tensor
from torch.nn.functional import softmax

def arg_parser():
	parser = argparse.ArgumentParser()

	parser.add_argument("-i", "--input_data", required=True, type=str, help="Bulk data to deconvolve")
	parser.add_argument("-m", "--model_dir", required=True, type=str, help="Trained methylbert model")
	parser.add_argument("-g", "--gt", required=True, type=float, help="Ground-truth tumour purity")
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

	return np.argmax(grid, axis=1)*(1/n_grid)	


def grid_search_regions(logits, margins, n_grid, regions):
	print(logits.shape, regions.shape)
	regions = pd.DataFrame({
		"logit1" : logits[:, 0],
		"logit2" : logits[:, 1],
		"region" : regions if isinstance(regions, list) else regions.tolist()
		})

	dmr_labels = regions["region"].unique()
	region_purity = np.zeros(len(dmr_labels))
	for idx, dmr_label in enumerate(dmr_labels):
		dmr_logits = regions[regions["region"] == dmr_label]
		region_purity[idx] =grid_search(np.array(dmr_logits.loc[:, ["logit1", "logit2"]]), margins, n_grid, verbose=False)

	init_region_purity = region_purity.copy()

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

	final_region_purity = np.multiply(np.array(region_purity), weights)
	
	return estimates, init_region_purity, final_region_purity


def cancer_detector_estimate(logits, margins, n_grid, regions, lambda_=0.5):
	print(logits.shape, regions.shape)
	regions = pd.DataFrame({
		"logit1" : logits[:, 0],
		"logit2" : logits[:, 1],
		"region" : regions if isinstance(regions, list) else regions.tolist()
		})

	dmr_labels = regions["region"].unique()
	good_reads_idx = regions.index

	# Infer tumour purity of each reagion
	region_purity = np.zeros(len(dmr_labels))
	for idx, dmr_label in enumerate(dmr_labels):
		dmr_logits = regions[regions["region"] == dmr_label]
		region_purity[idx] =grid_search(np.array(dmr_logits.loc[:, ["logit1", "logit2"]]), margins, n_grid, verbose=False)
	theta_std = np.std(1-region_purity)

	init_region_purity = region_purity.copy()
	print("Run CancerDetector")

	# 20 round of EM
	prev_good_region_ids = list(range(100))
	for iterration in tqdm(range(20)):
		estimates = grid_search(np.array(regions.loc[[r in np.array(prev_good_region_ids) for r in regions["region"]], ["logit1", "logit2"]]), margins, n_grid, verbose=False)
		estimates = estimates[0]
		cutoff = (1-estimates) - theta_std * lambda_
		good_region_ids = list(np.where((1-region_purity) > cutoff)[0])
		if len(good_region_ids) == len(prev_good_region_ids):
			break
		prev_good_region_ids = good_region_ids

	final_region_purity = init_region_purity[good_region_ids]
	#print(good_region_ids)
	#for idx, dmr_label in enumerate(good_region_ids):
	#		dmr_logits = regions[regions["region"] == dmr_label]
	#	final_region_purity[idx] =grid_search(np.array(dmr_logits.loc[:, ["logit1", "logit2"]]), margins, n_grid, verbose=False)
	
	return estimates, init_region_purity, final_region_purity, good_region_ids


def deconvolve(logits, margins, n_grid=10000, method="grid", dmr_labels=None, cancer_detector=False):
	'''

	method (str): method to optimise theta, either "grid" or "gradient"
	'''

	if (dmr_labels is not None) and (not cancer_detector):
		res = grid_search_regions(logits, margins, n_grid, dmr_labels)
	elif cancer_detector:
		res = cancer_detector_estimate(logits, margins, n_grid, dmr_labels)
	elif method == "grid":
		res = grid_search(logits, margins, n_grid)
	else:
		logging.error("Please choose \"grid\" or \"gradient\" for the optimisation method")
	
	return res

def plot_comparison(init_region_purity, final_region_purity, tumour_pred_ratio, res_path, gt):
	
	if os.path.exists(res_path):
		os.mkdir(res_path)

	color = ["blue", "orange"]
	label = ["without adjustment", "with adjustment"]
	
	fig, ax = plt.subplots(1, figsize=(3,5))	
	min_y, max_y = np.min(final_region_purity) - 0.1, np.max(final_region_purity) + 0.05

	for idx, purities in enumerate([init_region_purity, final_region_purity]):
		ax.scatter(x = range(100), y= purities, s=5, c=color[idx], label=label[idx])
		if idx == 0:
			ax.hlines(y=args.gt, color="grey", xmax=100, xmin=0, label="ground-truth")
		else:
			ax.hlines(y=tumour_pred_ratio, color="red", linestyle='dashed', xmax=100, xmin=0, label="estimated tumour purity")

	ax.set_xlabel("DMR labels")
	ax.set_ylabel("Estimated region-wise purity")
	ax.set_ylim([min_y, max_y])
	ax.set_title("MethylBERT calibration (abs err = %.3f)"%(abs(gt-tumour_pred_ratio)))
	plt.legend(loc="lower left",  prop = { "size": 8})
	plt.tight_layout()

	plt.savefig("%s/region_purity.png"%res_path, dpi=300, bbox_inches="tight")
	plt.savefig("%s/region_purity.pdf"%res_path, format="pdf", dpi=300, bbox_inches="tight")

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

	# Save result in a different dir
	args.output_path = args.output_path+"/plot/"
	if not os.path.exists(args.output_path):
		os.mkdir(args.output_path)
	print("Save plot in %s"%args.output_path)

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

	# Tumour-normal deconvolution
	margins = df_train.value_counts("ctype", normalize=True)[["N","T"]].tolist()
	print(total_res.head())
	if ("T" in total_res["ctype"]) and ("N" in total_res["ctype"]):
		print(total_res.value_counts("ctype", normalize=True)[["N","T"]].tolist(), margins)

	logits = logits[total_res["n_cpg"]>0,]
	total_res  = total_res[total_res["n_cpg"]>0]
	
	tumour_pred_ratio = deconvolve(logits=logits, margins=margins, dmr_labels=total_res["dmr_label"])
	
	tumour_pred_ratio, init_region_purity, final_region_purity = tumour_pred_ratio

	print("Deconvolution result: ", tumour_pred_ratio)

	if not os.path.exists(args.output_path+"/deconvolution.csv"):
		pd.DataFrame.from_dict({"cell_type":["T", "N"],
								"pred":[tumour_pred_ratio, 1-tumour_pred_ratio]}).to_csv(args.output_path+"/deconvolution.csv", sep="\t", header=True, index=False)
	

	color = ["blue", "orange"]
	label = ["without adjustment", "with adjustment"]
	fig, ax = plt.subplots(1, figsize=(3,5))	
	min_y, max_y = np.min(final_region_purity) - 0.1, np.max(final_region_purity) + 0.05

	for idx, purities in enumerate([init_region_purity, final_region_purity]):
		ax.scatter(x = range(100), y= purities, s=5, c=color[idx], label=label[idx])
		if idx == 0:
			ax.hlines(y=args.gt, color="grey", xmax=100, xmin=0, label="ground-truth")
		else:
			ax.hlines(y=tumour_pred_ratio, color="red", linestyle='dashed', xmax=100, xmin=0, label="estimated tumour purity")

	ax.set_xlabel("DMR labels")
	ax.set_ylabel("Estimated region-wise purity")
	ax.set_ylim([min_y, max_y])
	ax.set_title("MethylBERT adjustment (abs err = %.3f)"%(abs(args.gt-tumour_pred_ratio)))
	plt.legend(loc="lower left",  prop = { "size": 8})
	plt.tight_layout()
	plt.savefig("%s/region_purity.png"%args.output_path, dpi=300, bbox_inches="tight")
	plt.savefig("%s/region_purity.pdf"%args.output_path, format="pdf", dpi=300, bbox_inches="tight")
	
	# Cancer detector
	tumour_pred_ratio = deconvolve(logits=logits, margins=margins, dmr_labels=total_res["dmr_label"], cancer_detector=True)

	tumour_pred_ratio, init_region_purity, final_region_purity, dmr_ids = tumour_pred_ratio

	print("Cancer Detector result: ", tumour_pred_ratio)
	pd.DataFrame.from_dict({"cell_type":["T", "N"],
							"pred":[tumour_pred_ratio, 1-tumour_pred_ratio]}).to_csv(args.output_path+"/cancer_detector_deconvolution.csv", sep="\t", header=True, index=False)
	
	color = ["blue", "orange"]
	label = ["without adjustment", "with adjustment"]
	fig, ax = plt.subplots(1, figsize=(3,5))	
	for idx, p in enumerate(zip([init_region_purity, final_region_purity],
								[list(range(100)), dmr_ids])):
		purities, xticks = p
		dist = abs(args.gt - np.mean(purities))
		ax.scatter(x = xticks, y= purities, s=5, c=color[idx], label=label[idx])
		if idx == 0:
			ax.hlines(y=args.gt, color="grey", xmax=100, xmin=0, label="ground-truth")
		else:
			ax.hlines(y=tumour_pred_ratio, color="red", linestyle='dashed', xmax=100, xmin=0, label="estimated tumour purity")

	ax.set_xlabel("DMR labels")
	ax.set_ylabel("Estimated region-wise purity")
	ax.set_ylim([min_y, max_y])
	ax.set_title("CancerDetector adjustment (abs err = %.3f)"%(abs(args.gt-tumour_pred_ratio)))
	plt.legend(loc="lower left",  prop = { "size": 8})
	plt.tight_layout()
	plt.savefig("%s/cancer_detector_region_purity.png"%args.output_path, dpi=300, bbox_inches="tight")
	plt.savefig("%s/cancer_detector_region_purity.pdf"%args.output_path, format="pdf", dpi=300, bbox_inches="tight")
