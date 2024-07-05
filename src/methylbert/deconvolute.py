import argparse, os, logging
import pandas as pd
import numpy as np

from methylbert.trainer import MethylBertFinetuneTrainer
from methylbert.data.vocab import MethylVocab
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

def likelihood_fun(theta, margins, prob):
    '''
    theta should be 2 x 1
    prob should be reads x 2
    '''
    #margins, prob = args
    prob = np.divide(prob, margins)
    if prob.shape[1] != theta.shape[0]:
        raise ValueError(f"Dimensions are wrong: theta {theta.shape}, prob {prob.shape}")
    prob = np.log(np.matmul(theta.T, prob.T))
    return  np.sum(prob)


def grid_search(logits, margins, n_grid, verbose=True):
	grid = np.zeros([1, n_grid])
	
	if verbose:
		logging.info("Grid search (n=%d) for deconvolution", n_grid)
		pbar = tqdm(total=n_grid)
	for m_theta in range(0, n_grid):
		theta = m_theta*(1/n_grid)
		grid[0, m_theta] = likelihood_fun(np.array([1-theta, theta]).reshape([2,1]),
							   margins, logits)
		if verbose:
			pbar.update(1)
	if verbose:
		pbar.close()

	# Fisher info calculation
	fi = np.var([grid[0, f+1] -  grid[0, f] * n_grid for f in range(n_grid-1)])
	argmax_idx = np.argmax(grid, axis=1)
	estimates =  float(argmax_idx)*(1/n_grid)
	return [estimates, 1-estimates], fi, grid[0, argmax_idx]


def grid_search_regions(logits, margins, n_grid, regions):

	def skewness_test(x, *args):
		region_purity, estimates = args
		a = np.multiply(np.array(region_purity), x)
		m2 = _moment(a, 2, axis=0, mean=estimates)
		m3 = _moment(a, 3, axis=0, mean=estimates)
		with np.errstate(all='ignore'):
			zero = (m2 <= (np.finfo(m2.dtype).resolution * estimates)**2)
			vals = np.where(zero, np.nan, m3 / m2**1.5)
		return (vals[()])**2 

	regions = pd.DataFrame({
		"logit1" : logits[:, 0],
		"logit2" : logits[:, 1],
		"region" : regions if isinstance(regions, list) else regions.tolist()
		})

	dmr_labels = regions["region"].unique()
	region_purity = np.zeros(len(dmr_labels))
	fi, likelihood = np.zeros(len(dmr_labels)), np.zeros(len(dmr_labels))
	for idx, dmr_label in enumerate(dmr_labels):
		dmr_logits = regions[regions["region"] == dmr_label]
		purities, fi[idx], likelihood[idx] = grid_search(np.array(dmr_logits.loc[:, ["logit1", "logit2"]]), margins, n_grid, verbose=False)
		region_purity[idx] = purities[0]

	# EM algorithm for the adjustment
	weights = np.ones(len(dmr_labels))
	prev_mean = np.inf
	estimates = np.mean(np.multiply(region_purity,weights))

	for iterration in tqdm(range(10)):
		prev_mean = estimates
		x = minimize(skewness_test, weights, args=(region_purity, estimates)) 
		weights = x["x"]
		estimates = np.mean(np.multiply(region_purity,weights))

		if abs(estimates - prev_mean) < 0.0001:
			break
	
	estimates = np.clip(estimates, 0, 1)

	return [estimates, 1-estimates], list(fi), dmr_labels, list(likelihood), list(region_purity), list(weights)

def deconvolute(trainer : MethylBertFinetuneTrainer, 
				data_loader : DataLoader, 
				df_train : pd.DataFrame, 
				tokenizer : MethylVocab,
				output_path : str = "./", 
				n_grid : int = 10000, 
				adjustment : bool = False):
	'''
	Tumour deconvolution for the given bulk

	trainer: MethylBertFinetuneTrainer
		Fine-tuned methylbert model contained in a MethylBertFinetuneTrainer object
	data_loader: torch.utils.data.DataLoader
		DataLoader containing sequencing reads from the bulk
	df_train: pandas.DataFrame
		DataFrame containing the training data. This is for calculating margins (marginal probability in the Bayes' theorem)
	output_path: str (defalut: "./")
		Directory to save the results
	n_grid: int (default: 10000)
		Number of grids for the grid-search algorithm. The higher the number is, the more precise the tumour purity estimation will be
	adjustment: bool (default: False)
		Whether you want to conduct the estimation adjustment or not
	'''

	if not os.path.exists(output_path):
		os.mkdir(output_path)

	# Read classification
	total_res, logits = trainer.read_classification(data_loader=data_loader,
												tokenizer=tokenizer,
												logit=True)
	total_res = total_res.drop(columns=["ctype_label"])

	# Save the classification results 
	total_res["n_cpg"]=total_res["methyl_seq"].apply(lambda x: x.count("0") + x.count("1"))
	total_res["P_ctype"] = logits[:,1]
	total_res.to_csv(output_path+"/res.csv", sep="\t", header=True, index=False)
	total_res["P_N"] = logits[:,0]
	
	# Tumour-normal deconvolution
	# Margins from train data
	margins = df_train.value_counts("ctype", normalize=True)[["N","T"]].tolist()
	print("Margins : ", margins)
	print(total_res.head())

	total_res  = total_res[total_res["n_cpg"]>0]

	assert total_res.shape[0] != 0, "There are no reads selected for deconvolution. It may mean all of the reads do not have CpG methylation."
	
	if adjustment:
		estimation, fi, dmr_labels, likelihood = grid_search_regions(total_res.loc[:,["P_N", "P_ctype"]].to_numpy(), 
											 margins, n_grid, total_res["dmr_label"])
	else:
		estimation, fi, likelihood = grid_search(logits, margins, n_grid)

	# Save the results 
	res = pd.DataFrame.from_dict({"cell_type":["T", "N"],
							"pred":estimation}).to_csv(output_path+"/deconvolution.csv", sep="\t", header=True, index=False)
	if type(fi) is not list:
		fi = [fi]
		pd.DataFrame.from_dict({"fi":fi,
							    "likelihood": likelihood}).to_csv(output_path+"/FI.csv", sep="\t", header=True, index=False)
	else:
		pd.DataFrame.from_dict({"dmr_label":dmr_labels, 
								"fi":fi,
								"likelihood": likelihood}).sort_values("dmr_label").to_csv(output_path+"/FI.csv", sep="\t", header=True, index=False)

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
		tokenizer=MethylVocab(k=int(params["n_mers"]))
		dataset = MethylBertFinetuneDataset(args.input_data, tokenizer, seq_len=int(params["seq_len"]))
		data_loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=40)
		logging.info("Bulk data (%s) is loaded", args.input_data)

		restore_dir = os.path.join(args.model_dir, "bert.model/")
		trainer = MethylBertFinetuneTrainer(len(tokenizer), save_path='./test',
											train_dataloader=data_loader, 
											test_dataloader=data_loader,
											methyl_learning=params["methyl_learning"] if "methyl_learning" in params.keys() else "cnn",
											loss=params["loss"] if "loss" in params.keys() else "bce")
		trainer.load(restore_dir, load_fine_tune=True)
		logging.info("Trained model (%s) is restored", restore_dir)

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

	# convert the output of methylbert into np.array
	#logits = [l.detach().numpy() for l in logits]
	#logits = np.concatenate(logits, axis=0) # merge

	
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
			tumour_pred_ratio, fi = deconvolute(logits=logits, margins=margins, dmr_labels=total_res["dmr_label"])
		else:
			tumour_pred_ratio, fi = deconvolute(logits=logits, margins=margins)
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
			tumour_pred_ratio = deconvolute(logits=ctype_logits, margins=[margins["N"], margins[dmr_ctype]])
			estimated_tumour_proportions[dmr_ctype] = tumour_pred_ratio

		pd.DataFrame.from_dict({"cell_type":list(estimated_tumour_proportions.keys()),
								"pred":list(estimated_tumour_proportions.values())}).to_csv(args.output_path+"/deconvolution.csv", sep="\t", header=True, index=False)
