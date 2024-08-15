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

def likelihood_fun(theta, margins, prob):
	'''
	theta should be 2 x 1
	prob should be reads x 2
	'''
	#margins, prob = args
	prob = np.divide(prob, margins)
	prob = np.log(np.matmul(theta.T, prob.T))
	return  np.sum(prob)

def nll_multi_celltype(thetas, *args):

	df_res, margins = args
	nll = 0

	for idx, ctype in enumerate(margins.keys()):
		ctype_margins = np.array([1-margins[ctype], margins[ctype]])
		prob = df_res.loc[df_res["dmr_ctype"]==ctype, "P_ctype"].to_numpy()
		prob = np.concatenate([np.array(1-prob).reshape(-1,1), 
							   np.array(prob).reshape(-1, 1)], axis=1)
		prob = np.divide(prob, ctype_margins)
		prob = np.log(np.matmul(np.array([1-thetas[idx], thetas[idx]]).reshape([1,2]),
								prob.T))
		nll -= np.sum(prob)
	return nll

def grid_search(logits, margins, n_grid, verbose=True):
	'''
	return estimated_proportions (list), fisher info, likelihood
	'''
	grid = np.zeros([1, n_grid])
	
	if verbose:
		logging.info("Grid search (n=%d) for deconvolution", n_grid)
		pbar = tqdm(total=n_grid)
	for m_theta in range(0, n_grid):
		theta = m_theta*(1/n_grid)
		theta = np.array([1-theta, theta]).reshape([2,1])
		if logits.shape[1] != theta.shape[0]:
			raise ValueError(f"Dimensions are wrong: theta {theta.shape}, prob {logits.shape}")
		grid[0, m_theta] = likelihood_fun(theta, margins, logits)
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

	return [estimates, 1-estimates], list(fi), dmr_labels, list(likelihood), #list(region_purity), list(weights)

def optimise_nll_deconvolute(reads : pd.DataFrame,
							 margins : pd.Series):
	'''
	Deconvolution for multiple cell types
	'''

	estimates = minimize(nll_multi_celltype, 
							margins.to_numpy(),
							args=(reads, margins), 
							method='SLSQP',
							bounds=[(1e-10, 1-1e-10) for i in range(margins.shape[0])],
							constraints={'type': 'eq', 'fun': lambda x: np.sum(x)-1})
	return pd.DataFrame.from_dict({"cell_type":margins.keys().tolist(), 
								   "pred":estimates.x})

def purity_estimation(reads : pd.DataFrame,
					  margins : pd.Series, 
					  n_grid : int,
					  adjustment : bool):

	margins = margins[["N","T"]].tolist()
	
	# Tumour-normal deconvolution
	if adjustment:
		# Adjustment applied
		estimation, fi, dmr_labels, likelihood = \
		grid_search_regions(reads.loc[:,["P_N", "P_ctype"]].to_numpy(), 
							margins, 
							n_grid, 
							reads["dmr_label"])
	else:
		estimation, fi, likelihood = grid_search(reads.loc[:,["P_N", "P_ctype"]].to_numpy(),
												 margins, 
												 n_grid)

	# Dataframe for the results 
	deconv_res = pd.DataFrame.from_dict({"cell_type":["T", "N"], "pred":estimation})
	if type(fi) is not list:
		fi = [fi]
		fi_res = pd.DataFrame.from_dict({"fi":fi, "likelihood": likelihood})
	else:
		fi_res = pd.DataFrame.from_dict({"dmr_label":dmr_labels, 
										"fi":fi,
										"likelihood": likelihood}).sort_values("dmr_label")

	return deconv_res, fi_res

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
	
	# Select reads which contain methylation patterns 
	total_res  = total_res[total_res["n_cpg"]>0]
	assert total_res.shape[0] != 0, "There are no reads selected for deconvolution. It may mean all of the reads do not have CpG methylation."

	# Calculate prior from training data 
	margins = df_train.value_counts("ctype", normalize=True)

	print("Margins : ", margins)
	print(total_res.head())

	if len(margins.keys()) == 2:
		# purity estimation
		deconv_res, fi_res = purity_estimation(reads = total_res,
											   margins = margins, 
											   n_grid = n_grid,
											   adjustment = adjustment)
		deconv_res.to_csv(output_path+"/deconvolution.csv", sep="\t", header=True, index=False)
		fi_res.to_csv(output_path+"/FI.csv", sep="\t", header=True, index=False)
	elif len(margins.keys()) > 2:
		# multiple cell-type deconvolution 
		deconv_res = optimise_nll_deconvolute(reads = total_res, margins = margins)
		deconv_res.to_csv(output_path+"/deconvolution.csv", sep="\t", header=True, index=False)
	else:
		raise RuntimeError(f"There are less than two cell types in the training data set. {margins.keys()} Neither purity estimation nor deconvolution can be performed.")

	