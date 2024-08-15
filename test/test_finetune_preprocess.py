from methylbert.data import finetune_data_generate as fdg
import pandas as pd
import os

def test_split_ratio(bam_file: str, f_dmr: str, f_ref: str, out_dir = "tmp/", split_ratio=0.5):
	fdg.finetune_data_generate(
		input_file = bam_file,
		f_dmr = f_dmr,
		f_ref = f_ref,
		output_dir=out_dir,
		split_ratio = split_ratio,
		n_cores=1
	)

	assert os.path.exists(out_dir+"train_seq.csv")
	assert os.path.exists(out_dir+"test_seq.csv")
	assert os.path.exists(out_dir+"dmrs.csv")

	n_test_seqs = pd.read_csv(out_dir+"test_seq.csv", sep="\t").shape[0] 
	n_train_seqs = pd.read_csv(out_dir+"train_seq.csv", sep="\t").shape[0] 
	assert (n_train_seqs/(n_train_seqs+n_test_seqs) <= split_ratio + 0.05) and (n_train_seqs/(n_train_seqs+n_test_seqs) >= split_ratio - 0.05)

	print("test_split_ratio passed!")

def test_multi_cores(bam_file: str, f_dmr: str, f_ref: str, out_dir = "tmp/", n_cores=4):
	fdg.finetune_data_generate(
		input_file = bam_file,
		f_dmr = f_dmr,
		f_ref = f_ref,
		output_dir=out_dir,
		split_ratio = 1.0,
		n_cores=n_cores
	)

	assert os.path.exists(out_dir+"data.csv")
	assert os.path.exists(out_dir+"dmrs.csv")

	print("test_multi_cores passed!")

def test_dmr_subset(bam_file: str, f_dmr: str, f_ref: str, out_dir = "tmp/", n_dmrs=10):
	fdg.finetune_data_generate(
		input_file = bam_file,
		f_dmr = f_dmr,
		f_ref = f_ref,
		output_dir=out_dir,
		split_ratio = 1.0,
		n_cores=1,
		n_dmrs=n_dmrs
	)

	assert pd.read_csv(out_dir+"dmrs.csv", sep="\t").shape[0] == n_dmrs

	print("test_dmr_subset passed!")

def test_list_bam_file(f_bam_file_list: str, f_dmr: str, f_ref: str, out_dir = "tmp/"):
	fdg.finetune_data_generate(
		sc_dataset = f_bam_file_list,
		f_dmr = f_dmr,
		f_ref = f_ref,
		output_dir=out_dir,
		split_ratio = 1.0,
		n_cores=1
	)

	assert os.path.exists(out_dir+"data.csv")
	assert os.path.exists(out_dir+"dmrs.csv")

	res  = pd.read_csv(out_dir+"data.csv", sep="\t")
	assert "T" in res["ctype"].tolist()
	assert "N" in res["ctype"].tolist()

	print("test_list_bam_file passed!")

def test_single_bam_file(bam_file: str, f_dmr: str, f_ref: str, out_dir = "tmp/"):
	fdg.finetune_data_generate(
		input_file = bam_file,
		f_dmr = f_dmr,
		f_ref = f_ref,
		output_dir=out_dir,
		split_ratio = 1.0,
		n_cores=1
	)

	assert os.path.exists(out_dir+"data.csv")
	assert os.path.exists(out_dir+"dmrs.csv")

	print("test_single_bam_file passed!")

def test_dorado_aligned_file(bam_file: str, f_dmr: str, f_ref: str, out_dir = "tmp/"):
	fdg.finetune_data_generate(
		input_file = bam_file,
		f_dmr = f_dmr,
		f_ref = f_ref,
		output_dir=out_dir,
		split_ratio = 1.0,
		n_cores=1, 
		methyl_caller="dorado"
	)

	assert os.path.exists(out_dir+"data.csv")
	assert os.path.exists(out_dir+"dmrs.csv")

	print("test_dorado_aligned_file passed!")


def test_multi_cell_type(f_bam_file_list: str, f_dmr: str, f_ref: str, out_dir = "tmp/"):
	fdg.finetune_data_generate(
		sc_dataset = f_bam_file_list,
		f_dmr = f_dmr,
		f_ref = f_ref,
		output_dir=out_dir,
		split_ratio = 0.8,
		n_cores=1
	)

	assert os.path.exists(out_dir+"train_seq.csv")
	assert os.path.exists(out_dir+"test_seq.csv")
	assert os.path.exists(out_dir+"dmrs.csv")

	res  = pd.read_csv(out_dir+"train_seq.csv", sep="\t")
	assert "T" in res["ctype"].tolist()
	assert "N" in res["ctype"].tolist()
	assert "P" in res["ctype"].tolist()

	print("test_multi_cell_type passed!")


if __name__=="__main__":
	f_bam = "data/T_sample.bam"
	f_bam_list = "data/bam_list.txt"
	f_dmr = "data/dmrs.csv"
	f_ref = "data/genome.fa"

	
	test_single_bam_file(bam_file = f_bam, f_dmr=f_dmr, f_ref=f_ref)
	test_list_bam_file(f_bam_file_list = f_bam_list, f_dmr=f_dmr, f_ref=f_ref)
	test_dmr_subset(bam_file = f_bam, f_dmr=f_dmr, f_ref=f_ref, n_dmrs=10)
	test_multi_cores(bam_file = f_bam, f_dmr=f_dmr, f_ref=f_ref, n_cores=4)
	test_split_ratio(bam_file = f_bam, f_dmr=f_dmr, f_ref=f_ref, split_ratio=0.7)
	

	f_bam_list = "data/multi_cell_type/bam_list.txt"
	f_dmr = "data/multi_cell_type/dmrs.csv"
	out_dir = "data/multi_cell_type/"
	test_multi_cell_type(f_bam_file_list = f_bam_list, f_dmr=f_dmr, f_ref=f_ref, out_dir=out_dir)
	
	f_dorado = "data/dorado_aligned.bam"
	f_ref_hg38="data/hg38_genome.fa"
	test_dorado_aligned_file(bam_file = f_dorado, f_dmr=f_dmr, f_ref=f_ref_hg38)
	print("Everything passed!")
	