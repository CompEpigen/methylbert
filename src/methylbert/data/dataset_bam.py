from torch.utils.data import Dataset
import torch

import numpy as np
from copy import deepcopy
import multiprocessing as mp
from functools import partial
import random

from methylbert.data.vocab import WordVocab

def _line2tokens(l, tokenizer, max_len=120):
	'''
			convert a text line into a list of tokens converted by tokenizer 
			
	'''

	l = l.strip().split(" ")

	tokened = [tokenizer.to_seq(b) for b in l]
	if len(tokened) > max_len:
		return tokened[:max_len]
	else:
		return tokened + [[tokenizer.pad_index] for k in range(max_len-len(tokened))]


def _line2tokens_finetune(l, tokenizer, max_len=120):
	# Separate n-mers tokens and labels from each line 

	l = l.strip().split("\t")
	
	methyl_seq = list(l[1])
	dmr_label = l[4]
	ctype_label = l[2] == l[3]

	if len(l) > 5:
		dna_seq = l[5]
		xm_tag = l[6]

	else:
		dna_seq=None
		xm_tag = None
	l = l[0].split(" ")


	# Tokenisation
	tokened = [tokenizer.to_seq(b) for b in l]
	if len(tokened) > max_len:
		return {"seq":tokened[:max_len], 
				"methyl_seq":methyl_seq[:max_len],
				"dmr_label": dmr_label, 
				"ctype_label": ctype_label,
				"dna_seq": dna_seq, 
				"xm_tag":xm_tag}
	else:
		return {"seq": tokened + [[tokenizer.pad_index] for k in range(max_len-len(tokened))], 
				"methyl_seq":methyl_seq + [2 for k in range(max_len-len(tokened))],
				"dmr_label": dmr_label, 
				"ctype_label": ctype_label,
				"dna_seq": dna_seq, 
				"xm_tag":xm_tag}



class MethylBertDataset(Dataset):
	def __init__(self):
		pass
			
	def __len__(self):
		return self.lines.shape[0]


class MethylBertPretrainDataset(MethylBertDataset):
	def __init__(self, f_path: str, vocab: WordVocab, seq_len: int, random_len=False, n_cores=50):

		self.vocab = vocab
		self.seq_len = seq_len
		self.f_path = f_path
		self.random_len = random_len

		# Define a range of tokens to mask based on k-mers 
		self.mask_list = self._get_mask()

		# Read all text files and convert the raw sequence into tokens
		with open(self.f_path, "r") as f_input:
			print("Open data : %s"%f_input)
			raw_seqs = f_input.read().splitlines()

		print("Total number of sequences : ", len(raw_seqs))

		# Multiprocessing for the sequence tokenisation
		with mp.Pool(n_cores) as pool:
			line_labels = pool.map(partial(_line2tokens, tokenizer=self.vocab, max_len=self.seq_len), raw_seqs)
			del raw_seqs
			print("Lines are processed")
			self.lines = torch.squeeze(torch.tensor(np.array(line_labels, dtype=np.int16)))
		del line_labels

	def __getitem__(self, index): 

		dna_seq = self.lines[index].clone()

		# Random len
		if self.random_len and index % 2 == 0:
			dna_seq = dna_seq[:random.randint(5, self.seq_len)] 
		
		# Padding
		if dna_seq.shape[0] < self.seq_len:
			pad_num = self.seq_len+2-dna_seq.shape[0]
			dna_seq = torch.cat((dna_seq, 
								torch.tensor([self.vocab.pad_index for i in range(pad_num)])))
		# Mask 
		masked_dna_seq, dna_seq, bert_mask = self._masking(dna_seq)
		
		return {"input": masked_dna_seq,
				"label": dna_seq,
				"mask" : bert_mask}
	
	def subset_data(self, n_seq: int):
		self.lines = random.sample(self.lines, n_seq)

	def _get_mask(self):
		'''
			Relative positions from the centre of masked region 
			e.g) [-1, 0, 1] for 3-mers 
		'''
		half_length = int(self.vocab.kmers/2)
		mask_list = [-1*half_length + i for i in range(half_length)] + [i for i in range(1, half_length+1)]
		if self.vocab.kmers % 2 == 0:
			mask_list = mask_list[:-1]

		return mask_list

	def _masking(self, inputs: torch.Tensor, threshold=0.15):
		""" 
			Moidfied version of masking token function
			Originally developed by Huggingface (datacollator) and DNABERT
			
			https://github.com/huggingface/transformers/blob/9a24b97b7f304fa1ceaaeba031241293921b69d3/src/transformers/data/data_collator.py#L747

			https://github.com/jerryji1993/DNABERT/blob/bed72fc0694a7b04f7e980dc9ce986e2bb785090/examples/run_pretrain.py#L251

			Added additional tasks to handle each sequence
			Lines using tokenizer were modified due to different tokenizer object structure

		"""

		labels = inputs.clone()

		# Sample tokens with given probability threshold
		probability_matrix = torch.full(labels.shape, threshold) # tensor filled with 0.15

		# Handle special tokens and padding
		special_tokens_mask = [
			val < 5 for val in labels.tolist()
		]
		probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
		padding_mask = labels.eq(self.vocab.pad_index)
		probability_matrix.masked_fill_(padding_mask, value=0.0)

		masked_indices = torch.bernoulli(probability_matrix).bool() # get bernoulli for non-special token masks

		# change masked indices
		masked_index = deepcopy(masked_indices)
		
		# This function handles each sequence
		end = torch.where(probability_matrix!=0)[0].tolist()[-1] # end of the sequence
		mask_centers = set(torch.where(masked_index==1)[0].tolist()) # mask locations

		new_centers = deepcopy(mask_centers)
		for center in mask_centers:
			for mask_number in self._mask_list:# add neighbour loci 
				current_index = center + mask_number 
				if current_index <= end and current_index >= 1:
					new_centers.add(current_index)

		new_centers = list(new_centers)
		masked_indices[new_centers] = True
		
		# Avoid loss calculation on unmasked tokens
		labels[~masked_indices] = -100 

		'''
		# Deal with mask
		if torch.sum(labels==4).tolist() > 0:
			print(labels, inputs)
		'''


		# 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
		indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
		inputs[indices_replaced] = self.vocab.mask_index

		# 10% of the time, we replace masked input tokens with random word
		indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
		random_words = torch.randint(len(self.vocab), labels.shape, dtype=torch.int32)
		inputs[indices_random] = random_words[indices_random]

		# The rest of the time (10% of the time) we keep the masked input tokens unchanged

		# Special tokens (SOS, EOS)
		if end < inputs.shape[0]:
			inputs[end] = self.vocab.eos_index
		else:
			inputs[-1] = self.vocab.eos_index

		labels = torch.cat((torch.tensor([-100]), labels))
		inputs = torch.cat((torch.tensor([self.vocab.sos_index]), inputs))
		masked_index = torch.cat((torch.tensor([False]), masked_index))


		return inputs, labels, masked_index

class MethylBertFinetuneDataset(MethylBertDataset):
	def __init__(self, f_path: str, vocab: WordVocab, seq_len: int, f_ref: str, label_converter=None,n_cores=10):

		self.vocab = vocab
		self.seq_len = seq_len
		
		# files 
		self.f_bam = f_path
		self.f_ref = f_ref

		# Reference genome
		self.dict_ref = self._get_ref_genome(self.f_ref)

		# Input bam file
		self.bam_aln = pysam.AlignmentFile(self.f_bam, "rb").fetch(until_eof=True)
		self.is_single_end = self._is_single_end()

	def _get_ref_genome(self, f_fasta):
		# Reference genome
		record_iter = SeqIO.parse(f_fasta, "fasta")

		# Save the reference genome into a dictionary with chr as a key value
		#global dict_ref # NEED TO FIX IT AT SOME POINT 
		dict_ref = dict()
		for r in record_iter:
		    dict_ref[r.id] = str(r.seq.upper())
		del record_iter

		return dict_ref

	def _is_single_end(self):
        single_end=True
        for program_line in self.bam_aln.header.as_dict()["PG"]:
            if "bismark" in program_line["CL"]:
                parsed_line = program_line["CL"].split(" ")
                for p in parsed_line:
                    if "-2" in p:
                        single_end = False
        return single_end

	def _get_cls_num(self):
		# unique labels
		labels = list(set(self.ctype_labels))
		label_count = np.zeros(len(labels))
		for l in labels:
			label_count[l] = sum(np.array(self.ctype_labels) == l)
		return label_count

	def num_dmrs(self):
		return len(set(self.dmr_labels))
	
	def subset_data(self, n_seq):
		self.lines = self.lines[:n_seq]
		self.methyl_lines = self.methyl_lines[:n_seq]
		self.dmr_labels = self.dmr_labels[:n_seq]
		self.ctype_labels = self.ctype_labels[:n_seq]
	

	def __getitem__(self, index): 

		try:
		    read = next(self.bam_aln)
		except StopIteration:
			# The iterator reached at the end
			# Create a new alignmet iterator 
		    self.bam_aln = pysam.AlignmentFile(self.f_bam, "rb").fetch(until_eof=True)
		    read = next(self.bam_aln)

		processed_read = self._process_read(read)
		if not processed_read:
			continue

		dna_seq = deepcopy(self.lines[index])
		methyl_seq = deepcopy(self.methyl_lines[index])
		dmr_label = deepcopy(self.dmr_labels[index])
		ctype_label = deepcopy(self.ctype_labels[index])

		# Special tokens (SOS, EOS)

		end = torch.where(dna_seq!=self.vocab.pad_index)[0].tolist()[-1] + 1 # end of the read
		if end < dna_seq.shape[0]:
			dna_seq[end] = self.vocab.eos_index
			methyl_seq[end] = 2
		else:
			dna_seq[-1] = self.vocab.eos_index
			methyl_seq[-1] = 2
		
		dna_seq = torch.cat((torch.tensor([self.vocab.sos_index]), dna_seq))
		methyl_seq = torch.cat((torch.tensor([2]), methyl_seq))
		
		return {"dna_seq": dna_seq,
				"methyl_seq": methyl_seq,
				"dmr_label": dmr_label,
				"ctype_label": ctype_label,
				"query": self.dna_seq[index],
				"xm_tag": self.xm_tag[index]}
	
	def _process_read(self, read):
		ref_seq = self.dict_ref[chromo][reads.pos:(reads.pos+reads.query_alignment_end)].upper() # Remove case-specific mode occured by the quality
        xm_tag = read.get_tag("XM")
        cigarstring = read.cigarstring
        
        if xm_tag.count(".") == len(xm_tag):
        	# Methyl call is empty 
            return None
        
        ref_seq, xm_tag = handling_cigar(ref_seq, xm_tag, cigarstring)
        
        # Extract all cytosines
        methylatable_sites = [idx for idx, r in enumerate(ref_seq) if ref_seq[idx:idx+2] == "CG"]

        if len(methylatable_sites) == 0:
            return None 
        
        # Disregard CHH context (h and H)
        for idx in methylatable_sites:
            methyl_state = None
            methyl_idx = -1
            # Taking the complemented cytosine's methylation for the reversed reads
            if idx >= len(xm_tag):
                methyl_state = "."
                methyl_idx=idx
            elif (not reads.is_reverse and single_end) or (reads.is_reverse != reads.is_read1 and not single_end):
                methyl_state = xm_tag[idx]
                methyl_idx = idx 
            elif idx+1 < len(xm_tag): 
                methyl_state = xm_tag[idx+1]
                methyl_idx = idx+1
            else:
                methyl_state = "."
                methyl_idx = idx+1
                
            if methyl_state is not None:
                if methyl_state in (".", "D"): # Missing or occured by deletion
                    methyl_state = "C"
                    
                elif (methyl_state in ["x", "h", "X", "H"]):
                    if (xm_tag[idx] in ["D"]) or (xm_tag[idx+1] in ["D"]):
                        methyl_state="C"
                    else:
                        raise ValueError("Error in the conversion: %s %s %s %s %s"%(xm_tag[idx],   
                                         methyl_state, "Reverse" if reads.is_reverse else "Forward",
                                         ref_seq, xm_tag))
                ref_seq = ref_seq[:idx] + methyl_state + ref_seq[idx+1:]

        # Remove inserted and soft clip bases 
        ref_seq = ref_seq.replace("I", "")
        ref_seq = ref_seq.replace("S", "")