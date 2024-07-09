from typing import Tuple, List

def parse_cigar(cigar: str):
	num = 0
	cigar_char = list()
	cigar_num = list()
	cigar = list(cigar)
	
	for c in cigar:
		if c.isdigit() : 
			num = num*10 + int(c)
		else:
			cigar_char.append(c)
			cigar_num.append(num)
			num = 0
	return cigar_char, cigar_num

def handling_cigar(methyl_seq: str, cigarstring: str):
	# Handle cigar strings
	cigar_list, num_list = parse_cigar(cigarstring)
	start_idx  = 0
	new_seq, new_methyl_seq = "", ""

	for c, n in zip(cigar_list, num_list):
		if c not in ["D", "S", "I"]:
			new_methyl_seq += methyl_seq[start_idx:start_idx+n]
		elif c in ["D", "N"]:
			new_methyl_seq += "".join(["D" for nn in range(n)])
			continue
		start_idx += n

	return new_methyl_seq

def process_bismark_read(ref_seq, read):
	cigarstring = read.cigarstring

	# XM tag stores cytosine methyl patterns in bismark 
	xm_tag = read.get_tag("XM") 

	# No CpG methylation on the read
	if xm_tag.count("z") + xm_tag.count("Z") == 0:
		return None
	
	# Handle the insertion and deletion 
	xm_tag = handling_cigar(xm_tag, cigarstring)

	# Extract all CpGs
	methylatable_sites = [idx for idx, r in enumerate(ref_seq) if ref_seq[idx:idx+2] == "CG"]

	# if there's no CpGs on the read
	if len(methylatable_sites) == 0:
		return None 

	# Paired-end or single-end
	is_single_end = not bool(read.flag % 2)

	for idx in methylatable_sites:
		methyl_state = None
		methyl_idx = -1

		# Taking the complement cytosine's methylation for the reversed read
		if idx >= len(xm_tag):
			methyl_state = "."
			methyl_idx=idx
		elif (not read.is_reverse and is_single_end) or (read.is_reverse != read.is_read1 and not is_single_end):
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
				
			elif (methyl_state in ["x", "h", "X", "H"]): # non-CpG methyl
				if (xm_tag[idx] in ["D"]) or (xm_tag[idx+1] in ["D"]):
					methyl_state="C"
				else:
					raise ValueError("Error in the conversion: %d %s %s %s %s\nrefe_seq %s\nread_seq %s\nxmtag_seq %s\n%s\n%s %s %d"%(
										   idx, 
										   xm_tag[idx],   
										   methyl_state, 
										   "Reverse" if read.is_reverse else "Forward",
										   "Single" if is_single_end else "Paired",
									 		ref_seq, 
									 		read.query_alignment_sequence, 
									 		xm_tag, 
									 		cigarstring, 
									 		read.query_name, 
									 		read.reference_name, 
									 		read.pos))
			
			ref_seq = ref_seq[:idx] + methyl_state + ref_seq[idx+1:]

	# Remove inserted and soft clip bases 
	#ref_seq = ref_seq.replace("I", "")
	#ref_seq = ref_seq.replace("S", "")

	return ref_seq


def compare_modifications(mod_base1, mod_base2):
	'''
	Evaluate whether two modifications were measured at the same base
	'''
	if len(mod_base1) != len(mod_base2):
		return False
	for b1, b2 in zip(mod_base1, mod_base2):
		if b1[0] != b2[0]:
			return False
	return True

def process_dorado_read(ref_seq, read):
	'''
	Process reference sequences with methylation patterns from Dorado
	'''

	# read sequence bases including soft clipped bases + get forward if the read is aligned with reverse strand
	read_seq = read.query_sequence 
	
	if (read_seq is None ) or ("H" in read.cigarstring):
		# Secondary alignment and hard-clipped reads are ignored
		return None 
	
	if ref_seq.count("CG") == 0:
		return None

	# Get modified bases indicationg methylation 
	methyl_seq = [2 for i in range(len(read_seq))]
	modified_bases = read.modified_bases.copy() # the instance of read cannot be modified
	ch_key, cm_key = ('C', int(read.is_reverse), 'h'), ('C', int(read.is_reverse), 'm')

	if (ch_key not in modified_bases.keys()) and (cm_key not in modified_bases.keys()):
		# no cytosine modification
		return None
	elif cm_key not in modified_bases.keys():
		#  5-Methylcytosine  missing, add Cm keys with likelihood 0 
		modified_bases.update({cm_key: list()})
		for base_mod in modified_bases[ch_key]:
			modified_bases[cm_key].append((base_mod[0], 0))
	elif ch_key not in modified_bases.keys():
		# 5-Hydroxymethylcytosine missing, add Ch keys with likelihood 0 
		modified_bases.update({ch_key: list()})
		for base_mod in modified_bases[cm_key]:
			modified_bases[ch_key].append((base_mod[0], 0))
	elif not compare_modifications(modified_bases['C', int(read.is_reverse), 'h'], 
								 modified_bases['C', int(read.is_reverse), 'm']):
		raise ValueError(f"Modifications are not aligned: {modified_bases['C', int(read.is_reverse), 'h']}, {modified_bases['C', int(read.is_reverse), 'm']}")

	for ch, cm in zip(modified_bases[ch_key], 
					  modified_bases[cm_key]):
		sum_prob = ch[1]+cm[1]
		methyl_pattern = 1 if sum_prob >= 178 else 0 # consider methylated when the likelihood is > .5
		cg_idx = ch[0] - 1 if read.is_reverse else ch[0]
		methyl_seq[cg_idx] = methyl_pattern

	methyl_seq = "".join(list(map(str, methyl_seq)))

	original_methyl = methyl_seq
	# Handle cigar strings
	methyl_seq = handling_cigar(methyl_seq, read.cigarstring)
	
	#if len(ref_seq) != len(methyl_seq):
	#	raise ValueError(f"DNA seq and methylation seq have different lengths - {len(ref_seq)}, {len(methyl_seq)}")

	# Match reference seq and methyl pattern 
	for idx in range(len(ref_seq)):
		if idx >= len(methyl_seq):
			# methyl seq shorter than ref seq
			break
		if methyl_seq[idx] in ["2", "D"]:
			continue
			
		if ref_seq[idx:idx+2] != "CG":
			# Occured because of variant
			continue
		
		ref_seq = ref_seq[:idx] + ("z" if methyl_seq[idx] == "0" else "Z") + ref_seq[idx+1:]
	
	return ref_seq