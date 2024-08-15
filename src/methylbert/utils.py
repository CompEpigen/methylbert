import warnings, random 
import numpy as np
import torch

def get_dna_seq(tokens, tokenizer):
	# Convert n-mers tokens into a DNA sequence
	seq = tokenizer.from_seq(tokens)
	seq = [s for s in seq if "<" not in s]
	
	seq = seq[0][0] + "".join([s[1] for s in seq]) + seq[-1][-1]
	
	return seq

def set_seed(seed: int):
	"""
	Helper function for reproducible behavior to set the seed in ``random``, ``numpy``, ``torch`` and/or ``tf`` (if
	installed).

	Args:
		seed (:obj:`int`): The seed to set.
	"""
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	

def _moment(a, moment, axis, *, mean=None):
	if np.abs(moment - np.round(moment)) > 0:
		raise ValueError("All moment parameters must be integers")

	# moment of empty array is the same regardless of order
	if a.size == 0:
		return np.mean(a, axis=axis)

	dtype = a.dtype.type if a.dtype.kind in 'fc' else np.float64

	if moment == 0 or (moment == 1 and mean is None):
		# By definition the zeroth moment is always 1, and the first *central*
		# moment is 0.
		shape = list(a.shape)
		del shape[axis]

		if len(shape) == 0:
			return dtype(1.0 if moment == 0 else 0.0)
		else:
			return (np.ones(shape, dtype=dtype) if moment == 0
					else np.zeros(shape, dtype=dtype))
	else:
		# Exponentiation by squares: form exponent sequence
		n_list = [moment]
		current_n = moment
		while current_n > 2:
			if current_n % 2:
				current_n = (current_n - 1) / 2
			else:
				current_n /= 2
			n_list.append(current_n)

		# Starting point for exponentiation by squares
		mean = (a.mean(axis, keepdims=True) if mean is None
				else dtype(mean))
		a_zero_mean = a - mean

		eps = np.finfo(a_zero_mean.dtype).resolution * 10
		with np.errstate(divide='ignore', invalid='ignore'):
			rel_diff = np.max(np.abs(a_zero_mean), axis=axis) / np.abs(mean)
		with np.errstate(invalid='ignore'):
			precision_loss = np.any(rel_diff < eps)
		n = a.shape[axis] if axis is not None else a.size
		if precision_loss and n > 1:
			message = ("Precision loss occurred in moment calculation due to "
					   "catastrophic cancellation. This occurs when the data "
					   "are nearly identical. Results may be unreliable.")
			warnings.warn(message, RuntimeWarning, stacklevel=4)

		if n_list[-1] == 1:
			s = a_zero_mean.copy()
		else:
			s = a_zero_mean**2

		# Perform multiplications
		for n in n_list[-2::-1]:
			s = s**2
			if n % 2:
				s *= a_zero_mean
		return np.mean(s, axis)