import numpy as np

def psort(a, n, ind, ni):

	if n < 0 or ni < 0:raise NamedError("n and ni must both be non-negative")

	if n < 2 or ni == 0: raise NamedError("n cannot be less than 2 and ni cannot be 0")

	jl = 1
	ju = ni
	indl = [1]
	indu = [ni]
	i = 1
	j = n
	m = 1

	# 161
	if (i < j) # go to 10

	# 166
	m = m - 1
	if (m == 0) return
	i = il[m]
	j = iu[m]
	jl = indl[m]
	ju = indu[m]
	if (!jl < ju) :
		# go to go to 166

	# 173
	if !(j - i > 10) # go to 174

	# 10
	k = i
	ij = (i + j)/2

	if (a[i] > t):
		a[ij] = a[i]
		a[i] = t
		t = a[ij]

	l = j
	
	if (a[j] < t):
		a[ij] = a[j]
		a[j] = t
		t = a[ij]

		if a[i] > t:
			a[ij] = a[i]
			a[i] = t
			t = a[ij]


	# 181