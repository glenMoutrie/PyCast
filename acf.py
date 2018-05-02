import numpy as np

def acfCalc(x, lag_max, type):
	
	# TODO manage na removal in a contiguous fashion
	# TODO manage frequency estimation

	if type == "partial":
		# Match R functionality of pacf
		print("")

	# Get the shape dimesions
	dim = x.shape
	d = dim[0]
	n = dim[1]

	lag_max = np.min([lag_max, n - 1L])

	if lag_max < 0:
		raise NameError("lag_max must be at least zero")

	if demean:
		mean = x.mean(axis = 1)
		x = x - A[:, np.newaxis]

	lag = np.ones((d, d))
	lag[np.tril_indices(d, k = -1)] = -1

	acf = acfCalc(x, lag_max, type == "correlation")

def acfCalc0(x, n, ns, nl, correlation) :
	# acf = np.array((ns,nl + 1,ns))
	acf = np.zeros((ns*(nl + 1)*ns))

	# Calculate the covariance over every combination of columns
	# For each lag
	# TODO make this less C more pythonic...
	d1 = nl + 1
	d2 = ns * d1
	for u in range(0, ns):
		for v in range(0, ns):
			for lag in range(0,nl + 1):
				sum = 0.
				nu = 0;
				for i in range(0, n - lag):
					nu += 1
					sum += x[i + lag + n*u] * x[i + n*v]

				acf[lag + (d1 * u) + (d2 * v)] = np.where(nu > 0 , sum/(nu + lag), None)


	if correlation:

		if n == 1:
			for u in range(0,ns):
				acf[0 + d1*u + d2*u] = 1.
		else:
			se = np.ones(ns)

			for u in range(0, ns):
				se[u] = np.sqrt(acf[0 + (d1*u) + (d2*u)])

			for u in range(0,ns):
				for v in range(0, ns):
					for lag in range(0,nl + 1):
						a = np.divide(acf[lag + d1*u + d2*v],(se[u]*se[v]))
						acf[lag + d1*u + d2*v] = np.where(a > 1., 1., np.where(a < -1., -1. ,a))

	return acf.reshape((ns, ns, nl + 1))



def acf(x, lag_max = None, type = ("correlation", "covariance", "partial"),
	plot = True, na_action = "na_contiguous", demean = True):
	
	type = type[1]

	dim = x.shape()
	n <- dim[0]
	d <- dim[1]

	if lag_max is None :
		# TODO create frequency method to replace hard coded one
		lag_max = np.max([np.floor(10 * (np.log10(n, 10) - np.log10(d))), 2*1])

	acf_out = afcCalc(x, lag_max, type)

	# TODO add plotting functionality


	return acf_out

if __name__ == "__main__":
	x = np.array((1, 2, 3, 4, 5, 6))
	print(acfCalc0(x, 3, 2, 2, False))
	print(acfCalc0(x, 3, 2, 2, True))
