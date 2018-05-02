from sklearn import linear_model
import numpy as np
from scipy import signal
import math

def detrend(x):

	linear_trend = np.array([range(1, x.shape[0] + 1)]).transpose()

	reg = linear_model.LinearRegression()
	reg.fit(linear_trend,x)

	residuals = x - reg.predict(linear_trend)

	return residuals

def which_max(x, first = True):

	x_max = max(x)
	max_index = [i for i, j in enumerate(x) if j == x_max]

	if first:
		return max_index[0]
	else:
		return max_index

def estimateFrequency(x):

	n = x.shape[0]

	# I suspect this value is set to 500 as arima can churn with a seasonal length
	# greater than 500 in R. Using 500 here may be sub-optimal, and may need to be
	# reconsidered.
	# TODO check validity of setting n_freq to 500
	n_freq = 500

	# TODO consider better detrending options
	detrended = detrend(x)

	f, pxx = signal.periodogram(detrended)

	# 10 is a arbitary threshold chosen by Rob Hyndaman through trial and error
	# This may not be applicable to a fft based approach
	# TODO test this and consider alternative bounds. May not be needed...
	if max(pxx) > 10:
		period = math.floor(1/f[which_max(pxx)] + 0.5)

		if math.isinf(period):
			j = [i for i, j in enumerate(pxx) if j > 0]

			if len(j) > 0:
				next_max = j[0] + which_max(pxx[range(j[0] + 1, n_freq)])

				if next_max < f.size:
					period = math.floor(1/f[next_max] + 0.5)

				else:
					period = 1

			else:
				period = 1

	else:
		period = 1

	return int(period)



if __name__ == '__main__':

	from matplotlib import pyplot

	y = np.array([112, 118, 132, 129, 121, 135, 148, 148, 136, 119, 104, 118,
					 115, 126, 141, 135, 125, 149, 170, 170, 158, 133, 114, 140,
					 145, 150, 178, 163, 172, 178, 199, 199, 184, 162, 146, 166,
					 171, 180, 193, 181, 183, 218, 230, 242, 209, 191, 172, 194,
					 196, 196, 236, 235, 229, 243, 264, 272, 237, 211, 180, 201,
					 204, 188, 235, 227, 234, 264, 302, 293, 259, 229, 203, 229,
					 242, 233, 267, 269, 270, 315, 364, 347, 312, 274, 237, 278,
					 284, 277, 317, 313, 318, 374, 413, 405, 355, 306, 271, 306,
					 315, 301, 356, 348, 355, 422, 465, 467, 404, 347, 305, 336,
					 340, 318, 362, 348, 363, 435, 491, 505, 404, 359, 310, 337,
					 360, 342, 406, 396, 420, 472, 548, 559, 463, 407, 362, 405,
					 417, 391, 419, 461, 472, 535, 622, 606, 508, 461, 390, 432])

	f, pxx = signal.periodogram(detrend(y))



	print(estimateFrequency(y))
	pyplot.plot(f, pxx)
	pyplot.show()