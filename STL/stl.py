import numpy
from matplotlib import pyplot

from math import ceil

from stl_helpers import *
from TimeSeriesOO.TimeSeries import TimeSeries

# double
# precision
# y(n), rw(n), season(n), trend(n),
# & work(n + 2 * np, 5)
#       n                   : length(y)
#       ns, nt, nl          : spans        for `s', `t' and `l' smoother
#       isdeg, itdeg, ildeg : local degree for `s', `t' and `l' smoother
#       nsjump,ntjump,nljump: ........     for `s', `t' and `l' smoother
#       ni, no              : number of inner and outer (robust) iterations

# R PARAMETERS (REMEMBER THE INDEXING IS 1 BASED)
# STL PARAMETERS:
# n:  144 		 period:  12
# s.window:  1441 	 t.window:  19 	 l.window:  13
# s.degree:  0 		 t.degree:  1 	 l.degree:  1
# s.jump:  145 		 t.jump:  2 	 l.jump:  2
# inner:  2 		 outer:  0

class STL:

	def __init__(self, x, period = None, s_window = None, s_degree = 0,t_window = None, t_degree = 1, l_window = None, l_degree = None,
		s_jump = None, t_jump = None, l_jump = None, robust = False, inner = None, outer = None, verbose = False):

		# Set the parameters
		if isinstance(x, TimeSeries):
			self.x = x.getValues()
			self.period = x.getFrequency()
		else:

			if period == None:
				raise Exception("If x is not a TimeSeries object then a period value must be provided")

			self.x = x
			self.period = period

		self.n = self.x.shape[0]
		self.s_window = s_window
		self.s_degree = s_degree
		self.t_window = t_window
		self.t_degree = t_degree
		self.l_window = l_window
		self.l_degree = l_degree
		self.s_jump = s_jump
		self.t_jump = t_jump
		self.l_jump = l_jump
		self.robust = robust
		self.inner = inner
		self.outer = outer
		self.estimated = False

		self.robust = robust


		if self.s_window is None:
			self.periodic = True
			self.s_window = 10 * self.n  # + 1
			self.s_degree = 0

		if self.t_window is None:
			self.t_window = nextOdd(ceil((1.5 * self.period / (1 - 1.5 / self.s_window))))

		if self.l_window is None:
			self.l_window = nextOdd(self.period)

		# Add some smart defaults where needed
		if self.l_degree is None:
			self.l_degree = self.t_degree

		if self.s_jump is None:
			self.s_jump = ceil(self.s_window * 1. / 10)

		if self.t_jump is None:
			self.t_jump = ceil((self.t_window * 1. / 10))

		if self.l_jump is None:
			self.l_jump = ceil((self.l_window * 1. / 10))

		# Smart defaults for inner and outer loop settings
		if self.inner is None:
			if self.robust:
				self.inner = 1
			else:
				self.inner = 2

		if self.outer is None:
			if self.robust:
				self.outer = 15
			else:
				# Has to be 1 not 0 as in R due to looping format in Fortran
				self.outer = 1

		if len(self.x.shape) > 1:
			raise Exception('x should be a one dimensional array')

		if self.period < 2 or self.n <= 2 * self.period:
			raise Exception('Series is not periodic or has less than two periods')

		self.s_degree = degCheck(self.s_degree)
		self.t_degree = degCheck(self.t_degree)
		self.l_degree = degCheck(self.l_degree)

		if verbose:
			self.print_parameter_values()

		self.season = numpy.zeros(self.n)
		self.trend = numpy.zeros(self.n)
		self.work = numpy.zeros((5, (self.n + 2 * self.period)))



	def fit(self):
		self.season, self.trend, self.work = stl(y=self.x, n=self.n,
												 np=self.period, ns=self.s_window, nt=self.t_window, nl=self.l_window,
												 isdeg=self.s_degree, itdeg=self.t_degree, ildeg=self.l_degree,
												 nsjump=self.s_jump, ntjump=self.t_jump, nljump=self.l_jump,
												 ni=self.inner, no=self.outer,
												 season=self.season, trend=self.trend,
												 work=self.work)

		self.estimated = True


	def __repr__(self):

		return "STL(period %s, s_window = %s)" % (str(self.period), str(self.s_window))


	def __str__(self):

		output = "\nSTL PARAMETERS:\n"
		output += "n: " + str(self.n) + "\t\t\t period: " + str(self.period) + "\n"

		output += "s_window: " + str(self.s_window) + "\t t_window: " + str(self.t_window)
		output += "\t l_window: " + str(self.l_window) + "\n"

		output += "s_degree: " + str(self.s_degree) + "\t\t t_degree: "
		output += str(self.t_degree) + "\t l_degree: " + str(self.l_degree) + "\n"

		output += "s_jump: " + str(self.s_jump) + "\t t_jump: " + str(self.t_jump)
		output += "\t l_jump: " + str(self.l_jump) + "\n"

		output += "inner: " + str(self.inner) + "\t\t outer: " + str(self.outer) + "\n"

		return output

	def generate_plot(self):

		if not self.estimated:
			raise Exception("STL components must be estimated before plotting can be executed.")

		subplot_num = 411

		for i in [self.x, self.season, self.trend, self.x - self.season - self.trend]:
			self.decomp_plot = pyplot.figure(1)
			self.decomp_plot = pyplot.subplot(subplot_num)
			subplot_num += 1
			self.decomp_plot = pyplot.plot(i)

# This is the workhorse function that does the bulk of the work
def stlstp(y,n,np,ns,nt,nl,isdeg,itdeg,ildeg,nsjump,ntjump,nljump,ni,userw,rw,season,trend,work):


	# (For the below assume the timeseries goes from 1 to N
	# WORK BREAKDOWN
	# Work0: Low Pass filter of size N
	# Work1: Starts off as y, then is steadily detrended Y_v
	# Work2: Cycle subseries of size N + 2 * np. Time index is -np + 1 to N + np
	# work3: Stores the weights from the cycle subseries smoothing
	# Work4

	# Inner loop from Cleveland et al.
	for j in range(0,ni):

		# Detrend
		for i in range(0,n):
			work[0][i] = y[i] - trend[i]

		# Cycle subseries smoothing
		# Fortran funciton
		# stlss(y = work[0], n = n, np = np, ns = ns, isdeg = isdeg, nsjump = nsjump, userw = userw, rw = rw, season = work[1],
		# & work1, work2, work3, work4)
		# season, work[2], work[3], work[4], work[1] = stlss(y = work[0], n = n, np = np, ns = ns,
		# 												   isdeg = isdeg, nsjump = nsjump, userw = userw,
		# 												   rw = rw, season = work[1],
		# 												   work1 = work[2], work2 = work[3],
		# 												   work3 = work[4], work4 = season)
		stlss(y=work[0], n=n, np=np, ns=ns,
			  isdeg = isdeg, nsjump = nsjump, userw = userw,
			  rw = rw, season = work[1],
			  work1 = work[2], work2 = work[3],
			  work3 = work[4], work4 = season)
		# Low-pass filter of smoothed cycle subseries
		# 1) Remove moving average components first using stlfts
		# 2) Removing an additional LOESS component with d = 1, q = n(l)
		stlfts(x = work[1],n = n + 2 * np, np = np,
			   trend = work[2], work = work[0])

		stless(y = work[2], n = n, len = nl, ideg = ildeg,
								  njump = nljump, userw =  False,
								  rw = work[3], ys = work[0], res = work[4])

		# Detrending of smoothed cycle series
		for i in range(0,n):
			season[i] = work[1][np + i - 1] - work[0][i]

		# Deseasonalise the time series
		for i in range(0,n - 1):
			work[0][i] = y[i] - season[i]

		# Trend smoothing using LOESS q = n(t) d = 1
		stless(y = work[0], n = n, len = nt,
			   ideg = itdeg, njump = ntjump,
			   userw = userw, rw = rw,
			   ys = trend, res = work[2])

	return season, trend, work


# Low-pass filter of smoothed cycle subseries
# This function takes three moving averages, as stipulated in the paper. This is using
# work zero, the detrended series.
def stlfts(x, n, np, trend, work):

	stlma(x, n, np, trend)
	stlma(trend, n-np, np , work)
	stlma(work, n-2*np, 3, trend)

	return trend, work

# Call from stlss:
# stless(work1, k, ns, isdeg, nsjump, userw, work3, work2, work4)

# y: object to estimate loess
# n: size of object
# len: length of local area
# ideg: number of degrees for fit, either 1 or 2
# njump:
# rw: weights
# ys: updated fit
def stless(y,n,len,ideg,njump, userw,rw,ys,res,ok = False):

	# If only two observations set work2 to be the averages
	if n < 1:
		ys[0] = y[0]
		return

	newnj = int(min(njump, n - 2))

	# If the local area length is longer than the number of
	# observations
	if len >= n:
		nleft = 0
		nright = n - 1

		# Estimate parameters and fit for the full range of weights with a full span
		for i in range(0,n,newnj):
			ys[i], ok = stlest(y, n, len, ideg,i, ys[i],
				   nleft, nright, res, userw,
				   rw, ok)

			if not ok:
				ys[i] = y[i]
	else:

		if newnj == 1:
			nsh = int((len + 1)*0.5)
			nleft = 0
			nright = len - 1

			# Estimate all parameters between zero and N using a seasons width
			for i in range(0,n):
				if i > nsh and nright < n - 1:
					nleft += 1
					nright += 1

				ys[i], ok = stlest(y, n, len, ideg, i, ys[i], nleft, nright, res, userw, rw, ok)
				if not ok:
					ys[i] = y[i]

		else:


			nsh = (len) * 0.5

			# Only estimate parameters with a gap of either njump or n - 2 (largest possible)
			for i in range(0, n, int(newnj)):

				#  Code to ensure the nleft and nright is within each seasonal period ie a year
				# handling the first and last seasons being the size of half a season.
				if i < nsh:
					nleft = 0
					nright = len - 1
				elif i >= n - nsh:
					nleft = n - len
					nright = n - 1
				else:
					nleft = i - nsh
					nright = len + i - nsh - 1

				# stlest(y, n, len, ideg, xs, ys, nleft, nright, w, userw, rw, ok)
				ys[i], ok = stlest(y = y, n = n, len = len, ideg = ideg, xs = i, ys = ys[i],
					   nleft = nleft, nright = nright, w = res,
					   userw = userw, rw = rw, ok = ok)

				if not ok:
					ys[i] = y[i]


	if not newnj == 1:
		for i in range(0, n - newnj, newnj):
			delta = (ys[i + newnj] - ys[i])/newnj

			# TODO this works for now, but doesn't match up with what is needed in the code
			for j in range(i + 1, i + newnj * 2):
				ys[j] = ys[i] + delta * (j - i)

		k = int((n - 1)/newnj) * newnj + 1

		if not k == n - 1:
			ys[i], ok = stlest(y = y, n = n, len = len, ideg = ideg, xs = n, ys = ys[i],
				   nleft = nleft, nright = nright, w = res, userw = userw,
				   rw = rw, ok = ok)

			if not ok:
				ys[n - 1] = y[n - 1]

			if not k == n - 2:
				delta = (ys[n - 1] - ys[k])/(n - k)

				for j in range(k + 1, n - 1):
					ys[j] = ys[k] + delta * (j - k)

	return ys, res

# CYCLE-SUBSERIES SMOOTHING
#
# Example call for stlss
# stlss(work(1,1),n,np,ns,isdeg,nsjump,userw,rw,work(1,2),
#      &        work(1,3),work(1,4),work(1,5),season)
# stlss(work[0], n, np, ns, isdeg, nsjump, userw, rw,
# 												   work[1], work[2], work[3], work[4],season)
#
# y: time series
# n: time series lenght
# np: season
# ns: seasonal span or s_window default is n * 10 + 1
# isdeg: seasonal smoother
# nsjump:
# rw:
#
# OK is not set in Fortran code explicitly, but a boolean value automatically takes on a value of false
def stlss(y,n,np,ns,isdeg,nsjump,userw,rw,season,work1,work2,work3,work4, ok = False):

	if np < 1:
		return

	# Loop over each time period of the season. For example, if the data is monthly and the seasonal length (np)
	# is 12 then this is effectively looping over each month
	for j in range(0, np):

		# Number of occurances of this part of the season throughout the entire series
		k = int((n - j)/np + 1)

		for i in range(0,k):
			# Each i is one seasonal period
			# Work one stores the y value for each time period of the season across all seasons.
			# For example every y for september
			work1[i] = y[i * np + j]

		if userw:
			for i in range(0, k):
				work3[i] = rw[i * np + j]

		# stless(y, n, len, ideg, njump, userw, rw, ys, res, ok=False):
		# Call from R version only passes the second index.... this is odd....
		# call stless(work1,k,ns,isdeg,nsjump,userw,work3,work2(2),work4)
		stless(y = work1, n = k, len = ns,
							  ideg = isdeg, njump = nsjump, userw = userw,
							  rw = work3, ys = work2[1:], res = work4)

		xs = 0

		# This will almost always be k, which is larger than n itslf (n * 10 + 1)
		nright = min(ns, k)

		work2[0], ok = stlest(y = work1, n = k, len = ns, ideg = isdeg, xs = xs, ys = work2[0],
								 nleft = 0, nright = nright, w = work4, userw = userw, rw = work3, ok = ok)

		if not ok:
			work2[0] = work2[1]

		xs = k + 1

		nleft = max(0, k - ns - 1)

		work2[k], ok = stlest(y = work1, n = k, len = ns, ideg = isdeg, xs = xs, ys = work2[k],
									 nleft = nleft, nright = k, w = work4, userw = userw, rw = work3, ok = ok)

		if not ok:
			work2[k] = work2[k - 1]

		# for j in range(np):
		# 	print [i for i in map(lambda m: (m*np) + j, range(0, k ))]


		for m in range(0, k + 2):
			season[(m * np) + j - 1] = work2[m]

	# return work1, work2, work3, work4, season


# WEIGHT CALCULATOR AND LOESS ESTIMATOR
#
# Only does this for a single point, given the parameters of how far the
# next largest values are. Some heavy calculations need to be made first for the indexing.
#
# Altered inputs
# ys and w
# n: length of y vector
# len: length of season period
# ideg: number of degrees
# nleft, nright; number of observations to the left and right to consider
# w: weights
#
# The indexes put in SHOULD BE ZERO indexed not 1 indexed. The calculatins have been altered accordingly.
#
# Call from stless
# stlest(y, n, len, ideg,i, ys[i], nleft, nright, res, userw, rw, ok)
def stlest(y, n, len, ideg, xs, ys, nleft, nright, w, userw, rw, ok):

	# I think these may need to be put one forward due to zero indexing...
	# TODO: validate this assumption
	nleft = int(nleft)
	nright = int(nright)
	y_range = n - 1.

	# Referenced in paper as lambda_q(x), can be achievd using indexes
	h = (max(xs - nleft, nright - xs)) * 1.0

	if len > n:
		h += ((len - n)/2.)

	h9 = 0.999 * h
	h1 = 0.001 * h

	a = 0.0


	# Calculate the local weights for each point around the focal area
	for j in range(nleft, nright):

		# Referenced in paper as |x_i - x|
		r = abs(j - xs)

		# Calculate the tricube weight function
		# W(u) = (1-u^3)^3 for 0 <= u < 1
		# W(u) = 0 for u >= 1
		if r <= h9:
			if r <= h1:
				w[j] = 1.0
			else:
				w[j] = (1.0 - (r/h) ** 3) ** 3

			if userw:
				w[j] = rw[j] * w[j]

			a += w[j]

		else:
			w[j] = 0.


	# Only proceed if the total weights are greater than zero
	ok = a > 0.

	# This is where the loess is fitted
	if ok:

		# Divide the weights by their total sum
		for j in range(nleft, nright):
			w[j] = w[j]/a

		# If a polynomial fit is needed
		if h > 0. and ideg > 0.:
			a = 0.

			for j in range(nleft, nright):
				# Changed from source, adding one to j due to zero indexing
				a += w[j] * (j + 1)

			b = xs - a +1
			c = 0.

			for j in range(nleft, nright):
				c += w[j] * (j - a + 1)**2

			if c ** 0.5 > 0.001 * y_range:
				b = b/c

				for j in range(nleft, nright):
					w[j] = w[j] * (b * (j - a + 1) + 1)

		ys = 0.
		for j in range(nleft, nright):
			ys += w[j] * y[j]
	return ys, ok

# Calculate the robustness weights, uses psort
# Robustness Weights
#       rw_i := B( |y_i - fit_i| / (6 M) ),   i = 1,2,...,n
#               where B(u) = (1 - u^2)^2  * 1[|u| < 1]   {Tukey's biweight}
#               and   M := median{ |y_i - fit_i| }
def stlrwt(y, n, fit, rw):

	rw = numpy.abs(y - fit)

	mid = [int(n/2) + 1, None]
	mid[1] = n - mid[0] + 1

	# TODO use psort instead of .sort
	# psort(y, n, mid, 2)
	rw.sort()

	# 6((midpoint_1 + midpoint_2)/2)
	cmad = 3*(rw[mid[0]] + rw[mid[1]])

	c_9 = 0.999*cmad
	c_1 = 0.001*cmad

	for i in range(n):
		r = numpy.abs(y[i] - fit[i])

		if r < c_1:
			rw[i] = 1.
		elif r < c_9:
			rw[i] = (1. - (r/cmad)**2)**2
		else:
			rw[i] = 0.

	return rw

# Potential unit test for stlrwt
# print stlrwt(numpy.linspace(1,10,10),10,numpy.linspace(11,20,10),numpy.zeros(10))


# STL converted from R stats procedure in FORTRAN
#
# The aim of this piece of work is to understand the inner workings of STL, and be able
# to replicate the functionality of stl in Python without having to reference the R Stats
# package. Further more it may be possible to get speed and scalability gains, as the R
# version does not use any paralellisaiton or vectorisation.
#
# Cleveland et al.: STL Seasonal-Trend Decomposition Procedure based on Loess


def stl(y, n, np, ns, nt, nl, isdeg, itdeg, ildeg,
		nsjump, ntjump, nljump, ni, no, season, trend, work):

	rw = numpy.zeros(n)

	userw = False

	trend = numpy.zeros(n)

	newns = max(3, ns)
	newnt = max(3, nt)
	newnl = max(3, nl)

	if newns % 2 == 0: newns += 1
	if newnt % 2 == 0: newnt += 1
	if newnl % 2 == 0: newnl += 1

	newnp = max(2, np)


	# Outer loop in Cleveland et al.
	for k in range(0, no):

		# stlstp(y, n, np, ns, nt, nl, isdeg, itdeg, ildeg, nsjump, ntjump, nljump, ni, userw, rw, season, trend, work):
		stlstp(y = y, n = n, np = newnp, ns = newns, nt = newnt, nl = newnl,
			   isdeg = isdeg, itdeg = itdeg, ildeg = ildeg,
			   nsjump = nsjump, ntjump =  ntjump, nljump =  nljump, ni = ni,
			   userw = userw, rw = rw, season = season, trend = trend, work = work)

		# TODO run through and check that n + 1 or 1 is replaced with 0 is used where needed in for loops
		# as indexing in Fortran is one indexed...
		for i in range(0, n):
			work[0][i] = trend[i] + season[i]

		stlrwt(y, n, work[0][0:n], rw)

		userw = True

	# Robustness weights when there were no robustness iterations
	if no <= 1:
		rw = numpy.linspace(1, 1, n)

	return season, trend, work


################
# Testing area #
################

if __name__ == '__main__':
	y = numpy.array([112, 118, 132, 129, 121, 135, 148, 148, 136, 119, 104, 118,
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

	season_decomp = STL(x = y, period = 12)

	season_decomp_TS = STL(TimeSeries(y))

	print(season_decomp_TS)

	season_decomp.fit()
	season_decomp_TS.fit()

	season_decomp_TS.generate_plot()
	pyplot.show()

