import numpy
from matplotlib import pyplot
from math import ceil
from stl_helpers import *

# double
# precision
# y(n), rw(n), season(n), trend(n),
# & work(n + 2 * np, 5)
#       n                   : length(y)
#       ns, nt, nl          : spans        for `s', `t' and `l' smoother
#       isdeg, itdeg, ildeg : local degree for `s', `t' and `l' smoother
#       nsjump,ntjump,nljump: ........     for `s', `t' and `l' smoother
#       ni, no              : number of inner and outer (robust) iterations

# This is the workhorse function that does the bulk of the work
def stlstp(y,n,np,ns,nt,nl,isdeg,itdeg,ildeg,nsjump,ntjump,nljump,ni,userw,rw,season,trend,work):


	# (For the below assume the timeseries goes from 1 to N
	# WORK BREAKDOWN
	# Work0: Low Pass filter of size N
	# Work1: Starts off as y, then is steadily detrended Y_v
	# Work2: Cycle subseries of size N + 2 * np. Time index is -np + 1 to N + np
	# work3: Stores the weights from the cycle subseries smoothing


	# Inner loop from Cleveland et al.
	for j in range(0,ni):

		# Detrend
		for i in range(0,n):
			work[1][i] = y[i] - trend[i]

		# Cycle subseries smoothing
		work[2], work[3], work[4], season = stlss(work[0], n, np, ns, isdeg, nsjump, userw, rw,
												   season, work[1], work[2], work[3], work[4])

		# Low-pass filter of smoothed cycle subseries
		# 1) Remove moving average components first using stlfts
		# 2) Removing an additional LOESS component with d = 1, q = n(l)
		work[2], work[0] = stlfts(work[1], n + 2 * np, np, work[2], work[0])
		work[0], work[4] = stless(work[2], n, nl, ildeg, nljump, False, work[3], work[0], work[4])

		# Detrending of smoothed cycle series
		for i in range(0,n):
			season[i] = work[1][np + i] - work[0][i]

		# Deseasonalise the time series
		for i in range(0,n):
			work[0][i] = y[i] - season[i]

		# Trend smoothing using LOESS q = n(t) d = 1
		trend, work[2] = stless(work[0], n, nt, itdeg, ntjump,userw,rw,trend,work[2])

	return season, trend, work


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

		k = (n - j)/np + 1

		for i in range(0,k):
			# Each i is one seasonal period
			# Work one stores the y value for each time period of the season across all seasons.
			# For example every y for september
			work1[i] = y[i * np + j]

		if userw:
			for i in range(0, k):
				work3[i] = rw[i * np + j - 1]

		work2, work4 = stless(work1, k, ns, isdeg, nsjump, userw, work3, work2, work4)

		xs = 0

		nright = min(ns, k)

		work2[0], work4 = stlest(work1, k, ns, isdeg, xs, work2[0], 1, nright, work4, userw, work3, ok)

		if not ok:
			work2[0] = work2[1]

		xs = k + 1

		nleft = max(0, k - ns)

		work2[k + 2], work4 = stlest(work1, k, ns, isdeg, xs, work2[k + 2], nleft, k, work4, userw, work3, ok)

		if not ok:
			work2[k + 2] = work2[k + 1]

		for m in range(0, k + 1):
			season[(m-1) * np + j - 1] = work2[m]

	return work2, work3, work4, season


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
	if n < 2:
		ys[0] = y[0]
		return

	newnj = int(min(njump, n - 1))

	# If the local area length is longer than the number of
	# observations
	if len >= n:
		nleft = 1
		nright = n

		# Estimate parameters and fit for the full range of weights with a full span
		for i in range(0,n,newnj):
			ys[i], res = stlest(y, n, len, ideg,i, ys[i], nleft, nright, res, userw, rw, ok)
			if not ok:
				ys[i] = y[i]
	else:

		if newnj == 1:
			nsh = (len + 1)*0.5
			nleft = 0
			nright = len

			for i in range(0,n):
				if i > nsh and not nright == n:
					nleft += 1
					nright += 1

				ys[i], res = stlest(y, n, len, ideg, i, ys[i], nleft, nright, res, userw, rw, ok)
				if not ok:
					ys[i] = y[i]

		else:
			nsh = (len + 1) * 0.5

			for i in range(0, n, int(newnj)):
				if i < nsh:
					nleft = 0
					nright = len
				elif i >= n - nsh + 1:
					nleft = n - len + 1
					nright = n
				else:
					nleft = i - nsh + 1
					nright = len + i - nsh

				ys[i], res = stlest(y, n, len, ideg, i, ys[i], nleft, nright, res, userw, rw, ok)

				if not ok:
					ys[i] = y[i]


	if not newnj == 1:
		for i in range(0, n - newnj, newnj):
			delta = (ys[i + newnj] - ys[i])/newnj

			for j in range(i, i + newnj):
				ys[j] = ys[i] + delta * (j - i)

		k = ((n - 1)/newnj)*newnj

		if not k == n:
			ys[i], res = stlest(y, n, len, ideg, n, ys[i], nleft, nright, res, userw, rw, ok)

			if not ok:
				ys[n - 1] = y[n - 1]

			if not k == n - 1:
				delta = (ys[n - 1] - ys[k])/(n - k)

				for j in range(k, n):
					ys[j] = ys[k] + delta * (j - k)

	return ys, res

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
# Call from stless
# stlest(y, n, len, ideg,i, ys[i], nleft, nright, res, userw, rw, ok)
def stlest(y, n, len, ideg, xs, ys, nleft, nright, w, userw, rw, ok):

	nleft = int(nleft) - 1
	nright = int(nright)
	y_range = n - 1.

	# Referenced in paper as lambda_q(x), can be achievd using indexes
	h = max(xs - nleft, nright - xs)

	if len > n:
		h += ((len - 2)/2.)

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

	# I believe that this is where the LOESS fit is actually estimated.
	if ok:
		for j in range(nleft, nright):
			w[j] = w[j]/a

		if h > 0. and ideg > 0.:
			a = 0.

			for j in range(nleft, nright):
				a += w[j] * j

			b = xs - a
			c = 0.

			for j in range(nleft, nright):
				c += w[j]*(j - a)**2

			if c ** 0.5 > 0.001 * y_range:
				b = b/c

				for j in range(nleft, nright):
					w[j] = w[j] * (b * (j - a) + 1)

			ys = 0.
			for j in range(nleft, nright):
				ys += w[j] * y[j]
	return ys, w


def stlfts(x, n, np, trend, work):
	trend = stlma(x, n, np, trend)
	work = stlma(trend, n-np + 1, np , work)
	trend = stlma(work, n-(2*np)+2, 3, trend)
	return trend, work



# Calculate the robustness weights, uses psort
# Robustness Weights
#       rw_i := B( |y_i - fit_i| / (6 M) ),   i = 1,2,...,n
#               where B(u) = (1 - u^2)^2  * 1[|u| < 1]   {Tukey's biweight}
#               and   M := median{ |y_i - fit_i| }
def stlrwt(y, n, fit, rw):

	rw = numpy.abs(y - fit)

	mid = [n/2 + 1, None]
	mid[1] = n - mid[0] + 1

	# TODO use psort instead of .sort
	# psort(y, n, mid, 2)
	rw = rw.copy()
	rw.sort()

	# 6((midpoint_1 + midpoint_2)/2)
	cmad = 3.*(rw[mid[0]] + rw[mid[1]])

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

print stlrwt(numpy.linspace(1,10,10),10,numpy.linspace(11,20,10),numpy.zeros(10))


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

	userw = True

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

		season, trend, work = stlstp(y, n, newnp, newns, newnt, newnl, isdeg, itdeg, ildeg, nsjump, ntjump, nljump, ni,
									 userw, rw, season, trend, work)

		# TODO run through and check that n + 1 or 1 is replaced with 0 is used where needed in for loops
		# as indexing in Fortran is one indexed...
		for i in range(0, n):
			work[0][i] = trend[i] + season[i]

		rw = stlrwt(y, n, work[0][0:n], rw)

		userw = True

	# Robustness weights when there were no robustness iterations
	if no <= 0:
		rw = numpy.linspace(1, 1, n)

	return season, trend, work, rw


def nextOdd(x):
	x = round(x)

	if x % 2 == 0:
		x += 1

	return x

def degCheck(deg):
	if deg < 0 or deg > 1:
		raise Exception('Degrees must be 0 or 1')

	return int(deg)

def STL(x, period, s_window = None, s_degree = 0, t_window = None, t_degree = 1, l_window = None, l_degree = None,
		s_jump = None, t_jump = None, l_jump = None, robust = False, inner = None, outer = None):

	n = x.shape[0]

	if s_window is None:
		periodic = True
		s_window = 10 * n + 1
		s_degree = 0

	if t_window is None:
		t_window = nextOdd(ceil((1.5 * period / (1- 1.5/s_window))))

	if l_window is None:
		l_window = nextOdd(period)

	# Add some smart defaults where needed
	if l_degree is None:
		l_degree = t_degree

	if s_jump is None:
		s_jump = ceil(s_window*1./10)

	if t_jump is None:
		t_jump = ceil((t_window*1./10))

	if l_jump is None:
		l_jump = ceil((l_window*1./10))

	# Smart defaults for inner and outer loop settings
	if inner is None:
		if robust:
			inner = 1
		else:
			inner = 2

	if outer is None:
		if robust:
			outer = 15
		else:
			# Has to be 1 not 0 as in R due to looping format in Fortran
			outer = 1

	if len(x.shape) > 1:
		raise Exception('x should be a one dimensional array')

	# TODO: create a proper function to find frequency, currently set by user
	# period = frequency(x)

	if period < 2 or n <= 2 * period:
		raise Exception('Series is not periodic or has less than two periods')

	s_degree = degCheck(s_degree)
	t_degree = degCheck(t_degree)
	l_degree = degCheck(l_degree)

	output = "\nSTL PARAMETERS:\n"
	output += "n: " + str(n) + "\t\t\t period: " + str(period) + "\n"
	output += "s_window: " + str(s_window) + "\t t_window: " + str(t_window) + "\t l_window: " + str(l_window) + "\n"
	output += "s_degree: " + str(s_degree) + "\t\t t_degree: " + str(t_degree) + "\t l_degree: " + str(l_degree) + "\n"
	output += "s_jump: " + str(s_jump) + "\t t_jump: " + str(t_jump) + "\t l_jump: " + str(l_jump) + "\n"
	output += "inner: " + str(inner) + "\t\t outer: " + str(outer) + "\n"

	print output

	season, trend, work, rw = stl(y = x, n = n,
			np = period, ns = s_window, nt= t_window, nl = l_window,
			isdeg = s_degree, itdeg= t_degree, ildeg = l_degree,
			nsjump = s_jump, ntjump = t_jump, nljump = l_jump,
			ni = inner, no = outer,
			season = numpy.zeros(n), trend = numpy.zeros(n), work = numpy.zeros((5,(n + 2 * period))))

	return season, trend, work




# stl(y, n, np, ns, nt, nl, isdeg, itdeg, ildeg,
# 		nsjump, ntjump, nljump, ni, no, season, trend, work)

y = numpy.array(
	[112, 118, 132, 129, 121, 135, 148, 148, 136, 119, 104, 118, 115, 126, 141, 135, 125, 149, 170, 170, 158, 133, 114,
	 140, 145, 150, 178, 163, 172, 178, 199, 199, 184, 162, 146, 166, 171, 180, 193, 181, 183, 218, 230, 242, 209, 191,
	 172, 194, 196, 196, 236, 235, 229, 243, 264, 272, 237, 211, 180, 201, 204, 188, 235, 227, 234, 264, 302, 293, 259,
	 229, 203, 229, 242, 233, 267, 269, 270, 315, 364, 347, 312, 274, 237, 278, 284, 277, 317, 313, 318, 374, 413, 405,
	 355, 306, 271, 306, 315, 301, 356, 348, 355, 422, 465, 467, 404, 347, 305, 336, 340, 318, 362, 348, 363, 435, 491,
	 505, 404, 359, 310, 337, 360, 342, 406, 396, 420, 472, 548, 559, 463, 407, 362, 405, 417, 391, 419, 461, 472, 535,
	 622, 606, 508, 461, 390, 432])

season, trend, work, rw = stl(y, 144, 12, 1441, 19, 13, 0, 1, 1, 145, 2, 2, 2, 0,
							  numpy.zeros(144), numpy.zeros(144), numpy.zeros((5,144 + 2 * 12)))

season, trend, work = STL(y, period = 12)

subplot_num = 411
# for i in [y, season, trend, y - season[0:144] - trend[0:144]]:
for i in [y, season, trend, y - season[0:144] - trend[0:144]]:
	pyplot.figure(1)
	pyplot.subplot(subplot_num)
	subplot_num += 1
	pyplot.plot(i)

work_size = work.shape[0]
for i in range(0,work_size):
	pyplot.figure(2)
	pyplot.subplot((work_size * 100) + 11 + i)
	pyplot.plot(work[i])

pyplot.show()