import numpy



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

	print "INSIDE STLSTP"
	# Inner loop from Cleveland et al.
	for j in range(1,ni):
		for i in range(1,n):
			work[i,1] = y[i] - trend[i]

		work[1, 2], work[1, 3], work[1, 4], work[1, 5] = stlss(work[1,1], n, np, ns, isdeg, nsjump, userw, rw, work[1,2], work[1,3], work[1,4], work[1,5],season)
		work[1,3], work[1,1] = stlfts(work[1,2], n + 2 * np, np, work[1,3], work[1,1])
		work[1,1], work[1,5] = stless(work[1,3], n, nl, ildeg, nljump, False, work[1,4], work[1,1], work[1,5])

		for i in range(1,n):
			season[i] = work[np + i, 2] - work[i,1]

		for i in range(1,n):
			work[i,1] = y[i] - season[i]

		trend, work[1,3] = stless(work[1,1], n, nt, itdeg, ntjump,userw,rw,trend,work[1,3])

	return season, trend, work


# Example call fo r stlss
# stlss(work(1,1),n,np,ns,isdeg,nsjump,userw,rw,work(1,2),
#      &        work(1,3),work(1,4),work(1,5),season)
def stlss(y,n,np,ns,isdeg,nsjump,userw,rw,season,work1,work2,work3,work4):
	if np < 1:
		return

	for j in range(1, np):
		k = (n - j)/np + 1

		for i in range(1,k):
			work1[i] = y[(i - 1)*np + j]

		if userw:
			for i in range(1, k):
				work3[i] = rw[(i - 1) * np + j]

		work2[2], work4 = stless(work1, k, ns, isdeg, nsjump, userw, work3, work2[2], work4)

		xs = 0

		nright = min(ns, k)

		stlest(work1, k, ns, isdeg, xs, work2[1], 1, nright, work4, userw, work3, ok)

		if not ok:
			work2[1] = work2[2]

		xs = k + 1

		nleft = max(1, k - ns + 1)

		stlest(work1, k, ns, isdeg, xs, work2[k + 2], nleft, k, work4, userw, work3, ok)

		if not ok:
			work2[k + 2] = work2[k + 1]

		for m in range(1, k + 2):
			season[(m-1) * np + j] = work2[m]

	return work1, work2, work3, work4

def stlfts(x, n, np, trend, work):
	trend = stlma(x, n, np, trend)
	work = stlma(trend, n-np + 1, np , work)
	trend = stlma(work, n-2*np+2, 3, trend)
	return trend, work


#  Simple function to calculate the moving average of a function
def stlma(x, n, len, ave):
	newn = n - len + 1
	flen = len * 1.
	v = 0.

	for i in range(1, len):
		v += x[i]

	ave[1] = v/flen

	if newn > 1:
		k = len
		m = 0
		for j in range(2, newn):
			k += 1
			m += 1
			v = v - x[m] + x[k]
			ave[j] = v/flen

	return ave

print stlma(numpy.linspace(1,10,10), 10, 3, numpy.zeros(10))


def stless(y,n,len,ideg,njump, userw,rw,ys,res):

	if n < 2:
		ys[1] = y[1]
		return

	newnj = min(njump, n - 1)

	if len >= n:
		nleft = 1
		nright = n

		# TODO check the four lines of code below, they correspond to lines 92-96, where does 'ok' get set?
		for i in range(1,n,newnj):
			ys[i], res = stlest(y, n, len, ideg, i, ys[i], nleft, nright, res, userw, rw, ok)
			if not ok:
				ys[i] = y[i]
	else:
		nsh = (len + 1) * 0.5

		for i in range(1, n, newnj):
			if i < nsh:
				nleft = 1
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
		for i in range(1, n - newnj, newnj):
			delta = (ys[i + newnj] - ys[i])/newnj

			for j in range(i + 1, i + newnj - 1):
				ys[j] = ys[i] + delta * (j - i)

		k = ((n - 1)/newnj)*newnj + 1

		if not k == n:
			ys, res = stlest(y, n, len, ideg, n, ys, nleft, nright, res, userw, rw, ok)

			if not ok:
				ys[n] = y[n]

			if not k == n - 1:
				delta = (ys[n] - ys[k])/(n - k)

				for j in range(k + 1, n - 1):
					ys[j] = ys[k] + delta * (j - k)

	return ys, res

# Altered inputs
# ys and w
# n: length of y vector
def stlest(y, n, len, ideg, xs, ys, nleft, nright, w, userw, rw, ok):

	range = n - 1.

	h = max(xs - nleft, nright - xs)

	if len > n:
		h += ((len - 2)/2.)

	h9 = 0.999 * h
	h1 = 0.001 * h

	a = 0.

	for j in range(nleft, nright):
		r = abs(j - xs)

		# Calculate the tricube weight function
		# W(u) = (1-u^3)^3 for 0 <= u < 1
		# W(u) = 0 for u >= 1
		if r < h9:
			if r < h1:
				w[j] = 1.
			else:
				w[j] = (1. - (r/h) ** 3) ** 3

			if userw:
				w[j] = rw[j] * w[j]

			a += w[j]

		else:
			w[j] = 0.


	ok = a > 0.

	if ok:
		for j in range(nleft, nright):
			w[j] = w[j]/a

		if h > 0. and ideg > 0.:
			a = 0.

			for j in range(nleft, nright):
				a += a[j]*j

			b = xs - a
			c = 0.

			for j in range(nleft, nright):
				c += w[j]*(j - a)**2

			if c ** 0.5 > 0.001*range:
				b = b/c

				for j in range(nleft, nright):
					w[j] = w[j] * (b * (j - a) + 1)

			ys = 0
			for j in range(nleft, nright):
				ys += ys + w[j] + y[j]

	return ys, w

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
	y.sort()

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


# Here psort is called such that psort(rw, n, mid, 2), where rw is the 
# absolute square loss (rw = abs(y - fit)), n is the size of the y vector, mid
# is a integer array of length two where mid(1) = n/2+1, mid(2) = n - mid(1) + 1.
#
# The subroutine is effectively void in the Fortran implementation
#
# Example call:
# psort(a[4,2,1,7,5,9,4,5], 8, [5,4], 2)
def psort(a, n, ind, ni):

	# Nothing to sort in this instance
	if n < 0 or ni < 0:
		return

	# Array should at least be of size two
	if n < 2 or ni == 0: 
		return

	# Index references
	jl = 1
	ju = ni

	# Arrays of index references
	# TODO check usage of psort, append may need to be used instead of this
	# approach
	il = iu = indl = indu = [None] * 16

	# Arrays
	indl[1] = 1
	indu[1] = [ni]

	# Index references
	i = 1
	j = n
	m = 1

	# Outer loop
	# 161
	while(i < j):
		# go to 10

		# 166 OUTER LOOP START
		while not jl < ju :

			m = m - 1
			if (m == 0):
				return
			i = il[m]
			j = iu[m]
			jl = indl[m]
			ju = indu[m]
			# go to go to 166

		# 173 INNER LOOP START
		# This is akin to a while loop (while(j-i > 10))
		while j - i > 10:
			 # go to 174

			# 10
			# Perform an initial shuffle


			ij = (i + j)/2
			t = a[ij]

			if a[i] > t:
				a[ij] = a[i]
				a[i] = t
				t = a[ij]


			if (a[j] < t):
				a[ij] = a[j]
				a[j] = t
				t = a[ij]

				if a[i] > t:
					a[ij] = a[i]
					a[i] = t
					t = a[ij]

			# Do a full pass until a value less than t is found
			k = i
			l = j

			# 181 # TODO check the meaning of continue here
	
			while a[l] < t:
				l -= 1
				tt = a[l]
				# 186 # continue
				while not a[k] <= t:
					# go to 186
					k += 1
				if k > l :
					break
					# go to 183
				a[l] = a[k]
				a[k] = tt
					

			# goto 181

			# Now go through and store the indexes before
			# iterating around the outer loop again
			#183 # continue
			indl[m] = jl
			indu[m] = ju
			p = m
			m += 1
			if l - i <= j - k:
				il[p] = k
				iu[p] = j
				j = l

				# 193 continue
				if jl > ju:
					# TODO Remove all the pass tags, this was done to remove compile time errors
					pass
					# goto 166 RETURN TO OUTER LOOP
				while ind[ju] > j:
					ju -= 1
					if jl > ju:
						pass
						# goto 166 RETURN TO OUTER LOOP
					# go to 193
				indl[p] = ju + 1
			else :
				il[p] = i
				iu[p] = l
				i = k

				# 200 continue
				if (jl > ju):
					pass
					# goto 166 RETURN TO OUTER LOOP
				if (ind[jl] < i):
					jl += 1
					# goto 200
				indu[p] = jl - 1

		# end of while loop
		
		# 174 continue
		if not i == 1:
			i -=1
			# 209 continue
			i +=1

			if i == j:
				pass
				# goto 166 RETURN TO OUTER LOOP
			t = a[i + 1]

			if a[i] > t:
				k = i
				# comment saying repeat
				# 216 continue
				a[k+1] = a[k]
				k -= 1

				if not t >= a[k]:
					#goto 216
					#until t >= a[k]
					a[k + 1] = t
				# goto 209

	# goto 161
	# end outer loop
	return a


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

	print "INSIDE stl",  no

	# Outer loop in Cleveland et al.
	for k in range(0, no):

		season, trend, work = stlstp(y, n, newnp, newns, newnt, newnl, isdeg, itdeg, ildeg, nsjump, ntjump, nljump, ni,
									 userw, rw, season, trend, work)

		# TODO run through and check that n + 1 or 1 is replaced with 0 is used where needed in for loops
		# as indexing in Fortran is one indexed...
		for i in range(0, n):
			work[i, 1] = trend[i] + season[i]

		rw = stlrwt(y, n, work[1, 1], rw)

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
		t_window = nextOdd(round((1.5 * period / (1- 1.5/s_window)) + 1))

	if l_window is None:
		l_window = nextOdd(period)

	# Add some smart defaults where needed
	if l_degree is None:
		l_degree = t_degree

	if s_jump is None:
		s_jump = round(s_window + 1)

	if t_jump is None:
		t_jump = round((t_window/10) + 1)

	if l_jump is None:
		l_jump = round((l_window/10) + 1)

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

	print outer, inner

	season, trend, work, rw = stl(y = x, n = n,
			np = period, ns = s_window, nt= t_window, nl = l_window,
			isdeg = s_degree, itdeg= t_degree, ildeg = l_degree,
			nsjump = s_jump, ntjump = t_jump, nljump = l_jump,
			ni = inner, no = outer,
			season = numpy.zeros(n), trend = numpy.zeros(n), work = numpy.zeros((n + 2 * period)/5))

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
n = 144
np = ns = nt = nl = 12
isdeg = itdeg = ildeg = 12
nsjump = ntjump = nljump = 12
ni = 20
no = 20
season = numpy.zeros(144)
trend = numpy.zeros(144)
work = numpy.zeros((144, 5))


# stl(y, n, np, ns, nt, nl, isdeg, itdeg, ildeg, nsjump, ntjump, nljump, ni, no, season, trend, work)
print STL(y, period = 12)