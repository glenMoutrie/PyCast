import numpy

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
    while (i < j):
        # go to 10

        # 166 OUTER LOOP START
        while not jl < ju:

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


            ij = (i + j) / 2
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
                if k > l:
                    break
                # go to 183
                a[l] = a[k]
                a[k] = tt

            # goto 181

            # Now go through and store the indexes before
            # iterating around the outer loop again
            # 183 # continue
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
            else:
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
            i -= 1
            # 209 continue
            i += 1

            if i == j:
                pass
            # goto 166 RETURN TO OUTER LOOP
            t = a[i + 1]

            if a[i] > t:
                k = i
                # comment saying repeat
                # 216 continue
                a[k + 1] = a[k]
                k -= 1

                if not t >= a[k]:
                    # goto 216
                    # until t >= a[k]
                    a[k + 1] = t
                    # goto 209

    # goto 161
    # end outer loop
    return a


#  Simple function to calculate the moving average of a function
def stlma(x, n, len, ave):
    newn = n - len + 1
    flen = (len) * 1.
    v = 0.

    for i in range(0, len):
        v += x[i]

    ave[0] = v/flen

    if newn > 1:
        k = len - 1
        m = -1
        for j in range(1, newn):
            k += 1
            m += 1
            v = v - x[m] + x[k]
            ave[j] = v/flen

    return ave

test_x = numpy.linspace(1,10,10)
test_y = test_x.copy()

print test_x
print stlma(test_x, 10, 3, numpy.zeros(10))
print stlma(stlma(test_x, 10, 3, numpy.zeros(10)), 8, 3, numpy.zeros(10))