# from stl import STL
#
# class STLM:
#
#     def __init__(self,  s_window = 7, robust = False, method = ["ets", "arima"], model_funciton = None,
#                  model = None, etsmodel = "ZZN", lambda = NULL, biasadj = False, xreg = None,
#                  allow_multiplicative_trend = False):
#
#         pass
#
t = {"1": 3, "2": 2}
for i in sorted(t.iteritems(), key = lambda (k,v): (v,k)):
    print i