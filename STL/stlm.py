from STL.stl import STL
from enum import Enum



class STLMMethod(Enum):
	ETS = ets,
	ARIMA = arima



class STLM:

    def __init__(self, y,  s_window = 7, robust = False, method = ["ets", "arima"], model_funciton = None,
                 model = None, etsmodel = "ZZN", lambda = NULL, biasadj = False, xreg = None,
                 allow_multiplicative_trend = False):

        pass

