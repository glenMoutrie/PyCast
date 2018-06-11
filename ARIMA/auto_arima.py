from enum import Enum

from TimeSeriesOO.FrequencyEstimator import estimateFrequency

class informationCriteria(Enum):
    AICC = 1
    AIC = 2
    BIC = 3

class testProcedure(Enum):
    KPSS = 1
    ADF = 2
    PP = 3

class seasonalTest(Enum):
    SEAS = 1
    OCSB = 2
    HEGY = 3
    CH = 4

def approx(x):
    return len(x) > 150 or estimateFrequency(x) > 12

class autoArima:

    def __init__(self, x, d = None, D = None, max_p = 5, max_q = 5,
                 max_P = 2, max_Q = 2, max_order = 5, max_d = 2, max_D = 1,
                 start_p = 2, start_q = 2, start_P = 1, start_Q = 1,
                 stationary = False, seasonal = True, ic = informationCriteria.AICC,
                 stepwise = False, trace = False, approximation = approx,
                 truncate = None, xreg = None, test = testProcedure.KPSS,
                 seasonal_test = seasonalTest.SEAS,
                 allowdrift = True, allowmean = True, lmbd = None, biasadj = False,
                 parallel = False, num_cores = 2,):
        pass


if __name__ == "__main__":
    pass