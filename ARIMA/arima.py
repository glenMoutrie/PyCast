from statsmodels.tsa.statespace.sarimax import SARIMAX
from TimeSeriesOO.TimeSeries import TimeSeries

import numpy as np
from matplotlib import pyplot

class ARIMA:

    def __init__(self, x : TimeSeries, order = (0,0,0), seasonal = (0,0,0), constant = True, ic = "aic",
                 trace = False, approximation = False,  offset = 0, xreg = None):

        self.n = x.getSeriesLength()
        self.m = x.getFrequency()

        self.use_season = sum(seasonal) > 0 and self.m > 0

        self.diffs = order[1] + seasonal[2]

        if xreg is None:
            xreg = np.ndarray(None)

        elif not isinstance(xreg, np.ndarray):
            raise Error("xreg should be a numpy ndarray")

        # Inherited from R fit options in stats::arima
        # TODO: look for equivalent in ARIMA from stats models or depricate
        # if approximation:
        #     self.method = "CSS"
        # else:
        #     self.method = "CSS-ML"

        if self.diffs == 1 and constant:
            pass





if __name__ == "__main__":

    column_references = {'dates': "Month", 'values': 'Air_Passengers'}
    date_format = "%Y-%m-%d"

    ts = TimeSeries(csv_dir="~/PycharmProjects/PyCast/AirPassengers.csv",col_ref= column_references,
               date_format = date_format)

    print(ts)

    model = SARIMAX(ts.getValues(), order = (1,1,1),
                    seasonal_order= (1,1,1,ts.getFrequency()))

    results = model.fit()

    print(results.predict(start = 144, end = 156, dynamic = True))