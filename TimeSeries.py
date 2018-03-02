import numpy as np
import pandas as pd
from scipy import interpolate as interp
from matplotlib import pyplot

class TimeSeries:

	def __init__(self, series = None, dates = None, column_ref = None,
				 interpolation_method = interp.PchipInterpolator):

		if isinstance(series, pd.DataFrame):

			if column_ref is None:
				raise Exception("column_ref must be given if series is a pandas DataFrame")

			self.getTimeSeriesFromPandasDF(series, column_ref)

		elif isinstance(series, np.ndarray):
			self.series = series

		elif isinstance(dates, np.ndarray):
			self.dates = dates

		self.checkTimeSeriesIsClean()


	def getTimeSeriesFromPandasDF(self, data_frame, column_ref):

		self.series = data_frame.as_matrix([column_ref['series']])
		self.dates = data_frame.as_matrix([column_ref['date_index']])

		self.n = data_frame.shape[0]

	def print_details(self):

		output = "Number of observations: " + str(self.n) + "\n"
		output += "Number of NaN values: " + str(self.total_missing) + "\n"
		output += "Cleaned time series: " + str(self.clean) + "\n"

		print output

	def checkTimeSeriesIsClean(self):

		self.total_missing = np.isnan(self.series).sum()
		self.clean = self.total_missing == 0

	# TODO complete interpolation structure
	def cleanSeries(self):

		self.clean = True

	def plotTimeSeries(self):
		self.plot = pyplot.figure(1)
		self.plot = pyplot.plot(self.dates, self.series)
		pyplot.show()


test_data = pd.read_csv("~/PycharmProjects/PyCast/AirPassengers.csv")
# print test_data
# print test_data["Month"][:]
# print test_data.as_matrix(["Month"])
reference = {'date_index': "Month", 'series': "Air_Passengers"}

test = TimeSeries(series = test_data, column_ref = reference)
test.print_details()
test.plotTimeSeries()

sparse_data = pd.read_csv("~/PycharmProjects/PyCast/AirPassengersMissingValues.csv")
test_sparse = TimeSeries(series = sparse_data, column_ref= reference)
test_sparse.print_details()
test_sparse.plotTimeSeries()


print type(test_sparse.dates[0][0])