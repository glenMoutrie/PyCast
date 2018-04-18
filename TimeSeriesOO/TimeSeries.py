import numpy as np
import pandas as pd
from scipy import interpolate as interp
from matplotlib import pyplot

class TimeSeries:

	def __init__(self, csv_loc = None, series = None, column_ref = None, date_format = None, source = "csv"
				 interpolation_method = interp.PchipInterpolator):

		get_from_csv = not csv_loc is None
		get_from_csv = get_from_csv and not column_references is None
		get_from_csv = get_from_csv and not date_format is None


		if source == "csv":

			if csv_loc is None:
				pass

			self.data = consumeTimeSeriesfromCSV(csv_loc,column_references,date_format)

		self.checkTimeSeriesIsClean()


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


def dateParser(x, date_format):

    try:
        x = pd.datetime.strptime(x, date_format)

    except ValueError:
        x = None

    return x



def consumeTimeSeriesfromCSV(location, col_ref, date_format):

    local = pd.read_csv(location, parse_dates= [col_ref['dates']], date_parser= lambda x: dateParser(x, date_format))

    local = local.sort_values(col_ref['dates'])

    local = local.dropna(subset = [col_ref['dates']])

    average_step = local[col_ref['dates']].diff()
    print "Average Step = " + str(average_step.median()) + " SD Step = " + str(average_step.std())

    return local


column_references = {'dates': "Month", 'values': 'Air_Passengers'}
date_format = "%Y-%m-%d"

print consumeTimeSeriesfromCSV("~/PycharmProjects/PyCast/AirPassengers.csv",  column_references, date_format)
print consumeTimeSeriesfromCSV("~/PycharmProjects/PyCast/AirPassengersMissingValues.csv",  column_references, date_format)


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