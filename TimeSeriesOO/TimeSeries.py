import pandas as pd
import numpy as np
from enum import Enum
from TimeSeriesOO.FrequencyEstimator import estimateFrequency

class FileType(Enum):
	CSV = 1
	PANDAS_DF = 2
	NUMPY_ARRAY = 3
	UNSPECIFIED = 4

class TimeSeries:

	def __init__(self, data = None, freq = None, csv_dir = None, col_ref = None, date_format = "%Y-%m-%d", file_type = FileType.UNSPECIFIED):

		self.data = data
		self.csv_dir = csv_dir
		self.col_ref = col_ref
		self.date_format = date_format
		self.file_type = file_type
		self.freq = freq

		self.checkFileInput()
		self.consumeData()


	def __str__(self):
		output = "TimeSeries(file_type = " + str(self.file_type.name) + ")"
		output += "\nNumber of Observations: " + str(self.data.shape[0])
		return output

	def __repr__(self):
		return self.__str__()

	def consumeData(self):

		if self.file_type == FileType.CSV:
			self.consumeTimeSeriesFromCSV(self.csv_dir, self.col_ref, self.date_format)

		elif self.file_type == FileType.NUMPY_ARRAY:
			self.consumeTimeSeriesFromNumpy()

	def canReadCSV(self):

		can_read = True

		can_read &= isinstance(self.csv_dir, str)
		can_read &= isinstance(self.col_ref, dict)

		return can_read

	def checkFileInput(self):

		if self.file_type is FileType.UNSPECIFIED:

			if isinstance(self.data, pd.DataFrame):
				self.file_type = FileType.PANDAS_DF
				return

			elif isinstance(self.data, np.ndarray):
				self.file_type = FileType.NUMPY_ARRAY
				return

			elif self.canReadCSV():
				self.file_type = FileType.CSV
				return

			else:
				raise ValueError("Unclear what data source to use. Check constructor inputs.")


	def consumeTimeSeriesFromCSV(self, location, col_ref, date_format):
		self.col_ref = col_ref

		local = pd.read_csv(location, parse_dates=[col_ref['dates']], date_parser=lambda x: dateParser(x, date_format))

		local = local.sort_values(col_ref['dates'])

		local = local.dropna(subset=[col_ref['dates']])

		self.data = local

	def consumeTimeSeriesFromNumpy(self):
		self.data = pd.DataFrame({"values": self.data})
		self.col_ref = {'values': 'values'}

	def getMetrics(self):
		average_step = self.data[self.col_ref['dates']].diff()
		print("Average Step = " + str(average_step.median()) + "\nSD Step = " + str(average_step.std()), "\n")

	def plot(self):
		pass

	def getValues(self):
		return self.data[self.col_ref["values"]].values

	def getDates(self):
		return self.data[self.col_ref["dates"]].values

	def getFrequency(self):

		if self.freq is None:
			self.freq = estimateFrequency(self.getValues())

		return self.freq




def dateParser(x, date_format):

	try:
		x = pd.datetime.strptime(x, date_format)

	except ValueError:
		x = None

	return x


if __name__ == "__main__":

	column_references = {'dates': "Month", 'values': 'Air_Passengers'}
	date_format = "%Y-%m-%d"

	TS = TimeSeries(csv_dir="~/PycharmProjects/PyCast/AirPassengers.csv",col_ref= column_references, date_format = date_format)
	TS.getMetrics()
	print(TS)
	print(TS.getValues())

	TS.consumeTimeSeriesfromCSV("~/PycharmProjects/PyCast/AirPassengersMissingValues.csv", column_references, date_format)
	TS.getMetrics()