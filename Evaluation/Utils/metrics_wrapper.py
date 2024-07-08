import pandas as pd
from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.metrics import mean_squared_error

import numpy as np

class MetricsDataCreator:

	def __init__(self, metrics_names, algorithm_name, experiment_name, directory='./'):

		self.metrics_names = metrics_names
		self.algorithm_name = algorithm_name
		self.data = []
		self.directory = directory
		self.experiment_name = experiment_name
		self.base_df = None

	def register_step(self, run_num, step, metrics, algorithm_name = None):

		if algorithm_name is None:
			algorithm_name = self.algorithm_name

		""" Append the next step value of metrics """
		self.data.append([algorithm_name, run_num, step, *metrics])

	def register_experiment(self):

		df = pd.DataFrame(data = self.data, columns=['Algorithm', 'Run', 'Step', *self.metrics_names])

		if self.base_df is None:
			df.to_csv(self.directory + self.experiment_name + '.csv', sep = ',')
			return df
		else:
			self.base_df = pd.concat((self.base_df, df), ignore_index = True)
			self.base_df.to_csv(self.directory + self.experiment_name + '.csv', sep = ',')
			return self.base_df

	def load_df(self, path):

		self.base_df = pd.read_csv(path, sep=',')







