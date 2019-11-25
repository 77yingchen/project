import numpy as np
import csv
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
from sklearn.svm import SVR
from keras import models
from keras import layers

from scipy.stats import pearsonr
from sklearn.metrics import r2_score
import random

np.random.seed(0)
random.seed(0)

color_set = ['brown', 'black', 'blue', 'red', 'grey', 'white', 'gold', 'rose', 'pink', 'silver']

all_item = []

with open('itemSet3.txt') as file:
	for l in file:
		if (l != '\n'):
			all_item.append(l.strip())

data = pd.read_csv('final_output.csv')
groupedData = data.groupby('bundle')

x_train = []
x_test = []
y_train = []
y_test = []


groupedData = list(groupedData)

for i in range(0, len(groupedData)):
# for i in range(0, 20):
	print i
	# 253 item + 10 color + price mean, sum, std + likes mean, sum, std + outfit_count mean, sum, std = 263+9 numbers
	bundle_feature = [0]*263
	bundle_item = list(groupedData[i][1]['match'])

	for j in range(0, len(bundle_item)):
		if (str(bundle_item[j]) != 'None'):
			try:
				bundle_feature[all_item.index(bundle_item[j])] = 1
			except:
				pass

	bundle_color = ''.join([x for x in list(groupedData[i][1]['color']) if str(x) != 'nan'])
	for m in range(0, len(color_set)):
		if color_set[m] in bundle_color:
			bundle_feature[m+253] = 1
			# print color_set[m]
	bundle_feature.append(float(groupedData[i][1]['price'].mean()))
	bundle_feature.append(float(groupedData[i][1]['price'].sum()))
	bundle_feature.append(float(groupedData[i][1]['price'].std()))
	bundle_feature.append(float(groupedData[i][1]['likes'].mean()))
	bundle_feature.append(float(groupedData[i][1]['likes'].sum()))
	bundle_feature.append(float(groupedData[i][1]['likes'].std()))
	bundle_feature.append(float(groupedData[i][1]['outfit_count'].mean()))
	bundle_feature.append(float(groupedData[i][1]['outfit_count'].sum()))
	bundle_feature.append(float(groupedData[i][1]['outfit_count'].std()))

	if (i%5 == 0):
		x_train.append(bundle_feature)
		y_train.append(float(groupedData[i][1]['bundle_like'].mean()))
	else:
		x_test.append(bundle_feature)
		y_test.append(float(groupedData[i][1]['bundle_like'].mean()))

param_grid_rf = {
		'max_depth': [2, 4, 8],
		'n_estimators': [300, 400, 500, 600]
	}
rf = GridSearchCV(RandomForestRegressor(),
					  param_grid_rf, n_jobs=-1,
					  scoring='r2', verbose=0)
rf.fit(x_train, y_train)
print rf.best_estimator_
y_pred_rf = list(rf.predict(x_test))
r2Score_rf = r2_score(y_test, y_pred_rf)
pearson_rf = pearsonr(y_test, y_pred_rf)
print r2Score_rf
print pearson_rf
print '-'*50

param_grid_xgb = {
		'n_estimators': [300, 400, 500],
		'max_depth': [2, 4, 8],
		'reg_alpha': [0.2, 0.3, 0.4],
		'reg_lambda': [0.2, 0.4, 0.6, 0.8],
		'learning_rate': [0.05, 0.1]
	}
model_xgb = GridSearchCV(XGBRegressor(),
							param_grid=param_grid_xgb,
							n_jobs=-1,
							scoring='r2', verbose=0)
model_xgb.fit(x_train, y_train)
print model_xgb.best_estimator_
y_pred_xgb = list(model_xgb.predict(x_test))
r2Score_xgb = r2_score(y_test, y_pred_xgb)
pearson_xgb = pearsonr(y_test, y_pred_xgb)
print r2Score_xgb
print pearson_xgb
print '-'*50


param_grid_svr = {
		'C': [13, 14, 15, 16, 18, 20, 25, 30],
		'gamma': [0.02, 0.03, 0.04, 0.05, 0.1, 0.5]
	}
model_svr = GridSearchCV(SVR(), n_jobs=-1, param_grid=param_grid_svr,
							scoring='r2', verbose=0)
model_svr.fit(x_train, y_train)
print model_svr.best_estimator_
y_pred_svr = list(model_svr.predict(x_test))
r2Score_svr = r2_score(y_test, y_pred_svr)
pearson_svr = pearsonr(y_test, y_pred_svr)
print r2Score_svr
print pearson_svr
print '-'*50

# x_train = np.array(x_train)
# x_test = np.array(x_test)
# y_train = np.array(y_train)
# # y_test = np.array(y_test)

# model = models.Sequential()
# model.add(layers.Dense(64, activation='relu', input_shape=(x_train.shape[1],)))
# model.add(layers.Dropout(0.2))
# model.add(layers.Dense(64, activation='relu'))
# model.add(layers.Dropout(0.2))
# model.add(layers.Dense(1))
# model.compile(optimizer='sgd', loss='mse', metrics=['mae'])

# model.fit(x_train, y_train, epochs=150, batch_size=32, verbose=1, shuffle = True)

# y_pred = list(model.predict(x_test))
# r2Score = r2_score(y_test, y_pred)
# pearson = pearsonr(y_test, y_pred)
# print r2Score
# print pearson
# print '-'*50







