# s es es, ,
import json
import numpy as np
import pandas as pd
import string
from collections import Counter
from operator import itemgetter


all_item = []

with open('itemSet5.txt') as file:
	for l in file:
		if (l != '\n'):
			all_item.append(l.strip())

#print all_item


with open('bundle_products_data.json', 'r') as f:
    bundle_products_data = json.load(f)


df = pd.DataFrame(bundle_products_data)

#print bundle_products_data[1]['likes']

all_feature = []

all_label = []
for i in range (0, df.shape[0]):
	#print i
	element_tmp = bundle_products_data[i]
	all_label.append(element_tmp['likes'])
	value = []
	for j in range(0, len(element_tmp['products'])):
		item_tmp = element_tmp['products'][j]
		value_tmp = item_tmp['title'].lower().split(' ')
		value = value+value_tmp

	# print value
	feature_tmp = [0]*len(all_item)

	for m in range(0, len(value)):
		if (value[m] in all_item):
			feature_tmp[all_item.index(value[m])] = 1
		elif ((value[m]+'s') in all_item):
			feature_tmp[all_item.index((value[m]+'s'))] = 1
		elif ((value[m]+'es') in all_item):
			feature_tmp[all_item.index((value[m]+'es'))] = 1
		elif ((value[m]+'es,') in all_item):
			feature_tmp[all_item.index((value[m]+'es,'))] = 1
		elif ((value[m]+',') in all_item):
			feature_tmp[all_item.index((value[m]+','))] = 1
		# the possible titles are 'watch,', 'watch', 'watches', 'watches,'
		# our logic cannot find the 'watch,', so added it as the case below
		elif (value[m] == 'watch,'):
			feature_tmp[104] = 1

	all_feature.append(feature_tmp)
	

x_train = []
y_train = []
x_test = []
y_test = []

k = 0
for i in range(0, len(all_label)):
	if (k == 4 and len(y_test) < 5000):
		x_test.append(all_feature[i])
		y_test.append(all_label[i])
		k = 0
	else:
		x_train.append(all_feature[i])
		y_train.append(all_label[i])
		k = k+1

x_test = np.array(x_test)
x_train = np.array(x_train)
y_test = np.array(y_test)
y_train = np.array(y_train)

np.save('x_train.npy', x_train)
np.save('y_train.npy', y_train)
np.save('x_test.npy', x_test)
np.save('y_test.npy', y_test)

print x_train.shape
print y_train.shape
print x_test.shape
print y_test.shape












