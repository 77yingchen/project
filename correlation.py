import json
import numpy as np
import pandas as pd
import string
from collections import Counter
from operator import itemgetter


all_item = []

with open('itemSet3.txt') as file:
	for l in file:
		if (l != '\n'):
			all_item.append(l.strip())

#print all_item


with open('bundle_products_data.json', 'r') as f:
    bundle_products_data = json.load(f)


df = pd.DataFrame(bundle_products_data)

n = 0
all_dict = {}
for i in range (0, df.shape[0]):
	element_tmp = bundle_products_data[i]
	
	value = []
	for j in range(0, len(element_tmp['products'])):
		item_tmp = element_tmp['products'][j]
		value_tmp = item_tmp['title'].lower().split(' ')
		value = value+value_tmp
	all_dict[n] = value
	n = n+1

final_list = []
for i in range(0, len(all_item)):
	tmp_items = []
	for j in range(0, len(all_dict)):
		#print all_dict[j]
		small_tmp = []
		bool_tmp = False
		for k in range(0, len(all_dict[j])):
			#print k
			if ((all_dict[j][k] in all_item) and (all_dict[j][k] != all_item[i])):
				small_tmp.append(all_dict[j][k])
			if (all_item[i] == all_dict[j][k]):
				bool_tmp = True
		if bool_tmp:
			tmp_items = tmp_items+small_tmp
	tmp_freq = Counter(tmp_items)
	#print tmp_freq.most_common(5)
	if (len(tmp_freq.most_common(5)) < 5):
		print(all_item[i] + ": does not have enough correlation")
		print(tmp_freq.most_common(5))
	else:
		final_list.append([all_item[i], tmp_freq.most_common(5)[0][0], tmp_freq.most_common(5)[1][0], 
						tmp_freq.most_common(5)[2][0], tmp_freq.most_common(5)[3][0], tmp_freq.most_common(5)[4][0]])


print final_list



# a = [1, 1, 1, 2, 2, 2, 2, 3, 4]
# b = []
# print a.most_common(3)[0][0]





