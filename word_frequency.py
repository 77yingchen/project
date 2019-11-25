import json
import numpy as np
import pandas as pd
import string
from collections import Counter
from operator import itemgetter


# a = {1 : ['tmp','test']}
# a[2] = ['tmp','test','new']

# print a


with open('bundle_products_data.json', 'r') as f:
    bundle_products_data = json.load(f)


df = pd.DataFrame(bundle_products_data)


word_list = dict()
i = 1
# go through the original json file, find id and title, add title and id to dict
for i in range (0, df.shape[0]):
	element_tmp = bundle_products_data[i]
	for j in range(0, len(element_tmp['products'])):
		item_tmp = element_tmp['products'][j]
		value_tmp = item_tmp['title'].split(' ')[-1]
		word_list[i] = value_tmp
		i = i+1

c = Counter(word_list.values())
print (c)







