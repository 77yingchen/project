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

# create a dict to store the id and title info
title_id_dict = dict()

# go through the original json file, find id and title, add title and id to dict
for i in range (0, df.shape[0]):
	element_tmp = bundle_products_data[i]
	id_tmp = element_tmp['_id']
	value = []
	for j in range(0, len(element_tmp['products'])):
		item_tmp = element_tmp['products'][j]
		value_tmp = item_tmp['title'].split(' ')
		value = value+value_tmp
	title_id_dict[id_tmp] = value


print title_id_dict[105]
# the output of the id 105 satisfy the title in 'data structure.txt', check point passed
# [u'Rose', u'stretch-knit', u'turtleneck', u'mini', u'dress', u'Sam', u'Edelman', 
# u'Sable', u'Boot', u'in', u'Camel', u'Suede', u'Textured-leather', 
# u'bucket', u'bag']


# the code commented below is to store the dict to a json file named title_id_dict.json
# jsObj = json.dumps(title_id_dict)
# fileObject = open('title_id_dict.json', 'w')
# fileObject.write(jsObj)
# fileObject.close()



