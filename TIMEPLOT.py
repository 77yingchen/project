#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 21:54:05 2019

@author: yingchenliu
"""

import json
import numpy as np
import pandas as pd
import string
import csv
from datetime import datetime

with open('succint_bundle_data.json','r') as f:
    bundle_products_data = json.load(f)
    
df = pd.DataFrame(bundle_products_data)

print(bundle_products_data[1]['products'][0]['title'])
print(bundle_products_data[1]['products'][0]['price'])
print(bundle_products_data[1]['products'][0]['brand'])
print(bundle_products_data[1]['products'][0]['color'])
print(bundle_products_data[1]['products'][0]['likes'])
print(bundle_products_data[1]['products'][0]['outfit_count'])
print(bundle_products_data[1]['products'][0]['created'])
print(bundle_products_data[1]['products'][0]['updated'])



def changeWord(w):
    n1 = w.replace('T', ' ')
    n2 = n1.replace('Z', '')
    return(n2)
def changeDate(date_time_str):
    date_time_obj = datetime.strptime(date_time_str,
                                               '%Y-%m-%d %H:%M:%S.%f')
    return(date_time_obj)


t1 = datetime.now() 
duration_likes = []
#labels = ['duration','likes']
#duration_likes.append(labels)
for i in range (0, df.shape[0]):
    for j in range(0, len(bundle_products_data[i]['products'])):
        product_tmp = []
        tmp = str(bundle_products_data[i]['products'][j]['created'])
        bundle_products_data[i]['products'][j]['created']=changeDate(changeWord(tmp))
        time_diff = t1 - bundle_products_data[i]['products'][j]['created'] 
        product_tmp.append(time_diff.days)
        product_tmp.append(bundle_products_data[i]['products'][j]['likes'])
        duration_likes.append(product_tmp)
        
with open("duration_likes.csv", 'w') as resultFile:
    wr = csv.writer(resultFile, dialect = 'excel')
    wr.writerows(duration_likes)


all_result = []
labels = ['bundle','title','price',
              'brand','color','likes',
              'outfit_count','time_created','time_updated']
all_result.append(labels)
for i in range (0, df.shape[0]):
    for j in range(0, len(bundle_products_data[i]['products'])):
        product_tmp = []
        product_tmp.append(i)
        product_tmp.append(bundle_products_data[i]['products'][j]['title'])
        product_tmp.append(bundle_products_data[i]['products'][j]['price'])
        product_tmp.append(bundle_products_data[i]['products'][j]['brand'])
        product_tmp.append(bundle_products_data[i]['products'][j]['color'])
        product_tmp.append(bundle_products_data[i]['products'][j]['likes'])
        product_tmp.append(bundle_products_data[i]['products'][j]['outfit_count'])
        product_tmp.append(bundle_products_data[i]['products'][j]['created'])
        product_tmp.append(bundle_products_data[i]['products'][j]['updated'])
       

        all_result.append(product_tmp)
        
with open("output.csv", 'w') as resultFile:
    wr = csv.writer(resultFile, dialect = 'excel')
    wr.writerows(all_result)
