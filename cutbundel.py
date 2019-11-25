#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 19:30:12 2019

@author: yingchenliu
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df= outputcsv

list=[]
#compute the avg bundle duration

for i in range (0, 23352):
    list_tmp=[]
    for j in(0,df.shape[0]):
        if (df['bundle'][j] == i):
            list_tmp.extend(df['duration'][j])
    list.append(list_tmp)
print(list)