import csv
import numpy as np
from scipy.stats.stats import pearsonr
import pandas as pd

data = pd.read_csv('predict_and_real.csv')
print pearsonr(data['predict'],data['real'])