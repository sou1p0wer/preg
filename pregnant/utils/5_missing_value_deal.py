import pandas as pd
import numpy as np
"""
方法说明：
1. dropna
"""

# 1. dropna
def drop_na(data):
    return data.dropna()

data = pd.read_csv('pregnant/data/final/input1.csv', low_memory=False)
data = drop_na(data)
data.to_csv('pregnant/data/final/base/input1.csv')