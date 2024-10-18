import pandas as pd
import numpy as np
"""
方法说明：
1. 筛选孩子和母亲能匹配上的数据
2. 划分训练集和测试集

"""


# 1. 筛选孩子和母亲能匹配上的数据
def match_child_mother():
    # 读output和input
    output1 = pd.read_csv('pregnant/data/extract_feature1_csv/output1.csv', low_memory=False, index_col=0)
    output1 = output1.drop('taishu', axis=1)
    input = pd.read_csv('pregnant/data/extract_feature1_csv/input_deal_data.csv', low_memory=False)
    # 合并
    data1 = pd.merge(input, output1, on='母亲ID', how='inner')
    data1 = data1.drop('母亲ID', axis=1)
    data1 = data1.drop('儿童ID', axis=1)
    data1 = data1.dropna()
    data1.to_csv('pregnant/data/final/base/input1.csv', index=False)

    # TODO data2
    # output2 = pd.read_csv('pregnant/data/final/ouput2.csv', low_memory=False)
    # output2 = output2.drop('taishu', axis=1)

# 2. 划分训练集和测试集
def split_train_val():
    data = pd.read_csv('pregnant/data/final/base/input1.csv', low_memory=False)
    total_rows = data.shape[0]
    validation_indices = np.random.choice(total_rows, size=10000, replace=False)
    val_data = data.iloc[validation_indices]
    train_data = data.drop(validation_indices)
    val_data.to_csv('pregnant/data/final/base/val.csv', index=False)
    train_data.to_csv('pregnant/data/final/base/train.csv', index=False)
    
# match_child_mother()
split_train_val()