import pandas as pd
import numpy as np
"""
1. 输出：premature的NA处理0，其余按程度递增处理，提取数据
2. 输入：处理下frist里的产次，处理成>=1
3. 输入：统一mother表和first表的身高体重
4. 输入：将胎数拼接为输入
5. 输入：将first表中的指标转为数字
6. 输入：处理bmi的相关指标
"""

# 1. premature的NA处理0，其余按程度递增处理，提取数据
def deal_premature():
    data = pd.read_csv('pregnant/data/original_csv/birth_2tai.csv', low_memory=False)
    data['premature'] = data['premature'].fillna(0)
    
    mapping = {'极早产（不足28周）': 3, 
               '中至晚期早产（32至37周）': 2, 
               '早期早产（28至32周）': 1}
    # unique_values = data['premature'].unique()
    # 这个写法真牛波，他map后，没有映射的值会设为NAN，再用原始值覆盖
    data['premature'] = data['premature'].map(mapping).fillna(data['premature'])
    data = data[['母亲ID', '儿童ID', 'taishu', 'birth_weight', 'body_length', 'gw', 'SGA_S1', 'LGA_S1', 'low_BW', 'macrosomia', 'premature', 'foetal_death', 'stillbirth', 'death_7days', 'malformation']]
    data.to_csv('pregnant/data/digital_code/output2.csv', index=False)

# 2. 处理下frist里的产次和孕次，处理成>=1
def deal_chanci_yunci(data):
    # 处理产次
    data.loc[data['产次'] == 0.0, '产次'] = 1.0
    data['产次'] = data['产次'].fillna(1.0)
    # 处理孕次
    data.loc[data['孕次'] == 0.0, '孕次'] = 1.0
    data['孕次'] = data['孕次'].fillna(1.0)    
    return data

# 3. 统一mother表和first表的身高体重
def unify_weight_height(data):
    data['height_first'] = data['height_first'].fillna(data['height_mo'])
    data['weight_first'] = data['weight_first'].fillna(data['weight_mo']) 
    data = data.drop(['height_mo', 'weight_mo'], axis=1)
    return data

# 4. 将胎数拼接为输入
def deal_taichu(data):    
    # 将胎数添加到输入表中
    data1 = pd.read_csv('pregnant/data/digital_code/output1.csv', low_memory=False)
    data1_selected = data1[['母亲ID', 'taishu']]
    data2 = pd.read_csv('pregnant/data/digital_code/output2.csv', low_memory=False)
    data2_selected = data2[['母亲ID', 'taishu']]
    concatenated_df = pd.concat([data1_selected, data2_selected])
    # 去下重
    df_unique = concatenated_df.drop_duplicates()
    data = pd.merge(data, df_unique, on='母亲ID', how='inner')
    return data
    
# 5. 将first表中的指标转为数字
def deal_first_to_num(df):
    mapping = {'阴性': 0, '阳性': 1}
    df['弓形体'] = df['弓形体'].map(mapping)
    df['巨细胞病毒'] = df['巨细胞病毒'].map(mapping)
    df['风疹病毒'] = df['风疹病毒'].map(mapping)
    df['单纯疱疹病毒'] = df['单纯疱疹病毒'].map(mapping)
    
    
    # 处理另外复杂的三列
    # 尿蛋白,尿糖,尿酮体
    columns_to_process = ['尿蛋白', '尿糖', '尿酮体']
    keywords = ['阳', '+', '有', '微量', '自述诉异常', '≥','自诉异常']
    # 处理每一列
    for column in columns_to_process:
        null_column = df[column].isnull()
        df[column] = df[column].apply(lambda x: 1.0 if (any(keyword in str(x) for keyword in keywords) or (isinstance(x, (int, float)) and x > 0)) else 0.0)
        df.loc[null_column, column] = np.nan
    return df           
    
# 6. 处理bmi的相关指标   
def deal_bmi(df):
    # 计算 BMI
    def calculate_bmi(row):
        if pd.isnull(row['height_first']) or pd.isnull(row['weight_first']):
            return float('NaN')
        else:
            return row['weight_first'] / ((row['height_first'] / 100) ** 2)

    # 判断是否肥胖，超重，过瘦的函数
    def is_overweight(row):
        if pd.isnull(row['bmi_mo']):
            return float('NaN')
        else:
            return 1.0 if row['bmi_mo'] > 25 and row['bmi_mo'] <= 27 else 0.0
    def is_overobesity(row):
        if pd.isnull(row['bmi_mo']):
            return float('NaN')
        else:
            return 1.0 if row['bmi_mo'] > 27 else 0.0    
    def is_thin(row):
        if pd.isnull(row['bmi_mo']):
            return float('NaN')
        else:
            return 1.0 if row['bmi_mo'] < 18.5 else 0.0
    # 计算
    df['bmi_mo'] = df.apply(calculate_bmi, axis=1)
    df['ob_mo'] = df.apply(is_overobesity, axis=1)
    df['thin_mo'] = df.apply(is_thin, axis=1)
    df['ow_mo'] = df.apply(is_overweight, axis=1)
    return df



# deal_premature()
data = pd.read_csv('pregnant/data/digital_code/input.csv', low_memory=False)
data = deal_chanci_yunci(data)
data = unify_weight_height(data)
data = deal_taichu(data)
data = deal_first_to_num(data)
data = deal_bmi(data)
data.to_csv('pregnant/data/digital_code/input_clean_data.csv', index=False)