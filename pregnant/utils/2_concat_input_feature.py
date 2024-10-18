import pandas as pd
import os

result_path = 'pregnant/data/extract_feature1_csv'
root = 'pregnant/data/original_csv'
file_names = ['first', 'mother']
target_feature = {
    'birth_1tai': ['母亲ID', '儿童ID', 'taishu', 'birth_weight', 'body_length', 'gw', 'SGA_S1', 'LGA_S1', 'low_BW', 'macrosomia', 'premature', 'foetal_death', 'stillbirth', 'death_7days', 'malformation'],
    'birth_2tai': ['母亲ID', '儿童ID', 'taishu', 'birth_weight', 'body_length', 'gw', 'SGA_S1', 'LGA_S1', 'low_BW', 'macrosomia', 'premature', 'foetal_death', 'stillbirth', 'death_7days', 'malformation'],
    'first': ['母亲ID', '孕次', '产次', '弓形体' ,'巨细胞病毒', '风疹病毒', '单纯疱疹病毒', '自然流产', '人工流产', '死胎次数', '死产数', '新生儿死亡', '出生缺陷儿', '血糖', '血红蛋白', '尿蛋白', '尿糖', '尿酮体', '血清谷丙转氨酶', '血清谷草转氨酶', '白蛋白', '总胆红素', '结合胆红素', 
              '血清肌酐', '血尿素氮', 'height_first', 'weight_first', 'sbp', 'dbp'],
    'mother': ['母亲ID', 'age_mo', 'height_mo', 'weight_mo', 'folic_pre', 'folic_dur', 'health_mo', 'bmi_mo', 'ob_mo', 'ow_mo', 'thin_mo'],
    'highrisk': ['母亲ID', 'highrisk_name', 'riskclass_code']
}
merged_data = None
for name in file_names:
    data = pd.read_csv(os.path.join(root, name+'.csv'), low_memory=False)
    features = target_feature[name]
    data_select = data[features]
    if merged_data is None:
        merged_data = data_select
    else:
        merged_data = pd.merge(merged_data, data_select, on='母亲ID', how='outer')
# 合并high_risk_one_hot
data_high_risk_one_hot = pd.read_csv('pregnant/data/original_csv/hightrisk_onehot.csv', low_memory=False)
merged_data = pd.merge(merged_data, data_high_risk_one_hot, on='母亲ID', how='outer')
# 将没有high_risk的都设为0.0
risk_columns = data_high_risk_one_hot.columns[1:]
merged_data[risk_columns] = merged_data[risk_columns].fillna(0.0)
merged_data.to_csv('pregnant/data/extract_feature1_csv/input.csv', index=False)
