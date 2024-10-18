import pandas as pd
from sklearn.model_selection import train_test_split
import yaml
import pickle
import numpy as np

def encode_FE(df_train, df_test, df_val, cols):
    for col in cols:
        vc = df_train[col].value_counts(dropna=True, normalize=True).to_dict()
        df_train[col] = df_train[col].map(vc)
        df_train[col] = df_train[col].astype('float32')
        df_test[col] = df_test[col].map(vc) 
        df_test[col] = df_test[col].astype('float32')
        df_val[col] = df_val[col].map(vc) 
        df_val[col] = df_val[col].astype('float32')
        # logger.info(col,', ',end='')

def encode_label(df_train, df_test, df_val, cols):
    for col in cols:
        df_train[col], encoding_dict = pd.factorize(df_train[col])
        df_test[col] = pd.Categorical(df_test[col], categories=encoding_dict).codes
        df_val[col] = pd.Categorical(df_val[col], categories=encoding_dict).codes

# def deal_bmi(df):
#     # 计算 BMI
#     def calculate_bmi(row):
#         if pd.isnull(row['height_first']) or pd.isnull(row['weight_first']):
#             return float('NaN')
#         else:
#             return row['weight_first'] / ((row['height_first'] / 100) ** 2)

#     # 判断是否肥胖，超重，过瘦的函数
#     def is_overweight(row):
#         if pd.isnull(row['bmi_mo']):
#             return float('NaN')
#         else:
#             return 1.0 if row['bmi_mo'] > 25 and row['bmi_mo'] <= 27 else 0.0
#     def is_overobesity(row):
#         if pd.isnull(row['bmi_mo']):
#             return float('NaN')
#         else:
#             return 1.0 if row['bmi_mo'] > 27 else 0.0    
#     def is_thin(row):
#         if pd.isnull(row['bmi_mo']):
#             return float('NaN')
#         else:
#             return 1.0 if row['bmi_mo'] < 18.5 else 0.0
#     # 计算
#     df['bmi_mo'] = df.apply(calculate_bmi, axis=1)
#     df['ob_mo'] = df.apply(is_overobesity, axis=1)
#     df['thin_mo'] = df.apply(is_thin, axis=1)
#     df['ow_mo'] = df.apply(is_overweight, axis=1)
#     return df


# Load Data
data = pd.read_csv('/vepfs-sha/xiezixun/pregnant/data/input_set_123_test/clean_bug_input.csv', low_memory=False)
# 合并三列为一列
data['death'] = (data['foetal_death'].astype(int) | data['stillbirth'].astype(int) | data['death_7days'].astype(int)).astype(int)
# 删除原始列
data = data.drop(['foetal_death', 'stillbirth', 'death_7days', '母亲ID', '分娩时间'], axis=1)
# 将premature设置为0，1二分类任务
data['premature'] = data['premature'].replace([1, 2, 3], 1)

# 编码
# 时间编码：
encode_times = ['lmp', '首检日期', '孕产期']
for encode_time in encode_times:
    data[encode_time] = pd.to_datetime(data[encode_time])
    # 拆分为年、月、日三列
    data[encode_time+'_year'] = data[encode_time].dt.year
    data[encode_time+'_month'] = data[encode_time].dt.month
    data[encode_time+'_day'] = data[encode_time].dt.day
    data = data.drop(encode_time, axis=1)

# 'birth_weight', 'body_length', 'gw',这三个先不考虑，最后看哪种好一些
# 'foetal_death', 'stillbirth', 'death_7days' 三合一，death
target_cols = ['premature', 'low_BW', 'macrosomia', 'death', 'malformation','分娩方式','产后出血','SGA_S1', 'LGA_S1']
train_cols = [x for x in list(data.columns) if x not in target_cols]

# 其他分类列，顺序编码，0填充编码缺失值，将数值列用均值填充缺失值
categorical_columns = []
categorical_dims =  {}
for col in train_cols:
    if data[col].nunique() < 200:    # 分类量编码填充
        data[col], _ = pd.factorize(data[col])
        data[col] += 1  # pd.factorize默认编码nan = -1，然后0，1，2这样顺序编码，整体加1，方便后续embedding
        categorical_columns.append(col)
        categorical_dims[col] = data[col].nunique()
    else: # 连续量，用均值填充缺失值
        data[col] = data[col].fillna(data[col].mean())

# 获取分类变量的idx及种类数量，方便后续embedding
cat_idxs = [ i for i, f in enumerate(train_cols) if f in categorical_columns]
cat_dims = [ categorical_dims[f] for f in train_cols if f in categorical_columns]

# 无法split分层，只存在一个的lable行，放入测试集
# special_rows = [1402486, 75272, 371480, 750878, 1390206, 207783, 317173, 505064, 702547, 766337, 548030, 1376716, 574211, 215386, 804351, 1570992, 1386415, 459293, 518737, 1085799, 1287955, 1296928, 749321, 656231, 867150, 846717, 1412169, 900277, 1173472, 215841, 1157973, 1288159, 465091, 423075, 1111558, 1506946, 1377126, 738436, 164691, 886174, 857749, 613026, 179958, 1147128, 667341, 576873, 1232152, 1383533, 211497, 459547, 1232415, 1483063, 114729, 200882, 310848, 834517, 831510, 857729, 1346870, 552494, 1546073, 1443487, 615008, 1480332]
# special_rows = [1402486, 505064, 1376716, 574211, 548030, 215386, 749321, 1111558, 1173472, 846717, 215841, 465091, 738436, 164691, 886174, 179958, 1147128, 857749, 613026, 667341, 576873, 1232152, 1383533, 211497, 459547, 1232415, 114729, 834517, 857729, 1346870, 552494, 1546073, 1443487, 615008, 1480332]
special_rows = [2529833,2513475,3299436]
# 提取特殊行
sp_data = data.loc[special_rows].copy()
data = data.drop(special_rows)

X = data[train_cols]
y = data[target_cols]

# 进行分层抽样
# X_tr, X_test, y_tr, y_test = train_test_split(X, y, test_size=0.01, stratify=data[['xth_child'] + target_cols], random_state=42)
# X_train, X_val, y_train, y_val = train_test_split(X_tr, y_tr, test_size=0.01, stratify=pd.concat([X_tr['xth_child'], y_tr], axis=1), random_state=42)
X_tr, X_test, y_tr, y_test = train_test_split(X, y, test_size=0.01, stratify=data[target_cols], random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_tr, y_tr, test_size=0.01, stratify=y_tr, random_state=42)
X_test = pd.concat([X_test, sp_data[train_cols]])
y_test = pd.concat([y_test, sp_data[target_cols]])

# # NAN processing
# # 1. 超过80%的nan列+某些列，连续值，采用均值
# continuous_nan_cols = ['结合胆红素', '血尿素氮', '血清肌酐', '血清谷草转氨酶', '白蛋白', '总胆红素', '血清谷丙转氨酶', '血糖', 'weight_first', 'height_first']
# for col in continuous_nan_cols:
#     mean = X_train[col].mean()
#     X_train[col] = X_train[col].fillna(mean)
#     X_test[col] = X_test[col].fillna(mean)
#     X_val[col] = X_val[col].fillna(mean)

# X_train = deal_bmi(X_train)
# X_test = deal_bmi(X_test)
# X_val = deal_bmi(X_val)

# # 用特殊值直接填充，已经验证过效果更好
# X_train = X_train.fillna(-1)
# X_test = X_test.fillna(-1)
# X_val = X_val.fillna(-1)

# # 编码，这个频率的，后期属于特征工程了，看有没有提高，加辅助列
# encode_FE(X_train, X_test, X_val, ['文化程度', '职业', '服用叶酸', '全面两孩', '单独两孩', '痛经', '母亲健康状况', '月经颜色', '月经量', '月经血块', '早孕反应', '病毒感染', '孕期服药', '营养', '发育'])
# encode_label(X_train, X_test, X_val, ['父亲民族', '母亲民族'])

# 将数值列归一化
for col in train_cols:
    if col not in categorical_columns:    # 连续量，归一化
        X_train[col] = (X_train[col] - X_train[col].mean()) / X_train[col].std()
        X_test[col] = (X_test[col] - X_train[col].mean()) / X_train[col].std()
        X_val[col] = (X_val[col] - X_train[col].mean()) / X_train[col].std()

# 保存cat_idxs和cat_dims，训练时使用
with open('/vepfs-sha/xiezixun/pregnant/data/bishe_data_test/cat_info.pkl', 'wb') as file:
    pickle.dump((cat_idxs, cat_dims), file)

# 将x和y concat，分别保存
train_data = pd.concat([X_train, y_train], axis=1).to_csv('/vepfs-sha/xiezixun/pregnant/data/bishe_data_test/train.csv', index=False)
val_data = pd.concat([X_val, y_val], axis=1).to_csv('/vepfs-sha/xiezixun/pregnant/data/bishe_data_test/val.csv', index=False)
test_data = pd.concat([X_test, y_test], axis=1).to_csv('/vepfs-sha/xiezixun/pregnant/data/bishe_data_test/test.csv', index=False)
toy_data = pd.concat([X_train, y_train], axis=1).loc[:20000, :].to_csv('/vepfs-sha/xiezixun/pregnant/data/bishe_data_test/toy.csv', index=False)