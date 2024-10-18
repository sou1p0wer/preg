# General imports
import numpy as np
import pandas as pd
import os, sys, gc, warnings, random, datetime, math
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluation.metrics import metirc_auc
from utils.log import create_logger
from sklearn.metrics import mean_squared_error
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedShuffleSplit, KFold, train_test_split
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from math import sqrt
import time
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline
from imblearn.combine import SMOTEENN
warnings.filterwarnings('ignore')
## Seeder
# :seed to make all processes deterministic     # type: int
def seed_everything(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    
## Memory Reducer
# :df pandas dataframe to reduce size             # type: pd.DataFrame()
# :verbose                                        # type: bool
def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: logger.info('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df

# Encoding Functions
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



logger = create_logger()
logger.info('############模型信息############')
SEED = 42
seed_everything(SEED)
logger.info(f'种子: {SEED}')

# Load Data
data = pd.read_csv('/data/Project_1_cataract/xzx/pregnant/data/input_set_123/clean_bug_input.csv', low_memory=False)
data = data.drop(['母亲ID','分娩时间'], axis=1)




# 'birth_weight', 'body_length', 'gw',这三个先不考虑，最后看哪种好一些
target_cols = ['SGA_S1', 'LGA_S1', 'low_BW',
       'macrosomia', 'premature', 'foetal_death', 'stillbirth', 'death_7days',
       'malformation']
target_3_cols = ['SGA_S1', 'LGA_S1']
target_4_cols = 'premature'
train_cols = [x for x in list(data.columns) if x not in target_cols]

# 无法split分层，只存在一个的lable行，放入测试集
special_rows = [1407262, 75274, 371482, 751432, 1394982, 207785, 317175, 505073, 703101, 766891, 548143, 1381440, 574688, 215388, 804905, 1575787, 1391191, 459302, 518746, 1090463, 1292619, 1301592, 749875, 656785, 867891, 847458, 1416945, 904488, 1178136, 215843, 1162637, 1292823, 465100, 423081, 1116222, 1511741, 1381878, 738990, 164693, 886915, 858490, 613580, 179960, 1151792, 667895, 577381, 1236816, 1388309, 211499, 459556, 1237079, 1487858, 114731, 200884, 310850, 835093, 832067, 858470, 1351534, 552658, 1550868, 1448263, 615562, 1485127]


# 提取特殊行
sp_data = data.loc[special_rows].copy()
data = data.drop(special_rows)


logger.info(f'target_cols: {target_cols}')
logger.info(f'train_cols: {train_cols}')
X = data[train_cols]
y = data[target_cols]

mth_mAP = 0
mth_AUC = 0

for xth in [9, 42]:
    logger.info('#' * 10)
    logger.info('#' * 10)
    logger.info('#' * 10)
    logger.info(f'随机种子为：{xth}号开始迭代')
    
    # 按照胎数，进行分层抽样，随机五次进行验证
    X_tr, X_test, y_tr, y_test = train_test_split(X, y, test_size=0.01, stratify=data[['xth_child'] + target_cols], random_state=xth)
    X_train, X_val, y_train, y_val = train_test_split(X_tr, y_tr, test_size=0.01, stratify=pd.concat([X_tr['xth_child'], y_tr], axis=1), random_state=xth)
    X_test = pd.concat([X_test, sp_data[train_cols]])
    y_test = pd.concat([y_test, sp_data[target_cols]])
    
    # NAN processing
    # 1. 超过80%的nan列，连续值，采用均值
    continuous_nan_cols = ['结合胆红素', '血尿素氮', '血清肌酐', '血清谷草转氨酶', '白蛋白', '总胆红素', '血清谷丙转氨酶', '血糖', 'weight_first', 'height_first']
    # test = ['血红蛋白', '婚龄', 'dbp', 'sbp', 'age_fa', 'age_mo']
    # for col in continuous_nan_cols + test:
    for col in continuous_nan_cols:
        mean = X_train[col].mean()
        X_train[col] = X_train[col].fillna(mean)
        X_test[col] = X_test[col].fillna(mean)
        X_val[col] = X_val[col].fillna(mean)
    
    X_train = deal_bmi(X_train)
    X_test = deal_bmi(X_test)
    X_val = deal_bmi(X_val)

    # discrete_nan_cols = ['单独两孩', '母亲健康状况', '职业', '月经颜色', '月经量', '全面两孩', '病毒感染', '发育', '营养', '孕期服药', '服用叶酸', '父亲民族', '母亲民族', '文化程度', '痛经', '月经血块', 'edu_high_mo', 'edu_low_mo', 'nation_han_mo', 'nation_han_fa', 'work',
    # 'lmp', '首检日期', '孕产期']
    # for col in discrete_nan_cols:
    #     mode = X_train[col].mode()[0]
    #     X_train[col] = X_train[col].fillna(mode)
    #     X_test[col] = X_test[col].fillna(mode)
    #     X_val[col] = X_val[col].fillna(mode)

    # 用特殊值直接填充，已经验证过效果更好
    # special_nan_cols = ['早孕反应', '初次妊娠', '经产妇', '检查孕周', '尿酮体', '尿蛋白', '尿糖', '人工流产', '单纯疱疹病毒', '风疹病毒', '巨细胞病毒', '弓形体', '出生缺陷儿', '死产数', '新生儿死亡', '死胎次数', '自然流产']
    X_train = X_train.fillna(-1)
    X_test = X_test.fillna(-1)
    X_val = X_val.fillna(-1)

    # 编码
    encode_FE(X_train, X_test, X_val, ['文化程度', '职业', '服用叶酸', '全面两孩', '单独两孩', '痛经', '母亲健康状况', '月经颜色', '月经量', '月经血块', '早孕反应', '病毒感染', '孕期服药', '营养', '发育'])
    encode_label(X_train, X_test, X_val, ['父亲民族', '母亲民族'])
    encode_times = ['lmp', '首检日期', '孕产期']
    for encode_time in encode_times:
        X_train[encode_time] = pd.to_datetime(X_train[encode_time]).astype(int) // 10**9
        X_test[encode_time] = pd.to_datetime(X_test[encode_time]).astype(int) // 10**9
        X_val[encode_time] = pd.to_datetime(X_val[encode_time]).astype(int) // 10**9

    # 训练
    mean_mAP = []
    mean_AUC = []

    for col in target_cols:
        logger.info(f'start train on: {col}')
        if col == target_4_cols:
            es = xgb.callback.EarlyStopping(
                rounds=100,
                maximize=True
            )
            clf4 = xgb.XGBClassifier( 
                n_estimators=2000,
                max_depth=12, 
                learning_rate=0.02, 
                subsample=0.8,
                colsample_bytree=0.4, 
                missing=-999, 
                seed=42,
                early_stopping_rounds=100,
                # USE GPU
                tree_method='hist',
                device='gpu',
                callbacks=[es],
                scale_pos_weight=10,

                objective='multi:softprob',
                eval_metric=average_precision_score,
                num_class=4
            )
            clf4.fit(X_train, y_train[col], 
                eval_set=[(X_val,y_val[col])],
                verbose=50)
            logger.info(f'Stopping. Best iteration:[{clf4.best_iteration}]   {clf4.best_score}')
            y_pred=clf4.predict_proba(X_test)
        elif col in target_3_cols:
            es = xgb.callback.EarlyStopping(
                rounds=100,
                maximize=True
            )
            clf3 = xgb.XGBClassifier( 
                n_estimators=2000,
                max_depth=12, 
                learning_rate=0.02, 
                subsample=0.8,
                colsample_bytree=0.4, 
                missing=-999, 
                seed=42,
                early_stopping_rounds=100,
                # USE GPU
                tree_method='hist',
                device='gpu',
                callbacks=[es],
                scale_pos_weight=10,

                objective='multi:softprob',
                eval_metric=average_precision_score,
                num_class=3
            )
            clf3.fit(X_train, y_train[col], 
                eval_set=[(X_val,y_val[col])],
                verbose=50)     
            logger.info(f'Stopping. Best iteration:[{clf3.best_iteration}]   {clf3.best_score}')
            y_pred=clf3.predict_proba(X_test)
        else:
            es = xgb.callback.EarlyStopping(
                rounds=100,
                maximize=True
            )
            clf2 = xgb.XGBClassifier( 
                n_estimators=2000,
                max_depth=12, 
                learning_rate=0.02, 
                subsample=0.8,
                colsample_bytree=0.4, 
                missing=-999, 
                seed=42,
                early_stopping_rounds=100,
                # USE GPU
                tree_method='hist',
                device='gpu',                           
                callbacks=[es],
                scale_pos_weight=10,

                eval_metric=average_precision_score
            )
            clf2.fit(X_train, y_train[col], 
            eval_set=[(X_val,y_val[col])],
            verbose=50)   
            logger.info(f'Stopping. Best iteration:[{clf2.best_iteration}]   {clf2.best_score}')
            y_pred=clf2.predict_proba(X_test)[:,1]

        auc = metirc_auc(y_test[col], y_pred)
        mAP = average_precision_score(y_test[col], y_pred)
        logger.info(f'预测结果: \nmAP: {mAP}\nAUC: {auc}')
        mean_mAP.append(mAP)
        mean_AUC.append(auc)

    logger.info(f'\n{xth}号随机种子的平均预测值：\nmean_mAP: {sum(mean_mAP) / len(mean_mAP)}\nmean_AUC: {sum(mean_AUC) / len(mean_AUC)}')
    mth_mAP += sum(mean_mAP) / len(mean_mAP)
    mth_AUC += sum(mean_AUC) / len(mean_AUC)
logger.info(f'\n平均预测值：\nmean_mAP: {mth_mAP / 2}\nmean_AUC: {mth_AUC / 2}')