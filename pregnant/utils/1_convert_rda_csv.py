import pyreadr
import pandas as pd

# 将hight_risk表项进行one-hot编码，并赋值其危险程度
def deal_hight_risk_one_hot():
    data = pd.read_csv('pregnant/data/original_csv/highrisk.csv')
    data = data[['母亲ID', 'highrisk_name', 'riskclass_code']]
    # 对"highrisk_name"列进行One-Hot编码
    one_hot_encoded = pd.get_dummies(data['highrisk_name'], prefix='highrisk')
    # 逐列对'highrisk_name'赋值 "riskclass_code" 的值
    for column in one_hot_encoded.columns:
        one_hot_encoded[column] *= data['riskclass_code']
    data = pd.concat([data['母亲ID'], one_hot_encoded], axis=1)
    # 合并具有相同ID的编码结果
    data = data.groupby(data['母亲ID']).sum().reset_index()
    data.to_csv('pregnant/data/original_csv/hightrisk_onehot1.csv', index=False)

def convert_rda_csv():
    name = 'highrisk'     # 将名字换了就行
    data = pyreadr.read_r(f'pregnant/data/rda/{name}.rda')
    data = data[name]
    df = pd.DataFrame(data)
    df.to_csv(f'pregnant/data/original_csv/{name}.csv', index=False)

# deal_hight_risk_one_hot()
# convert_rda_csv()