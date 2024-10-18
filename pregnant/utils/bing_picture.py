import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("/vepfs-sha/xiezixun/high_risk_pregnant/pregnant/data/original_csv/highrisk.csv")
value_counts = df['highrisk_name'].value_counts().reset_index()
value_counts.columns = ['highrisk_name', 'count']
# 计算占比
value_counts['percentage'] = (value_counts['count'] / value_counts['count'].sum()) * 100

# 将占比小于1%的类别合并为 "其他"
value_counts['highrisk_name'] = value_counts['highrisk_name'].where(value_counts['percentage'] >= 1, '其他')

# 按照 "highrisk_name" 分组，将 "其他" 的类别合并
result = value_counts.groupby('highrisk_name').agg({'count': 'sum', 'percentage': 'sum'}).reset_index()

# 准备数据
labels = result['highrisk_name']
sizes = result['percentage']

# 定义自然的配色方案，不包含红色和绿色
colors = [
    '#FFB6C1', '#FFD700', '#87CEFA', '#FF69B4', '#EEE8AA', '#8A2BE2',
    '#DA70D6', '#D8BFD8', '#20B2AA', '#9370DB', '#FFDEAD', '#8B4513',
    '#6495ED', '#FA8072', '#F4A460', '#CD5C5C'
]


# 绘制饼图
plt.figure(figsize=(10, 8))
plt.pie(
    sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140, textprops={'fontsize': 12}
)
plt.title('高危因素数量占比', fontsize=16)
plt.savefig("bingtu.png")