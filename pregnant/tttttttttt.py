# 各类别的比例
class_ratios = [0.972560372195939, 0.023043461648835292, 0.0013099203853881335, 0.0030862457698375536]

# 计算倒数作为权重
weights = [1.0 / ratio for ratio in class_ratios]

# 归一化权重，使权重和等于1（可以不归一化）
normalized_weights = [weight / sum(weights) for weight in weights]
print(normalized_weights)