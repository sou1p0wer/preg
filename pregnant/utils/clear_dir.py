import os

def clear_directory(directory):
    """
    清空指定目录下的所有文件
    
    参数:
    - directory: 目录路径
    """
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)    
    print(f"目录已清空：{directory}")


# 清空指定目录下的所有文件
root = 'pregnant/outputs'
for n in ['checkpoints', 'logs', 'train_loss']:
    clear_directory(os.path.join(root, n))