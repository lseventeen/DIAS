import cv2
import os
import numpy as np

# 定义文件夹路径
folder_path = "/home/lwt/code_data/data/vessel/DIAS/DSA1/DSA/labels"

# 获取文件夹中的所有文件
files = os.listdir(folder_path)

# 创建一个字典来存储每个序列ID的图像列表
sequence_images = {}

# 遍历文件夹中的每个文件
for file in files:
    # 分割文件名以获取序列ID和图像ID
    parts = file.split("-")
    sequence_id = parts[0]

    # 读取图像
    image = cv2.imread(os.path.join(folder_path, file), cv2.IMREAD_GRAYSCALE)

    # 如果序列ID不在字典中，则将其添加
    if sequence_id not in sequence_images:
        sequence_images[sequence_id] = [image]
    else:
        sequence_images[sequence_id].append(image)

# 创建一个新的文件夹来保存合并的图像
output_folder = "/home/lwt/code_data/data/vessel/DIAS/DSA1/DSA/MIP_labels"
os.makedirs(output_folder, exist_ok=True)

# 合并每个序列ID的图像并实现最大密度投影
for sequence_id, images in sequence_images.items():
    # 将图像堆叠在一起
    stacked_images = np.stack(images, axis=0)

    # 计算最大密度投影
    max_density_projection = np.max(stacked_images, axis=0)
    max_density_projection = np.where(max_density_projection > 100,255,0)

    # 保存最大密度投影图像
    output_file = os.path.join(output_folder, f"{sequence_id}.jpg")
    cv2.imwrite(output_file, max_density_projection)

print("合并和保存完成。")