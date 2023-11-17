import cv2
import os
import numpy as np

# 定义输入和输出文件夹路径
input_folder = "/home/lwt/code_data/data/vessel/DIAS/DSA1/DSA/MIP_labels"
output_folder = "/home/lwt/code_data/data/vessel/DIAS/DSA1/DSA/RMS_labels"

# 创建输出文件夹
os.makedirs(output_folder, exist_ok=True)

# # 定义要去除的小血管的最大宽度（以像素为单位）
# max_vessel_width = 1

# # 获取输入文件夹中的所有文件
# files = os.listdir(input_folder)

# # 遍历文件夹中的每个文件
# for file in files:
#     # 读取二值血管图像
#     input_path = os.path.join(input_folder, file)
#     blood_vessel_image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE) // 255

#     # 使用距离变换来确定每个像素点到最近的背景像素的距离
#     dist_transform = cv2.distanceTransform(blood_vessel_image, cv2.DIST_L2, 3)

#     # 根据距离变换结果，确定小血管的区域
#     small_vessels = (dist_transform < max_vessel_width).astype(np.uint8)

#     # 将小血管从原始图像中去除
#     removed_small_vessels = cv2.subtract(blood_vessel_image, small_vessels)

#     # 保存处理后的图像到输出文件夹
#     output_path = os.path.join(output_folder, file)
#     cv2.imwrite(output_path, removed_small_vessels*255)
#     # cv2.imwrite(output_path, np.uint16(dist_transform/dist_transform.max()*255))
 
# print("处理完成，图像已保存到输出文件夹。")

# 定义形态学操作的内核大小和迭代次数
min_width = 1

# 获取输入文件夹中的所有文件
files = os.listdir(input_folder)

# 遍历文件夹中的每个文件
for file in files:
    # 读取二值血管图像
    input_path = os.path.join(input_folder, file)
    blood_vessel_image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)// 255

    # 执行形态学操作，去除小血管
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (min_width * 2 + 1, min_width * 2 + 1))
    removed_small_vessels = cv2.morphologyEx(blood_vessel_image, cv2.MORPH_OPEN,kernel)

    # 保存处理后的图像到输出文件夹
    output_path = os.path.join(output_folder, file)
    cv2.imwrite(output_path, removed_small_vessels*255)

print("处理完成，图像已保存到输出文件夹。")




