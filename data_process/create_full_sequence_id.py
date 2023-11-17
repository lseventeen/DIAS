import os
import shutil

# 源目录和目标目录
source_dir = "/home/lwt/code_data/data/vessel/DIAS/DSA1/DSA_ysl/ICAS（重）/复发/173、赵现华-4(new)/Series_005_Arch"
target_dir = "/home/lwt/code_data/data/vessel/DIAS/full_sequence"
os.makedirs(target_dir,exist_ok=True)

mame = "1.3.12.2.1107.5.4.5.154061.30000017030100350167100000774.4_Frame"
new_prefix = "60"

# 获取源目录中的所有文件列表
file_list = os.listdir(source_dir)

# 遍历文件列表
for filename in file_list:
    # 检查文件是否是文件而不是目录
    if os.path.isfile(os.path.join(source_dir, filename)):
        # 获取文件名和扩展名
        base_name, ext = os.path.splitext(filename)
        if mame in base_name:
        # 新的文件名前缀（这里使用"_new_"作为前缀，你可以根据需要修改）
            
        
            # 构建新的文件名
            new_filename = new_prefix +"_" +base_name.split(mame)[1] + ext
        
        # 构建源文件的完整路径和目标文件的完整路径
            source_path = os.path.join(source_dir, filename)
            target_path = os.path.join(target_dir, new_filename)
        
        # 复制文件到目标目录
            shutil.copy(source_path, target_path)
