import os
import shutil
import random

def process_dataset(src_dir, dst_dir, num_images, txt_file):
    # 创建目标目录
    os.makedirs(os.path.join(dst_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(dst_dir, 'labels'), exist_ok=True)

    # 读取txt文件中的图片名称
    with open(os.path.join(src_dir, txt_file), 'r') as f:
        lines = f.readlines()

    # 随机选择指定数量的图片
    selected_images = random.sample(lines, num_images)

    # 复制选择的图片和对应的标签文件到目标目录
    copied_images = []
    for line in selected_images:
        image_name = line.strip().split('/')[-1]
        label_name = image_name.replace('.jpg', '.lines.json')

        src_image_path = os.path.join(src_dir, 'images', image_name)
        dst_image_path = os.path.join(dst_dir, 'images', image_name)

        src_label_path = os.path.join(src_dir, 'labels', label_name)
        dst_label_path = os.path.join(dst_dir, 'labels', label_name)

        # 检查图片和标签文件是否存在
        if os.path.exists(src_image_path) and os.path.exists(src_label_path):
            shutil.copy(src_image_path, dst_image_path)
            shutil.copy(src_label_path, dst_label_path)
            copied_images.append(line)

    # 将实际复制的图片名称写入新的txt文件
    with open(os.path.join(dst_dir, txt_file), 'w') as f:
        f.write(''.join(copied_images))

# # 处理训练集
# process_dataset('train', 'new_train', 20480, 'train.txt')

# # 处理测试集
# process_dataset('test', 'new_test', 4096, 'test.txt')

# 处理验证集
process_dataset('valid', 'new_valid', 4096, 'valid.txt')