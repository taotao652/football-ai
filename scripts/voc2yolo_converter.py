#!/usr/bin/env python3
# 将 VOC 格式 (xml) 的数据集转换为 YOLO 格式 (txt)

import xml.etree.ElementTree as ET
import os
import shutil
from pathlib import Path

def convert_voc_to_yolo(voc_dir, output_dir, class_names):
    """
    将 VOC 格式的数据集转换为 YOLO 格式。
    
    Args:
        voc_dir (str): VOC 格式数据集的根目录，应包含 'Annotations' 和 'JPEGImages' 子目录。
        output_dir (str): 转换后 YOLO 格式数据集的输出目录。
        class_names (list): 类别名称列表，例如 ['person', 'ball']。
    """
    # 创建输出目录结构
    yolo_images_dir = Path(output_dir) / 'images'
    yolo_labels_dir = Path(output_dir) / 'labels'
    yolo_images_dir.mkdir(parents=True, exist_ok=True)
    yolo_labels_dir.mkdir(parents=True, exist_ok=True)

    annotations_dir = Path(voc_dir) / 'Annotations'
    jpeg_images_dir = Path(voc_dir) / 'JPEGImages'

    if not annotations_dir.exists() or not jpeg_images_dir.exists():
        print(f"错误：在 {voc_dir} 目录下未找到 'Annotations' 或 'JPEGImages' 文件夹。")
        return

    xml_files = list(annotations_dir.glob('*.xml'))
    if not xml_files:
        print(f"在 {annotations_dir} 目录下未找到任何 .xml 文件。")
        return

    # 创建类别到ID的映射
    class_to_id = {name: idx for idx, name in enumerate(class_names)}

    for xml_path in xml_files:
        # 解析 XML 文件
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # 获取对应的图片文件名
        img_filename = root.find('filename').text
        img_path = jpeg_images_dir / img_filename
        if not img_path.exists():
            print(f"警告：找不到关联的图片文件 {img_path}，跳过 {xml_path.name}")
            continue

        # 获取图像尺寸
        size = root.find('size')
        img_width = float(size.find('width').text)
        img_height = float(size.find('height').text)

        # 复制图片到目标目录
        shutil.copy2(img_path, yolo_images_dir / img_filename)

        # 转换标注
        yolo_lines = []
        for obj in root.findall('object'):
            class_name = obj.find('name').text
            if class_name not in class_to_id:
                print(f"警告：在 {xml_path.name} 中发现未定义的类别 '{class_name}'，已跳过。")
                continue

            class_id = class_to_id[class_name]
            bndbox = obj.find('bndbox')
            xmin = float(bndbox.find('xmin').text)
            ymin = float(bndbox.find('ymin').text)
            xmax = float(bndbox.find('xmax').text)
            ymax = float(bndbox.find('ymax').text)

            # 转换为 YOLO 格式：<class_id> <x_center> <y_center> <width> <height>
            x_center = (xmin + xmax) / 2.0 / img_width
            y_center = (ymin + ymax) / 2.0 / img_height
            width = (xmax - xmin) / img_width
            height = (ymax - ymin) / img_height

            yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

        # 保存 YOLO 标签文件
        if yolo_lines:
            label_filename = Path(img_filename).stem + '.txt'
            label_path = yolo_labels_dir / label_filename
            with open(label_path, 'w') as f:
                f.write('\n'.join(yolo_lines))
        else:
            print(f"警告：文件 {xml_path.name} 没有生成任何 YOLO 标注，可能没有符合类别的物体。")

    print(f"转换完成！YOLO 格式数据集已保存到: {output_dir}")
    print(f"请检查 {output_dir}/images 和 {output_dir}/labels 目录。")

def main():
    # --- 这里需要你根据自己的数据集进行修改 ---
    # 1. VOC格式数据集的路径
    voc_directory = "./"  # 当前目录，应包含 'Annotations' 和 'JPEGImages'
    
    # 2. YOLO格式数据集的输出路径
    yolo_output_directory = "./yolo_dataset"
    
    # 3. 你的数据集中包含的类别名称列表（注意顺序！ID从0开始）
    # 重要：请先查看 Annotations 下的任意一个 .xml 文件，找到 <name> 标签里的内容
    # 例如，如果里面是 <name>person</name>，就写 ['person']
    # 如果还有 <name>ball</name>，就写 ['person', 'ball']
    classes = ['person','football']  # <--- 这里一定要改成你数据集中实际的类别名称！
    # --- 修改结束 ---
    
    convert_voc_to_yolo(voc_directory, yolo_output_directory, classes)

if __name__ == "__main__":
    main()
