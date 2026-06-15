#!/usr/bin/env python3
import xml.etree.ElementTree as ET
import os
import shutil
from pathlib import Path

def convert_voc_to_yolo(voc_dir, output_dir, class_names):
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
        tree = ET.parse(xml_path)
        root = tree.getroot()

        img_filename = root.find('filename').text
        img_path = jpeg_images_dir / img_filename
        if not img_path.exists():
            print(f"警告：找不到关联的图片文件 {img_path}，跳过 {xml_path.name}")
            continue

        size = root.find('size')
        img_width = float(size.find('width').text)
        img_height = float(size.find('height').text)

        shutil.copy2(img_path, yolo_images_dir / img_filename)

        yolo_lines = []
        for obj in root.findall('object'):
            class_name = obj.find('name').text
            if class_name not in class_to_id:
                print(f"警告：未定义的类别 '{class_name}'，已跳过。")
                continue

            class_id = class_to_id[class_name]
            bndbox = obj.find('bndbox')
            xmin = float(bndbox.find('xmin').text)
            ymin = float(bndbox.find('ymin').text)
            xmax = float(bndbox.find('xmax').text)
            ymax = float(bndbox.find('ymax').text)

            x_center = (xmin + xmax) / 2.0 / img_width
            y_center = (ymin + ymax) / 2.0 / img_height
            width = (xmax - xmin) / img_width
            height = (ymax - ymin) / img_height

            yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

        if yolo_lines:
            label_filename = Path(img_filename).stem + '.txt'
            label_path = yolo_labels_dir / label_filename
            with open(label_path, 'w') as f:
                f.write('\n'.join(yolo_lines))

    print(f"转换完成！YOLO 格式数据集已保存到: {output_dir}")
    print(f"图片数量: {len(list(yolo_images_dir.glob('*')))}")
    print(f"标签数量: {len(list(yolo_labels_dir.glob('*')))}")

def main():
    # --- 这里修改成你的实际路径 ---
    voc_directory = "./football_person_dataset"      # VOC格式数据集的根目录
    yolo_output_directory = "./person_yolo_dataset"  # YOLO格式数据集的输出目录
    classes = ['football', 'person']                  # 类别列表
    # --- 修改结束 ---
    
    convert_voc_to_yolo(voc_directory, yolo_output_directory, classes)

if __name__ == "__main__":
    main()
