import os
import cv2
import numpy as np
import mediapipe as mp
from glob import glob
import argparse
from tqdm import tqdm
import multiprocessing as mp_lib
from functools import partial
import time

# 嘴唇关键点索引（MediaPipe定义的468点中的嘴唇区域）
LIPS_LANDMARKS = [
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
    291, 146, 91, 181, 84, 17, 314, 405, 320, 307,
    375, 321, 311, 308, 324, 318, 402, 317, 14, 87
]

def get_face_landmarks_batch(image_paths):
    """
    批量处理图像获取面部关键点，使用单个FaceMesh实例提高效率
    """
    # 初始化 MediaPipe FaceMesh
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)
    
    for image_path in image_paths:
        try:
            # 读取图像
            image = cv2.imread(image_path)
            if image is None:
                save_landmarks(image_path, None)
                continue
            
            # 转换为RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # 处理图像获取面部关键点
            result = face_mesh.process(image_rgb)
            
            if result.multi_face_landmarks:
                # 获取第一个面部的关键点
                face_landmarks = result.multi_face_landmarks[0]
                
                # 提取嘴唇关键点坐标
                h, w, _ = image.shape
                lip_landmarks = []
                for idx in LIPS_LANDMARKS:
                    lm = face_landmarks.landmark[idx]
                    x, y = lm.x * w, lm.y * h
                    lip_landmarks.append([x, y])
                
                save_landmarks(image_path, np.array(lip_landmarks))
            else:
                save_landmarks(image_path, None)
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            save_landmarks(image_path, None)
    
    # 关闭资源
    face_mesh.close()

def process_image_batch(args):
    """
    处理图像批次的函数，用于多进程
    """
    image_paths, batch_size = args
    get_face_landmarks_batch(image_paths)



def process_folder(data_root, subfolder, num_workers, batch_size):
    """
    处理单个文件夹中的图像
    """
    # 构造完整的子文件夹路径
    full_subfolder_path = os.path.join(data_root, subfolder)
    
    # 检查子文件夹是否存在
    if not os.path.exists(full_subfolder_path):
        print(f"Warning: Subfolder {full_subfolder_path} does not exist, skipping...")
        return 0
    
    # 获取子文件夹中的所有.jpg文件
    jpg_files = glob(os.path.join(full_subfolder_path, "*.jpg"))
    
    # 检查每个.jpg文件是否已有对应的landmarks文件
    images_to_process = []
    for jpg_file in jpg_files:
        # 检查是否已有对应的landmarks文件
        base_name = os.path.splitext(jpg_file)[0]
        landmarks_path = base_name + ".npy"
        if not os.path.exists(landmarks_path):
            images_to_process.append(jpg_file)
    
    # 如果没有需要处理的图像，直接返回
    if not images_to_process:
        print(f"No images need to be processed in folder: {full_subfolder_path}")
        return 0
    
    print(f"\nProcessing folder: {full_subfolder_path} ({len(images_to_process)} images)")
    
    # 将当前文件夹的图像路径分批
    batches = []
    for i in range(0, len(images_to_process), batch_size):
        batch = images_to_process[i:i+batch_size]
        batches.append((batch, batch_size))
    
    # 处理当前文件夹的批次
    folder_start_time = time.time()
    
    if num_workers > 1:
        with mp_lib.Pool(processes=num_workers) as pool:
            # 使用imap进行进度跟踪
            list(tqdm(pool.imap(process_image_batch, batches), 
                     total=len(batches), 
                     desc=f"Processing batches in {os.path.basename(full_subfolder_path)}"))
    else:
        # 单进程处理
        for batch in tqdm(batches, desc=f"Processing batches in {os.path.basename(full_subfolder_path)}"):
            process_image_batch(batch)
    
    folder_end_time = time.time()
    folder_elapsed_time = folder_end_time - folder_start_time
    folder_processed = len(images_to_process)
    
    print(f"Completed folder {os.path.basename(full_subfolder_path)}: {folder_processed} images in {folder_elapsed_time:.2f} seconds")
    return folder_processed

def save_landmarks(image_path, landmarks):
    """
    保存关键点数据
    """
    base_name = os.path.splitext(image_path)[0]
    landmarks_path = base_name + ".npy"
    
    if landmarks is not None:
        np.save(landmarks_path, landmarks)
    else:
        # 如果没有检测到面部，保存一个空数组
        np.save(landmarks_path, np.array([]))

def main(data_root, abc_file_path, num_workers=4, batch_size=10):
    """
    主函数：优化的预计算面部关键点
    """
    print("Processing folders...")
    
    # 读取指定的文件获取子文件夹路径列表
    if not os.path.exists(abc_file_path):
        print(f"Error: abc file not found at {abc_file_path}")
        return
    
    with open(abc_file_path, 'r') as f:
        subfolders = [line.strip() for line in f.readlines() if line.strip()]
    
    if not subfolders:
        print("No subfolders found in abc file.")
        return
    
    start_time = time.time()
    total_processed = 0
    
    # 遍历每个子文件夹并立即处理
    for subfolder in subfolders:
        folder_processed = process_folder(data_root, subfolder, num_workers, batch_size)
        total_processed += folder_processed
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print(f"\nLandmarks precomputation completed!")
    print(f"Processed {total_processed} images in {elapsed_time:.2f} seconds")
    print(f"Average processing speed: {total_processed/elapsed_time:.2f} images/second")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimized precompute facial landmarks for SyncNet training")
    parser.add_argument("--data_root", type=str, default="training_data", help="Root directory of training data")
    parser.add_argument("--abc_file", type=str, default="filelists/train.txt", help="Path to the file containing subfolder paths")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of worker processes")
    parser.add_argument("--batch_size", type=int, default=10, help="Batch size for processing")
    
    args = parser.parse_args()
    
    main(args.data_root, args.abc_file, args.num_workers, args.batch_size)
