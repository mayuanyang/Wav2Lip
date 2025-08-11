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
    
    results = []
    for image_path in image_paths:
        try:
            # 读取图像
            image = cv2.imread(image_path)
            if image is None:
                results.append(None)
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
                
                results.append(np.array(lip_landmarks))
            else:
                results.append(None)
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            results.append(None)
    
    # 关闭资源
    face_mesh.close()
    
    return list(zip(image_paths, results))

def process_image_batch(args):
    """
    处理图像批次的函数，用于多进程
    """
    image_paths, batch_size = args
    return get_face_landmarks_batch(image_paths)

def collect_all_images(data_root):
    """
    收集所有需要处理的图像路径
    """
    image_paths = []
    for root, dirs, files in os.walk(data_root):
        # 检查当前目录是否包含.jpg文件
        jpg_files = [f for f in files if f.endswith('.jpg')]
        if jpg_files:
            for jpg_file in jpg_files:
                image_path = os.path.join(root, jpg_file)
                # 检查是否已有对应的landmarks文件
                base_name = os.path.splitext(image_path)[0]
                landmarks_path = base_name + ".npy"
                if not os.path.exists(landmarks_path):
                    image_paths.append(image_path)
    
    return image_paths

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

def main(data_root, num_workers=4, batch_size=10):
    """
    主函数：优化的预计算面部关键点
    """
    print("Collecting images to process...")
    image_paths = collect_all_images(data_root)
    
    print(f"Found {len(image_paths)} images to process")
    
    if len(image_paths) == 0:
        print("No images need to be processed. All landmarks files already exist.")
        return
    
    # 将图像路径分批
    batches = []
    for i in range(0, len(image_paths), batch_size):
        batch = image_paths[i:i+batch_size]
        batches.append((batch, batch_size))
    
    print(f"Processing {len(batches)} batches with {num_workers} workers")
    
    start_time = time.time()
    
    # 使用多进程处理批次
    if num_workers > 1:
        with mp_lib.Pool(processes=num_workers) as pool:
            # 使用imap进行进度跟踪
            results = list(tqdm(pool.imap(process_image_batch, batches), 
                              total=len(batches), 
                              desc="Processing batches"))
    else:
        # 单进程处理
        results = []
        for batch in tqdm(batches, desc="Processing batches"):
            results.append(process_image_batch(batch))
    
    # 保存结果
    processed_count = 0
    for batch_result in results:
        for image_path, landmarks in batch_result:
            save_landmarks(image_path, landmarks)
            processed_count += 1
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print(f"Landmarks precomputation completed!")
    print(f"Processed {processed_count} images in {elapsed_time:.2f} seconds")
    print(f"Average processing speed: {processed_count/elapsed_time:.2f} images/second")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimized precompute facial landmarks for SyncNet training")
    parser.add_argument("--data_root", type=str, default="training_data", help="Root directory of training data")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of worker processes")
    parser.add_argument("--batch_size", type=int, default=10, help="Batch size for processing")
    
    args = parser.parse_args()
    
    main(args.data_root, args.num_workers, args.batch_size)
