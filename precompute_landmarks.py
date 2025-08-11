import os
import cv2
import numpy as np
import mediapipe as mp
from glob import glob
import argparse
from tqdm import tqdm

# 嘴唇关键点索引（MediaPipe定义的468点中的嘴唇区域）
LIPS_LANDMARKS = [
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
    291, 146, 91, 181, 84, 17, 314, 405, 320, 307,
    375, 321, 311, 308, 324, 318, 402, 317, 14, 87
]

def get_face_landmarks(image_path):
    """
    使用 MediaPipe 计算面部关键点
    """
    # 初始化 MediaPipe FaceMesh
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)
    
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        return None
    
    # 转换为RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 处理图像获取面部关键点
    results = face_mesh.process(image_rgb)
    
    # 关闭资源
    face_mesh.close()
    
    if results.multi_face_landmarks:
        # 获取第一个面部的关键点
        face_landmarks = results.multi_face_landmarks[0]
        
        # 提取嘴唇关键点坐标
        h, w, _ = image.shape
        lip_landmarks = []
        for idx in LIPS_LANDMARKS:
            lm = face_landmarks.landmark[idx]
            x, y = lm.x * w, lm.y * h
            lip_landmarks.append([x, y])
        
        return np.array(lip_landmarks)
    
    return None

def process_video_directory(video_dir):
    """
    处理单个视频目录中的所有图片
    """
    # 获取所有jpg图片
    image_paths = glob(os.path.join(video_dir, "*.jpg"))
    
    # 为每个图片计算并保存landmarks
    for image_path in tqdm(image_paths, desc=f"Processing {os.path.basename(video_dir)}"):
        # 生成landmarks文件路径
        base_name = os.path.splitext(image_path)[0]
        landmarks_path = base_name + ".npy"
        
        # 如果landmarks文件已存在，跳过
        if os.path.exists(landmarks_path):
            continue
        
        # 计算面部关键点
        landmarks = get_face_landmarks(image_path)
        
        # 保存关键点数据
        if landmarks is not None:
            np.save(landmarks_path, landmarks)
        else:
            # 如果没有检测到面部，保存一个空数组
            np.save(landmarks_path, np.array([]))

def main(data_root, num_workers=1):
    """
    主函数：遍历所有视频目录并预计算landmarks
    """
    # 获取所有视频目录
    video_dirs = []
    for root, dirs, files in os.walk(data_root):
        # 检查当前目录是否包含.jpg文件
        if any(file.endswith('.jpg') for file in files):
            video_dirs.append(root)
    
    print(f"Found {len(video_dirs)} video directories to process")
    
    # 使用多进程处理
    if num_workers > 1:
        from multiprocessing import Pool
        with Pool(processes=num_workers) as pool:
            pool.map(process_video_directory, video_dirs)
    else:
        # 单进程处理
        for video_dir in tqdm(video_dirs, desc="Processing video directories"):
            process_video_directory(video_dir)
    
    print("Landmarks precomputation completed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Precompute facial landmarks for SyncNet training")
    parser.add_argument("--data_root", type=str, default="training_data", help="Root directory of training data")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of worker processes")
    
    args = parser.parse_args()
    
    main(args.data_root, args.num_workers)
