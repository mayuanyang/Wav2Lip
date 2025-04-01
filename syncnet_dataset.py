from hparams import hparams, get_image_list
import multiprocessing
from os.path import dirname, join, basename, isfile
import os, random, cv2, argparse
from glob import glob
import numpy as np
import audio
import torch
import matplotlib.pyplot as plt
from PIL import Image
import mediapipe as mp

face_image_cache = multiprocessing.Manager().dict()
file_exist_cache = multiprocessing.Manager().dict()
orig_mel_cache = multiprocessing.Manager().dict()

"""
The FPS is set to 25 for video, 5/25 is 0.2, we need to have 0.2 seconds for the audio,
because the audio mel spectrogram ususlly has 80 frame per seconds, so 16/80 is 0.2 seconds
"""
syncnet_T = 5
syncnet_mel_step_size = 16
samples = [True, True,True, True,True, False,False, False, False, False]
negative_data_mode = "HARD" # SIMPLE, MEDIUM, HARD

# 嘴唇关键点索引（MediaPipe定义的468点中的嘴唇区域）
LIPS_LANDMARKS = [
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
    291, 146, 91, 181, 84, 17, 314, 405, 320, 307,
    375, 321, 311, 308, 324, 318, 402, 317, 14, 87
]


class Dataset(object):
    
    def __init__(self, split, data_root, train_root, use_augmentation, img_size_factor=1):
        print('-----')
        self.all_videos = get_image_list(data_root, split, train_root)
        self.use_augmentation = use_augmentation
        self.img_size_factor = img_size_factor
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)
        

    def get_frame_id(self, frame):
        return int(basename(frame).split('.')[0])

    def get_window(self, start_frame):
        start_id = self.get_frame_id(start_frame)
        vidname = dirname(start_frame)

        window_fnames = []
        for frame_id in range(start_id, start_id + syncnet_T):
            frame = join(vidname, '{}.jpg'.format(frame_id))
            
            if not frame in file_exist_cache:
              if not isfile(frame):
                return None    
            
            
            file_exist_cache[frame] = True
            window_fnames.append(frame)
        return window_fnames
    

    def crop_audio_window(self, spec, start_frame):
        # num_frames = (T x hop_size * fps) / sample_rate
        start_frame_num = self.get_frame_id(start_frame)

        """
        80. is a scaling factor used to convert the time in seconds to the index in the audio spectrogram.
        This scaling factor is related to how the audio spectrogram is calculated and the time resolution of the spectrogram.
        For instance, if the spectrogram has a time resolution of 12.5 ms per frame (which is typical for many audio processing tasks), 
        80 frames per second would correspond to 1.25 seconds. This means the spectrogram has a higher temporal resolution than the video.
        """
        start_idx = int(80. * (start_frame_num / float(hparams.fps)))

        end_idx = start_idx + syncnet_mel_step_size

        return spec[start_idx : end_idx, :]


    def __len__(self):
        return len(self.all_videos)

    def compute_alignment_score(self, chosen_id, wrong_img_id, max_difference):
        """
        Compute an alignment score based on the difference between two frame IDs.
        A perfect alignment (difference = 0) yields a score of 1.
        If the difference equals or exceeds max_difference, the score is 0.
        
        Args:
            chosen_id (int or float): Frame ID of the correct image.
            wrong_img_id (int or float): Frame ID of the misaligned image.
            max_difference (int or float): The maximum expected difference between IDs.
            
        Returns:
            score (float): A value between 0 and 1, where 1 means perfectly aligned.
        """
        # Compute the absolute difference.
        d = abs(chosen_id - wrong_img_id)
        
        # Normalize so that a difference of 0 yields 1 and a difference of max_difference yields 0.
        normalized = 1 - (d / max_difference)
        
        # Clamp the normalized score to be within [0, 1]
        score = max(0.0, min(1.0, normalized))
        return score

    def __getitem__(self, idx):
        """
        Randomly select a video and corresponding images.
        Randomly choose a correct or incorrect image pair.
        Read and preprocess the images and audio data.
        Handle exceptions and retries in case of read errors.
        Return the processed image data, audio features, and label.
        """
        should_load_diff_video = False
        while 1:
            if should_load_diff_video:
                idx = random.randint(0, len(self.all_videos) - 1)
                should_load_diff_video = False

            vidname = self.all_videos[idx]
            img_names = list(glob(join(vidname, '*.jpg')))
            
            if len(img_names) <= 3 * syncnet_T:
                should_load_diff_video = True
                print('The video has not enough frames, {0}'.format(vidname))
                continue
            
            img_name = random.choice(img_names)
            correct_window_images = self.get_window(img_name)
            while correct_window_images is None:
              img_name = random.choice(img_names)
              correct_window_images = self.get_window(img_name)

            chosen_id = self.get_frame_id(img_name)

            wrong_img_name = random.choice(img_names)          
            wrong_img_id = self.get_frame_id(wrong_img_name)
            wrong_window_images = self.get_window(wrong_img_name)
            
            """
            Changed by eddy, the following are the original codes, it uses random to get the wrong_img_name, 
            this might get an image that very close to the correct image(the next frame) which is a bit hard to learn.
            Eddy introduced a new algorithm that to get a image a bit futher from the img_name to have enough difference,
            this might help the model to converge.
            """
            attempt = 0
            while wrong_img_name == img_name or abs(wrong_img_id - chosen_id) < 15 or wrong_window_images is None:
                  wrong_img_name = random.choice(img_names)
                  wrong_img_id = self.get_frame_id(wrong_img_name)
                  wrong_window_images = self.get_window(wrong_img_name)
                  attempt += 1
                  if attempt > 5:
                      should_load_diff_video = True
                      break
            
            if should_load_diff_video:
                continue

            alignment_score = self.compute_alignment_score(chosen_id, wrong_img_id, 50)
            
            # We firstly to learn all the positive, once it reach the loss of less than 0.2, we incrementally add some negative samples 10% per step
            good_or_bad = True
            good_or_bad = random.choice(samples)
            
            
            #print('The chosen, wrong and alignment score', chosen_id, wrong_img_id, alignment_score)

            if good_or_bad:
                regression_y = 1.0
                classification_y = 1
                window_fnames = correct_window_images
            else:
                regression_y = alignment_score
                classification_y = 0
                
                if negative_data_mode == "SIMPLE":
                  window_fnames = correct_window_images[::-1] # reverse it
                elif negative_data_mode == "MEDIUM":
                  shuffled_list = correct_window_images.copy()[:]
                  while True:
                      random.shuffle(shuffled_list)
                      if shuffled_list != correct_window_images:
                          break
                  window_fnames = shuffled_list
                else:
                  window_fnames = wrong_window_images
                                

            face_window = []

            all_read = True
            for fname in window_fnames:
                if fname in face_image_cache:
                    img = face_image_cache[fname]
                else:
                    img = cv2.imread(fname)
                    if img is None:
                        all_read = False
                        break
                    try:
                        img = cv2.resize(img, (hparams.img_size * self.img_size_factor, hparams.img_size * self.img_size_factor))                            
                        
                        img = self.apply_lip_mask_single(img)
                        if len(face_image_cache) < hparams.syncnet_image_cache_size:
                          face_image_cache[fname] = img  # Cache the resized image
                        
                    except Exception as e:
                        all_read = False
                        break
                
                '''
                Data augmentation
                0 means unchange
                1 for grayscale
                2 for brightness
                3 for contrast
                '''
                if self.use_augmentation:
                  option = random.choices([0, 0, 0, 0, 0, 0, 0, 0, 4, 4])[0] 
                  
                  if option == 1:
                      img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                      img = cv2.merge([img_gray, img_gray, img_gray])
                  elif option == 2:
                      brightness_factor = np.random.uniform(0.7, 1.3)
                      img = cv2.convertScaleAbs(img, alpha=brightness_factor, beta=0)
                  elif option == 3:
                      contrast_factor = np.random.uniform(0.7, 1.3)
                      img = cv2.convertScaleAbs(img, alpha=contrast_factor, beta=0)
                  elif option == 4:
                      angle = np.random.uniform(-15, 15)  # Random angle between -15 and 15 degrees

                      # Get the image dimensions
                      (h, w) = img.shape[:2]

                      # Calculate the center of the image
                      center = (w // 2, h // 2)

                      # Get the rotation matrix
                      rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

                      # Perform the rotation
                      img = cv2.warpAffine(img, rotation_matrix, (w, h))

                face_window.append(img)

            if not all_read: continue

            try:
                wavpath = join(vidname, "audio.wav")

                if wavpath in orig_mel_cache:
                    orig_mel = orig_mel_cache[wavpath]
                    #print('The audio cache hit ', wavpath)
                else:
                    wav = audio.load_wav(wavpath, hparams.sample_rate)
                    orig_mel = audio.melspectrogram(wav).T
                    if len(orig_mel_cache) < hparams.syncnet_audio_cache_size:
                      orig_mel_cache[wavpath] = orig_mel
                
            except Exception as e:
                should_load_diff_video = True
                print('The audio is invalid, file name {0}, will retry with a differnt video'.format(join(vidname, "audio.wav")))
                continue
            
            mel = self.crop_audio_window(orig_mel.copy(), img_name)
            

            if (mel.shape[0] != syncnet_mel_step_size):
                should_load_diff_video = True
                #print("This specific audio is invalid {0}".format(join(vidname, "audio.wav")))
                continue
            
            
            #face_window = self.apply_lip_mask(face_window)
            #save_sample_images(face_window)
            
            # H x W x 3 * T
            x = np.concatenate(face_window, axis=2) / 255.
            x = x.transpose(2, 0, 1)
            x = x[:, x.shape[1]//2:]

            # Each face_window contains 5 images and each image has 3 channels, concatenate them through the channel channel yield a 15 channels image, the x shape is 15x96x192
            x = torch.FloatTensor(x)
            mel = torch.FloatTensor(mel.T).unsqueeze(0)

            return x, mel, regression_y, classification_y

    def blackout_non_lip(self, img, bbox):
        """
        Black out areas outside the lips region.
        
        Args:
            img (np.ndarray): Input image as a numpy array of shape (H, W, C).
            bbox (list or tuple): Normalized bounding box [x_min, y_min, x_max, y_max] 
                                  with values between 0 and 1.
        
        Returns:
            np.ndarray: The image with non-lip areas blacked out.
        """
        H, W = img.shape[:2]
        # Convert normalized coordinates to pixel coordinates.
        x_min = int(bbox[0])
        y_min = int(bbox[1])
        x_max = int(bbox[2])
        y_max = int(bbox[3])
        
        # Create a binary mask with 1s in the lips region and 0s elsewhere.
        mask = np.zeros((H, W), dtype=np.float32)
        mask[y_min:y_max, x_min:x_max] = 1.0
        
        # If the image has multiple channels, expand the mask.
        if img.ndim == 3:
            mask = np.expand_dims(mask, axis=-1)
        
        # Multiply the image by the mask to blackout non-lip regions.
        img_masked = img * mask
        return img_masked

    def apply_lip_mask_single(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = self.face_mesh.process(frame_rgb)

        if results.multi_face_landmarks:
            # Get the mouth landmarks (MediaPipe Face Mesh landmarks for mouth are from 61 to 80)
            mouth_points = []
            h, w, _ = frame.shape
            split_row = h // 2
            for idx in LIPS_LANDMARKS:
                lm = results.multi_face_landmarks[0].landmark[idx]
                x, y = int(lm.x * w), int(lm.y * h)
                mouth_points.append([x, y])

            # Convert the list of mouth points to a NumPy array for easier manipulation.
            mouth_points = np.array(mouth_points)

            # Compute the bounding rectangle coordinates.
            x_min = np.min(mouth_points[:, 0])
            x_max = np.max(mouth_points[:, 0])
            y_min = np.min(mouth_points[:, 1])
            y_max = np.max(mouth_points[:, 1])

            # Calculate the width and height of the mouth region.
            width = x_max - x_min
            height = y_max - y_min

            # Define a padding factor (e.g., 50% larger in each direction).
            pad_width_factor = 0.4  # Adjust this value as needed.
            pad_height_factor = 0.4  # Adjust this value as needed.
            pad_x = int(width * pad_width_factor)
            pad_y = int(height * pad_height_factor)

            # Expand the rectangle and ensure the coordinates stay within frame boundaries.
            x_min_expanded = max(x_min - pad_x, 0)
            y_min_expanded = max(y_min - pad_y, 0)
            x_max_expanded = min(x_max + pad_x, w)
            y_max_expanded = min(y_max + pad_y, h)

            bbox = [x_min_expanded, y_min_expanded, x_max_expanded, y_max_expanded]
            img_masked = self.blackout_non_lip(frame, bbox)
            
        else:
            bbox = [0.0, 0.0, 1.0, 1.0]
        
        return img_masked
      
      
    def apply_lip_mask(self, window):
            masked_frames = []

            for frame in window:
                #frame_rgb = (frame * 255).astype(np.uint8)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                results = self.face_mesh.process(frame_rgb)

                if results.multi_face_landmarks:
                    # Get the mouth landmarks (MediaPipe Face Mesh landmarks for mouth are from 61 to 80)
                    mouth_points = []
                    h, w, _ = frame.shape
                    split_row = h // 2
                    for idx in LIPS_LANDMARKS:
                        lm = results.multi_face_landmarks[0].landmark[idx]
                        x, y = int(lm.x * w), int(lm.y * h)
                        mouth_points.append([x, y])

                    # Convert the list of mouth points to a NumPy array for easier manipulation.
                    mouth_points = np.array(mouth_points)

                    # Compute the bounding rectangle coordinates.
                    x_min = np.min(mouth_points[:, 0])
                    x_max = np.max(mouth_points[:, 0])
                    y_min = np.min(mouth_points[:, 1])
                    y_max = np.max(mouth_points[:, 1])

                    # Calculate the width and height of the mouth region.
                    width = x_max - x_min
                    height = y_max - y_min

                    # Define a padding factor (e.g., 50% larger in each direction).
                    pad_width_factor = 0.4  # Adjust this value as needed.
                    pad_height_factor = 0.5  # Adjust this value as needed.
                    pad_x = int(width * pad_width_factor)
                    pad_y = int(height * pad_height_factor)

                    # Expand the rectangle and ensure the coordinates stay within frame boundaries.
                    x_min_expanded = max(x_min - pad_x, 0)
                    y_min_expanded = max(y_min - pad_y, 0)
                    x_max_expanded = min(x_max + pad_x, w)
                    y_max_expanded = min(y_max + pad_y, h)

                    bbox = [x_min_expanded, y_min_expanded, x_max_expanded, y_max_expanded]
                    img_masked = self.blackout_non_lip(frame, bbox)
                    masked_frames.append(img_masked)
                else:
                    bbox = [0.0, 0.0, 1.0, 1.0]
                    masked_frames.append(frame)
                    
            return masked_frames

def save_sample_images(x):
    
    for idx, img in enumerate(x):
      # Save the concatenated image
      cv2.imwrite('img_{0}_concatenated.jpg'.format(idx), img)
    