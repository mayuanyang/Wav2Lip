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
from torch import nn
import traceback
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter

image_cache = multiprocessing.Manager().dict()
orig_mel_cache = multiprocessing.Manager().dict()

syncnet_T = 5
syncnet_mel_step_size = 16


cross_entropy_loss = nn.CrossEntropyLoss()
recon_loss = nn.L1Loss()

class Dataset(object):
    def __init__(self, split, data_root, train_root, use_augmentation):
        self.all_videos = get_image_list(data_root, split, train_root)
        self.use_augmentation = use_augmentation

    def get_frame_id(self, frame):
        return int(basename(frame).split('.')[0])

    def get_window(self, start_frame):
        start_id = self.get_frame_id(start_frame)
        vidname = dirname(start_frame)

        window_fnames = []
        for frame_id in range(start_id, start_id + syncnet_T):
            frame = join(vidname, '{}.jpg'.format(frame_id))
            if not isfile(frame):
                return None
            window_fnames.append(frame)
        return window_fnames
    
    


    def read_window(self, window_fnames):
        if window_fnames is None: return None
        window = []
        for fname in window_fnames:
                #print('The image name ', fname)
                if fname in image_cache:
                    img = image_cache[fname]
                    #print('The image cache hit ', fname)
                else:
                    img = cv2.imread(fname)
                    if img is None:
                        break
                    try:
                        img = cv2.resize(img, (hparams.img_size, hparams.img_size))                       
                        if len(image_cache) < hparams.image_cache_size:
                          image_cache[fname] = img  # Cache the resized image and prevent OOM
                    
                        
                    except Exception as e:
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

                window.append(img)

        return window

    def crop_audio_window(self, spec, start_frame):
        if type(start_frame) == int:
            start_frame_num = start_frame
        else:
            start_frame_num = self.get_frame_id(start_frame) # 0-indexing ---> 1-indexing
        start_idx = int(80. * (start_frame_num / float(hparams.fps)))
        
        end_idx = start_idx + syncnet_mel_step_size

        return spec[start_idx : end_idx, :]

    def get_segmented_mels(self, spec, start_frame):
        mels = []
        assert syncnet_T == 5
        start_frame_num = self.get_frame_id(start_frame) + 1 # 0-indexing ---> 1-indexing
        if start_frame_num - 2 < 0: return None
        for i in range(start_frame_num, start_frame_num + syncnet_T):
            m = self.crop_audio_window(spec, i - 2)
            if m.shape[0] != syncnet_mel_step_size:
                return None
            mels.append(m.T)

        mels = np.asarray(mels)

        return mels

    def prepare_window(self, window):
        """
        3 x T x H x W
        Normalization: The pixel values of the images are divided by 255 to normalize them from a range of [0, 255] to [0, 1]. 
        This is a common preprocessing step for image data in machine learning to help the model converge faster during training.
        """
        x = np.asarray(window) / 255.

        """
        Transposition: The method transposes the dimensions of the array using np.transpose(x, (3, 0, 1, 2)).
        The original shape of x is assumed to be (T, H, W, C) where:
        T is the number of images (time steps if treating images as a sequence).
        H is the height of the images.
        W is the width of the images.
        C is the number of color channels (typically 3 for RGB images).
        The transposition changes the shape to (C, T, H, W) which means:
        C (number of channels) comes first.
        T (number of images) comes second.
        H (height of images) comes third.
        W (width of images) comes fourth.
        """
        x = np.transpose(x, (3, 0, 1, 2))

        return x

    def __len__(self):
        return len(self.all_videos)
    
    def get_ref_images(self, forbidden_images, img_names):
      
      # Filter out the forbidden images from img_names
      available_images = [img for img in img_names if img not in forbidden_images]

      # Initialize the list for reference window filenames
      ref_window_fnames = []

      needed_length = syncnet_T

      # Check if we have enough unique images
      if len(available_images) >= needed_length:
        # Randomly select unique images if there are enough available
        ref_window_fnames = random.sample(available_images, needed_length)
      else:
        # If not enough unique images, fill up with duplicates as necessary
        while len(ref_window_fnames) < needed_length:
          ref_window_fnames.extend(available_images)
          # Trim the list to the exact needed length
          ref_window_fnames = ref_window_fnames[:needed_length]

      return ref_window_fnames  

    def __getitem__(self, idx):
        #start_time = time.perf_counter()
        
        should_load_diff_video = False

        while 1:
            if should_load_diff_video:
                idx = random.randint(0, len(self.all_videos) - 1)
                should_load_diff_video = False

            vidname = self.all_videos[idx]

            img_names = list(glob(join(vidname, '*.jpg')))
            
            if len(img_names) <= 3 * syncnet_T:
                print('The length', len(img_names), vidname)
                should_load_diff_video = True
                continue
            

            img_name = random.choice(img_names)
            wrong_img_name = random.choice(img_names)
            while wrong_img_name == img_name:
                wrong_img_name = random.choice(img_names)

            window_fnames = self.get_window(img_name)
            wrong_window_fnames = self.get_window(wrong_img_name)

            if window_fnames is None or wrong_window_fnames is None:
                should_load_diff_video = True
                continue

            window = self.read_window(window_fnames)
            if window is None:
                should_load_diff_video = True
                continue

            wrong_window = self.read_window(wrong_window_fnames)
            if wrong_window is None:
                should_load_diff_video = True
                continue
            
            # Create a set of forbidden image names for faster lookup
            forbidden_images = set(window_fnames).union(set(wrong_window_fnames))
            
            # Initialize the list for reference window filenames
            ref1_window_fnames = self.get_ref_images(forbidden_images, img_names)

            forbidden_images = set(window_fnames).union(set(wrong_window_fnames)).union(set(ref1_window_fnames))
            ref2_window_fnames = self.get_ref_images(forbidden_images, img_names)
            
            ref1_window = self.read_window(ref1_window_fnames)
            if ref1_window is None:
                should_load_diff_video = True
                continue

            ref2_window = self.read_window(ref2_window_fnames)
            if ref2_window is None:
                should_load_diff_video = True
                continue
            
            try:
                wavpath = join(vidname, "audio.wav")

                if wavpath in orig_mel_cache:
                    orig_mel = orig_mel_cache[wavpath]
                    #print('The audio cache hit ', wavpath)
                else:
                    wav = audio.load_wav(wavpath, hparams.sample_rate)
                    orig_mel = audio.melspectrogram(wav).T
                    if len(orig_mel_cache) < hparams.audio_cache_size:
                      orig_mel_cache[wavpath] = orig_mel

                mel = self.crop_audio_window(orig_mel.copy(), img_name)
                
                if (mel.shape[0] != syncnet_mel_step_size):
                    continue

                indiv_mels = self.get_segmented_mels(orig_mel.copy(), img_name)
                if indiv_mels is None: continue

                window = self.prepare_window(window)
                y = window.copy()


                '''
                Set the second half of the images to be black, the window has 5 images
                The wrong_window contains images that do not align with the audio
                x contains 5 images and 6 channels each, the 5 images from window with second half black out, the images from wrong window are merged via channels
                so the final x still got 5 images and with the merged window and wrong_window
                indiv_mels contains the corresponding audio for the given window
                y is the window that without the second half black out
                '''

                window = self.apply_gaussian_blur_to_bottom_half_vectorized(window)

                wrong_window = self.prepare_window(wrong_window)

                ref1_window = self.prepare_window(ref1_window)

                ref2_window = self.prepare_window(ref2_window)

                # do not include the correct window so that no second half black
                x = np.concatenate([window, wrong_window, ref1_window, ref2_window], axis=0) # Concat via the channel axis
                

                x = torch.FloatTensor(x)
                mel = torch.FloatTensor(mel.T).unsqueeze(0)

                indiv_mels = torch.FloatTensor(indiv_mels).unsqueeze(1)

                y = torch.FloatTensor(y)

                return x, indiv_mels, mel, y

            except Exception as e:
                #print('An error has occured', vidname, img_name, wrong_img_name)
                traceback.print_exc()   
                continue
    
    def apply_gaussian_blur_to_bottom_half_vectorized(self, window, sigma=6):
        blurred_window = window.copy()
        split_row = blurred_window.shape[-2] // 2  # e.g., 96 for 192 height

        bottom_half = blurred_window[:, :, split_row:, :]  # Shape: (channels, frames, 96, 192)
        blurred_bottom_half = gaussian_filter(bottom_half, sigma=(0, 0, sigma, sigma))
        blurred_window[:, :, split_row:, :] = blurred_bottom_half

        return blurred_window


