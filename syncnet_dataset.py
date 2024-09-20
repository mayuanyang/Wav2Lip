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

face_image_cache =  multiprocessing.Manager().dict()
file_exist_cache = multiprocessing.Manager().dict()
orig_mel_cache = multiprocessing.Manager().dict()

"""
The FPS is set to 25 for video, 5/25 is 0.2, we need to have 0.2 seconds for the audio,
because the audio mel spectrogram ususlly has 80 frame per seconds, so 16/80 is 0.2 seconds
"""
syncnet_T = 5
syncnet_mel_step_size = 16
samples = [True, True,True, True,True, False,False, False, False, False]

class Dataset(object):
    
    def __init__(self, split, data_root, train_root):
        print('-----')
        self.all_videos = get_image_list(data_root, split, train_root)
        

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
                while wrong_img_name == img_name or abs(wrong_img_id - chosen_id) < 5 or wrong_window_images is None:
                      #print('The selected wrong image {0} is not far engough from {1}, diff {2}, window is None {3}'.format(wrong_img_id, chosen_id, abs(wrong_img_id - chosen_id), wrong_window_images is None))
                      wrong_img_name = random.choice(img_names)
                      wrong_img_id = self.get_frame_id(wrong_img_name)
                      wrong_window_images = self.get_window(wrong_img_name)
                      attempt += 1
                      if attempt > 5:
                          should_load_diff_video = True
                          break
                
                if should_load_diff_video:
                    continue

                
                # We firstly to learn all the positive, once it reach the loss of less than 0.2, we incrementally add some negative samples 10% per step
                good_or_bad = True
                good_or_bad = random.choice(samples)

                if good_or_bad:
                    y = 1
                    window_fnames = correct_window_images
                else:
                    y = 0
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
                            img = cv2.resize(img, (hparams.img_size, hparams.img_size))                            
                            
                            if len(face_image_cache) < hparams.image_cache_size:
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
                    option = random.choices([0, 1, 2, 3, 4])[0] 
                    
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
                        angle = np.random.uniform(1, 15)
                        angle = np.random.uniform(-angle, angle)

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
                        if len(orig_mel_cache) < hparams.audio_cache_size:
                          orig_mel_cache[wavpath] = orig_mel
                    
                except Exception as e:
                    should_load_diff_video = True
                    print('The audio is invalid, file name {0}, will retry with a differnt video'.format(join(vidname, "audio.wav")))
                    continue
                
                mel = self.crop_audio_window(orig_mel.copy(), img_name)

                if (mel.shape[0] != syncnet_mel_step_size):
                    should_load_diff_video = True
                    print("This specific audio is invalid {0}".format(join(vidname, "audio.wav")))
                    continue

                if idx % 1000 == 0:
                  save_sample_images(np.concatenate(face_window, axis=2), idx, mel)

                # H x W x 3 * T
                x = np.concatenate(face_window, axis=2) / 255.
                x = x.transpose(2, 0, 1)
                x = x[:, x.shape[1]//2:]

                x = torch.FloatTensor(x)
                mel = torch.FloatTensor(mel.T).unsqueeze(0)

                return x, mel, y

def save_sample_images(x, idx, orig_mel):
    
    x = x.transpose(2, 0, 1)  # Now x is of shape (3*T, H, W)

    x = x[:, x.shape[1] // 2:, :]

    x_final = x.transpose(1, 2, 0)  # Transpose back to H x W x C

    # Initialize an empty list to store each image
    images = []

    for i in range(5):  # There are 5 images, hence 5 sets of 3 channels
        img = x_final[:, :, i*3:(i+1)*3]  # Select the i-th image channels
        img = img.astype(np.uint8)  # Convert back to uint8 for saving
        images.append(img)

    # Concatenate the images horizontally
    concatenated_image = np.hstack(images)

    # Save the concatenated image
    cv2.imwrite('img_{0}_concatenated.jpg'.format(idx), concatenated_image)
    
    # Persist the mel-spectrogram as an image
    plt.figure(figsize=(10, 4))
    plt.imshow(orig_mel.T, aspect='auto', origin='lower', interpolation='none')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel-Spectrogram')
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.tight_layout()

    # Save as image
    plt.savefig("img_{0}_mel_spectrogram.png".format(idx))
    plt.close()