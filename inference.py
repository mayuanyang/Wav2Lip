from os import listdir, path
from os.path import dirname, join, basename, isfile
import numpy as np
import scipy, cv2, os, sys, argparse, audio
import json, subprocess, random, string
from tqdm import tqdm
from glob import glob
import torch, face_detection
from models import ResUNet384V2
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet
from PIL import Image
from scipy.ndimage import gaussian_filter

import platform

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description='Inference code to lip-sync videos in the wild using Wav2Lip models')

parser.add_argument('--checkpoint_path', type=str, 
          help='Name of saved checkpoint to load weights from', required=True)

parser.add_argument('--lora_checkpoint_path', type=str, 
          help='Name of saved lora checkpoint to load weights from', required=False)

parser.add_argument('--face', type=str, 
          help='Filepath of video/image that contains faces to use', required=True)
parser.add_argument('--audio', type=str, 
          help='Filepath of video/audio file to use as raw audio source', required=True)
parser.add_argument('--outfile', type=str, help='Video path to save result. See default for an e.g.', 
                default='results/result_voice.mp4')

parser.add_argument('--static', type=bool, 
          help='If True, then use only first video frame for inference', default=False)

parser.add_argument('--use_ref_img', default=True, type=str2bool)

parser.add_argument('--fps', type=float, help='Can be specified only if input is a static image (default: 25)', 
          default=25., required=False)

parser.add_argument('--pads', nargs='+', type=int, default=[0, 10, 0, 0], 
          help='Padding (top, bottom, left, right). Please adjust to include chin at least')

parser.add_argument('--face_det_batch_size', type=int, 
          help='Batch size for face detection', default=16)
parser.add_argument('--wav2lip_batch_size', type=int, help='Batch size for Wav2Lip model(s)', default=16)

parser.add_argument('--resize_factor', default=1, type=int, 
      help='Reduce the resolution by this factor. Sometimes, best results are obtained at 480p or 720p')

parser.add_argument('--crop', nargs='+', type=int, default=[0, -1, 0, -1], 
          help='Crop video to a smaller region (top, bottom, left, right). Applied after resize_factor and rotate arg. ' 
          'Useful if multiple face present. -1 implies the value will be auto-inferred based on height, width')

parser.add_argument('--box', nargs='+', type=int, default=[-1, -1, -1, -1], 
          help='Specify a constant bounding box for the face. Use only as a last resort if the face is not detected.'
          'Also, might work only if the face is not moving around much. Syntax: (top, bottom, left, right).')

parser.add_argument('--rotate', default=False, action='store_true',
          help='Sometimes videos taken from a phone can be flipped 90deg. If true, will flip video right by 90deg.'
          'Use if you get a flipped result, despite feeding a normal looking video')

parser.add_argument('--nosmooth', default=False, action='store_true',
          help='Prevent smoothing face detections over a short temporal window')

parser.add_argument('--model_layers', default=2, type=int, 
      help='The number of layers that the model has')

parser.add_argument('--use_esrgan', default=False, type=str2bool)

parser.add_argument('--iteration', type=int, help='Number of iteration to inference', default=2)

args = parser.parse_args()
args.img_size = 384

if os.path.isfile(args.face) and args.face.split('.')[1] in ['jpg', 'png', 'jpeg']:
  args.static = True

def get_smoothened_boxes(boxes, T):
  for i in range(len(boxes)):
    if i + T > len(boxes):
      window = boxes[len(boxes) - T:]
    else:
      window = boxes[i : i + T]
    boxes[i] = np.mean(window, axis=0)
  return boxes

def face_detect(images):
  detector = face_detection.FaceAlignment(face_detection.LandmarksType._2D, 
                      flip_input=False, device=device)

  batch_size = args.face_det_batch_size
  
  while 1:
    predictions = []
    try:
      for i in tqdm(range(0, len(images), batch_size)):
        predictions.extend(detector.get_detections_for_batch(np.array(images[i:i + batch_size])))
    except RuntimeError:
      if batch_size == 1: 
        raise RuntimeError('Image too big to run face detection on GPU. Please use the --resize_factor argument')
      batch_size //= 2
      print('Recovering from OOM error; New batch size: {}'.format(batch_size))
      continue
    break

  results = []
  pady1, pady2, padx1, padx2 = args.pads
  for rect, image in zip(predictions, images):
    if rect is None:
      cv2.imwrite('temp/faulty_frame.jpg', image) # check this frame where the face was not detected.
      raise ValueError('Face not detected! Ensure the video contains a face in all the frames.')

    y1 = max(0, rect[1] - pady1)
    y2 = min(image.shape[0], rect[3] + pady2)
    x1 = max(0, rect[0] - padx1)
    x2 = min(image.shape[1], rect[2] + padx2)
    
    results.append([x1, y1, x2, y2])

  boxes = np.array(results)
  if not args.nosmooth: boxes = get_smoothened_boxes(boxes, T=5)
  results = [[image[y1: y2, x1:x2], (y1, y2, x1, x2)] for image, (x1, y1, x2, y2) in zip(images, boxes)]

  del detector
  return results 

def prepare_window(window):
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

def apply_gaussian_blur_to_bottom_half_vectorized(window, sigma=8):
        blurred_window = window.copy()
        blurred_window = prepare_window(blurred_window)
        split_row = blurred_window.shape[-2] // 2  # e.g., 96 for 192 height

        bottom_half = blurred_window[:, :, split_row:, :]  # Shape: (channels, frames, 96, 192)
        blurred_bottom_half = gaussian_filter(bottom_half, sigma=(0, 0, sigma, sigma))
        blurred_window[:, :, split_row:, :] = blurred_bottom_half

        blurred_window = np.transpose(blurred_window, (1, 2, 3, 0)) * 255

        # frames, H, W, c = blurred_window.shape

        # output_dir = "checkpoints/wav2lip_checkpoint/no-blackout/"
        # for i in range(frames):
        #         # Extract the i-th frame
        #         frame = blurred_window[i]
        #         print('The image shape', frame.shape)

        #         # # Convert RGB to BGR since OpenCV uses BGR format
        #         #frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        #         # Construct the filename with zero-padding (e.g., frame_0001.png)
        #         filename = f"{i:04d}.png"
        #         filepath = os.path.join(output_dir, filename)

        #         # Save the image using OpenCV's imwrite
        #         success = cv2.imwrite(filepath, frame)

        #         print(f'Saving {success} to {filepath}')

        return blurred_window


def datagen(frames, mels, use_ref_img, ref_pool, iteration):
  img_batch, mel_batch, frame_batch, coords_batch, ref_batch, ref_batch2 = [], [], [], [], [], []

  if args.box[0] == -1:
    if not args.static:
      face_det_results = face_detect(frames) # BGR2RGB for CNN face detection
    else:
      face_det_results = face_detect([frames[0]])
  else:
    print('Using the specified bounding box instead of face detection...')
    y1, y2, x1, x2 = args.box
    face_det_results = [[f[y1: y2, x1:x2], (y1, y2, x1, x2)] for f in frames]

  should_fill_ref_pool = len(ref_pool) == 0

  ids_in_ref = []

  for i, m in enumerate(mels):
    idx = 0 if args.static else i%len(frames)
    frame_to_save = frames[idx].copy()
    face, coords = face_det_results[idx].copy()

    face = cv2.resize(face, (args.img_size, args.img_size))
    # Generate a unique filename
    if should_fill_ref_pool:
      ref_pool.append(face)
    
    
    if use_ref_img:
      if iteration <= 0:
          rdn_idx = random.randint(0, len(frames) - 1)

          #print('The rdn_idx and cache 1', rdn_idx, ids_in_ref)
          while rdn_idx == idx:
              rdn_idx = random.randint(0, len(frames) - 1)
          
          ref_face, _ = face_det_results[rdn_idx].copy()
          ref_face = cv2.resize(ref_face, (args.img_size, args.img_size))
          ref_batch.append(ref_face)
          ids_in_ref.append(rdn_idx)

          rdn_idx = random.randint(0, len(frames) - 1)

          #print('The rdn_idx and cache 2', rdn_idx, ids_in_ref)
          while rdn_idx == idx or rdn_idx in ids_in_ref:
              rdn_idx = random.randint(0, len(frames) - 1)
          
          ref_face2, _ = face_det_results[rdn_idx].copy()
          ref_face2 = cv2.resize(ref_face2, (args.img_size, args.img_size))
          ref_batch2.append(ref_face2)
          ids_in_ref.append(rdn_idx)
      else:
          rdn_idx = random.randint(0, len(ref_pool) - 1)

          #print('The rdn_idx and cache 3', rdn_idx, ids_in_ref)
          while rdn_idx == idx or rdn_idx in ids_in_ref:
              rdn_idx = random.randint(0, len(ref_pool) - 1)
          
          ref_batch.append(ref_pool[rdn_idx])
          ids_in_ref.append(rdn_idx)

          rdn_idx = random.randint(0, len(ref_pool) - 1)

          #print('The rdn_idx and cache 4', rdn_idx, ids_in_ref)
          while rdn_idx == idx or rdn_idx in ids_in_ref:
              rdn_idx = random.randint(0, len(ref_pool) - 1)
          
          ref_batch2.append(ref_pool[rdn_idx])
          ids_in_ref.append(rdn_idx)

    else:
      ref_batch.append(face)
      ref_batch2.append(face)
    
      
    img_batch.append(face)
    mel_batch.append(m)
    frame_batch.append(frame_to_save)
    coords_batch.append(coords)

    if len(img_batch) >= args.wav2lip_batch_size:
      img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)
      ref_batch = np.asarray(ref_batch)
      ref_batch2 = np.asarray(ref_batch2)

      img_masked = img_batch.copy()

      # img_masked[:, args.img_size//2:] = 0
      img_masked = apply_gaussian_blur_to_bottom_half_vectorized(img_masked)
      print('The image shape 1', img_masked.shape)

      img_batch = np.concatenate((img_masked, img_batch, ref_batch, ref_batch2), axis=3) / 255.
      mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

      yield img_batch, mel_batch, frame_batch, coords_batch
      img_batch, mel_batch, frame_batch, coords_batch, ref_batch, ref_batch2 = [], [], [], [], [], []
    
    ids_in_ref = []

  if len(img_batch) > 0:
    img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)
    img_masked = img_batch.copy()

    
    #img_masked[:, args.img_size//2:] = 0
    img_masked = apply_gaussian_blur_to_bottom_half_vectorized(img_masked)

    print('The image shape 2', img_masked.shape)

    if use_ref_img:
      ref_batch = np.asarray(ref_batch)
      ref_batch2 = np.asarray(ref_batch2)
      img_batch = np.concatenate((img_masked, img_batch, ref_batch, ref_batch2), axis=3) / 255.
    else:
      img_batch = np.concatenate((img_masked, img_batch, ref_batch, ref_batch2), axis=3) / 255.
    mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

    yield img_batch, mel_batch, frame_batch, coords_batch

mel_step_size = 16
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} for inference.'.format(device))

def _load(checkpoint_path):
  if device == 'cuda':
    checkpoint = torch.load(checkpoint_path)
  else:
    checkpoint = torch.load(checkpoint_path,
                map_location=lambda storage, loc: storage)
  return checkpoint

def load_model(path, lora_path=None, model_layers=1):
  model = ResUNet384V2(model_layers)
  print("Load checkpoint from: {}".format(path))
  checkpoint = _load(path)
  s = checkpoint["state_dict"]
  new_s = {}
  for k, v in s.items():
    new_s[k.replace('module.', '')] = v
  model.load_state_dict(new_s, strict=False)
  if lora_path:
    print('I got lora')
    lora_params = torch.load(lora_path)
    model.load_state_dict(lora_params, strict=False)
  

  model = model.to(device)
  return model.eval()

def load_esrgan_model(checkpoint_path='checkpoints/RealESRGAN_x4plus.pth', device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Loads the pre-trained Real-ESRGAN model for image enhancement.
    """
    # Define the model architecture for Real-ESRGAN
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)

    # Instantiate RealESRGANer with the loaded model
    esrgan = RealESRGANer(
        scale=4,  # 4x upscaling
        model_path=checkpoint_path,
        model=model,
        tile=0,  # No tiling by default
        tile_pad=10,
        pre_pad=0,
        half=True,  # Use half-precision if supported
        device=device
    )

    print("Real-ESRGAN model loaded successfully.")
    return esrgan

def enhance_image_with_esrgan(model, image):
    """
    Enhances an image using the Real-ESRGAN model.
    Args:
        model (RealESRGANer): The loaded Real-ESRGAN model.
        image (PIL Image or numpy array): Image to enhance.
    Returns:
        Enhanced image as a PIL Image.
    """
    # Convert the image to a NumPy array if it's a PIL Image
    if isinstance(image, Image.Image):
        image = np.array(image)

    # Enhance the image using Real-ESRGAN
    enhanced_img_np, _ = model.enhance(image, outscale=2)  # Scale factor of 4

    # Convert the enhanced image back to a PIL image
    enhanced_img_pil = Image.fromarray(enhanced_img_np)
    return enhanced_img_pil

def main():
  print('use gan', args.use_esrgan, args.use_ref_img)
  global esrgan_model
  if args.use_esrgan:
    esrgan_model = load_esrgan_model()  # Load the ESRGAN model
    print("ESRGAN model loaded for image enhancement.")

  if not os.path.isfile(args.face):
    raise ValueError('--face argument must be a valid path to video/image file')

  elif args.face.split('.')[1] in ['jpg', 'png', 'jpeg']:
    full_frames = [cv2.imread(args.face)]
    fps = args.fps


  if not args.audio.endswith('.wav'):
    print('Extracting raw audio...')
    command = 'ffmpeg -y -i {} -strict -2 {}'.format(args.audio, 'temp/temp.wav')

    subprocess.call(command, shell=True)
    args.audio = 'temp/temp.wav'

  wav = audio.load_wav(args.audio, 16000)
  mel = audio.melspectrogram(wav)

  if np.isnan(mel.reshape(-1)).sum() > 0:
    raise ValueError('Mel contains nan! Using a TTS voice? Add a small epsilon noise to the wav file and try again')

  video_stream = cv2.VideoCapture(args.face)
  fps = video_stream.get(cv2.CAP_PROP_FPS)

  print('Initial fps', fps)

  mel_chunks = []
  mel_idx_multiplier = 80./fps 
  i = 0
  while 1:
    start_idx = int(i * mel_idx_multiplier)
    if start_idx + mel_step_size > len(mel[0]):
      mel_chunks.append(mel[:, len(mel[0]) - mel_step_size:])
      break
    mel_chunks.append(mel[:, start_idx : start_idx + mel_step_size])
    i += 1

  batch_size = args.wav2lip_batch_size
  ref_pool = []

  model = load_model(args.checkpoint_path, args.lora_checkpoint_path, args.model_layers)
  print ("Model loaded")

  for x in range(args.iteration):
    
    if x > 0:
      print('Reading video frames...')
      video_stream = cv2.VideoCapture(f'temp/result_{x-1}.avi')
      fps = video_stream.get(cv2.CAP_PROP_FPS)
    

    full_frames = []

    index = 0
    while 1:
      still_reading, frame = video_stream.read()
      if not still_reading:
        video_stream.release()
        break
      if args.resize_factor > 1:
        frame = cv2.resize(frame, (frame.shape[1]//args.resize_factor, frame.shape[0]//args.resize_factor))

      if args.rotate:
        frame = cv2.rotate(frame, cv2.cv2.ROTATE_90_CLOCKWISE)

      y1, y2, x1, x2 = args.crop
      if x2 == -1: x2 = frame.shape[1]
      if y2 == -1: y2 = frame.shape[0]

      frame = frame[y1:y2, x1:x2]
      
      full_frames.append(frame)

      index += 1
    
    print ("Number of frames available for inference: "+str(len(full_frames)))

    temp = full_frames[1: -1:]
    print(f'Iteration {x}, length of chunks {len(mel_chunks)} and length of full frames {len(temp)} and fps {fps}')
    
    input_frames = temp[:len(mel_chunks)]  

    gen = datagen(input_frames.copy(), mel_chunks, args.use_ref_img, ref_pool, x)

    for i, (img_batch, mel_batch, frames, coords) in enumerate(tqdm(gen, 
                        total=int(np.ceil(float(len(mel_chunks))/batch_size)))):
      if i == 0:
        frame_h, frame_w = input_frames[0].shape[:-1]
        out = cv2.VideoWriter(f'temp/result_{x}.avi', 
                    cv2.VideoWriter_fourcc(*'DIVX'), fps, (frame_w, frame_h))
      
      img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(device)
      
      mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(device)

      
      with torch.no_grad():
        pred = model(mel_batch, img_batch)

      pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.
      
      i = 0
      for p, f, c in zip(pred, frames, coords):
        y1, y2, x1, x2 = c
        p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))

        if args.use_esrgan:
          p = enhance_image_with_esrgan(esrgan_model, p)  # Apply ESRGAN enhancement
          p = np.array(p)  # Convert PIL image to NumPy array if needed
          p = cv2.resize(p, (x2 - x1, y2 - y1))  # Resize to match the original region shape

        #cv2.imwrite('{}/{}_{}.jpg'.format('checkpoints/wav2lip_checkpoint/face-enhancer', x, i), p)

        f[y1:y2, x1:x2] = p
        out.write(f)
        i += 1

    out.release()

  command = 'ffmpeg -y -i {} -i {} -strict -2 -q:v 1 {}'.format(args.audio, f'temp/result_{args.iteration - 1}.avi', args.outfile)
  subprocess.call(command, shell=platform.system() != 'Windows')

if __name__ == '__main__':
  main()
