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
import mediapipe as mp

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

LIPS_LANDMARKS = [
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
    291, 146, 91, 181, 84, 17, 314, 405, 320, 307,
    375, 321, 311, 308, 324, 318, 402, 317, 14, 87
]

def apply_dynamic_blur(window, sigma=12):
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)
        # This function assumes window has shape (C, T, H, W)
        # It applies a gaussian blur to the mouth region and gradually diffuses it outward.
        
        
        C, T, H, W = window.shape
        frames = window.copy()  # now shape: (T, H, W, C)
        blurred_frames = []
        

        for frame in frames:
            frame_rgb = (frame).astype(np.uint8)

            results = face_mesh.process(frame_rgb)

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
                pad_width_factor = 0.6  # Adjust this value as needed.
                pad_height_factor = 0.6  # Adjust this value as needed.
                pad_x = int(width * pad_width_factor)
                pad_y = int(height * pad_height_factor)

                # Expand the rectangle and ensure the coordinates stay within frame boundaries.
                x_min_expanded = max(x_min - pad_x, 0)
                y_min_expanded = max(y_min - pad_y, 0)
                x_max_expanded = min(x_max + pad_x, w)
                y_max_expanded = min(y_max + pad_y, h)

                # Black out the expanded rectangular region.
                frame[y_min_expanded:y_max_expanded, x_min_expanded:x_max_expanded] = [0, 0, 0]
                blurred_frames.append(frame)
            else:
                
                h, w, _ = frame.shape
                split_row = h // 2

                # Split the frame into the top and bottom halves.
                top_half = frame[:split_row, :, :]
                bottom_half = frame[split_row:, :, :]

                # For clarity, compute the height of the bottom half.
                bottom_height = h - split_row

                # Define the rectangle size as a percentage of the bottom half's dimensions.
                rectangle_height = int(bottom_height * 0.65)  # 30% of the bottom half height
                rectangle_width = int(w * 0.8)              # 30% of the full frame width

                # Calculate coordinates to center the rectangle in the bottom half.
                start_x = (w - rectangle_width) // 2
                end_x = start_x + rectangle_width
                start_y = (bottom_height - rectangle_height) // 2
                end_y = start_y + rectangle_height

                print('Rectangle dimensions and coordinates:', rectangle_height, rectangle_width, start_x, end_x, start_y, end_y)

                # Fill the specific rectangle in the bottom half with black.
                bottom_half[start_y:end_y, start_x:end_x] = [0, 0, 0]

                # Reassemble the full frame from the top and modified bottom halves.
                frame_masked = np.vstack([top_half, bottom_half])
                blurred_frames.append(frame_masked)     

        # Reassemble the frames and convert back to (C, T, H, W)
        result = np.stack(blurred_frames, axis=0)  # shape: (T, H, W, C)
        #result = np.transpose(result, (0, 1, 2, 3))  # shape: (C, T, H, W)
        return result


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
      img_masked = apply_dynamic_blur(img_masked)
      #print('The image shape 1', img_masked.shape, img_batch.shape)

      img_batch = np.concatenate((img_masked, img_batch, ref_batch, ref_batch2), axis=3) / 255.
      mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

      yield img_batch, mel_batch, frame_batch, coords_batch
      img_batch, mel_batch, frame_batch, coords_batch, ref_batch, ref_batch2 = [], [], [], [], [], []
    
    ids_in_ref = []

  if len(img_batch) > 0:
    img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)
    img_masked = img_batch.copy()

    
    #img_masked[:, args.img_size//2:] = 0
    img_masked = apply_dynamic_blur(img_masked)

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
