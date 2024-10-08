from os.path import dirname, join, basename, isfile
from tqdm import tqdm

from models import SyncNet_color as SyncNet
import audio

import torch
from torch import nn
from torch import optim
import torch.backends.cudnn as cudnn
from torch.utils import data as data_utils
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau

from glob import glob

import os, random, cv2, argparse
from hparams import hparams, get_image_list

from models.conv import Conv2d, Conv2dTranspose

# import module 
import traceback
import wandb
import multiprocessing

from PIL import Image

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description='Code to train the expert lip-sync discriminator')

parser.add_argument("--data_root", help="Root folder of the preprocessed LRS2 dataset", required=True)

parser.add_argument('--checkpoint_dir', help='Save checkpoints to this directory', required=True, type=str)
parser.add_argument('--checkpoint_path', help='Resumed from this checkpoint', default=None, type=str)
parser.add_argument('--use_cosine_loss', help='Whether to use cosine loss', default=True, type=str2bool)
parser.add_argument('--sample_mode', help='easy or random', default=True, type=str)

args = parser.parse_args()


global_step = 1
global_epoch = 1
use_cuda = torch.cuda.is_available()
use_cosine_loss=True
sample_mode='random'
global_cache = multiprocessing.Manager().dict()
orig_mel_cache = multiprocessing.Manager().dict()

current_training_loss = 0.6
learning_step_loss_threshhold = 0.3
consecutive_threshold_count = 0
samples = [True, True,True, True,True, True,True, False,False, False]

print('use_cuda: {}'.format(use_cuda))

"""
The FPS is set to 25 for video, 5/25 is 0.2, we need to have 0.2 seconds for the audio,
because the audio mel spectrogram ususlly has 80 frame per seconds, so 16/80 is 0.2 seconds
"""
syncnet_T = 5
syncnet_mel_step_size = 16

class Dataset(object):
    def __init__(self, split, use_image_cache):
        print('A new dataset')
        self.all_videos = get_image_list(args.data_root, split)
        self.image_cache = {}  # Initialize the cache
        self.orig_mel_cache = {}
        self.file_exist_cache = {}       
        

    def get_frame_id(self, frame):
        return int(basename(frame).split('.')[0])

    def get_window(self, start_frame):
        start_id = self.get_frame_id(start_frame)
        vidname = dirname(start_frame)

        window_fnames = []
        for frame_id in range(start_id, start_id + syncnet_T):
            frame = join(vidname, '{}.jpg'.format(frame_id))
            
            if not frame in self.file_exist_cache:
              if not isfile(frame):
                return None    
            
            
            self.file_exist_cache[frame] = True
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
        
        #print("image cache", len(image_cache))
        start_time = time.perf_counter()
        #print("working on", self.all_videos[idx])
        while 1:
            #idx = random.randint(0, len(self.all_videos) - 1)
            vidname = self.all_videos[idx]

            img_names = list(glob(join(vidname, '*.jpg')))
            
            if len(img_names) <= 3 * syncnet_T:
                continue
            
            """
            Changed by eddy, the following are the original codes, it uses random to get the wrong_img_name, 
            this might get an image that very close to the correct image(the next frame) which is a bit hard to learn.
            Eddy introduced a new algorithm that to get a image a bit futher from the img_name to have enough difference,
            this might help the model to converge.
            However we can't just learn the easy samples, so we use a flag to control that
            img_name = random.choice(img_names)
            wrong_img_name = random.choice(img_names)
            """
            
            img_name = random.choice(img_names)
            if sample_mode == 'random':
              wrong_img_name = random.choice(img_names)
              print("The random mode image", wrong_img_name)
            else:
              chosen_id = self.get_frame_id(img_name)
              # Find the position of the last slash
              last_slash_index = img_name.rfind('/')

              # Get the substring up to and including the last slash
              dir_name = img_name[:last_slash_index+1]
              index = 10

              wrong_img_name = dir_name + str(chosen_id + index) + ".jpg"

              while wrong_img_name not in img_names:
                index -= 1
                wrong_img_name = dir_name + str(chosen_id + index) + ".jpg"
                if index < 5:
                    wrong_img_name = random.choice(img_names)
                    #print("Cannot find a good one, use random instead, original img_name {0} and the not good one is {1}".format(img_name, wrong_img_name))
                    continue

            while wrong_img_name == img_name:
                wrong_img_name = random.choice(img_names)


            # We firstly to learn all the positive, once it reach the loss of less than 0.3, we incrementally add some negative samples 10% per step
            

            good_or_bad = True

            good_or_bad = random.choice(samples)

            if good_or_bad:
                y = 1
                chosen = img_name
            else:
                y = 0
                chosen = wrong_img_name

            
            window_fnames = self.get_window(chosen)
            if window_fnames is None:
                continue
            
            window = []

            all_read = True
            for fname in window_fnames:
                #print('The image name ', fname)
                if fname in global_cache:
                    img = global_cache[fname]
                    #print('The image cache hit ', fname)
                else:
                    img = cv2.imread(fname)
                    if img is None:
                        all_read = False
                        break
                    try:
                        img = cv2.resize(img, (hparams.img_size, hparams.img_size))
                        global_cache[fname] = img  # Cache the resized image
                        
                    except Exception as e:
                        all_read = False
                        break

                window.append(img)

            if not all_read: continue

            try:
                wavpath = join(vidname, "audio.wav")

                if wavpath in orig_mel_cache:
                    orig_mel = orig_mel_cache[wavpath]
                    #print('The audio cache hit ', wavpath)
                else:
                    wav = audio.load_wav(wavpath, hparams.sample_rate)
                    orig_mel = audio.melspectrogram(wav).T
                    orig_mel_cache[wavpath] = orig_mel
                
            except Exception as e:
                print('error', e)
                traceback.print_exc() 
                continue

            mel = self.crop_audio_window(orig_mel.copy(), img_name)

            if (mel.shape[0] != syncnet_mel_step_size):
                print('The mel shape is {}, but it should be {} and start num is {}'.format(mel.shape[0], syncnet_mel_step_size, img_name))
                continue
            
            # Save the sample images
            # if idx % 100 == 0:
            #   print('The video is ', vidname)
            #   for i, img in enumerate(window):
            #         img_to_save = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            #         img_to_save.save(f'temp1/saved_image_{idx}_{i}.png')

            # H x W x 3 * T
            x = np.concatenate(window, axis=2) / 255.
            x = x.transpose(2, 0, 1)
            x = x[:, x.shape[1]//2:]

            x = torch.FloatTensor(x)
            mel = torch.FloatTensor(mel.T).unsqueeze(0)

            end_time = time.perf_counter()
            execution_time = (end_time - start_time) * 1000  # Convert seconds to milliseconds
            #print(f"The method took {execution_time:.2f} milliseconds to execute.")

            return x, mel, y


logloss = nn.BCELoss()
def cosine_loss(a, v, y):
    """
    Computes the cosine similarity between audio and face embeddings.
    Reshapes the similarity scores for compatibility with ground truth labels.
    Uses logistic loss (binary cross-entropy) to penalize incorrect similarity predictions.
    """
    d = nn.functional.cosine_similarity(a, v)
    loss = logloss(d.unsqueeze(1), y)

    return loss

def cosine_bce_loss(a, v, y):
    """
    Computes the cosine similarity between audio and face embeddings.
    Applies a sigmoid to the similarity scores for compatibility with ground truth labels.
    Uses binary cross-entropy loss to penalize incorrect similarity predictions.
    """
    
    # Compute cosine similarity
    cosine_sim = nn.functional.cosine_similarity(a, v)

    # Apply sigmoid to map similarity scores to [0, 1]
    sigmoid_sim = torch.sigmoid(cosine_sim)

    # Reshape for BCELoss compatibility
    sigmoid_sim = sigmoid_sim.unsqueeze(1)

    # Compute binary cross-entropy loss
    loss = logloss(sigmoid_sim, y)

    return loss
    

def contrastive_loss(a, v, y, margin=0.5):
    """
    Contrastive loss tries to minimize the distance between similar pairs and maximize the distance between dissimilar pairs up to a margin.
    """
    d = nn.functional.pairwise_distance(a, v)
    loss = torch.mean((1 - y) * torch.pow(d, 2) + y * torch.pow(torch.clamp(margin - d, min=0.0), 2))
    return loss

def contrastive_loss2(a, v, y, margin=0.3):
        # Compute the Euclidean distance between the embeddings
        euclidean_distance = nn.functional.pairwise_distance(a, v)

        # Compute contrastive loss
        euclidean_distance = (1 - y) * 0.5 * torch.pow(euclidean_distance, 2) + \
               y * 0.5 * torch.pow(torch.clamp(margin - euclidean_distance, min=0.0), 2)
        
        # Apply sigmoid to the Euclidean distance to get a probability-like value
        sigmoid_distance = torch.sigmoid(euclidean_distance).unsqueeze(1)
        
        # Compute binary cross-entropy loss using the sigmoid-transformed distance
        bce_loss = logloss(sigmoid_distance, y)
        
        return bce_loss

# added by eddy
# Register hooks to print gradient norms
def print_grad_norm(module, grad_input, grad_output):
    for i, grad in enumerate(grad_output):
        if grad is not None and global_step % 100 == 0:
            print(f'{module.__class__.__name__} - grad_output[{i}] norm: {grad.norm().item()}')

# end added by eddy


def train(device, model, train_data_loader, test_data_loader, optimizer,
          checkpoint_dir=None, checkpoint_interval=None, nepochs=None, should_print_grad_norm=False):

    
    global global_step, global_epoch, consecutive_threshold_count, current_training_loss
    resumed_step = global_step
    print('start training data folder', train_data_loader)
    patience = 40

    current_lr = get_current_lr(optimizer)
    print('The learning rate is: {0}'.format(current_lr))

    # Added by eddy
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=patience, verbose=True)
    
    if should_print_grad_norm:
      for name, module in model.named_modules():
        if isinstance(module, (Conv2d, Conv2dTranspose, nn.Linear)):
            module.register_backward_hook(print_grad_norm)
    
    # end



    while global_epoch < nepochs:
        running_loss = 0.
        prog_bar = tqdm(enumerate(train_data_loader))
        current_lr = get_current_lr(optimizer)
        for step, (x, mel, y) in prog_bar:
            
            model.train()
            optimizer.zero_grad()

            # Transform data to CUDA device
            x = x.to(device)

            mel = mel.to(device)

            a, v = model(mel, x)
            y = y.to(device)

            loss = cosine_bce_loss(a, v, y) #if (global_epoch // 50) % 2 == 0 else contrastive_loss2(a, v, y)
            loss.backward()
            optimizer.step()

            global_step += 1
            cur_session_steps = global_step - resumed_step
            running_loss += loss.item()

            if global_step == 1 or global_step % checkpoint_interval == 0:
                save_checkpoint(
                    model, optimizer, global_step, checkpoint_dir, global_epoch)

            if global_step % hparams.syncnet_eval_interval == 0:
                with torch.no_grad():
                    eval_model(test_data_loader, global_step, device, model, checkpoint_dir, scheduler)
                
            current_training_loss = running_loss / (step + 1)
            prog_bar.set_description('Global Step: {0}, Epoch: {1}, Loss: {2}, current learning rate: {3}'.format(global_step, global_epoch, current_training_loss, current_lr))
            metrics = {"train/train_loss": current_training_loss, 
                       "train/step": global_step, 
                       "train/epoch": global_epoch,
                       "train/learning_rate": current_lr}
            
            wandb.log({**metrics})

        if current_training_loss < 0.3:
          consecutive_threshold_count += 1
        else:
          consecutive_threshold_count = 0
            
        if consecutive_threshold_count >= 10:
          false_count = samples.count(False)
          if false_count < 5:
            # Find the index of the first occurrence of True
            first_true_index = samples.index(True)
            # Change the element at that index to False
            samples[first_true_index] = False

            print('Adding negative samples, the current samples are', samples)
                
            
        global_epoch += 1
        # if should_print_grad_norm or global_step % 20==0:
        #   for param in model.parameters():
        #         if param.grad is not None:
        #             print('The gradient is ', param.grad.norm())
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        

# Added by eddy
def get_current_lr(optimizer):
    # Assuming there is only one parameter group
    for param_group in optimizer.param_groups:
        return param_group['lr']


def eval_model(test_data_loader, global_step, device, model, checkpoint_dir, scheduler):
    #eval_steps = 1400
    eval_steps = 20
    eval_loop = 20
    current_step = 1

        
    print()
    print('Evaluating for {0} steps of total steps {1}'.format(eval_steps, len(test_data_loader)))
    losses = []
    while 1:
        for step, (x, mel, y) in enumerate(test_data_loader):

            model.eval()

            # Transform data to CUDA device
            x = x.to(device)

            mel = mel.to(device)

            a, v = model(mel, x)
            y = y.to(device)

            loss = cosine_bce_loss(a, v, y)
            losses.append(loss.item())

            #print('Step: {0}, Cosine Loss: {1}'.format(step, loss))

            if step > eval_steps: break

        averaged_loss = sum(losses) / len(losses)
        print('The avg eval loss is: {0}'.format(averaged_loss))

        metrics = {"val/loss": averaged_loss, 
                    "val/step": global_step, 
                    "val/epoch": global_epoch}
            
        wandb.log({**metrics})
        # Scheduler step with average training loss
        scheduler.step(averaged_loss)
        current_step += 1
        if current_step > eval_loop: 
            return

def save_checkpoint(model, optimizer, step, checkpoint_dir, epoch):

    checkpoint_path = join(
        checkpoint_dir, "checkpoint_step{:09d}.pth".format(global_step))
    optimizer_state = optimizer.state_dict() if hparams.save_optimizer_state else None
    torch.save({
        "state_dict": model.state_dict(),
        "optimizer": optimizer_state,
        "global_step": step,
        "global_epoch": epoch,
    }, checkpoint_path)
    print("Saved checkpoint:", checkpoint_path)

def _load(checkpoint_path):
    if use_cuda:
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path,
                                map_location=lambda storage, loc: storage)
    return checkpoint

def load_checkpoint(path, model, optimizer, reset_optimizer=False):
    global global_step
    global global_epoch

    print("Load checkpoint from: {}".format(path))
    checkpoint = _load(path)
    model.load_state_dict(checkpoint["state_dict"])
    if not reset_optimizer:
        optimizer_state = checkpoint["optimizer"]
        if optimizer_state is not None:
            print("Load optimizer state from {}".format(path))
            optimizer.load_state_dict(checkpoint["optimizer"])
    global_step = checkpoint["global_step"]
    global_epoch = checkpoint["global_epoch"]

    # Reset the new learning rate
    # for param_group in optimizer.param_groups:
    #     param_group['lr'] = 0.0001

    return model

if __name__ == "__main__":
    checkpoint_dir = args.checkpoint_dir
    checkpoint_path = args.checkpoint_path
    use_cosine_loss = args.use_cosine_loss
    sample_mode = args.sample_mode
    print("The use_cosine_loss value", use_cosine_loss)
    print("The sample mode value", sample_mode)

    wandb.init(
      # set the wandb project where this run will be logged
      project="my-wav2lip",

      # track hyperparameters and run metadata
      config={
      "learning_rate": hparams.syncnet_lr,
      "architecture": "Syncnet",
      "dataset": "MyOwn",
      "epochs": 200000,
      }
    )

    if not os.path.exists(checkpoint_dir): os.mkdir(checkpoint_dir)

    # Dataset and Dataloader setup
    train_dataset = Dataset('train', False)
    test_dataset = Dataset('val', False)
    #print(train_dataset.all_videos)

    train_data_loader = data_utils.DataLoader(
        train_dataset, batch_size=hparams.syncnet_batch_size, shuffle=True,
        num_workers=hparams.num_workers)

    test_data_loader = data_utils.DataLoader(
        test_dataset, batch_size=hparams.syncnet_batch_size,
        num_workers=8)

    device = torch.device("cuda" if use_cuda else "cpu")

    # Model
    model = SyncNet().to(device)
    print('total trainable params {}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad],
                           lr=hparams.syncnet_lr,betas=(0.9, 0.999), weight_decay=1e-4)

    if checkpoint_path is not None:
        load_checkpoint(checkpoint_path, model, optimizer, reset_optimizer=False)

    train(device, model, train_data_loader, test_data_loader, optimizer,
          checkpoint_dir=checkpoint_dir,
          checkpoint_interval=hparams.syncnet_checkpoint_interval,
          nepochs=hparams.nepochs, should_print_grad_norm=True)
