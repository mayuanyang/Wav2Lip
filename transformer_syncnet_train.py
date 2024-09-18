from os.path import dirname, join, basename, isfile
from tqdm import tqdm

from models import TransformerSyncnet as TransformerSyncnet
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
from syncnet_dataset import Dataset, samples

# import module 
import traceback
import wandb


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
parser.add_argument('--use_wandb', help='Whether to use wandb', default=True, type=str2bool)

args = parser.parse_args()

# Define lip landmark indices according to MediaPipe documentation
LIP_LANDMARKS = list(range(61, 79)) + list(range(191, 209))

global_step = 1
global_epoch = 1
use_cuda = torch.cuda.is_available()
use_cosine_loss=True
sample_mode='random'
use_wandb=True


current_training_loss = 0.6
learning_step_loss_threshhold = 0.3
consecutive_threshold_count = 0


print('use_cuda: {}'.format(use_cuda))


cross_entropy_loss = nn.CrossEntropyLoss()

logloss = nn.BCELoss()
def cosine_loss(a, v, y):
    d = nn.functional.cosine_similarity(a, v)
    
    # Scale cosine similarity to range [0, 1]
    cos_sim_scaled = (1 + d) / 2.0
    
    # Calculate the loss: the target is 1 for similar pairs and 0 for dissimilar pairs
    loss = nn.functional.mse_loss(cos_sim_scaled, y.float())
    
    return loss



def get_lip_landmark(image, face_mesh):
  try:
    
    # Load an image (make sure the image path is correct)
    height, width, _ = image.shape

    # Convert the image from BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image to extract face landmarks
    results = face_mesh.process(image_rgb)

        
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Create an empty list to store the lip landmarks
            lip_embeddings = []
            
            # Iterate through the specific lip landmarks
            for idx in LIP_LANDMARKS:
                landmark = face_landmarks.landmark[idx]
                # Extract the x, y, z coordinates
                x = landmark.x
                y = landmark.y
                z = landmark.z
                # Append the coordinates to the embeddings list
                lip_embeddings.append([x, y, z])
            
            # Convert the list to a numpy array (optional, for easier manipulation)
            lip_embeddings = np.array(lip_embeddings)
            break

    # Normalize the embedding (optional but recommended)
    # This normalization ensures that all coordinates are on the same scale (between 0 and 1)
    lip_embeddings = lip_embeddings / np.linalg.norm(lip_embeddings)

    # Print the embedding
    #print("Lip Embedding:", lip_embedding)

    
    return lip_embeddings
  except Exception as e:
    print('error', e)
    traceback.print_exc() 
    return None

# added by eddy
# Register hooks to print gradient norms
def print_grad_norm(module, grad_input, grad_output):
    for i, grad in enumerate(grad_output):
        if grad is not None and global_step % 500 == 0:
            print(f'{module.__class__.__name__} - grad_output[{i}] norm: {grad.norm().item()}')

# end added by eddy


def train(device, model, train_data_loader, test_data_loader, optimizer,
          checkpoint_dir=None, checkpoint_interval=None, nepochs=None, should_print_grad_norm=False):

    
    global global_step, global_epoch, consecutive_threshold_count, current_training_loss
    resumed_step = global_step
    print('start training data folder', train_data_loader)
    patience = 50

    current_lr = get_current_lr(optimizer)
    print('The learning rate is: {0}'.format(current_lr))

    # Added by eddy
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=patience, verbose=True)
    
    if should_print_grad_norm:
      for name, module in model.named_modules():
        if isinstance(module, (Conv2d, Conv2dTranspose, nn.Linear, nn.TransformerEncoderLayer)):
            module.register_backward_hook(print_grad_norm)
    
    # end
    
    ce_min = 1000.
    ce_max = 0.
    cos_min = 1000.
    cos_max = 0.
    
    while global_epoch < nepochs:
        
        avg_ce_loss = 0.
        avg_cos_loss = 0.
        avg_mse_loss = 0.
        prog_bar = tqdm(enumerate(train_data_loader))
        current_lr = get_current_lr(optimizer)
        for step, (x, mel, y) in prog_bar:
            
            model.train()
            optimizer.zero_grad()

            # Transform data to CUDA device
            x = x.to(device)

            mel = mel.to(device)

            output, audio_embedding, face_embedding = model(x, mel)
            
            y = y.to(device)                        
            
            ce_loss = cross_entropy_loss(output, y)
            if ce_loss.item() < ce_min:
                ce_min = ce_loss.item()
            
            if ce_loss.item() > ce_max:
                ce_max = ce_loss.item()

            cos_loss = cosine_loss(audio_embedding, face_embedding, y)
            if cos_loss.item() < cos_min:
                cos_min = cos_loss.item()
            
            if cos_loss.item() > cos_max:
                cos_max = cos_loss.item()

            

            # Normalize losses
            normalized_ce_loss = (ce_loss.item() - ce_min) / (ce_max - ce_min + 1e-8)
            normalized_cos_loss = (cos_loss.item() - cos_min) / (cos_max - cos_min + 1e-8)

            print('ce min {0}, ce max {1}, cos min {2}, cos max {3}, norm ce {4}, norm cos{5}'.format(ce_min, ce_max, cos_min, cos_max, normalized_ce_loss, normalized_cos_loss))

            mse_loss = nn.functional.mse_loss(output, nn.functional.one_hot(y, num_classes=2).float())

            if normalized_ce_loss < normalized_cos_loss:
                back_loss = ce_loss
            else:
                back_loss = cos_loss

            #back_loss = 0.5 * ce_loss + 0.5 * cos_loss

            back_loss.backward()
            optimizer.step()

            global_step += 1
            avg_ce_loss += ce_loss.item()
            avg_cos_loss += cos_loss.item()
            avg_mse_loss += mse_loss.item()


            if global_step == 1 or global_step % checkpoint_interval == 0:
                save_checkpoint(
                    model, optimizer, global_step, checkpoint_dir, global_epoch)

            if global_step % hparams.syncnet_eval_interval == 0:
                with torch.no_grad():
                    eval_model(test_data_loader, global_step, device, model, checkpoint_dir, scheduler)
                
            current_training_loss = avg_ce_loss / (step + 1)
            current_cos_loss = avg_cos_loss / (step + 1)
            avg_mse_loss = avg_mse_loss / (step + 1)
            prog_bar.set_description('Global Step: {0}, Epoch: {1}, CE Loss: {2}, Cos Loss: {3}, MSE Loss: {4}, LR: {5}'.format(global_step, global_epoch, current_training_loss, current_cos_loss, avg_mse_loss, current_lr))
            metrics = {"train/ce_loss": current_training_loss, 
                       "train/cos_loss": current_cos_loss, 
                       "train/mse_loss": avg_mse_loss, 
                       "train/step": global_step, 
                       "train/epoch": global_epoch,
                       "train/learning_rate": current_lr}
            
            if use_wandb:
              wandb.log({**metrics})

        if current_training_loss < 0.25:
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


    print('Evaluating for {0} steps of total steps {1}'.format(eval_steps, len(test_data_loader)))
    prog_bar = tqdm(enumerate(test_data_loader))
    losses = []
    while 1:
        for step, (x, mel, y) in enumerate(test_data_loader):

            model.eval()

            # Transform data to CUDA device
            x = x.to(device)

            mel = mel.to(device)

            output, audio_embedding, face_embedding = model(x, mel)
            y = y.to(device)                

            loss = cross_entropy_loss(output, y) #if (global_epoch // 50) % 2 == 0 else contrastive_loss2(a, v, y)
            
            losses.append(loss.item())

            if step > eval_steps: break

        averaged_loss = sum(losses) / len(losses)
        
        prog_bar.set_description('Step: {0}/{1}, Loss: {2}'.format(current_step, eval_loop, averaged_loss))

        metrics = {"val/loss": averaged_loss, 
                    "val/step": global_step, 
                    "val/epoch": global_epoch}
        
        if use_wandb: 
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
    for param_group in optimizer.param_groups:
        param_group['lr'] = 0.00002

    return model

if __name__ == "__main__":
    checkpoint_dir = args.checkpoint_dir
    checkpoint_path = args.checkpoint_path
    sample_mode = args.sample_mode
    use_wandb = args.use_wandb

    if use_wandb: 
      wandb.init(
        # set the wandb project where this run will be logged
        project="my-wav2lip-syncnet",

        # track hyperparameters and run metadata
        config={
        "learning_rate": hparams.syncnet_lr,
        "architecture": "TransformerSyncnet",
        "dataset": "MyOwn",
        "epochs": 200000,
        }
      )

    if not os.path.exists(checkpoint_dir): os.mkdir(checkpoint_dir)

    # Dataset and Dataloader setup
    train_dataset = Dataset('train', args.data_root)
    test_dataset = Dataset('val', args.data_root)
    #print(train_dataset.all_videos)

    train_data_loader = data_utils.DataLoader(
        train_dataset, batch_size=hparams.syncnet_batch_size, shuffle=True,
        num_workers=hparams.num_workers)

    test_data_loader = data_utils.DataLoader(
        test_dataset, batch_size=hparams.syncnet_batch_size,
        num_workers=8)

    device = torch.device("cuda" if use_cuda else "cpu")

    # Model
    model = TransformerSyncnet(num_heads=8, num_encoder_layers=6).to(device)
    print('total trainable params {}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad],
                           lr=hparams.syncnet_lr,betas=(0.8, 0.999), weight_decay=1e-5)

    if checkpoint_path is not None:
        load_checkpoint(checkpoint_path, model, optimizer, reset_optimizer=False)

    train(device, model, train_data_loader, test_data_loader, optimizer,
          checkpoint_dir=checkpoint_dir,
          checkpoint_interval=hparams.syncnet_checkpoint_interval,
          nepochs=hparams.nepochs, should_print_grad_norm=True)
