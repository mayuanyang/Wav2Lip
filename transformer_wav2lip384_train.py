from os.path import dirname, join, basename, isfile
from tqdm import tqdm

from models import TransformerSyncnet
from models import ResUNet384, ResUNet384V2
import torch

import wandb

from torch import nn
from torch import optim
import torch.backends.cudnn as cudnn
from torch.utils import data as data_utils
import numpy as np
import torchvision.models as models
import lpips

from glob import glob

import os, cv2, argparse
from hparams import hparams, get_image_list

from torch.optim.lr_scheduler import ReduceLROnPlateau
from models.conv import Conv2d, Conv2dTranspose
from wav2lip_dataset import Dataset, syncnet_T
import torch.nn.functional as F
from pytorch_msssim import ms_ssim
from torch.cuda.amp import autocast

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description='Code to train the Wav2Lip model without the visual quality discriminator')

parser.add_argument("--data_root", help="Root folder of the preprocessed LRS2 dataset", required=True, type=str)

parser.add_argument('--checkpoint_dir', help='Save checkpoints to this directory', required=True, type=str)
parser.add_argument('--syncnet_checkpoint_path', help='Load the pre-trained Expert discriminator', required=True, type=str)

parser.add_argument('--checkpoint_path', help='Resume from this checkpoint', default=None, type=str)
parser.add_argument('--use_wandb', help='Whether to use wandb', default=True, type=str2bool)
parser.add_argument('--wandb_run_id', help='The run ID for wandb', required=False, type=str)
parser.add_argument('--use_augmentation', help='Whether to use data augmentation', default=True, type=str2bool)
parser.add_argument('--train_root', help='the folder that contains train.txt and val.txt', default='filelists', type=str)
parser.add_argument('--num_of_unet_layers', help='The num of layers for resunet', default=2, type=int)
parser.add_argument('--version', help='The train.txt and val.txt directory', default='v1', type=str)
args = parser.parse_args()


global_step = 0
global_epoch = 0
num_of_unet_layers = 2
use_wandb=True
use_augmentation= True
version = 'v1'
use_cuda = torch.cuda.is_available()


print('use_cuda: {}'.format(use_cuda))

def save_sample_images(x, g, gt, global_step, checkpoint_dir):
    '''
    refs: Reference images (extracted from the input x with channels 3 onward).
    inps: Input images (extracted from the input x with the first 3 channels).
    g: Generated images by the model.
    gt: Ground truth images.
    '''
    x = (x.detach().cpu().numpy().transpose(0, 2, 3, 4, 1) * 255.).astype(np.uint8)
    g = (g.detach().cpu().numpy().transpose(0, 2, 3, 4, 1) * 255.).astype(np.uint8)
    gt = (gt.detach().cpu().numpy().transpose(0, 2, 3, 4, 1) * 255.).astype(np.uint8)

    refs, inps = x[..., 9:], x[..., :3]
    folder = join(checkpoint_dir, "samples_step{:09d}".format(global_step))
    if not os.path.exists(folder): os.mkdir(folder)
    collage = np.concatenate((refs, inps, g, gt), axis=-2)
    for batch_idx, c in enumerate(collage):
        for t in range(len(c)):
            cv2.imwrite('{}/{}_{}.jpg'.format(folder, batch_idx, t), c[t])



device = torch.device("cuda" if use_cuda else "cpu")
syncnet = TransformerSyncnet(num_heads=8, num_encoder_layers=6).to(device)
for p in syncnet.parameters():
    p.requires_grad = False


cross_entropy_loss = nn.CrossEntropyLoss()
recon_loss = nn.L1Loss()

def get_sync_loss(mel, g):
    
    B, C, T, H, W = g.shape

    # Reshape to (B*T, C, H, W) for interpolation
    g_reshaped = g.view(B * T, C, H, W)

    # Resize H and W to half using bilinear interpolation
    g_resized = F.interpolate(g_reshaped, scale_factor=0.5, mode='bilinear', align_corners=False)

    # Reshape back to (B, T, C, H//2, W//2)
    H_half, W_half = g_resized.shape[2], g_resized.shape[3]
    
    g_resized = g_resized.view(B, C, T, H_half, W_half)

    # =========================
    # Step 2: Persist Resized `g`
    # =========================
    # 
    # import torchvision.transforms as transforms
    # import datetime
    # transform = transforms.ToPILImage()
    # g_cpu = g.detach().cpu()

    # # Iterate over batch and temporal dimensions
    # for b in range(B):
    #     for t in range(T):
    #         img_tensor = g_cpu[b, t]  # Shape: (C, H, W)

    #         # Handle different channel scenarios
    #         if C == 1:
    #             # Grayscale image
    #             img_tensor = img_tensor.squeeze(0)  # Shape: (H, W)
    #             mode = 'L'
    #         elif C == 3:
    #             # RGB image
    #             mode = 'RGB'
    #         else:
    #             # Other channels: Handle accordingly or skip
    #             print(f"Skipping image with {C} channels.")
    #             continue

    #         # Convert tensor to PIL Image
    #         img = transform(img_tensor)

    #         # Generate a unique filename
    #         timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    #         filename = f"b{b}_t{t}_{timestamp}.png"

    #         file_path = os.path.join('temp/sync_imgs', filename)

    #         # Save the image
    #         img.save(file_path)
    # g = g.view(B, C, T, H_half, W_half)

    # Continue with existing processing
    # Proceed with the rest of your code
    # print('The resized shape', g_resized.shape)
    g_resized = g_resized[:, :, :, g_resized.size(3)//2:]
    g_resized = torch.cat([g_resized[:, :, i] for i in range(syncnet_T)], dim=1)
    # B, 3 * T, H//2, W
    output, audio_embedding, face_embedding = syncnet(g_resized, mel)

    y = torch.ones(g_resized.size(0), dtype=torch.long).squeeze().to(device)
    
    return cross_entropy_loss(output, y)


def print_grad_norm(module, grad_input, grad_output):
    for i, grad in enumerate(grad_output):
        if grad is not None and global_step % 1000 == 0:
            print(f'{module.__class__.__name__} - grad_output[{i}] norm: {grad.norm().item()}')

# Added by eddy
def get_current_lr(optimizer):
    # Assuming there is only one parameter group
    for param_group in optimizer.param_groups:
        return param_group['lr']

def train(device, model, train_data_loader, test_data_loader, optimizer, 
          checkpoint_dir=None, checkpoint_interval=None, nepochs=None, should_print_grad_norm=False):

    global global_step, global_epoch
    resumed_step = global_step

    patience = 5000

    current_lr = get_current_lr(optimizer)
    print('The learning rate is: {0}'.format(current_lr))

    # Added by eddy
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=patience, verbose=True)

    if should_print_grad_norm:
      for name, module in model.named_modules():
        if isinstance(module, (Conv2d, Conv2dTranspose, nn.Linear)):
            module.register_backward_hook(print_grad_norm)

    # Initialize LPIPS model
    lpips_loss = lpips.LPIPS(net='vgg').to(device)  # You can choose 'alex', 'vgg', or 'squeeze'

    eval_loss = 0.0

    syncnet_wt = hparams.syncnet_wt
    sync_loss = 0.

    while global_epoch < nepochs:
        current_lr = get_current_lr(optimizer)
                
        #print('Starting Epoch: {}'.format(global_epoch))
        running_sync_loss, running_l1_loss = 0., 0.
        prog_bar = tqdm(enumerate(train_data_loader))
        running_img_loss = 0.0
        running_disc_loss = 0.0
        running_bottom_disc_loss = 0.0
        running_bottom_l1_loss = 0.0
        running_bottom_l1_loss = 0.0
        running_ssim_loss = 0.0

        running_triplet_loss = 0.0
        for step, (x, indiv_mels, mel, gt) in prog_bar:
            #print("The x shape", x.shape)
            if x.shape[0] == hparams.batch_size:
              model.train()
              optimizer.zero_grad()

              # Move data to CUDA device
              x = x.to(device)
              mel = mel.to(device)
              indiv_mels = indiv_mels.to(device)

              gt = gt.to(device)

              with autocast():
                g = model(indiv_mels, x)
                
                # Compare two images
                '''
                The g and gt shape is torch.Size([2, 3, 5, 192, 192]), and vgg is expecting [batch, channels, h, w]
                the 5 here represent the number of frames, so we either need to loop through them or combine them
                we choose to collapse
                '''
                num_of_frames = g.shape[2]
                full_losses = []
                bottom_losses = []
                ssim_losses = []
                full_disc_loss = 0
                bottom_disc_loss = 0

                if hparams.disc_wt > 0:
                  for i in range(num_of_frames):
                    # Extract the i-th frame from gen_image and gt_image
                    gen_frame = g[:, :, i, :, :]  # Shape: [batch_size, 3, 192, 192]
                    gt_frame = gt[:, :, i, :, :]    # Shape: [batch_size, 3, 192, 192]

                    _, _, H, _ = gen_frame.shape
                    g_bottom = gen_frame[:, :, H//2:, :]
                    gt_bottom = gt_frame[:, :, H//2:, :]

                    # Now you can process the individual frames, e.g., pass them through a model
                    # For example:
                    full_frame_loss = lpips_loss(gen_frame.to(device), gt_frame.to(device))
                    bottom_frame_loss = lpips_loss(g_bottom.to(device), gt_bottom.to(device))

                    ms_ssim_value = 0.0

                    if hparams.ssim_wt > 0:
                      ms_ssim_value = 1 - ms_ssim(gen_frame, gt_frame, data_range=1.0, size_average=True)
                      ssim_losses.append(ms_ssim_value)
                    
                    full_losses.append(full_frame_loss)
                    bottom_losses.append(bottom_frame_loss)
                    
                  
                  # Average the loss over all frames
                  full_disc_loss = torch.mean(torch.stack(full_losses))
                  running_disc_loss += full_disc_loss.item()

                  bottom_disc_loss = torch.mean(torch.stack(bottom_losses))
                  running_bottom_disc_loss += bottom_disc_loss.item()

                  ssim_loss = 0.0
                  if len(ssim_losses) > 0:
                    ssim_loss = torch.mean(torch.stack(ssim_losses))
                    running_ssim_loss += ssim_loss.item()
                  

                if hparams.syncnet_wt > 0.:
                    sync_loss = get_sync_loss(mel, g)
                else:
                    sync_loss = 0.

                l1loss = recon_loss(g, gt)

                running_l1_loss += l1loss.item()

                _, _, _, H, _ = g.shape

                bottom_l1loss = recon_loss(g[:, :, :, H//2:, :], gt[:, :, :, H//2:, :])

                running_bottom_l1_loss += bottom_l1loss.item()
                
                loss = syncnet_wt * sync_loss + hparams.l1_wt * l1loss + hparams.bottom_l1_wt * bottom_l1loss + hparams.disc_wt * full_disc_loss + hparams.bottom_disc_wt * bottom_disc_loss + hparams.ssim_wt * ssim_loss
                
              loss.backward()
              optimizer.step()

              # **Apply Gradient Clipping Here**
              torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

              if global_step % checkpoint_interval == 0:
                  save_sample_images(x, g, gt, global_step, checkpoint_dir)

              global_step += 1

              running_img_loss += loss.item()

              if hparams.syncnet_wt > 0.:
                  running_sync_loss += sync_loss.item()
              else:
                  running_sync_loss += 0.

              if global_step == 1 or global_step % checkpoint_interval == 0:
                  save_checkpoint(
                      model, optimizer, global_step, checkpoint_dir, global_epoch)

              avg_img_loss = (running_img_loss) / (step + 1)

              avg_l1_loss = running_l1_loss / (step + 1)

              avg_bottom_l1_loss = running_bottom_l1_loss / (step + 1)

              avg_disc_loss = running_disc_loss / (step + 1)

              avg_bottom_disc_loss = running_bottom_disc_loss / (step + 1)

              avg_ssim_loss = running_ssim_loss / (step + 1)

              
              if global_step % hparams.eval_interval == 0:
                with torch.no_grad():
                  eval_loss = eval_model(test_data_loader, global_step, device, model, checkpoint_dir, scheduler, 20)

              prog_bar.set_description(f"Epoch: {global_epoch}, Step: {global_step:.0f}, Img Loss: {avg_img_loss:.5f}, Sync Loss: {running_sync_loss / (step + 1):.5f}, L1: {avg_l1_loss:.5f}, Bottom L1: {avg_bottom_l1_loss:.5f}, Full Disc: {avg_disc_loss:.5f}, Bottom Disc: {avg_bottom_disc_loss:.5f}, SSIM: {avg_ssim_loss:.5f}, LR: {current_lr:.7f}")
              
              scheduler.step(loss)
              
              metrics = {
                  "train/overall_loss": avg_img_loss, 
                  "train/avg_l1": avg_l1_loss, 
                  "train/avg_bottom_l1": avg_bottom_l1_loss, 
                  "train/sync_loss": running_sync_loss / (step + 1), 
                  "train/disc_loss": avg_disc_loss,
                  "train/bottom_disc_loss": avg_bottom_disc_loss,
                  "train/ssim_loss": avg_ssim_loss,
                  "params/step": global_step,
                  "params/learning_rate": current_lr,
                  "params/l1_wt": hparams.l1_wt,
                  "params/bottom_l1_wt": hparams.bottom_l1_wt,
                  "params/syncnet_wt": hparams.syncnet_wt,
                  "params/disc_wt": hparams.disc_wt,
                  "params/bottom_disc_wt": hparams.bottom_disc_wt,
                  }
              if use_wandb: 
                wandb.log({**metrics})

        global_epoch += 1


def eval_model(test_data_loader, global_step, device, model, checkpoint_dir, scheduler, eval_steps = 100):
    print('Evaluating for {} steps'.format(eval_steps))
    sync_losses, recon_losses = [], []
    step = 0
    while 1:
        for x, indiv_mels, mel, gt in test_data_loader:
            if x.shape[0] == hparams.batch_size:
              step += 1
              model.eval()

              # Move data to CUDA device
              x = x.to(device)
              gt = gt.to(device)
              indiv_mels = indiv_mels.to(device)
              mel = mel.to(device)

              g = model(indiv_mels, x)

              sync_loss = get_sync_loss(mel, g)
              
              l1loss = recon_loss(g, gt)

              sync_losses.append(sync_loss.item())
              recon_losses.append(l1loss.item())

              averaged_sync_loss = sum(sync_losses) / len(sync_losses)
              averaged_recon_loss = sum(recon_losses) / len(recon_losses)

              print('Eval Loss, L1: {}, Sync loss: {}'.format(averaged_recon_loss, averaged_sync_loss))

              metrics = {"val/l1_loss": averaged_recon_loss, 
                       "val/sync_loss": averaged_sync_loss, 
                       "val/epoch": global_epoch,
                       }
              if use_wandb:
                wandb.log({**metrics})

              scheduler.step(averaged_sync_loss + averaged_recon_loss)

              if step > eval_steps: 
                return averaged_sync_loss
            

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

def load_checkpoint(path, model, optimizer, reset_optimizer=False, overwrite_global_states=True):
    global global_step
    global global_epoch

    print("Load checkpoint from: {}".format(path))
    checkpoint = _load(path)
    s = checkpoint["state_dict"]
    new_s = {}
    for k, v in s.items():
        new_s[k.replace('module.', '')] = v
    model.load_state_dict(new_s, strict=False)
    if not reset_optimizer:
        optimizer_state = checkpoint["optimizer"]
        if optimizer_state is not None:
            print("Load optimizer state from {}".format(path))
            optimizer.load_state_dict(checkpoint["optimizer"])
    if overwrite_global_states:
        global_step = checkpoint["global_step"]
        global_epoch = checkpoint["global_epoch"]

    if optimizer != None:
      for param_group in optimizer.param_groups:
        param_group['lr'] = 0.0005

    # for name, param in model.named_parameters():
    #   if 'face_enhancer' not in name:
    #     param.requires_grad = False
    #   else:
    #      print('Not freeze', name)

    return model

if __name__ == "__main__":
    checkpoint_dir = args.checkpoint_dir
    use_wandb = args.use_wandb
    use_augmentation = args.use_augmentation
    version = args.version

    # Dataset and Dataloader setup
    train_dataset = Dataset('train', args.data_root, args.train_root, use_augmentation, img_size_factor=2)
    test_dataset = Dataset('val', args.data_root, args.train_root, False, img_size_factor=2)

    train_data_loader = data_utils.DataLoader(
        train_dataset, batch_size=hparams.batch_size, shuffle=True,
        num_workers=hparams.resunet_num_workers)

    test_data_loader = data_utils.DataLoader(
        test_dataset, batch_size=hparams.batch_size,
        num_workers=4)

    device = torch.device("cuda" if use_cuda else "cpu")

    # Model
    if version == 'v1':
      model = ResUNet384(args.num_of_unet_layers).to(device)
    else:
      model = ResUNet384V2(args.num_of_unet_layers).to(device)

    print('total trainable params {}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad],
                           lr=hparams.initial_learning_rate)

    if args.checkpoint_path is not None:
        load_checkpoint(args.checkpoint_path, model, optimizer, reset_optimizer=True)
        
    load_checkpoint(args.syncnet_checkpoint_path, syncnet, None, reset_optimizer=True, overwrite_global_states=False)

    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)

    
    if use_wandb:
      wandb.init(
        # set the wandb project where this run will be logged
        project="my-wav2lip",
        id=args.wandb_run_id, 
        resume="allow",
        # track hyperparameters and run metadata
        config={
        "learning_rate": hparams.initial_learning_rate,
        "architecture": "Wav2lip",
        "dataset": "MyOwn",
        "epochs": 2000000,
        }
      )

    # Train!
    train(device, model, train_data_loader, test_data_loader, optimizer,
              checkpoint_dir=checkpoint_dir,
              checkpoint_interval=hparams.checkpoint_interval,
              nepochs=hparams.nepochs)
