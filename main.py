import argparse
import os
from datetime import datetime
import torch
import torch.optim as optim
from dataloaders import get_girls_dataloader
from model import Generator, Discriminator
from training import Trainer
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser()
parser.add_argument("path", type=str, help="path to folder containing images")
parser.add_argument("--log_dir", type=str, default=os.getcwd(), help="directory to save results")
parser.add_argument("--pretrained_models", type=int, default=None, help="number epoch for loading models")
parser.add_argument("--n_epochs", type=int, default=10, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=16, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0001, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.9, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.99, help="adam: decay of first order momentum of gradient")
parser.add_argument("--img_size", type=int, default=128, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=1, help="interval between image sampling")
opt = parser.parse_args()
print(opt)

os.makedirs(os.path.join(opt.log_dir, 'images'), exist_ok=True)
os.makedirs(os.path.join(opt.log_dir, 'checkpoints', 'generators'), exist_ok=True)
os.makedirs(os.path.join(opt.log_dir, 'checkpoints', 'discriminators'), exist_ok=True)
os.makedirs(os.path.join(opt.log_dir, 'metrics'), exist_ok=True)

loger = SummaryWriter(os.path.join(opt.log_dir, 'metrics', datetime.now().strftime('%Y_%m_%d %H_%M_%S')))

if os.path.exists(os.path.join(opt.log_dir, 'metrics', 'logs.txt')):
    if opt.pretrained_models is not None:
        with open(os.path.join(opt.log_dir, 'metrics', 'logs.txt'), 'r') as f:
            f.seek(0)
            lines = f.readlines()
            lines = lines[:int(opt.pretrained_models)]
            for line in lines:
                epoch, g_loss, d_loss = line.split(' ')
                loger.add_scalar("Generator loss", float(g_loss), int(epoch))
                loger.add_scalar("Discriminator loss", float(d_loss), int(epoch))
        with open(os.path.join(opt.log_dir, 'metrics', 'logs.txt'), 'a') as f:
            f.seek(0)
            f.truncate()
            f.writelines(lines)
    else:
        os.remove(os.path.join(opt.log_dir, 'metrics', 'logs.txt'))

data_loader = get_girls_dataloader(opt.path, batch_size=opt.batch_size)

generator = Generator(resnet=True, z_size=128, channel=opt.channels, output_size=opt.img_size)
discriminator = Discriminator(resnet=True, spectral_normed=True, num_rotation=4, channel=opt.channels, ssup=True)


# Initialize optimizers
G_optimizer = optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
D_optimizer = optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

# Train model
trainer = Trainer(generator, discriminator, G_optimizer, D_optimizer,
                  weight_rotation_loss_d=1.0, weight_rotation_loss_g=0.5,
                  use_cuda=torch.cuda.is_available(), logger=loger, opt=opt)
trainer.train(data_loader, opt.n_epochs, save_training_gif=True)

print("Training was finished!")
