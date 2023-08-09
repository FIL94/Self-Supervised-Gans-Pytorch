import torch
import torch.optim as optim
from dataloaders import get_girls_dataloader
from model import Generator, Discriminator
from training import Trainer

data_loader = get_girls_dataloader(r"C:\Users\Admin\Desktop\Flowers", batch_size=2)

generator = Generator(resnet=True, z_size=128, channel=3, output_size=128)
discriminator = Discriminator(resnet=True, spectral_normed=True, num_rotation=4, channel=3, ssup=True)


# Initialize optimizers
lr = 1e-4
betas = (.9, .99)
G_optimizer = optim.Adam(generator.parameters(), lr=lr, betas=betas)
D_optimizer = optim.Adam(discriminator.parameters(), lr=lr, betas=betas)

# Train model
epochs = 5
trainer = Trainer(generator, discriminator, G_optimizer, D_optimizer,
                  weight_rotation_loss_d=1.0, weight_rotation_loss_g=0.5,
                  use_cuda=torch.cuda.is_available())
trainer.train(data_loader, epochs, save_training_gif=False)

# Save models
name = 'girls_model'
torch.save(trainer.G.state_dict(), './gen_' + name + '.pt')
torch.save(trainer.D.state_dict(), './dis_' + name + '.pt')
