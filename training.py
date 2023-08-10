"""
    Pytorch implementation of Self-Supervised GAN
    Reference: "Self-Supervised GANs via Auxiliary Rotation Loss"
    Authors: Ting Chen,
                Xiaohua Zhai,
                Marvin Ritter,
                Mario Lucic and
                Neil Houlsby
    https://arxiv.org/abs/1811.11212 CVPR 2019.
    Script Author: Vandit Jain. Github:vandit15
"""
import os.path
import sys

import imageio
import numpy as np
import torch
from torchvision.utils import make_grid
from torch.autograd import Variable
from torch.autograd import grad as torch_grad
import torch.nn.functional as F


class Trainer:
    def __init__(self, generator, discriminator, gen_optimizer, dis_optimizer,
                 weight_rotation_loss_d, weight_rotation_loss_g, gp_weight=10, critic_iterations=1,
                 print_every=1, use_cuda=False, logger=None, opt=None):
        self.G = generator
        self.G_opt = gen_optimizer
        self.D = discriminator
        self.D_opt = dis_optimizer
        self.losses = {'G': [], 'D': [], 'GP': [], 'gradient_norm': []}
        self.num_steps = 0
        self.use_cuda = use_cuda
        self.gp_weight = gp_weight
        self.critic_iterations = critic_iterations
        self.print_every = print_every
        self.weight_rotation_loss_d = weight_rotation_loss_d
        self.weight_rotation_loss_g = weight_rotation_loss_g
        self.logger = logger
        self.opt = opt

        if self.use_cuda:
            self.G.cuda()
            self.D.cuda()

    def _critic_train_iteration(self, data, generated_data, batch_size):
        """ """
        # Calculate probabilities on real and generated data
        data = Variable(data)
        if self.use_cuda:
            data = data.cuda()
        _, d_real_pro_logits, d_real_rot_logits, d_real_rot_prob = self.D(data)
        _, g_fake_pro_logits, g_fake_rot_logits, g_fake_rot_prob = self.D(generated_data)

        # Get gradient penalty
        gradient_penalty = self._gradient_penalty(data, generated_data)
        self.losses['GP'].append(gradient_penalty.data)

        # Create total loss and optimize
        self.D_opt.zero_grad()
        d_loss = torch.sum(g_fake_pro_logits) - torch.sum(d_real_pro_logits) + gradient_penalty

        # Add auxiliary rotation loss
        rot_labels = torch.zeros(4 * batch_size).cuda()
        for i in range(4 * batch_size):
            if i < batch_size:
                rot_labels[i] = 0
            elif i < 2 * batch_size:
                rot_labels[i] = 1
            elif i < 3 * batch_size:
                rot_labels[i] = 2
            else:
                rot_labels[i] = 3

        rot_labels = F.one_hot(rot_labels.to(torch.int64), 4).float()
        d_real_class_loss = torch.sum(F.binary_cross_entropy_with_logits(
            input=d_real_rot_logits.repeat(4, 1),
            target=rot_labels))

        d_loss += self.weight_rotation_loss_d * d_real_class_loss
        d_loss.backward(retain_graph=True)

        self.D_opt.step()

        # Record loss
        self.losses['D'].append(d_loss.data)

    def _generator_train_iteration(self, generated_data, batch_size):
        """ """
        self.G_opt.zero_grad()

        # Calculate loss and optimize
        _, g_fake_pro_logits, g_fake_rot_logits, g_fake_rot_prob = self.D(generated_data)
        g_loss = - torch.sum(g_fake_pro_logits)

        # add auxiliary rotation loss
        rot_labels = torch.zeros(4 * batch_size, ).cuda()
        for i in range(4 * batch_size):
            if i < batch_size:
                rot_labels[i] = 0
            elif i < 2 * batch_size:
                rot_labels[i] = 1
            elif i < 3 * batch_size:
                rot_labels[i] = 2
            else:
                rot_labels[i] = 3

        rot_labels = F.one_hot(rot_labels.to(torch.int64), 4).float()
        g_fake_class_loss = torch.sum(F.binary_cross_entropy_with_logits(
            input=g_fake_rot_logits.repeat(4, 1),
            target=rot_labels))

        g_loss += self.weight_rotation_loss_g * g_fake_class_loss

        g_loss.backward(retain_graph=True)
        self.G_opt.step()

        # Record loss
        self.losses['G'].append(g_loss.data)

    def _gradient_penalty(self, real_data, generated_data):
        batch_size = real_data.size()[0]

        # Calculate interpolation
        alpha = torch.rand(batch_size, 1, 1, 1)
        alpha = alpha.expand_as(real_data)
        if self.use_cuda:
            alpha = alpha.cuda()
        interpolated = alpha * real_data.data + (1 - alpha) * generated_data.data
        interpolated = Variable(interpolated, requires_grad=True)
        if self.use_cuda:
            interpolated = interpolated.cuda()

        # Calculate probability of interpolated examples
        _, prob_interpolated, _, _ = self.D(interpolated)

        # Calculate gradients of probabilities with respect to examples
        gradients = torch_grad(outputs=prob_interpolated, inputs=interpolated,
                               grad_outputs=torch.ones(
                                   prob_interpolated.size()).cuda() if self.use_cuda else torch.ones(
                                   prob_interpolated.size()),
                               create_graph=True, retain_graph=True)[0]

        # Gradients have shape (batch_size, num_channels, img_width, img_height),
        # so flatten to easily take norm per example in batch
        gradients = gradients.view(batch_size, -1)
        self.losses['gradient_norm'].append(gradients.norm(2, dim=1).sum().data)

        # Derivatives of the gradient close to 0 can cause problems because of
        # the square root, so manually calculate norm and add epsilon
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

        # Return gradient penalty
        return self.gp_weight * ((gradients_norm - 1) ** 2).mean()

    def _train_epoch(self, data_loader):
        for i, data in enumerate(data_loader):
            # Get generated data
            data = data
            batch_size = data.size()[0]
            if batch_size == 1:
                continue
            generated_data = self.sample_generator(batch_size)

            x = generated_data
            x_90 = x.transpose(-2, -1)
            x_180 = x.flip(-2, -1)
            x_270 = x.transpose(-2, -1).flip(-1, -2)
            generated_data = torch.cat((x, x_90, x_180, x_270), 1)

            x = data
            x_90 = x.transpose(-2, -1)
            x_180 = x.flip(-2, -1)
            x_270 = x.transpose(-2, -1).flip(-1, -2)
            data = torch.cat((x, x_90, x_180, x_270), 1)

            self.num_steps += 1
            self._critic_train_iteration(data, generated_data, batch_size)
            # Only update self.G every |critic_iterations| iterations
            if self.num_steps % self.critic_iterations == 0:
                self._generator_train_iteration(generated_data, batch_size)

            if i % self.print_every == 0:
                print("Iteration {}".format(i + 1))
                print("D: {}".format(self.losses['D'][-1]))
                print("GP: {}".format(self.losses['GP'][-1]))
                print("Gradient norm: {}".format(self.losses['gradient_norm'][-1]))
                if self.num_steps > self.critic_iterations:
                    print("G: {}".format(self.losses['G'][-1]))

    def train(self, data_loader, epochs, save_training_gif=True):

        if self.opt.pretrained_models is not None:
            try:
                self.G.load_state_dict(
                    torch.load(os.path.join(self.opt.log_dir, 'checkpoints', 'generators',
                                            f'model_{self.opt.pretrained_models}.pt')))
                self.G.train()
                self.G_opt.load_state_dict(
                    torch.load(os.path.join(self.opt.log_dir, 'checkpoints', 'generators',
                                            f'optimizer_{self.opt.pretrained_models}.pt')))
                self.D.load_state_dict(
                    torch.load(os.path.join(self.opt.log_dir, 'checkpoints', 'discriminators',
                                            f'model_{self.opt.pretrained_models}.pt')))
                self.D.train()
                self.D_opt.load_state_dict(
                    torch.load(os.path.join(self.opt.log_dir, 'checkpoints', 'discriminators',
                                            f'optimizer_{self.opt.pretrained_models}.pt')))
                start_point = self.opt.pretrained_models
            except:
                print("Pretrained models was not found!")
                sys.exit()
        else:
            start_point = 0

        if self.G_opt.param_groups[0]['lr'] != self.opt.lr:
            for g in self.G_opt.param_groups:
                g['lr'] = self.opt.lr
        if self.D_opt.param_groups[0]['lr'] != self.opt.lr:
            for g in self.D_opt.param_groups:
                g['lr'] = self.opt.lr

        if save_training_gif:
            # Fix latents to see how image generation improves during training
            fixed_latents = Variable(self.G.sample_latent(64))
            if self.use_cuda:
                fixed_latents = fixed_latents.cuda()
            training_progress_images = []

        for epoch in range(start_point + 1, start_point + self.opt.n_epochs + 1):
            print("\nEpoch {}".format(epoch))
            self._train_epoch(data_loader)

            self.logger.add_scalar("Generator loss", self.losses['G'][-1], epoch)
            self.logger.add_scalar("Discriminator loss", self.losses['D'][-1], epoch)
            self.logger.flush()

            with open(os.path.join(self.opt.log_dir, 'metrics', 'logs.txt'), 'a') as f:
                f.write(f"{epoch} {self.losses['D'][-1]} {self.losses['D'][-1]}\n")

            torch.save(self.G.state_dict(),
                       os.path.join(self.opt.log_dir, 'checkpoints', 'generators', f'model_{epoch}.pt'))
            torch.save(self.G_opt.state_dict(),
                       os.path.join(self.opt.log_dir, 'checkpoints', 'generators', f'optimizer_{epoch}.pt'))
            torch.save(self.D.state_dict(),
                       os.path.join(self.opt.log_dir, 'checkpoints', 'discriminators', f'model_{epoch}.pt'))
            torch.save(self.D_opt.state_dict(),
                       os.path.join(self.opt.log_dir, 'checkpoints', 'discriminators', f'optimizer_{epoch}.pt'))

            if save_training_gif:
                # Generate batch of images and convert to grid
                img_grid = make_grid(self.G(fixed_latents).cpu().data)
                # Convert to numpy and transpose axes to fit imageio convention
                # i.e. (width, height, channels)
                img_grid = np.transpose(img_grid.numpy(), (1, 2, 0))
                # Add image grid to training progress
                training_progress_images.append((img_grid*255).astype(np.uint8))

                if epoch % self.opt.sample_interval == 0:
                    imageio.mimsave(os.path.join(self.opt.log_dir, 'images', '{}.gif'.format(epoch)),
                                    training_progress_images)

        self.logger.close()

    def sample_generator(self, num_samples):
        latent_samples = Variable(self.G.sample_latent(num_samples))
        if self.use_cuda:
            latent_samples = latent_samples.cuda()
        generated_data = self.G(latent_samples)
        return generated_data

    def sample(self, num_samples):
        generated_data = self.sample_generator(num_samples)
        # Remove color channel
        return generated_data.data.cpu().numpy()[:, 0, :, :]
