#!/usr/bin/env python
# coding: utf-8

# In[4]:


import torch
from dataset import flowermonetDataset
import sys
from utils import save_checkpoint, load_checkpoint
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import config
from tqdm import tqdm
from torchvision.utils import save_image
from discriminator_model import Discriminator
from generator_model import Generator

#  Hooorse = Flooower
# Zeeebra = Mooonet
def train_fn(
    disc_F, disc_M, gen_M, gen_F, loader, opt_disc, opt_gen, l1, mse, d_scaler, g_scaler
):
    F_reals = 0
    F_fakes = 0
    loop = tqdm(loader, leave=True)

    for idx, (monet, flower) in enumerate(loop):
        monet = monet.to(config.DEVICE)
        flower = flower.to(config.DEVICE)

        # Train Discriminators for flower and Monet Pictures
        with torch.cuda.amp.autocast():
            fake_flower = gen_F(monet)
            D_F_real = disc_F(flower)
            D_F_fake = disc_F(fake_flower.detach())
            F_reals += D_F_real.mean().item()
            F_fakes += D_F_fake.mean().item()
            D_F_real_loss = mse(D_F_real, torch.ones_like(D_F_real))
            D_F_fake_loss = mse(D_F_fake, torch.zeros_like(D_F_fake))
            D_F_loss = D_F_real_loss + D_F_fake_loss

            fake_monet = gen_M(flower)
            D_M_real = disc_M(monet)
            D_M_fake = disc_M(fake_monet.detach())
            D_M_real_loss = mse(D_M_real, torch.ones_like(D_M_real))
            D_M_fake_loss = mse(D_M_fake, torch.zeros_like(D_M_fake))
            D_M_loss = D_M_real_loss + D_M_fake_loss

            # put it togethor
            D_loss = (D_F_loss + D_M_loss) / 2

        opt_disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # Train Generators for flowers and monet pictures
        with torch.cuda.amp.autocast():
            # adversarial loss for both generators
            D_F_fake = disc_F(fake_flower)
            D_M_fake = disc_M(fake_monet)
            loss_G_F = mse(D_F_fake, torch.ones_like(D_F_fake))
            loss_G_M = mse(D_M_fake, torch.ones_like(D_M_fake))

            # cycle loss
            cycle_monet = gen_M(fake_flower)
            cycle_flower = gen_F(fake_monet)
            cycle_monet_loss = l1(monet, cycle_monet)
            cycle_flower_loss = l1(flower, cycle_flower)

            # identity loss (remove these for efficiency if you set lambda_identity=0)
            identity_monet = gen_M(monet)
            identity_flower = gen_F(flower)
            identity_monet_loss = l1(monet, identity_monet)
            identity_flower_loss = l1(flower, identity_flower)

            # add all togethor
            G_loss = (
                loss_G_M
                + loss_G_F
                + cycle_monet_loss * config.LAMBDA_CYCLE
                + cycle_flower_loss * config.LAMBDA_CYCLE
                + identity_flower_loss * config.LAMBDA_IDENTITY
                + identity_monet_loss * config.LAMBDA_IDENTITY
            )

        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        if idx % 200 == 0:
            save_image(fake_flower * 0.5 + 0.5, f"saved_images/flower_{idx}.png")
            save_image(fake_monet * 0.5 + 0.5, f"saved_images/monet_{idx}.png")

        loop.set_postfix(F_real=F_reals / (idx + 1), F_fake=F_fakes / (idx + 1))


def main():
    disc_F = Discriminator(in_channels=3).to(config.DEVICE)
    disc_M = Discriminator(in_channels=3).to(config.DEVICE)
    gen_M = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
    gen_F = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
    opt_disc = optim.Adam(
        list(disc_F.parameters()) + list(disc_M.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    opt_gen = optim.Adam(
        list(gen_M.parameters()) + list(gen_F.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    L1 = nn.L1Loss()
    mse = nn.MSELoss()

    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_GEN_F,
            gen_F,
            opt_gen,
            config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_GEN_M,
            gen_M,
            opt_gen,
            config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_CRITIC_F,
            disc_F,
            opt_disc,
            config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_CRITIC_M,
            disc_M,
            opt_disc,
            config.LEARNING_RATE,
        )

    dataset = flowermonetDataset(
        root_flower="./data/train/flower",
        root_monet= "./data/train/monet",
        transform=config.transforms,
    )
    val_dataset = flowermonetDataset(
        root_flower= "./data/val/flower",
        root_monet="./data/val/monet",
        transform=config.transforms,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()

    for epoch in range(config.NUM_EPOCHS):
        train_fn(
            disc_F,
            disc_M,
            gen_M,
            gen_F,
            loader,
            opt_disc,
            opt_gen,
            L1,
            mse,
            d_scaler,
            g_scaler,
        )

        if config.SAVE_MODEL:
            save_checkpoint(gen_F, opt_gen, filename=config.CHECKPOINT_GEN_F)
            save_checkpoint(gen_M, opt_gen, filename=config.CHECKPOINT_GEN_M)
            save_checkpoint(disc_F, opt_disc, filename=config.CHECKPOINT_CRITIC_F)
            save_checkpoint(disc_M, opt_disc, filename=config.CHECKPOINT_CRITIC_M)


if __name__ == "__main__":
    main()

