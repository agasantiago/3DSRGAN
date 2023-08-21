import os
from tqdm import tqdm
import numpy as np
from datetime import datetime

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader


from model import Generator, Discriminator
from loss import VggLoss
from Dataset3D import Dataset3D
from utils import save_model


def generator_loss_function(in_channels, vgg_coef=6.0e-6, bce_coef=1.0e-3, mse_coef=1.0):
    vgg_loss_fn = VggLoss(in_channels=in_channels)
    bce_loss_fn = nn.BCEWithLogitsLoss()
    mse_loss_fn = nn.MSELoss()
    def loss_fn(high_res, high_res_fake, clf_high_res_fake):
        adversarial_loss = bce_loss_fn(clf_high_res_fake, torch.ones_like(clf_high_res_fake))
        mse_loss = mse_loss_fn(high_res, high_res_fake)
        vgg_loss = vgg_loss_fn(high_res_fake.detach(), high_res.detach())
        loss = vgg_coef*vgg_loss + bce_coef*adversarial_loss + mse_coef*mse_loss
        return loss
    return loss_fn


def discriminator_loss_function():
    bce_loss_fn = nn.BCEWithLogitsLoss()
    def loss_fn(clf_high_res, clf_high_res_fake):
        loss_high_res = bce_loss_fn(clf_high_res, torch.ones_like(clf_high_res) - 1.0e-2*torch.rand_like(clf_high_res))
        loss_high_res_fake = bce_loss_fn(clf_high_res_fake, torch.zeros_like(clf_high_res_fake))
        loss = loss_high_res + loss_high_res_fake
        return loss
    return loss_fn


def train_generator(generator, optimizer, loader, epochs, device, to_print=10, path_to_save=None):
    generator.train()
    generator.to(device)
    loop = tqdm(range(epochs))
    loss_fn = nn.MSELoss()
    start_time = datetime.now().hour
    for e in loop:
        for X, y in loader:
            X = X.to(device).float()
            y = y.to(device).float()

            y_hat = generator(X).float()
            loss = loss_fn(y, y_hat)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (e + 1)%to_print == 0:
            print(f"[Loss - GENERATOR] Epoch: {e + 1}: {loss.item():.4e}")

        now_time = datetime.now().hour
        if abs(now_time - start_time) >= 2.0:
            save_model(generator, optimizer, path_to_save)

    return generator, optimizer



def _pack_vars(model, optimizer, loss_fn):
    setup = {
        'model': model,
        'optimizer': optimizer,
        'loss': loss_fn
    }
    return setup

def _unpack_vars(setup):
    model, optimizer, loss_fn = setup.values()
    return model, optimizer, loss_fn


def train_one_epoch(generator_pack, discriminator_pack, loader, device):

    generator, gen_optimizer, generator_loss_fn = _unpack_vars(generator_pack)
    discriminator, disc_optimizer, discriminator_loss_fn = _unpack_vars(discriminator_pack)

    generator.to(device)
    discriminator.to(device)

    generator.train()
    discriminator.train()

    for low_res, high_res in loader:
        disc_optimizer.zero_grad()
        gen_optimizer.zero_grad()

        low_res = low_res.to(device).float()
        high_res = high_res.to(device).float()

        high_res_fake = generator(low_res).float()

        clf_high_res = discriminator(high_res).float()
        clf_high_res_fake = discriminator(high_res_fake).float()

        discriminator_loss = discriminator_loss_fn(clf_high_res, clf_high_res_fake)
        discriminator_loss.backward(retain_graph=True)

        generator_loss = generator_loss_fn(high_res, high_res_fake, clf_high_res_fake)
        generator_loss.backward()

        disc_optimizer.step()
        gen_optimizer.step()

    generator_pack = _pack_vars(generator, gen_optimizer, generator_loss)
    discriminator_pack = _pack_vars(discriminator, disc_optimizer, discriminator_loss)

    return generator_pack, discriminator_pack


def train_fn(loader, epochs, lr, device, in_channels, to_print=10, path_to_save=None):
    generator = Generator(in_channels=in_channels).to(device)
    discriminator = Discriminator(in_channels=in_channels).to(device)

    gen_optimizer = Adam(generator.parameters(), lr=lr)
    disc_optimizer = Adam(discriminator.parameters(), lr=lr)

    generator_loss_fn = generator_loss_function(in_channels)
    discriminator_loss_fn = discriminator_loss_function()

    generator_pack = _pack_vars(generator, gen_optimizer, generator_loss_fn)
    discriminator_pack = _pack_vars(discriminator, disc_optimizer, discriminator_loss_fn)

    generator_loss_record = []
    discriminator_loss_record = []

    loop = tqdm(range(epochs))
    start = datetime.now().hour
    for e in loop:
        generator_pack, discriminator_pack = train_one_epoch(generator_pack, discriminator_pack, loader, device)

        generator, gen_optimizer, generator_loss = _unpack_vars(generator_pack)
        discriminator, disc_optimizer, discriminator_loss = _unpack_vars(discriminator_pack)

        if (e + 1)%to_print == 0:
            print(f"[Loss - Generator] Epoch: {e + 1}: {generator_loss.item():.4e}")
            print(f"[Loss - Discriminator] Epoch: {e + 1}: {discriminator_loss.item():.4e}")

        now = datetime.now().hour
        if abs(now - start) >= 2.0:
            save_model(generator, gen_optimizer, path_to_save)
            save_model(discriminator, disc_optimizer, path_to_save)


        generator_loss_record.append(generator_loss.item())
        discriminator_loss_record.append(discriminator_loss.item())

    generator_pack = _pack_vars(generator, gen_optimizer, generator_loss_record)
    discriminator_pack = _pack_vars(discriminator, discriminator_loss, discriminator_loss_record)
    return generator_pack, discriminator_pack
