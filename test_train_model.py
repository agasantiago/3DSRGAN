import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from model import Generator, Discriminator
from train import train_one_epoch, train_generator, train_fn
from Dataset3D import Dataset3D
from utils import save_model


def setup_vars():
    video_folder = '/home/agasantiago/Documents/Datasets/VideoDataset'
    high_res = (16, 16, 64)
    dataset = Dataset3D(video_folder, high_res, rgb=False)
    loader = DataLoader(dataset, batch_size=1)

    generator = Generator(in_channels=1)
    discriminator = Discriminator(in_channels=1)
    gen_optimizer = Adam(generator.parameters(), lr=1.0e-3)
    disc_optimizer = Adam(discriminator.parameters(), lr=1.0e-3)

    epochs = 1
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    return generator, discriminator, gen_optimizer, disc_optimizer, loader, epochs, device



def test_train_generator():
    generator, _, gen_optimizer, _, loader, epochs, device = setup_vars()
    generator, gen_optimizer = train_generator(generator, gen_optimizer, loader, epochs, device)
    save_model(generator, gen_optimizer)
    return


def test_train_fn():
    _, _, _, _, loader, epochs, device = setup_vars()
    in_channels = 1
    lr = 1.0e-3
    generator_pack, discriminator_pack = train_fn(loader, epochs, lr, device, in_channels)
    return

