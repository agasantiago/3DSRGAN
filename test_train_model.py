import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

from model import Generator, Discriminator
from train import train_generator, train_fn, pack_vars
from loss import generator_loss_function, discriminator_loss_function
from Dataset3D import Dataset3D
from utils import save_model
from main import evaluate_model


def setup_vars():
    video_folder = "/home/agasantiago/Documents/Datasets/VideoDataset"
    high_res = (16, 16, 64)
    batch_size = 1
    shuffle = False
    dataset = Dataset3D(video_folder, high_res, rgb=False)
    loader = DataLoader(dataset, batch_size=batch_size)

    generator = Generator(in_channels=1)
    discriminator = Discriminator(in_channels=1)
    gen_optimizer = Adam(generator.parameters(), lr=1.0e-3)
    disc_optimizer = Adam(discriminator.parameters(), lr=1.0e-3)

    epochs = 1
    device = "cuda" if torch.cuda.is_available() else "cpu"

    return (
        generator,
        discriminator,
        gen_optimizer,
        disc_optimizer,
        loader,
        dataset,
        batch_size,
        shuffle,
        epochs,
        device,
    )


def test_train_generator():
    generator, _, gen_optimizer, _, loader, _, _, _, epochs, device = setup_vars()
    generator, gen_optimizer = train_generator(
        generator, gen_optimizer, loader, epochs, device
    )
    save_model(generator, gen_optimizer)
    return


def test_train_fn():
    in_channels = 1
    (
        generator,
        discriminator,
        gen_optimizer,
        disc_optimizer,
        loader,
        dataset,
        batch_size,
        shuffle,
        epochs,
        device,
    ) = setup_vars()
    generator_pack = pack_vars(
        generator, gen_optimizer, generator_loss_function(in_channels=in_channels)
    )
    discriminator_pack = pack_vars(
        discriminator, disc_optimizer, discriminator_loss_function()
    )
    generator_pack, discriminator_pack = train_fn(
        generator_pack, discriminator_pack, loader, epochs, device
    )
    return


def test_evaluate_model():
    video_folder = "/home/agasantiago/Documents/Datasets/VideoDataset"
    high_res = (16, 16, 64)
    generator, discriminator, gen_optimizer, disc_optimizer, loader, dataset, batch_size, shuffle, epochs, _ =\
        setup_vars()
    generator_pack, discriminator_pack = evaluate_model(generator, discriminator, video_folder, 1.0e-3, batch_size, epochs, high_res)
    return 
