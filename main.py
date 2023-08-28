import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

from train import train_generator, train_fn
from loss import generator_loss_function, discriminator_loss_function
from Dataset3D import Dataset3D
from utils import save_model, pack_vars, unpack_vars


def evaluate_model(
    generator,
    discriminator,
    video_folder,
    learning_rate,
    batch_size,
    epochs,
    high_res,
    in_channels=1,
    low_res=None,
    path_to_save=None,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = Dataset3D(video_folder, high_res, low_res, rgb=(in_channels == 3))
    loader = DataLoader(dataset, batch_size=batch_size)

    generator = generator.to(device)
    gen_optimizer = Adam(generator.parameters(), lr=learning_rate)
    generator, gen_optimizer = train_generator(
        generator, gen_optimizer, loader, epochs, device
    )

    discriminator = discriminator.to(device)
    disc_optimizer = Adam(discriminator.parameters(), lr=learning_rate)

    generator_loss = generator_loss_function(in_channels=in_channels)
    discriminator_loss = discriminator_loss_function()

    generator_pack = pack_vars(generator, gen_optimizer, generator_loss)
    discriminator_pack = pack_vars(discriminator, disc_optimizer, discriminator_loss)

    generator_pack, discriminator_pack = train_fn(
        generator_pack, discriminator_pack, loader, epochs, device
    )

    generator, gen_optimizer, generator_loss = unpack_vars(generator_pack)
    discriminator, disc_optimizer, discriminator_loss = unpack_vars(discriminator_pack)
    save_model(generator, gen_optimizer, path_to_save)
    save_model(discriminator, disc_optimizer, path_to_save)

    return generator_pack, discriminator_pack
