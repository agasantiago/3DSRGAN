import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

from train import train_generator, train_fn
from Dataset3D import Dataset3D
from utils import save_model, pack_vars, unpack_vars


def evaluate_model(
    generator,
    discriminator,
    gen_optimizer,
    disc_optimizer,
    video_folder,
    batch_size,
    epochs,
    high_res,
    epochs_gen=None,
    transform=None,
    in_channels=1,
    low_res=None,
    to_print=50,
    path_to_save=None,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = Dataset3D(
        video_folder, high_res, low_res, transform=transform, rgb=(in_channels == 3)
    )
    loader = DataLoader(dataset, batch_size=batch_size)

    generator = generator.to(device)

    if epochs_gen:
        generator, gen_optimizer = train_generator(
            generator, gen_optimizer, loader, epochs_gen, device, to_print=to_print
        )

    discriminator = discriminator.to(device)

    generator_pack = pack_vars(generator, gen_optimizer)
    discriminator_pack = pack_vars(discriminator, disc_optimizer)

    (
        generator_pack,
        discriminator_pack,
        generator_loss_record,
        discriminator_loss_record,
    ) = train_fn(
        generator_pack,
        discriminator_pack,
        in_channels,
        loader,
        epochs,
        device=device,
        to_print=to_print,
    )

    generator, gen_optimizer = unpack_vars(generator_pack)
    discriminator, disc_optimizer = unpack_vars(discriminator_pack)
    save_model(generator, gen_optimizer, path_to_save)
    save_model(discriminator, disc_optimizer, path_to_save)

    return (
        generator_pack,
        discriminator_pack,
        generator_loss_record,
        discriminator_loss_record,
    )


if __name__ == "__main__":
    from torchio import transforms
    from model import Generator, Discriminator

    video_folder = "/home/agasantiago/Documents/Datasets/VideoDataset"
    high_res = (16, 16, 64)

    batch_size = 2
    shuffle = True
    lr = 1.0e-2
    in_channels = 1

    generator = Generator(in_channels=in_channels)
    discriminator = Discriminator(in_channels=in_channels)

    gen_optimizer = Adam(generator.parameters(), lr=lr)
    disc_optimizer = Adam(discriminator.parameters(), lr=lr)

    epochs = 1
    transform = transforms.ZNormalization()

    (
        generator_pack,
        discriminator_pack,
        generator_loss_record,
        discriminator_loss_record,
    ) = evaluate_model(
        generator,
        discriminator,
        gen_optimizer,
        disc_optimizer,
        video_folder,
        batch_size,
        epochs,
        high_res,
        transform=transform,
        to_print=10,
    )
