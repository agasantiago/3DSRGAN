from tqdm import tqdm
from datetime import datetime

from torch import nn


from utils import save_model, pack_vars, unpack_vars


def train_generator(
    generator, optimizer, loader, epochs, device, to_print=10, path_to_save=None
):
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

        if (e + 1) % to_print == 0:
            print(f"[Loss - GENERATOR] Epoch: {e + 1}: {loss.item():.4e}")

        now_time = datetime.now().hour
        if abs(now_time - start_time) >= 2.0:
            save_model(generator, optimizer, path_to_save)

    return generator, optimizer


def train_one_epoch(generator_pack, discriminator_pack, loader, device):
    generator, gen_optimizer, generator_loss_fn = unpack_vars(generator_pack)
    discriminator, disc_optimizer, discriminator_loss_fn = unpack_vars(
        discriminator_pack
    )

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

    generator_pack = pack_vars(generator, gen_optimizer, generator_loss)
    discriminator_pack = pack_vars(discriminator, disc_optimizer, discriminator_loss)

    return generator_pack, discriminator_pack


def train_fn(
    generator_pack,
    discriminator_pack,
    loader,
    epochs,
    device,
    to_print=10,
    path_to_save=None,
):
    generator, gen_optimizer, generator_loss_fn = unpack_vars(generator_pack)
    discriminator, disc_optimizer, discriminator_loss_fn = unpack_vars(
        discriminator_pack
    )

    generator_loss_record = []
    discriminator_loss_record = []

    loop = tqdm(range(epochs))
    start = datetime.now().hour
    for e in loop:
        generator_pack, discriminator_pack = train_one_epoch(
            generator_pack, discriminator_pack, loader, device
        )

        generator, gen_optimizer, generator_loss = unpack_vars(generator_pack)
        discriminator, disc_optimizer, discriminator_loss = unpack_vars(
            discriminator_pack
        )

        if to_print and (e + 1) % to_print == 0:
            print(f"[Loss - Generator] Epoch: {e + 1}: {generator_loss.item():.4e}")
            print(
                f"[Loss - Discriminator] Epoch: {e + 1}: {discriminator_loss.item():.4e}"
            )

        now = datetime.now().hour
        if abs(now - start) >= 2.0:
            save_model(generator, gen_optimizer, path_to_save)
            save_model(discriminator, disc_optimizer, path_to_save)

        generator_loss_record.append(generator_loss.item())
        discriminator_loss_record.append(discriminator_loss.item())

    generator_pack = pack_vars(generator, gen_optimizer, generator_loss_record)
    discriminator_pack = pack_vars(
        discriminator, discriminator_loss, discriminator_loss_record
    )

    save_model(generator, gen_optimizer, path_to_save)
    save_model(discriminator, disc_optimizer, path_to_save)

    return generator_pack, discriminator_pack
