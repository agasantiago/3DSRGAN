import torch
from torch import nn


from VggLoss import VggLoss


def generator_loss_function(
    in_channels, vgg_coef=6.0e-6, bce_coef=1.0e-3, mse_coef=1.0
):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    vgg_loss_fn = VggLoss(in_channels=in_channels).to(device)
    bce_loss_fn = nn.BCEWithLogitsLoss()
    mse_loss_fn = nn.MSELoss()

    def loss_fn(high_res, high_res_fake, clf_high_res_fake):
        adversarial_loss = bce_loss_fn(
            clf_high_res_fake, torch.ones_like(clf_high_res_fake)
        )
        mse_loss = mse_loss_fn(high_res, high_res_fake)
        vgg_loss = vgg_loss_fn(high_res_fake.detach(), high_res.detach())
        loss = vgg_coef * vgg_loss + bce_coef * adversarial_loss + mse_coef * mse_loss
        return loss

    return loss_fn


def discriminator_loss_function():
    bce_loss_fn = nn.BCEWithLogitsLoss()

    def loss_fn(clf_high_res, clf_high_res_fake):
        loss_high_res = bce_loss_fn(
            clf_high_res,
            torch.ones_like(clf_high_res) - 1.0e-2 * torch.rand_like(clf_high_res),
        )
        loss_high_res_fake = bce_loss_fn(
            clf_high_res_fake, torch.zeros_like(clf_high_res_fake)
        )
        loss = loss_high_res + loss_high_res_fake
        return loss

    return loss_fn
