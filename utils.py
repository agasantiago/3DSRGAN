import os
import torch
from datetime import datetime


def _get_path(model, message, path):
    model_name = type(model).__name__
    if not path:
        path = os.getcwd()
        path = os.path.join(path, f"{model_name}.pth")
    time = datetime.now()
    print(f"{message}: {model_name} at {time.strftime('%d/%m at %H:%M')}...")
    return path


def save_model(model, optimizer, path=None):
    path = _get_path(model, "Saving", path=path)
    info = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    torch.save(info, path)
    return


def load_model(model, optimizer, path=None):
    path = _get_path(model, "Loading", path)
    info = torch.load(path)
    model.load_state_dict(info["model_state_dict"])
    optimizer.load_state_dict(info["optimizer_state_dict"])
    return model, optimizer


def pack_vars(model, optimizer, loss_fn):
    setup = {"model": model, "optimizer": optimizer, "loss": loss_fn}
    return setup


def unpack_vars(setup):
    model, optimizer, loss_fn = setup.values()
    return model, optimizer, loss_fn
