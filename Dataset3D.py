import os
import cv2
import numpy as np
from glob import glob

from torch.utils.data import Dataset, DataLoader
import torchio as tio


def read_video_file(video_file, rgb):
    def cvtColor(frame):
        if rgb:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return np.expand_dims(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), axis=2)

    assert os.path.exists(video_file)
    video = cv2.VideoCapture(video_file)
    length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for _ in range(length):
        ret, frame = video.read()
        if ret:
            frame = cvtColor(frame)
            frames.append(frame)
        else:
            break
    frames = np.stack(frames, axis=3)
    frames = frames.transpose(2, 0, 1, 3)
    return frames


def _create_subject(frames):
    subject = tio.Subject(t1=tio.ScalarImage(tensor=frames))
    return subject


def _create_subject_dataset(videos, transforms):
    subject_list = [_create_subject(frames) for frames in videos]
    subject_dataset = tio.SubjectsDataset(subject_list, transforms)
    return subject_dataset


def _create_queue(subject_list, size):
    sampler = tio.data.UniformSampler(size)
    queue = tio.Queue(subject_list, 10, 4, sampler)
    return queue


def _create_dataset(video_folder, transform, size, rgb, ext):
    video_files = [file for file in glob(os.path.join(video_folder, f"*{ext}"))]
    videos = [read_video_file(video_file, rgb) for video_file in video_files]
    subject_dataset = _create_subject_dataset(videos, transform)
    queue = _create_queue(subject_dataset, size)
    return queue


class Dataset3D(Dataset):
    def __init__(
        self,
        video_folder,
        high_res,
        low_res=None,
        transform=None,
        rgb=False,
        ext=".mp4",
    ):
        super().__init__()
        if not low_res:
            h, w, d = high_res
            low_res = (h, w, d // 4)
        self._low_res_transf = tio.transforms.Resize(target_shape=low_res)
        self._dataset = _create_dataset(video_folder, transform, high_res, rgb, ext)

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, item):
        y = self._dataset[item]
        X = self._low_res_transf(y)
        return X["t1"]["data"], y["t1"]["data"]


if __name__ == "__main__":
    video_folder = "/home/agasantiago/Documents/Datasets/VideoDataset"
    high_res = (32, 32, 128)
    low_res = (32, 32, 64)
    volume_transforms = tio.transforms.ZNormalization()
    dataset = Dataset3D(
        video_folder, high_res, transform=volume_transforms, low_res=low_res, rgb=False
    )
    loader = DataLoader(dataset, batch_size=1)
    for X, y in loader:
        break

    print(X.max())
    print(y.shape)
