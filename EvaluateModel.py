import torchio


from Dataset3D import read_video_file


class Dataset:
    def __init__(self, video_filename, rgb=False):
        frames = read_video_file(video_filename, rgb=rgb)
        transforms = torchio.ZNormalization()
        subject = torchio.Subject(t1=torchio.ScalarImage(tensor=frames))
        self._subject = torchio.SubjectsDataset([subject], transforms)

    @property
    def subject(self):
        return self._subject[0]['t1']['data']


if __name__ == "__main__":
    video_filename = '/home/agasantiago/Documents/Datasets/HighResVideoDataset/HighRes1_0_5_9.mp4'
    data = Dataset(video_filename, rgb=True)
    subject = data.subject

    print(subject.shape)
