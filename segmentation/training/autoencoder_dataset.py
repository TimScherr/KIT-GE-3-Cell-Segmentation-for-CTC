import json
import numpy as np
import tifffile as tiff
from random import randint, shuffle
from torch.utils.data import Dataset


class AutoEncoderDataset(Dataset):
    """ Pytorch data set for instance cell nuclei segmentation """

    def __init__(self, root_dir, gt_train_dir, cell_type, transform=lambda x: x):
        """

        :param root_dir: Directory containing the Cell Tracking Challenge Data.
            :type root_dir: pathlib Path object.
        :param gt_train_dir: Directory containing the to the cell type belonging GT train data (only needed to get the
                                scale factor)
            :type gt_train_dir: pathlib Path object.
        :param cell_type: Cell type.
            :type cell_type: str
        :param transform: transforms/augmentations.
            :type transform:
        :return sample (image, image, scale).
        """

        self.img_ids = []

        img_ids_01 = sorted((root_dir / 'training_datasets' / cell_type / '01').glob('*.tif'))
        if len(img_ids_01) > 1500:
            img_ids_01 = img_ids_01[1500:]
        elif len(img_ids_01) > 1000:
            img_ids_01 = img_ids_01[1000:]
        while len(img_ids_01) > 75:
            img_ids_01 = img_ids_01[::5]
        if len(img_ids_01) > 15:
            shuffle(img_ids_01)
            img_ids_01 = img_ids_01[:15]

        img_ids_02 = sorted((root_dir / 'training_datasets' / cell_type / '02').glob('*.tif'))
        if len(img_ids_02) > 1500:
            img_ids_02 = img_ids_02[1500:]
        elif len(img_ids_02) > 1000:
            img_ids_02 = img_ids_02[1000:]
        while len(img_ids_02) > 75:
            img_ids_02 = img_ids_02[::5]
        if len(img_ids_02) > 15:
            shuffle(img_ids_02)
            img_ids_02 = img_ids_02[:15]

        self.img_ids = img_ids_01 + img_ids_02

        with open(gt_train_dir / "{}_GT".format(cell_type) / 'info.json') as f:
            self.scale = json.load(f)['scale']

        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):

        img_id = self.img_ids[idx]

        img = tiff.imread(str(img_id))

        if len(img.shape) == 2:
            img = img[..., None]
        else:
            img_mean, img_std = np.mean(img), np.std(img)
            i = randint(0, img.shape[0]-1)
            h = 0
            while np.mean(img[i]) < img_mean and h <= 10:
                i = randint(0, img.shape[0]-1)
                h += 1
            img = img[i, :, :, None]

        sample = {'image': img,
                  'label': img,
                  'scale': self.scale}

        sample = self.transform(sample)

        return sample
