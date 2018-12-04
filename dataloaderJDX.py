import torch
import numpy as np
import lmdb
from PIL import Image
import re
import io
import random

import skimage.io
import skimage.transform
import skimage.color
import skimage

from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler


class JDXTopViewPerson(Dataset):

    def __init__(self, lmdb_path = None, transform = None):

        """
        Args:
            :param root_dir: {string} JDXTopPerson directory
            :param dataset_name: {string} JDXTopPerson data set name
            :param transform: {callable, optional} :Optional transform to be applied on a sample
        """
        self.root_dir = lmdb_path
        self.transform = transform

        self.env = lmdb.open(self.root_dir,  max_dbs=3, max_readers=1, readonly=True, lock=False, readahead=False, meminit=False)
        self.data = self.env.open_db('data'.encode())
        self.annot = self.env.open_db('annot'.encode())

        with self.env.begin(write=False) as txn:
            idx = str(2)
            self.length = txn.stat(db=self.data)['entries']
            imgbuf = txn.get(idx.encode(), db=self.data)
            img = Image.open(io.BytesIO(imgbuf)).convert('RGB')
            img = np.array(img)
            self.width, self.height =img.shape[1], img.shape[0]

    def __len__(self):
        return int(self.length)

    def __getitem__(self, index):

        idx = str(index)
        with self.env.begin(write=False) as txn:
            imgbuf = txn.get(idx.encode(), db=self.data)
            annot  = txn.get(idx.encode(), db=self.annot)
        patten = '\s+'
        annot_ = [int(i) for i in re.split(patten, annot.decode())]
        annotation = np.zeros((1, 5))
        annotation[:, :4] = annot_[1:]
        annotation[:, 4] = 1# represent person type

        # transform from (x, y, w, h ) to (x1, y1, x2, y2)
        annotation[:, 2] = annotation[:, 0] + annotation[:, 2]
        annotation[:, 3] = annotation[:, 1] + annotation[:, 3]

        img = Image.open(io.BytesIO(imgbuf)).convert('RGB')
        img = np.array(img)
        img = img /255 #

        sample = {'img': img, 'annot': annotation}
        if self.transform:
            sample = self.transform(sample)
        return sample

    def num_class(self):
        return 1
    def image_aspect_ratio(self, index = None):
        return float(self.width) / float(self.height)


class NormalizerJDX(object):
    def __init__(self, mean=None, std=None):
        if mean is None:
            self.mean = np.array([[[0.032210593494720066, 0.0310730162881052, 0.038560610305837816]]])
        else:
            self.mean = mean
        if std is None:
            self.std = np.array([[[1.8993060797853687, 1.9120247198746647, 1.8979984319548366]]])
        else:
            self.std = std

    def __call__(self, sample):

        image, annot = sample['img'], sample['annot']

        return {'img': ((image.astype(np.float32) - self.mean) / self.std), 'annot': annot}

class UnNormalizer(object):
    def __init__(self, mean=None, std=None):
        if mean == None:
            self.mean = [0.032210593494720066, 0.0310730162881052, 0.038560610305837816]
        else:
            self.mean = mean
        if std == None:
            self.std = [1.8993060797853687, 1.9120247198746647, 1.8979984319548366]
        else:
            self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor

 #   def __call__(self, tensor):
        """
        
        :param tensor: {pytorch.Tensor} Tensor images of size (C, H, W)  
        :return: 
        """

class flip_xJDX(object):

    def __call__(self, sample, flip = 0.5):

        if np.random.rand() < flip:
            image, annot = sample['img'], sample['annot']

            image = image[:, ::-1, :]
            rows, cols, channels = image.shape

            x1 = annot[:, 0].copy()
            x2 = annot[:, 2].copy()

            x_tmp = x1.copy()

            annot[:, 0] = cols - x2
            annot[:, 2] = cols - x_tmp

            sample = {'img': image, 'annot': annot}
        return sample

class ResizerJDX(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample, min_side=608, max_side=1024):
        image, annots = sample['img'], sample['annot']

        rows, cols, cns = image.shape

        smallest_side = min(rows, cols)

        # rescale the image so the smallest side is min_side
        scale = min_side / smallest_side

        # check if the largest side is now greater than max_side, which can happen
        # when images have a large aspect ratio
        largest_side = max(rows, cols)

        if largest_side * scale > max_side:
            scale = max_side / largest_side

        # resize the image with the computed scale
        image = skimage.transform.resize(image, (int(round(rows*scale)), int(round((cols*scale)))), mode='reflect')
        rows, cols, cns = image.shape

        pad_w = 32 - rows%32
        pad_h = 32 - cols%32

        new_image = np.zeros((rows + pad_w, cols + pad_h, cns)).astype(np.float32)
        new_image[:rows, :cols, :] = image.astype(np.float32)

        annots[:, :4] *= scale

        return {'img': torch.from_numpy(new_image), 'annot': torch.from_numpy(annots), 'scale': scale}

class AspectRatioBasedSampler(Sampler):

    def __init__(self, data_source, batch_size, drop_last):
        self.data_source = data_source
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.groups = self.group_images()

    def __iter__(self):
        random.shuffle(self.groups)
        for group in self.groups:
            yield group

    def __len__(self):
        if self.drop_last:
            return len(self.data_source) // self.batch_size
        else:
            return (len(self.data_source) + self.batch_size - 1) // self.batch_size

    def group_images(self):
        # determine the order of the images
        order = list(range(len(self.data_source)))
        order.sort(key=lambda x: self.data_source.image_aspect_ratio(x))

        # divide into groups, one group = one batch
        return [[order[x % len(order)] for x in range(i, i + self.batch_size)] for i in range(0, len(order), self.batch_size)]


def collater(data):
    imgs = [s['img'] for s in data]
    annots = [s['annot'] for s in data]
    scales = [s['scale'] for s in data]

    widths = [int(s.shape[0]) for s in imgs]
    heights = [int(s.shape[1]) for s in imgs]
    batch_size = len(imgs)

    max_width = np.array(widths).max()
    max_height = np.array(heights).max()

    padded_imgs = torch.zeros(batch_size, max_width, max_height, 3)

    for i in range(batch_size):
        img = imgs[i]
        padded_imgs[i, :int(img.shape[0]), :int(img.shape[1]), :] = img

    max_num_annots = max(annot.shape[0] for annot in annots)

    if max_num_annots > 0:

        annot_padded = torch.ones((len(annots), max_num_annots, 5)) * -1

        if max_num_annots > 0:
            for idx, annot in enumerate(annots):
                # print(annot.shape)
                if annot.shape[0] > 0:
                    annot_padded[idx, :annot.shape[0], :] = annot
    else:
        annot_padded = torch.ones((len(annots), 1, 5)) * -1

    padded_imgs = padded_imgs.permute(0, 3, 1, 2)

    return {'img': padded_imgs, 'annot': annot_padded, 'scale': scales}


def test():
    lmdb_path = "/home/boby/Desktop/pengcheng_work_note/note/train_test/train_lmdb"
    dset = JDXTopViewPerson(lmdb_path=lmdb_path)
    sample = dset[2]
    img = sample['img']
    annot = sample['annot']

    print(img.shape)
    print(annot)

if __name__ == '__main__':
    test()