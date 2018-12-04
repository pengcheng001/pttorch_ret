import numpy as np
import torch
import time

from dataloaderJDX import JDXTopViewPerson, UnNormalizer, ResizerJDX, AspectRatioBasedSampler, collater,NormalizerJDX

from torchvision import transforms
from torch.utils.data import DataLoader

import cv2 as cv



def main():

    lmdb_path = './dataset/test_lmdb'
    model_path = '/home/boby/Desktop/pengcheng_work_note/src/JDX_src/deep-person-id/src/pytorch-retinanet/_retinanet_38.pt'

    dataset_val = JDXTopViewPerson(lmdb_path=lmdb_path, transform=transforms.Compose([NormalizerJDX(), ResizerJDX()]))

    sample_val = AspectRatioBasedSampler(dataset_val, batch_size=1, drop_last=False)

    dataloader_val  = DataLoader(dataset_val,num_workers=1, collate_fn=collater, batch_sampler=sample_val)

    retinanet = torch.load(model_path)

    if torch.cuda.is_available():
        retinanet = retinanet.cuda()
    retinanet.training = False

    retinanet.eval()

    unnormalize = UnNormalizer()

    def draw_caption(img, box, lables):
        b = np.array(box).astype(int)
        cv.putText(img, lables, (b[0], b[1] - 10), cv.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
        cv.putText(img, lables, (b[0], b[1] - 10), cv.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)

    for idx, data in enumerate(dataloader_val):
        with torch.no_grad():
            st = time.time()
            scores, classification, transformed_anchors = retinanet(data['img'].cuda().float())
            print('Elapsed time: {}'.format(time.time() - st))




            idxs = np.where(scores > 0.4)
            img = np.array( unnormalize(data['img'][0, :, :, :])).copy()

           # img[img < 0] = 0
          #  img[img > 255] = 255
            print(data['img'].size())
            img =  np.array(data['img'][0,:,:,:])


            img = np.transpose(img, (1, 2, 0))
            r = img[:,:,0].copy()
            g = img[:,:,1].copy()
            b = img[:,:,2].copy()
            img[:, :, 0] = b
            img[:, :, 1] = g
            img[:, :, 2] = r

            # img = cv.cvtColor(img.astype(np.uint8), cv.COLOR_BGR2RGB)

            for j in range(idxs[0].shape[0]):
                bbox = transformed_anchors[idxs[0][j], :]
                x1 = int(bbox[0])
                y1 = int(bbox[1])
                x2 = int(bbox[2])
                y2 = int(bbox[3])
               # label_name = dataset_val.labels[int(classification[idxs[0][j]])]

                draw_caption(img, (x1, y1, x2, y2), str(idxs[0][j]))

                cv.rectangle(img, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)
               # print(label_name)

            cv.imshow('img', img)
            cv.waitKey(0)


if __name__ == '__main__':
    main()






