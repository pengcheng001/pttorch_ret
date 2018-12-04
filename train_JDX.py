
import numpy as np
import torch
import model
from torchvision import transforms
import torch.optim as optim
import collections

import argparse

from dataloaderJDX import JDXTopViewPerson, NormalizerJDX, flip_xJDX, ResizerJDX,AspectRatioBasedSampler,collater
from torch.utils.data import DataLoader


assert torch.__version__.split('.')[1] == '4'


parser = argparse.ArgumentParser(description='JDX: training script for training a RetinaNet network')
#parser.add_argument('--train-lmdb-path', type=str, default='/home/boby/Desktop/pengcheng_work_note/note/train_test/train_lmdb')
parser.add_argument('--train-lmdb-path', type=str, default='/home/boby/Desktop/pengcheng_work_note/note/train_test/train_lmdb')
parser.add_argument('--test-lmdb-path', type=str, default='/home/boby/Desktop/pengcheng_work_note/note/train_test/test_lmdb')

parser.add_argument('--model-depth', type=int, default=50)
parser.add_argument('--epochs', type=int, default=100)

args = parser.parse_args()


print("CUDA avalibale: {}".format(torch.cuda.is_available()))

def main(args = None):


    train_lmdb = './dataset/train_lmdb'
    test_lmdb = './dataset/test_lmdb'


    dataset_train = JDXTopViewPerson(lmdb_path=train_lmdb, transform=transforms.Compose([NormalizerJDX(), flip_xJDX(), ResizerJDX()]))
    dataset_test = JDXTopViewPerson(lmdb_path=test_lmdb, transform=transforms.Compose([NormalizerJDX(), ResizerJDX()]))


    # train_dataset
    sampler = AspectRatioBasedSampler(dataset_train, batch_size=3, drop_last=False)
    dataloader_train = DataLoader(dataset_train, num_workers=3, collate_fn=collater, batch_sampler=sampler)

    # test_dataset
    sampler_test = AspectRatioBasedSampler(dataset_test, batch_size=1, drop_last=False)
    dataloader_test = DataLoader(dataset_test, num_workers=3, collate_fn=collater, batch_sampler=sampler_test)

    retinanet = model.resnet50(num_classes=dataset_train.num_class(), pretrained=True)

    if torch.cuda.is_available():
        retinanet.cuda()
        retinanet = torch.nn.DataParallel(retinanet).cuda()
    retinanet.training = True

    optimizer = optim.Adam(retinanet.parameters(), lr=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)
    loss_hist = collections.deque(maxlen=500)

    retinanet.train()

    print('Num training images: {}'.format(len(dataset_train)))

    for epoch in range(5):

        retinanet.train()

        epoch_loss = []

        for iter_idx, data in enumerate(dataloader_train):
            try:
                optimizer.zero_grad()
                classification_loss, regression_loss = retinanet([data['img'].cuda().float(), data['annot']])

                classification_loss = classification_loss.mean()
                regression_loss = regression_loss.mean()

                loss = classification_loss + regression_loss
                if bool( 0 == loss):
                    continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(retinanet.parameters(), 0.1)

                optimizer.step()
                loss_hist.append(float(loss))

                epoch_loss.append(float(loss))

                print(
                    'Epoch: {} | Iteration: {} | Classification loss: {:1.5f} | Regression loss: {:1.5f} | Running loss: {:1.5f}'.format(
                        epoch, iter_idx, float(classification_loss), float(regression_loss), np.mean(loss_hist)))
                del classification_loss
                del regression_loss
            except Exception as e:
                print("PCH ========== exception")
                #print(e)
                continue
            scheduler.step(np.mean(epoch_loss))
            torch.save(retinanet.module, '{}_retinanet_{}.pt'.format('./', epoch))
    retinanet.eval()

    torch.save(retinanet, 'model_final.pt'.format(epoch))




if __name__ == '__main__':
    main()