import argparse
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torchvision import transforms, datasets, models
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from matplotlib import pyplot as plt


os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"

def train(mobilenet, vgg, epochs, batch_size, lr, momentum, step_size, gamma, save_dir, data_dir, cuda):
    if cuda:
        device = '0'
    else:
        device = 'cpu'
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
        print('Create ' + save_dir + ' Success!')

    data_transforms = {
        'train': transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor()
            ]
        )
    }

    image_datasets = {
        mode: datasets.ImageFolder(
            os.path.join(data_dir, mode),
            data_transforms[mode]
        ) for mode in ['train']
    }

    dataloaders = {
        mode: DataLoader(
            image_datasets[mode], 
            batch_size=batch_size, 
            shuffle=True
        ) for mode in ['train']
    }

    mean = 0.
    std = 0.
    nb_samples = 0.
    for data, _ in dataloaders['train']:
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        nb_samples += batch_samples
    mean /= nb_samples
    std /= nb_samples
    print('mean:', mean)
    print('std:', std)

    with open(os.path.join(data_dir, 'mean_std.txt'), 'w') as f:
        f.write('mean: ' + str(mean) + '\n')
        f.write('std: ' + str(std))

    data_transforms = {
        'train': transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=mean,
                    std=std
                )
            ]
        ),
        'val': transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=mean,
                    std=std
                )
            ]
        )
    }

    image_datasets = {
        mode: datasets.ImageFolder(
            os.path.join(data_dir, mode),
            data_transforms[mode]
        ) for mode in ['train', 'val']
    }

    with open(os.path.join(data_dir, 'class_to_idx.txt'), 'w') as f:
        for classes in image_datasets['train'].class_to_idx:
            f.write(classes + ' ' + str(image_datasets['train'].class_to_idx[classes]) + '\n')

    dataloaders = {
        mode: DataLoader(
            image_datasets[mode], 
            batch_size=batch_size, 
            shuffle=True
        ) for mode in ['train', 'val']
    }

    dataset_sizes = {
        mode: len(image_datasets[mode]) for mode in ['train', 'val']
    }
    print(dataset_sizes)
    class_names = image_datasets['train'].classes
    print(class_names)
    device = ('cpu' if len(device) > 1 else 'cuda:' + device) if torch.cuda.is_available() else 'cpu'
    print('Use', device)

    if mobilenet:
        model_save_path = os.path.join(save_dir, 'MobileNet_v2_epoch{}_{}_ACC{}_LOSS{}.pt')
        model = models.mobilenet_v2(pretrained=True)
        fc_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(fc_features, len(class_names))
    elif vgg:
        model_save_path = os.path.join(save_dir, 'VGG16_epoch{}_{}_ACC{}_LOSS{}.pt')
        model = models.vgg16_bn(pretrained=True, progress=True)
        fc_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(fc_features, len(class_names))
    else:
        raise ValueError('MobileNet value or VGG value is not exist')
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    since = time.time()
    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch, epochs - 1))
        print('-' * 10)
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            running_loss = 0.0
            running_corrects = 0
            for inputs, labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            if phase == 'train':
                exp_lr_scheduler.step()
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            print('//////////////////////////////////////////')
            print(phase)
            if phase == 'val':
                save_name = model_save_path.format(
                    epoch,
                    time.strftime('%m%d-%H%M'),
                    '{:.4f}'.format(epoch_acc), 
                    '{:.4f}'.format(epoch_loss)
                )
                print(save_name)
                torch.save(model, save_name)
        
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, 
            time_elapsed % 60
        )
    )

def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--mobilenet', action='store_true', default=True)
    parser.add_argument('--vgg', action='store_true', default=False)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--cuda', action='store_true', default=False)
    parser.add_argument('--data-dir', type=str)
    parser.add_argument('--save-dir', type=str)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--step-size', type=int, default=7)
    parser.add_argument('--gamma', type=float, default=0.1)

    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt

def main(opt):
    train(**vars(opt))


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)