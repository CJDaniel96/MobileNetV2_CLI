import argparse
import copy
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
from tqdm import tqdm


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

def select_data_transforms(mode='default', mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    if mode == 'default':
        return {
            'train': transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]),
            'val': transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        }
    else:
        return {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(224), # 資料增補
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]),
            'val': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]),
        }

def get_image_datasets(data_dir, data_transforms):
    return {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}

def get_dataloaders(image_datasets):
    return {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=64, shuffle=True, num_workers=4) for x in ['train', 'val']}

def get_dataset_sizes(image_datasets):
    return {x: len(image_datasets[x]) for x in ['train', 'val']}

def get_class_names(image_datasets):
    return image_datasets['train'].classes

def save_classes(data_dir, image_datasets):
    with open(os.path.join(data_dir, 'classes.txt'), 'w') as f:
        for classes in image_datasets['train'].classes:
            f.write(f'{classes}\n')

def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, num_epochs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Use Device: {device}')

    early_stopper = EarlyStopper(patience=3)

    model = model.to(device)

    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # 逐批訓練或驗證
            for inputs, labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # 訓練時需要梯度下降
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # 訓練時需要 backward + optimize
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # 統計損失
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # 如果是評估階段，且準確率創新高即存入 best_model_wts
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

        if phase == 'val' and early_stopper.early_stop(epoch_loss):
            print('Early Stop')
            break

    time_elapsed = time.time() - since
    print(f'Training complete in {(time_elapsed // 60):.0f}m {(time_elapsed % 60):.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # 載入最佳模型
    model.load_state_dict(best_model_wts)
    return model

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str)
    parser.add_argument('--save-dir', type=str)
    parser.add_argument('--num-epochs', type=int, default=40)
    opt = parser.parse_args()

    return opt

def main(opt):
    data_transforms = select_data_transforms()
    image_datasets = get_image_datasets(opt.data_dir, data_transforms)
    dataloaders = get_dataloaders(image_datasets)
    dataset_sizes = get_dataset_sizes(image_datasets)
    class_names = get_class_names(image_datasets)
    print(f'class names: {class_names}')

    model_ft = models.mobilenet_v2(pretrained=True)
    num_ftrs = model_ft.classifier[-1].in_features
    model_ft.classifier = nn.Sequential(
        nn.Dropout(p=0.2), 
        nn.Linear(num_ftrs, len(class_names))
    )
    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
    model_ft = train_model(model_ft, dataloaders, dataset_sizes, criterion, optimizer_ft, exp_lr_scheduler, opt.num_epochs)

    if not os.path.exists(opt.save_dir):
        os.makedirs(opt.save_dir)
    torch.save(model_ft, os.path.join(opt.save_dir, 'best.pt'))

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)