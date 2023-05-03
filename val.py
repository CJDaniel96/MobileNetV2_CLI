import argparse
import glob
import os
import re
import torch
import cv2
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image


@torch.no_grad()
def run(data, cuda=False, weights='', batch_size=1, save_image_folder='', confidence=0.9):
    if cuda:
        device = 'cuda:0'
    else:
        device = 'cpu'
    model = torch.load(weights, map_location=device)

    class_names = []
    class_to_idx_dir_date = os.path.basename(os.path.dirname(weights))
    for dataset in os.listdir('./Datasets'):
        if class_to_idx_dir_date in dataset:
            class_to_idx = os.path.join('./Datasets', dataset, 'class_to_idx.txt')
            mean_std = os.path.join('./Datasets', dataset, 'mean_std.txt')
    with open(class_to_idx, 'r') as f:
        for class_name in f:
            class_names.append(class_name.split(' ')[0])
    with open(mean_std, 'r') as f:
        mean = eval(re.search('[[].*[]]', f.readline())[0])
        std = eval(re.search('[[].*[]]', f.readline())[0])

    '''
    Default:
        mean=[0.2682, 0.2322, 0.2276]
        std=[0.2368, 0.2380, 0.2188]
    '''
    data_transform = transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.ToTensor(), 
        transforms.Normalize(
            mean=mean, 
            std=std
        )
    ])

    for img in glob.glob(os.path.join(data, '**', '*.jpeg'), recursive=True):
        os.rename(img, os.path.splitext(img)[0] + '.jpg')
    images_list = glob.glob(os.path.join(data, '**', '*.jpg'), recursive=True)
    for idx, image_path in enumerate(images_list):
        print(image_path)
        print('now =', idx + 1, '/', len(images_list))
        img = Image.open(image_path).convert("RGB")
        image_tensor = data_transform(img)
        image_tensor = image_tensor.unsqueeze(0)
        image = DataLoader(
            image_tensor,
            batch_size=batch_size,
            shuffle=True
        )
        for inputs in image:
            inputs = inputs.to(device)
            output = model(inputs)
            output = F.softmax(output, dim=1)
            print('Score:', output)
            p, preds = torch.max(output, 1)
            if class_names[preds[0]] == 'OK':
                result = class_names[preds[0]] if float(p[0]) > confidence else os.path.basename(data)
            else:
                result = class_names[preds[0]]
            print('Predict:', result, 'value:', float(p[0]))
            print('AI predict =', result)

        image_1 = cv2.imread(image_path)
        image_1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2RGB)
        image_name_include_type = os.path.basename(image_path)

        if result in class_names:
            save_path = os.path.join(save_image_folder, result)
        else:
            save_path = os.path.join(save_image_folder, 'other')

        if save_image_folder != '':
            if not os.path.isdir(save_path):
                os.makedirs(save_path)
            cv2.imwrite(os.path.join(save_path, image_name_include_type), cv2.cvtColor(image_1, cv2.COLOR_RGB2BGR))

    print('Finish process')

def parse_opt():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data', type=str, default='')
    parser.add_argument('--weights', type=str, default='', help='model.pt path(s)')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    parser.add_argument('--confidence', type=float, default=0.9)
    parser.add_argument('--cuda', action='store_true', default=False)
    parser.add_argument('--save-image-folder', type=str, default='')

    opt = parser.parse_args()

    return opt

def main(opt):
    run(**vars(opt))
    

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)