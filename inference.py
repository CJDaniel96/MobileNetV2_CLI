import argparse
from glob import glob
import os
import shutil
import time
import torch
from torchvision import transforms
from PIL import Image


def get_classes(classes):
    class_list = []
    with open(classes, 'r') as f:
        for class_name in f.readlines():
            class_list.append(class_name.strip())

    return class_list

def inference(model, image_path, class_list, confidence, device):
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.ToTensor(), 
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    image = Image.open(image_path).convert("RGB")
    input_tensor = data_transforms(image)
    input_batch = input_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_batch)

    prediction = torch.nn.functional.softmax(output[0], dim=0)
    predicted_score = prediction.amax().item()
    predicted_class = prediction.argmax().item()

    if predicted_score > confidence:
        class_name = class_list[predicted_class]
    elif 'NG' in class_list:
        class_name = 'NG'
    else:
        class_name = 'other'

    return class_name

def save_image_result(save_image_folder, class_name, image_path):
    save_path = os.path.join(save_image_folder, class_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    shutil.copyfile(image_path, os.path.join(save_path, os.path.basename(image_path)))

def parse_opt():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data', type=str, default='')
    parser.add_argument('--weights', type=str, default='', help='model.pt path(s)')
    parser.add_argument('--confidence', type=float, default=0.0, help='give a confidence threshold to control inference result')
    parser.add_argument('--classes-file', type=str, default='', help='e.x. /path/to/classes.txt')
    parser.add_argument('--classes', type=str, default='', help='e.x. ["NG","OK"]')
    parser.add_argument('--save-image-folder', type=str, default='', help='save inference image to folder by result')

    opt = parser.parse_args()

    return opt

def main(opt):
    counter = 0

    if opt.classes_file:
        class_list = get_classes(opt.classes_file)
    elif opt.classes:
        class_list = eval(opt.classes)
    else:
        raise Exception('Please provides classes list or classes file')
    
    if os.path.isdir(opt.data):
        data = os.path.join(opt.data, '**', '*.jpg')
    elif os.path.isfile(opt.data):
        data = opt.data

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.load(opt.weights, map_location=device)

    total_since = time.time()
    data_list = glob(data, recursive=True)
    data_amount = len(data_list)

    for image_path in data_list:
        since = time.time()

        class_name = inference(model, image_path, class_list, opt.confidence, device)

        counter += 1
        print(f'{counter}/{data_amount}')
        print(f'Image Path: {image_path}')
        print(f'Prediction: {class_name}')
        print(f'inference time: {(time.time() - since):.2f}s')
        print()
        if opt.save_image_folder:
            save_image_result(opt.save_image_folder, class_name, image_path)

    time_elapsed = time.time() - total_since
    print(f'Inference Finish, Average Inference time: {(time_elapsed // 60):.0f}m {(time_elapsed % 60):.0f}s')


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)