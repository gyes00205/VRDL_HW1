import os
import torch
import torchvision
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image
import argparse
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision import models
import gc


def select_model(model_name: str):
    """Select model to predict images

    Parameters:
    -----------
    model_name: str
        Include resnext101, efficientnet b2~b4, resnet50, and regnet.

    Returns:
    -----------
    model:
        Use to predict images.
    """
    if model_name == 'resnext101':
        LOAD_MODEL_PATH = 'model/resnext101_batch4_epoch100_best.pth'
        torch.hub._validate_not_a_forked_repo = lambda a, b, c: True
        model = torch.hub.load(
            'pytorch/vision:v0.10.0',
            'resnext101_32x8d',
            pretrained=False
        )
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, 200)
        model.load_state_dict(torch.load(LOAD_MODEL_PATH))
        model = model.to(device)

    elif model_name == 'efficientnet_b4':
        LOAD_MODEL_PATH = 'model/efficient_b4_batch4_epoch100.pth'
        model = models.efficientnet_b4(pretrained=False)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(in_features, 200)
        )
        model.load_state_dict(torch.load(LOAD_MODEL_PATH))
        model = model.to(device)

    elif model_name == 'resnet50':
        LOAD_MODEL_PATH = 'model/resnet50_batch4_epoch100_best.pth'
        model = models.resnet50(pretrained=False)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, 200)
        model.load_state_dict(torch.load(LOAD_MODEL_PATH))
        model = model.to(device)

    elif model_name == 'efficientnet_b3':
        LOAD_MODEL_PATH = 'model/efficient_b3_batch4_epoch100.pth'
        model = models.efficientnet_b3(pretrained=False)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(in_features, 200)
        )
        model.load_state_dict(torch.load(LOAD_MODEL_PATH))
        model = model.to(device)

    elif model_name == 'efficientnet_b2':
        LOAD_MODEL_PATH = 'model/efficient_b2_batch4_epoch100.pth'
        model = models.efficientnet_b2(pretrained=False)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(in_features, 200)
        )
        model.load_state_dict(torch.load(LOAD_MODEL_PATH))
        model = model.to(device)

    elif model_name == 'regnet':
        LOAD_MODEL_PATH = 'model/regnet_x_8gf_batch4_epoch100_best.pth'
        model = models.regnet_x_8gf(pretrained=False)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, 200)
        model.load_state_dict(torch.load(LOAD_MODEL_PATH))
        model = model.to(device)

    return model


def parse_config():
    """Define parse config

    model: which model you would select
    ensemble: use ensemble or not

    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="resnext101", type=str)
    parser.add_argument("--ensemble", default=0, type=int)
    return parser.parse_args()


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    test_transformer = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])

    args = parse_config()
    f = open('classes.txt', 'r')
    class_list = []
    for i in f.readlines():
        class_name = i.strip('\n')
        class_list.append(class_name)
    f.close()
    print(len(class_list))

    with open('testing_img_order.txt') as f:
        test_images = [x.strip() for x in f.readlines()]
    if args.ensemble:
        model1 = select_model('efficientnet_b4')
        model1.eval()
        model2 = select_model('resnext101')
        model2.eval()
        model3 = select_model('resnet50')
        model3.eval()
        model4 = select_model('efficientnet_b3')
        model4.eval()
        model5 = select_model('efficientnet_b2')
        model5.eval()
        model6 = select_model('regnet')
        model6.eval()
    else:
        model = select_model(args.model)
        model.eval()

    submission = []
    outputs_list = []
    # image order is important to your result
    with torch.no_grad():
        for j, filename in enumerate(test_images):
            img = Image.open(os.path.join('test', filename)).convert('RGB')
            img_tensor = test_transformer(img)
            img_tensor = img_tensor.unsqueeze(0)
            img_tensor = img_tensor.to(device)
            if args.ensemble:
                outputs = model1(img_tensor) + model2(img_tensor) \
                        + model3(img_tensor) + model4(img_tensor) \
                        + model5(img_tensor) + model6(img_tensor)
            else:
                outputs = model(img_tensor)
            _, predicted = torch.max(outputs.data, 1)
            print(f'{filename}: {class_list[predicted]}')
            submission.append([filename, class_list[predicted]])
            del img, img_tensor, outputs, predicted
            gc.collect()
    np.savetxt('answer.txt', submission, fmt='%s')
