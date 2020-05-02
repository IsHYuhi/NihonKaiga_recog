from torchvision import models
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import torch
from utils.data_set import NishikaDataset, ImageTransform, make_datapath_list
from models.model import EfficientNet
import os
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import pandas as pd

torch.manual_seed(44)

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

device = 'cuda' if torch.cuda.is_available() else 'cpu'
net = EfficientNet.from_name('efficientnet-b7')
#net = models.resnet101(pretrained=False, num_classes=8)
#net = models.resnet152(pretrained=False, num_classes=8)

load_path = './checkpoints/efficient+auto_90.pth'
load_weights = torch.load(load_path)
net.load_state_dict(load_weights)
net.eval()

net = net.to(device)

torch.backends.cudnn.benchmark = True
"""use GPU in parallel"""
if device == 'cuda':
    net = torch.nn.DataParallel(net) # make parallel

print('setting network is done: loading weight and set train mode.')

criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(params=net.parameters(), lr=0.001, momentum=0.9)
#optimizer = optim.Adam(net.parameters(), lr=0.0001, betas=(0.9, 0.99))

def predict(net, test_dataloader):
    print('使用デバイス: ', device)
    net.eval()

    preds = []

    for inputs, _ in tqdm(test_dataloader):
        inputs = inputs.to(device)

        with torch.no_grad():
            outputs = net(inputs)
            preds += [int(output.argmax()) for output in outputs]

    return preds

def main():
    size = 224
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    test_list = make_datapath_list(phase="test")
    csv_path = './data/sample_submission.csv'

    test_dataset = NishikaDataset(file_list=test_list, transform=ImageTransform(size, mean, std), phase='test', csv_path=csv_path)

    batch_size = 32

    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    test_df = pd.read_csv(csv_path)
    preds = predict(net, test_dataloader)
    test_df['gender_status'] = preds
    save_path = './result/submission.csv'
    test_df.to_csv(save_path, index=False)

if __name__ == "__main__":
    main()