from torchvision import models
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import torch
from utils.data_set import NishikaDataset, ImageTransform, make_datapath_list
import os
import matplotlib.pyplot as plt

torch.manual_seed(44)

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

device = 'cuda' if torch.cuda.is_available() else 'cpu'
net = models.resnet101(pretrained=False, num_classes=8)

net = net.to(device)

torch.backends.cudnn.benchmark = True
"""use GPU in parallel"""
if device == 'cuda':
    net = torch.nn.DataParallel(net) # make parallel

print('setting network is done: loading weight and set train mode.')

criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(params=net.parameters(), lr=0.001, momentum=0.9)
#optimizer = optim.Adam(net.parameters(), lr=0.0001, betas=(0.9, 0.99))

def train_model(net, dataloaders_dict, criterion, optimizer, num_epochs):

    train_loss = []
    train_accuracy = []
    val_loss = []
    val_accuracy = []

    losses = {'train':train_loss, 'val':val_loss}
    accuracies = {'train':train_accuracy, 'val':val_accuracy}

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-------------')

        for phase in ['train', 'val']:
            if phase == 'train':
                net.train()
            else:
                net.eval()

            epoch_loss = 0.0
            epoch_corrects = 0

            if(epoch == 0) and (phase == 'train'):
                continue

            for inputs, labels in tqdm(dataloaders_dict[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1) # ラベルを予測

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    #total of loss
                    epoch_loss += loss.item() * inputs.size(0)

                    epoch_corrects += torch.sum(preds == labels.data)

            epoch_loss = epoch_loss / len(dataloaders_dict[phase].dataset)
            epoch_acc = epoch_corrects.double()/len(dataloaders_dict[phase].dataset)
            losses[phase].append(epoch_loss)
            accuracies[phase].append(epoch_acc)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            plt.plot(losses['train'], label='training loss')
            plt.plot(losses['val'], label='validation loss')
            plt.plot(accuracies['train'], label='train accuracy')
            plt.plot(accuracies['val'], label='val accuracy')
            plt.savefig('./result/logs.png')

            if((epoch+1)%10 == 0):
                torch.save(net.state_dict(), 'checkpoints/resnet101_'+str(epoch+1)+'.pth')


def main():
    size = 224
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    train_list, val_list = make_datapath_list(phase="train", rate=0.9)
    #val_list = make_datapath_list(phase="val")
    csv_path = './data/train.csv'

    train_dataset = NishikaDataset(file_list=train_list, transform=ImageTransform(size, mean, std), phase='train', csv_path=csv_path)
    val_dataset = NishikaDataset(file_list=val_list, transform=ImageTransform(size, mean, std), phase='val', csv_path=csv_path)

    batch_size = 32

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    dataloaders_dict = {"train": train_dataloader, "val": val_dataloader}
    num_epochs=300
    train_model(net, dataloaders_dict, criterion, optimizer, num_epochs=num_epochs)

if __name__ == "__main__":
    main()
