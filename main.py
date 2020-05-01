from torchvision import models
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import torch
from utils.data_set import NishikaDataset, ImageTransform, make_datapath_list
import os
from sklearn.metrics import f1_score
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
    train_f1 = []
    val_loss = []
    val_accuracy = []
    val_f1 = []

    losses = {'train':train_loss, 'val':val_loss}
    accuracies = {'train':train_accuracy, 'val':val_accuracy}
    f1s = {'train':train_f1, 'val':val_f1}

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-------------')

        for phase in ['train', 'val']:
            if phase == 'train':
                net.train()
            else:
                net.eval()

            f1score = 0.0
            epoch_loss = 0.0
            epoch_corrects = 0
            y_pred = []
            y_true = []

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
                    y_pred.extend(preds.tolist())
                    y_true.extend(labels.data.tolist())
            f1score = f1_score(y_true, y_pred, average='micro')
            epoch_loss = epoch_loss / len(dataloaders_dict[phase].dataset)
            epoch_acc = epoch_corrects.double()/len(dataloaders_dict[phase].dataset)
            losses[phase].append(epoch_loss)
            accuracies[phase].append(epoch_acc.tolist())
            f1s[phase].append(f1score)

            print('{} Loss: {:.4f} Acc: {:.4f} F1: {:.4f}'.format(phase, epoch_loss, epoch_acc, f1score))
            print(losses[phase])
            print(accuracies[phase])
            print(f1s[phase])
            plot_log('loss', losses)
            plot_log('acc', accuracies)
            plot_log('f1', f1s)

            if((epoch+1)%10 == 0):
                torch.save(net.state_dict(), 'checkpoints/resnet101_'+str(epoch+1)+'.pth')

def plot_log(types, data):
    plt.cla()
    plt.plot(data['train'], label='training '+types)
    plt.plot(data['val'], label='validation '+types)
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel(types)
    plt.title(types)
    plt.savefig('./result/'+ types +'.png')

def main():
    size = 256
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    train_list, val_list = make_datapath_list(phase="train", rate=0.9)
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
