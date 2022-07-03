import pandas as pd
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
from torch.nn import functional as F
from torch import nn
import os

batch_size = 32
num_workers = 0
initial_lr = 0.01
train_epoch = 50


class ResNetBasicBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(ResNetBasicBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(out_channels)
        self.conv2 = torch.nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        return F.relu(x + out)


class ResNetDownBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(ResNetDownBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride[0], padding=1)
        self.bn1 = torch.nn.BatchNorm2d(out_channels)
        self.conv2 = torch.nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=stride[1], padding=1)
        self.bn2 = torch.nn.BatchNorm2d(out_channels)
        self.extra = torch.nn.Sequential(torch.nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=stride[0], padding=0), torch.nn.BatchNorm2d(out_channels))

    def forward(self, x):
        extra_x = self.extra(x)
        output = self.conv1(x)
        out = F.relu(self.bn1(output))

        out = self.conv2(out)
        out = self.bn2(out)
        return F.relu(extra_x + out)


class ResNet18(torch.nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = torch.nn.BatchNorm2d(64)
        self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.l1 = torch.nn.Sequential(ResNetBasicBlock(
            64, 64, 1), ResNetBasicBlock(64, 64, 1))
        self.l2 = torch.nn.Sequential(ResNetDownBlock(
            64, 128, [2, 1]), ResNetBasicBlock(128, 128, 1))
        self.l3 = torch.nn.Sequential(ResNetDownBlock(
            128, 256, [2, 1]), ResNetBasicBlock(256, 256, 1))
        self.l4 = torch.nn.Sequential(ResNetDownBlock(
            256, 512, [2, 1]), ResNetBasicBlock(512, 512, 1))
        self.avgpool = torch.nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = torch.nn.Linear(512, 4)

    def forward(self, x):
        out = self.conv1(x)
        out = self.maxpool(out)
        out = self.l1(out)
        out = self.l2(out)
        out = self.l3(out)
        out = self.l4(out)
        out = self.avgpool(out)
        out = out.reshape(x.shape[0], -1)
        out = self.fc(out)
        return out


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    trainedNum = 0
    correct = 0

    for inputs, target in dataloader:
        inputs, target = inputs.to(device), target.to(device)

        pred = model(inputs)
        loss = loss_fn(pred, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss = loss.item()
        trainedNum += len(inputs)
        correct += (pred.argmax(1) == target).type(torch.float).sum().item()
        print(f"loss: {loss:>7f}  [{trainedNum:>5d}/{size:>5d}]")

    print(
        f"Train Error: \n Accuracy: {(100 * correct / size):>0.1f}%, Avg loss: {loss:>8f} \n")
    trainAccList.append(100 * correct / size)


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0

    with torch.no_grad():
        for inputs, target in test_loader:
            inputs, target = inputs.to(device), target.to(device)

            pred = model(inputs)
            test_loss += loss_fn(pred, target).item()
            correct += (pred.argmax(1) ==
                        target).type(torch.float).sum().item()
    test_loss /= num_batches

    print(
        f"Test Error: \n Accuracy: {(100 * correct / size):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    testAccList.append(100 * correct / size)


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    trainAccList = []
    testAccList = []

    # 数据输入和转换
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4, 0.4, 0.4], std=[0.2, 0.2, 0.2]),
        transforms.Grayscale(num_output_channels=1)
    ])
    train_dataset = ImageFolder('train/', transform=transform)
    test_dataset = ImageFolder('test/', transform=transform)
    full_dataset = train_dataset + test_dataset

    # 数据集切分
    train_size = int(0.9 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(
        full_dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, shuffle=True,
                              num_workers=num_workers, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, shuffle=False,
                             num_workers=num_workers, batch_size=batch_size)
    print('transform done')

    # 模型
    model = ResNet18().to(device=device)
    # ! 加载历史保存模型
    # if os.path.isfile("model.pth"):
    #     model.load_state_dict(torch.load("model.pth"))

    # 损失函数与优化器
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), initial_lr, momentum=0.9)
    # optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.1)
    # ! 可变学习率
    scheduler_lr = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[3, 8, 18], gamma=0.2)

    for t in range(train_epoch):
        print(f"Epoch {t + 1}\n-------------------------------")
        train(train_loader, model, loss_fn, optimizer)
        test(test_loader, model, loss_fn)
        # ! 可变学习率
        scheduler_lr.step()

    # 保存模型
#     torch.save(model.state_dict(), "model.pth")
#     print("Saved PyTorch Model State to model.pth")

# 保存训练过程中各epoch的正确率
#     accList = {'trainAcc': trainAccList, 'testAcc': testAccList}
#     accList = pd.DataFrame(accList)
#     writer = pd.ExcelWriter('accList.xlsx')
#     accList.to_excel(writer, 'page', float_format='%.5f')
#     writer.save()
#     writer.close()
#     print("Saved Acc Lists")
