import torch
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 4

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

import matplotlib.pyplot as plt
import numpy as np

# functions to show an image


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next() # 3, 32, 32

# # show images
# imshow(torchvision.utils.make_grid(images))
# # print labels
# print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))

import torch.nn as nn
import torch.nn.functional as F
from mixture_of_experts import MoE as newMoE


class Experts(nn.Module):
    def __init__(self, num_experts = 32):
        super().__init__()
        self.w1 = nn.Parameter(torch.randn(num_experts, 3072, 1024))
        self.w2 = nn.Parameter(torch.randn(num_experts, 1024, 256))
        self.act = nn.ReLU()

    def forward(self, x):
        hidden1 = self.act(torch.einsum('end,edh->enh', x, self.w1))
        out = torch.einsum('end,edh->enh', hidden1, self.w2)
        return self.act(out)

experts = Experts(num_experts = 32)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3072, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        return x, 0

# exp = 'normal' 
exp = 'moe'

if exp == 'normal':
    net = Net()
else:
    net = newMoE(dim = 3072, num_experts = 32, experts = experts)

print('exp: ', exp)
fc = nn.Linear(256, 10).cuda()      

net = net.cuda()

import torch.optim as optim

criterion = nn.CrossEntropyLoss().cuda()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(50):  # loop over the dataset multiple times

    running_loss = 0.0
    running_z = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs, labels = inputs.cuda(), labels.cuda()

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        inputs = inputs.view(inputs.shape[0], -1)
        outputs, z_loss = net(inputs)
        outputs = fc(outputs)
        loss = criterion(outputs, labels)

        # print statistics
        running_loss += loss.item()
        if torch.is_tensor(z_loss):
            running_z += z_loss.item()

        loss = loss + z_loss

        loss.backward()
        optimizer.step()

        if i % 2000 == 1999:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f} z_loss: {running_z / 2000:.3f}')
            running_loss = 0.0

print('Finished Training')

PATH = './cifar_net.pth'
torch.save(net.state_dict(), PATH)

