import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from themoe import MoE
from moe import cvMoE

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

class MLPP(nn.Module):
    def __init__(self, input_size=3072, output_size=768, hidden_size=768):
        super(MLPP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out



class MLP(nn.Module):
    def __init__(self, input_size=3072, output_size=64, hidden_size=64):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out, 0

h = 64
choose = 0 # MoE 0.014 50% E-10-k-4 E-32-k-10~13 0.019 49%
# choose  = 1 # Normal 0.08    50%
# choose =   2  #cvMoE 

print('h: ', h)
if choose == 0:
    net = MoE(input_size=3072, output_size=h, num_experts=10, hidden_size=h, noisy_gating=True, k=4)
    print('MoE!')
elif choose == 1:
    net = MLP(output_size=h, hidden_size=h)
    print('MLP!')
elif choose == 2:
    print('cvMoE!')
    ffd_exports = [
        MLPP()
        for _ in range(32)
    ]
    net = cvMoE(3072, ffd_exports, 2)

net = net.to(device)
fc = nn.Linear(h, 10).cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(list(net.parameters()) + list(fc.parameters()), lr=0.001, momentum=0.9)

net.train()
for epoch in range(100):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        inputs = inputs.view(inputs.shape[0], -1)
        if choose == 2:
            inputs = inputs.view(inputs.shape[0], 1, -1)
        outputs, aux_loss = net(inputs)
        outputs = fc(outputs).squeeze(1)
        loss = criterion(outputs, labels)
        total_loss = loss + aux_loss
        total_loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 100 == 99:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0

print('Finished Training')

correct = 0
total = 0
net.eval()
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        if choose == 2:
            outputs, _ = net(images.view(images.shape[0], 1, -1))
        else:
            outputs, _ = net(images.view(images.shape[0], -1))
        outputs = fc(outputs).squeeze(1)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))

# yields a test accuracy of around 34 %