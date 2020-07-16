import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms



from tinydfa import DFA, DFALayer, FeedbackPointsHandling

# Convoulutional neural network
class CIFAR10Convoulutional(nn.Module):
    def __init__(self, training_method='DFA'):
        super(CIFAR10Convoulutional, self).__init__()
        self.conv1 = nn.Conv2d(3,96,3)
        self.pool1 = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(96,128,3)
        self.pool2 = nn.MaxPool2d(2,2)
        self.conv3 = nn.Conv2d(128,256,3)
        self.pool3 = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(1024,2048)
        self.fc2 = nn.Linear(2048,2048)
        self.fc3 = nn.Linear(2048,10)
        

        self.training_method = training_method
        if self.training_method in ['DFA', 'SHALLOW']:
            self.dfa1, self.dfa2,self.dfa3, self.dfa4 ,self.dfa5  = DFALayer(), DFALayer(),DFALayer(),DFALayer(),DFALayer()
            self.dfa = DFA([self.dfa1, self.dfa2, self.dfa3, self.dfa4, self.dfa5], feedback_points_handling=FeedbackPointsHandling.LAST,
                           no_training=(self.training_method == 'SHALLOW'))

    def forward(self, x):
        
        if self.training_method in ['DFA', 'SHALLOW']:
            x = self.dfa1(self.pool1(torch.relu(self.conv1(x))))
            x = self.dfa2(self.pool2(torch.relu(self.conv2(x))))
            x = self.dfa3(self.pool3(torch.relu(self.conv3(x))))
            x = x.reshape(x.shape[0], -1)
            x = self.dfa4(torch.relu(self.fc1(x)))
            x = self.dfa5(torch.relu(self.fc2(x)))
            x = self.dfa(self.fc3(x))
        else:
            x = self.pool1(F.relu(self.conv1(x)))
            x = self.pool2(F.relu(self.conv2(x)))
            x = self.pool3(F.relu(self.conv3(x)))
            x = x.reshape(x.shape[0], -1)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
        return x
    
    


def train(args, train_loader, model, optimizer, device, epoch):
    model.train()
    for b, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if b%20==0:
            print(f"Training loss at batch {b}: {loss.item():.4f}\n", end='\r')
        
        


def test(args, test_loader, model, device, epoch):
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for b, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print(f"Epoch {epoch}: test loss {test_loss:.4f}, accuracy {correct / len(test_loader.dataset) * 100:.2f}.")



def main(args):
    use_gpu = not args.no_gpu and torch.cuda.is_available()
    device = torch.device(f"cuda:{args.gpu_id}" if use_gpu else "cpu")
    torch.manual_seed(args.seed)

    gpu_args = {'num_workers': 12, 'pin_memory': True} if use_gpu else {}
    CIFAR10_transform = transforms.Compose([transforms.ToTensor(),
                                                    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])
    train_loader = torch.utils.data.DataLoader(torchvision.datasets.CIFAR10(args.dataset_path, train=True, download=True,
                                                                          transform=CIFAR10_transform),
                                               batch_size=args.batch_size, shuffle=True, **gpu_args)
    test_loader = torch.utils.data.DataLoader(torchvision.datasets.CIFAR10(args.dataset_path, train=False,
                                                                         transform=CIFAR10_transform),
                                              batch_size=args.test_batch_size, shuffle=True, **gpu_args)

    model = CIFAR10Convoulutional( training_method=args.training_method).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum)

    for epoch in range(1, args.epochs + 1):
        train(args, train_loader, model, optimizer, device, epoch)
        test(args, test_loader, model, device, epoch)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tiny-DFA CIFAR10 Example')
    parser.add_argument('-t', '--training-method', type=str, choices=['BP', 'DFA', 'SHALLOW'], default='DFA',
                        metavar='T', help='training method to use, choose from backpropagation (BP), direct feedback '
                                          'alignment (DFA), or only topmost layer (SHALLOW) (default: DFA)')

    parser.add_argument('-b', '--batch-size', type=int, default=128, metavar='B',
                        help='training batch size (default: 128)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='B',
                        help='testing batch size (default: 1000)')

    parser.add_argument('-e', '--epochs', type=int, default=20, metavar='E',
                        help='number of epochs to train (default: 15)')
    parser.add_argument('-lr', '--learning-rate', type=float, default=0.01, metavar='LR',
                        help='SGD learning rate (default: 0.01)')
    parser.add_argument('-m', '--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')

    parser.add_argument('--no-gpu', action='store_true', default=False,
                        help='disables GPU training')
    parser.add_argument('-g', '--gpu-id', type=int, default=0, metavar='i',
                        help='id of the gpu to use (default: 0)')
    parser.add_argument('-s', '--seed', type=int, default=0, metavar='S',
                        help='random seed (default: 0)')
    parser.add_argument('-p', '--dataset-path', type=str, default='./data', metavar='P',
                        help='path to dataset (default: /data)')
    args = parser.parse_args()

    main(args)

