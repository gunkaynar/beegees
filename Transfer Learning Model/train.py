import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

data_dir = '/mnt/gunkaynar/project/beegees/data/train'

def load_split_train_test(datadir, valid_size = .2):
    train_transforms = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),])    
    validation_transforms = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),])    
    test_transforms = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),])    
    train_data = datasets.ImageFolder(datadir,transform=train_transforms)
    validation_data = datasets.ImageFolder(datadir,transform=validation_transforms)    
    test_data = datasets.ImageFolder(datadir,transform=test_transforms)    
    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))
    np.random.shuffle(indices)
    from torch.utils.data.sampler import SubsetRandomSampler
    train_idx, validation_idx,test_idx = indices[split:], indices[:round(split/2)], indices[round(split/2):split]
    train_sampler = SubsetRandomSampler(train_idx)
    validation_sampler = SubsetRandomSampler(validation_idx)
    test_sampler = SubsetRandomSampler(test_idx)
    trainloader = torch.utils.data.DataLoader(train_data, sampler=train_sampler, batch_size=64)
    validationloader = torch.utils.data.DataLoader(test_data, sampler=test_sampler, batch_size=64)
    testloader = torch.utils.data.DataLoader(test_data, sampler=test_sampler, batch_size=64)
    return trainloader, validationloader ,testloader
trainloader, validationloader,testloader = load_split_train_test(data_dir, .2)

print(trainloader.dataset.classes)
print(validationloader.dataset.classes)
print(testloader.dataset.classes)

gpu_id = int(input("GPU index: "))
if torch.cuda.is_available():
    device = torch.device(f"cuda:{gpu_id}")
    print(f"GPU {gpu_id} is available")
else:
    device = torch.device("cpu")
    print("GPU is not available")
print("We're using =>", device)
model = models.resnet50(pretrained=True)
print(model)

for param in model.parameters():
    param.requires_grad = False
    
model.fc = nn.Sequential(nn.Linear(2048, 512),nn.ReLU(),nn.Dropout(0.2),nn.Linear(512, 10),nn.LogSoftmax(dim=1))
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.003)
model.to(device)

epochs = 1
steps = 0
running_loss = 0
print_every = 10
train_losses, validation_losses = [], []
for epoch in range(epochs):
    for inputs, labels in trainloader:
        steps += 1
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        logps = model.forward(inputs)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
        if steps % print_every == 0:
            validation_loss = 0
            accuracy = 0
            model.eval()
            with torch.no_grad():
                for inputs, labels in validationloader:
                    inputs, labels = inputs.to(device),labels.to(device)
                    logps = model.forward(inputs)
                    batch_loss = criterion(logps, labels)
                    validation_loss += batch_loss.item()
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            train_losses.append(running_loss/len(trainloader))
            validation_losses.append(validation_loss/len(validationloader))                    
            print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Train loss: {running_loss/print_every:.3f}.. "
                  f"Validation loss: {validation_loss/len(validationloader):.3f}.. "
                  f"Validation accuracy: {accuracy/len(validationloader):.3f}")
            running_loss = 0
            model.train()
torch.save(model, 'beemodel.pth')


plt.plot(train_losses, label='Training loss')
plt.plot(validation_losses, label='Validation loss')
plt.legend(frameon=False)
plt.savefig("loss.png")

