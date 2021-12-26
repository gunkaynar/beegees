
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

data_dir = '/mnt/gunkaynar/project/beegees/data/train'
gpu_id = 6
if torch.cuda.is_available():
    device = torch.device(f"cuda:{gpu_id}")
print("We're using =>", device)
model=torch.load('beemodel.pth')
model.eval()
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

nb_classes = 2

confusion_matrix = torch.zeros(nb_classes, nb_classes)
accuracy = 0

with torch.no_grad():
    for inputs, labels in testloader:
        inputs, labels = inputs.to(device),labels.to(device)
        outputs = model(inputs)
        logps = model.forward(inputs)
        ps = torch.exp(logps)
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
        _, preds = torch.max(outputs, 1)
        for t, p in zip(labels.view(-1), preds.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1

print(confusion_matrix)
print(f"Test accuracy: {accuracy/len(testloader):.3f}")

