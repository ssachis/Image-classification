import argparse
import torch
from torch import nn
from collections import OrderedDict
from torch import optim
import json
from torchvision import datasets, transforms, models
import numpy as np
from PIL import Image
import pandas as pd

parser = argparse.ArgumentParser (description = "parser of train.py")
parser.add_argument ('--save_dir', help = 'saving directory.')
parser.add_argument ('--arch', help = 'Resnet50 or vgg16',type = str,default='vgg16')
parser.add_argument ('--lrate', help = 'Learning rate',  default=0.001)
parser.add_argument ('--epochs', help = 'Number of epochs', type = int)
parser.add_argument ('--GPU', help = "gpu", type = str, default="gpu")
args = parser.parse_args ()

if args.GPU == 'gpu' and torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

data_dir = 'ImageClassifier/flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224,224)), 
        transforms.RandomRotation(15),
        transforms.RandomVerticalFlip(),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean,std)
]),

'test':transforms.Compose([
        transforms.Resize((224,224)),   
        transforms.ToTensor(),
        transforms.Normalize(mean,std)
    ]),
    
'valid':transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean,std)
    ]),
}
dataset = {
    'train': datasets.ImageFolder(train_dir, transform=data_transforms['train']),
    'valid': datasets.ImageFolder(valid_dir, transform=data_transforms['valid']),
    'test': datasets.ImageFolder(test_dir, transform=data_transforms['test'])
}

train = torch.utils.data.DataLoader(dataset['train'], batch_size=32, shuffle=True)
val = torch.utils.data.DataLoader(dataset['valid'], batch_size =32,shuffle = True)
test = torch.utils.data.DataLoader(dataset['test'], batch_size = 32, shuffle = True)

dataloaders={
    'train':train,
    'test':test,
    'valid':val,
}

with open('ImageClassifier/cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
    
#model
def model(structure):
    if structure == 'vgg16':
        model = models.vgg16(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
            
        model.classifier = nn.Sequential(OrderedDict([

            ('inputs', nn.Linear(25088, 256)),
            ('relu1', nn.ReLU()),
            ('dropout',nn.Dropout(0.5)),
            ('hidden_layer1', nn.Linear(256, 128)), 
            ('relu2',nn.ReLU()),      
            ('hidden_layer2',nn.Linear(128,102)),
           ])).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.classifier.parameters(), args.lrate)
    elif (structure == 'resnet50'):
        model = models.resnet50(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False


        model.classifier = nn.Sequential(OrderedDict([
            ('input', nn.Linear(2048,256)),
            ('relu', nn.ReLU()),
            ('dropout', nn.Dropout(0.2)),
            ('hidden2', nn.Linear(256, 128)),
            ('relu', nn.ReLU()),
            ('dropout', nn.Dropout(0.2)),
            ('hidden3', nn.Linear(128, 102)),

            ])).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.classifier.parameters(), args.lrate)
    return model,criterion,optimizer

model,criterion,optimizer=model(args.arch)
#training

model.to(device)
def train_model(model, criterion, optimizer, num_epochs=3):
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)

        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                _, preds = torch.max(outputs, 1)
                running_loss += loss.detach() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataset[phase])
            epoch_acc = running_corrects.float() / len(dataset[phase])

            print('{} loss: {:.4f}, acc: {:.4f}'.format(phase,
                                                        epoch_loss.item(),
                                                        epoch_acc.item()))
    return model
if args.epochs:
    epochs = args.epochs
else:
    epochs = 10
    
   
model = train_model(model, criterion, optimizer, epochs)

#saving

model.class_to_idx = dataset['train'].class_to_idx


checkpoint = {'input_size': (3, 224, 224),
              'output_size': 102,
              'learning_rate':args.lrate ,
              'model_name': args.arch,
              'classifier': model.classifier,
              'state_dict': model.state_dict (),
             
              'class_to_idx':model.class_to_idx
             }
if args.save_dir:
    torch.save (checkpoint, args.save_dir )
else:
    torch.save (checkpoint, 'checkpoint.pth')

#loading
def loading_model (file_path):
    checkpoint = torch.load (file_path)
    if (args.arch=='resnet50'):
        model = models.resnet50 (pretrained = True)
    else:
        model = models.vgg16 (pretrained = True)
    model.classifier = checkpoint ['classifier']
    model.load_state_dict (checkpoint ['state_dict'])
    model.class_to_idx = checkpoint ['class_to_idx']

    for param in model.parameters():
        param.requires_grad = False 

    return model

model_1=loading_model('checkpoint.pth')


#testing
def check_accuracy_on_test(model):
    model.to(device)
    correct,total = 0,0
    with torch.no_grad():
        model.eval()
        for data in test:
       
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy on test images is: %d%%' % (100 * correct / total))

check_accuracy_on_test(model)
check_accuracy_on_test(model_1)

