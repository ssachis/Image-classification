
import torch
import numpy as np
from torch import nn
from torch import optim
from torchvision import datasets, models, transforms
from collections import OrderedDict
import PIL
from PIL import Image
import argparse
import json

parser = argparse.ArgumentParser (description = "Parser of predict.py")
parser.add_argument ('--arch', help = 'Resnet50 or vgg16',type = str,default='vgg16')
parser.add_argument ('--image_dir', help = 'path to image', default="ImageClassifier/flowers/test/15/image_06351.jpg")
parser.add_argument ('--load_dir', help = 'path to checkpoint', default='vgg16_10epochs.pth')
parser.add_argument ('--top_k', help = 'Top K most likely classes', type = int)
parser.add_argument ('--category_names', help = 'Mapping of categories to real names.', type = str,default='ImageClassifier/cat_to_name.json')
parser.add_argument ('--GPU', help = "Option to use GPU", type = str)

mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])
args = parser.parse_args ()
if args.GPU == 'gpu' and torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'


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


def process_image(image):

    img = PIL.Image.open(image)
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    img= transform(img)
    
    return img
    
with open(args.category_names, 'r') as f:
    cat_to_name = json.load(f)
device="cpu"  

def predict(image_path, model, top_k=5):
    model.to("cpu") 
    model.eval();
    image = process_image(image_path) 
  
    probs = model.forward (image.unsqueeze(0))

    top_probs, top_labels = probs.topk(top_k)
    
    top_probs = top_probs.detach().numpy().tolist()[0]
    top_labels = top_labels.numpy().tolist()[0]
    
    idx_to_class = {model.class_to_idx[a]: a for a in model.class_to_idx}
    top_labels = [idx_to_class[a] for a in top_labels]
    top_flowers = [cat_to_name[a] for a in top_labels]
    
    return top_probs, top_labels,top_flowers

model = loading_model (args.load_dir)

file_path="ImageClassifier/flowers/test/17/image_03830.jpg"
top_probs, top_labels, top_flowers = predict (file_path, model)

for i in range (5):
    print(top_flowers[i] ,top_probs[i])
print("the predicted flower is",top_flowers[0],"with probability",str(top_probs[0])[:5],"%")

