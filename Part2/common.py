import matplotlib.pyplot as plt 
import os 
import numpy as np
    
import torch 
from torch import nn, optim
import torch.nn.functional as F 
from torchvision import datasets, transforms, models   
import json
from PIL import Image
from utils_predict import process_image 


def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    
    arch = checkpoint['arch']
    hidden_units = checkpoint['hidden_units']
    model = build_model(arch, hidden_units)
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model 


def predict(image_path, model, topk=5):
    
    im = Image.open(image_path)
    im = process_image(im)
    im = torch.from_numpy(im)
    
    im.unsqueeze_(0)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    im = im.to(device, dtype=torch.float)
    model.to(device); 
    
    ps = torch.exp(model(im))    
    top_p, top_class = ps.topk(topk, dim=1)
    
    top_p = top_p.cpu().detach().numpy()
    top_p = top_p.flatten()    
    top_p = top_p.tolist()
    
    top_class = top_class.cpu().detach().numpy()    
    top_class = top_class.flatten()
    top_class = top_class.tolist()
    
    return top_p, top_class 


# TODO: Build your network
def build_model(arch, hidden_units):
        
    # The output of the classifier, this value depends on the problem.
    output_classifier = 102 
    
    if arch == "densenet121":
        model = models.densenet121(pretrained=True)
    elif arch == "resnet18":
        model = models.resnet18(pretrained=True)
    elif arch == "alexnet": 
        model = models.alexnet(pretrained=True)
    elif arch == "vgg16":
        model = models.vgg16(pretrained=True) 
    elif arch == "densenet161":
        model = models.densenet161(pretrained=True) 
    elif arch == "inception_v3": 
        model = models.inception_v3(pretrained=True) 
    else:
        model = models.vgg19(pretrained=True)
                   
    # Freeze parameters 
    for param in model.parameters():
        param.requires_grad = False

    from collections import OrderedDict 

    counter = 0 
    input_classifer = 0 

    for each in model.classifier.parameters():
        if counter == 1:
            break
        
        input_classifier = each.shape[1]
        counter += 1
    

    classifier = nn.Sequential(OrderedDict([
                                           ('fc1', nn.Linear(input_classifier, hidden_units)) , 
                                           ('relu1', nn.ReLU()), 
                                           ('dropout1', nn.Dropout(0.4)), 
                                           ('fc2', nn.Linear(hidden_units, output_classifier)),
                                           ('output', nn.LogSoftmax(dim=1))
                                           ]))

    model.classifier = classifier 
    
    return model 
        
 