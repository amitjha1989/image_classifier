# data loading

def load_data(data_dir):
        
    import matplotlib.pyplot as plt 
    import os 
    import numpy as np
    
    import torch 
    from torch import nn, optim 
    import torch.nn.functional as F 
    from torchvision import datasets, transforms, models   
    import json
    
    
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'    
    
    # TODO: Define your transforms for the training, validation, and testing sets
    data_transforms = {
        'training' : transforms.Compose([transforms.RandomResizedCrop(224),
                                         transforms.RandomRotation(35),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], 
                                                             [0.229, 0.224, 0.225])]),

        'validation' : transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], 
                                                               [0.229, 0.224, 0.225])]),

        'testing' : transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], 
                                                               [0.229, 0.224, 0.225])])
    }
    #train_transforms = transforms.Compose([transforms.RandomRotation(30), 
    #                                       transforms.RandomResizedCrop(224), 
    #                                       transforms.RandomHorizontalFlip(), 
    #                                       transforms.ToTensor(), 
    #                                       transforms.Normalize([0.485, 0.456, 0.406], 
    #                                                            [0.229, 0.224, 0.225])
    #                                      ])
    
    
    #valid_transforms = transforms.Compose([transforms.Resize(255), 
    #                                       transforms.CenterCrop(224), 
    #                                       transforms.ToTensor(), 
    #                                       transforms.Normalize([0.485, 0.456, 0.406], 
    #                                                            [0.229, 0.224, 0.225])])


    #test_transforms = transforms.Compose([transforms.Resize(255), 
    #                                      transforms.CenterCrop(224), 
    #                                      transforms.ToTensor(), 
    #                                      transforms.Normalize([0.485, 0.456, 0.406], 
    #                                                           [0.229, 0.224, 0.225])])

    #train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    #valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    #test_data  = datasets.ImageFolder(test_dir, transform=test_transforms)

    # TODO: Load the datasets with ImageFolder
    image_datasets = {
        'training' : datasets.ImageFolder(train_dir, transform=data_transforms['training']),
        'testing' : datasets.ImageFolder(test_dir, transform=data_transforms['testing']),
        'validation' : datasets.ImageFolder(valid_dir, transform=data_transforms['validation'])
    }
    #train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    #valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    #test_data  = datasets.ImageFolder(test_dir, transform=test_transforms)

    # TODO: Using the image datasets and the trainforms, define the dataloaders

    from selected_sample import ImbalancedDataset

    #batch_size = 64

    #trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, selected_sample=ImbalancedDataset(train_data) )

    #validloader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size)

    #testloader  = torch.utils.data.DataLoader(test_data, batch_size=batch_size)
     
    # TODO: Using the image datasets and the trainforms, define the dataloaders
    dataloaders = {
        'training' : torch.utils.data.DataLoader(image_datasets['training'], batch_size=64, sampler=ImbalancedDataset(image_datasets['training'])),
        'testing' : torch.utils.data.DataLoader(image_datasets['testing'], batch_size=32, shuffle=False),
        'validation' : torch.utils.data.DataLoader(image_datasets['validation'], batch_size=32, shuffle=True)
    }

    class_to_idx = image_datasets['training'].class_to_idx

    
    return dataloaders['training'], dataloaders['validation'], dataloaders['testing'], class_to_idx 
    #, trainloader, validloader, testloader
