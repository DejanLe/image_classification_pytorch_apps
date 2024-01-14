import torchvision
import torch 
import torchvision.transforms as transforms 
import os 
import matplotlib.pyplot as plt 
import numpy as np 

#print(os.listdir())

train_dataset_path = 'patch to your image example google drive'
test_dataset_path = 'patch to your image example google drive'

 
training_transforms = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])
train_dataset = torchvision.datasets.ImageFolder(root = train_dataset_path, transform = training_transforms)

train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size=32, shuffle=False)

def get_mean_and_std(loader):
    mean = 0.
    std = 0.
    total_images_count = 0
    for images, _ in loader:
        image_count_in_a_batch = images.size(0)
        #print(images.shape)
        images = images.view(image_count_in_a_batch, images.size(1), -1)
        #print(images.shape)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_images_count += image_count_in_a_batch

    mean /=total_images_count
    std /=total_images_count

    return mean, std 

get_mean_and_std(train_loader)
#print(get_mean_and_std(train_loader))
# cod radi

'''
(tensor([0.4928, 0.4805, 0.3984]), tensor([0.2043, 0.1919, 0.2064]))
'''
mean = [0.4928, 0.4805, 0.3984]
std = [0.2043, 0.1919, 0.2064]

train_transforms = transforms.Compose([
    transforms.Resize((244,244)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(torch.Tensor(mean), torch.Tensor(std))
    ])
test_transforms = transforms.Compose([
    transforms.Resize((244,244)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
     transforms.Normalize(torch.Tensor(mean), torch.Tensor(std))
    ])

train_dataset = torchvision.datasets.ImageFolder(root = train_dataset_path, transform = train_transforms)
test_dataset = torchvision.datasets.ImageFolder(root = test_dataset_path, transform = test_transforms)

#prikaze slike

def show_transformed_images(dataset):
    loader = torch.utils.data.DataLoader(dataset, batch_size=6, shuffle = True)
    batch = next(iter(loader))
    images, labels = batch
    grid = torchvision.utils.make_grid(images, nrow=3)
    plt.figure(figsize=(11,11))
    plt.imshow(np.transpose(grid, (1,2,0)))
    print('labels:', labels)

show_transformed_images(train_dataset)

# cod radi 




































