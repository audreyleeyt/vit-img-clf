# Load the packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
import sys
import time
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
import torchvision.transforms as transforms
from torchvision import models
from efficientnet_pytorch import EfficientNet


from skimage.util import random_noise
import cv2
import yaml

import argparse

# Parse command line arguments for YAML config files
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, help='YAML config file')
args = parser.parse_args()

config_file = args.config

# Load configuration from yaml file
with open(config_file, "r") as file:
    config = yaml.safe_load(file)

# Get the available device
if torch.cuda.is_available():
    dev = "cuda"  # Gpu
    num_gpu = torch.cuda.device_count()
else:
    dev = "cpu"
device = torch.device(dev)
#print("Device: ", device)

# Self-defined Gaussian noise transformation
class Noise:
    def __init__(self, mean=0., var=0.05, mode='gaussian'):
        self.var = var
        self.mean = mean
        self.mode = mode

    def __call__(self, tensor):
        #return tensor + torch.randn(tensor.size()) * self.std + self.mean
        img_np = tensor.numpy()
        if self.mode == 'gaussian':
            noisy_img = random_noise(img_np, mode='gaussian', mean=self.mean, var=self.var, clip=True)
        elif self.mode == 'salt':
            noisy_img = random_noise(img_np, mode='s&p', amount=self.var)
        elif self.mode == 'speckle':
            noisy_img = random_noise(img_np, mode='speckle', var=self.var)
        else:
            raise ValueError(f"Unsupported noise mode: {self.mode}")
        return torch.from_numpy(noisy_img)
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, var={1}, mode={2})'.format(self.mean, self.var, self.mode)

# Data augmentation and normalization for training
def create_transform(transform = 'default', params: dict = None):
    if params is None:
        params = {}

    basic_transforms = [transforms.ToTensor(),
                        transforms.Resize((224, 224)),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    # Default transformation
    if transform.lower() == 'default':
        return transforms.Compose(basic_transforms)
    
    if transform.lower() == 'gaussian blur':
        basic_transforms.append(transforms.GaussianBlur(params.get('kernel_size', 3),
                                                        sigma=params.get('sigma', (0.5, 3.0))))

    # Additional transformation options could be added as necessary
    elif transform.lower() == 'downsample':
        basic_transforms.append(transforms.Resize(params.get('size', 56)))

    elif transform.lower() == 'horizontal flip':
        basic_transforms.append(transforms.RandomHorizontalFlip())

    elif transform.lower() == 'vertical flip':
        basic_transforms.append(transforms.RandomVerticalFlip())

    elif transform.lower() == 'random rotation':
        basic_transforms.append(transforms.RandomRotation(params.get('degrees', 30)))

    elif transform.lower() == 'random crop':
        basic_transforms.append(transforms.RandomCrop(params.get('size', 224),
                                                      pad_if_needed=params.get('pad_if_needed', True),
                                                      padding_mode=params.get('padding_mode', 'reflect')))

    elif transform.lower() == 'random affine':
        basic_transforms.append(transforms.RandomAffine(params.get('degrees', 0),
                                                        translate=params.get('translate', None),
                                                        scale=params.get('scale', None),
                                                        shear=params.get('shear', None),
                                                        resample=params.get('resample', False),
                                                        fillcolor=params.get('fillcolor', 0)))

    elif transform.lower() == 'color jitter':
        basic_transforms.append(transforms.ColorJitter(brightness=params.get('brightness', 0),
                                                       contrast=params.get('contrast', 0),
                                                       saturation=params.get('saturation', 0),
                                                       hue=params.get('hue', 0)))

    elif transform.lower() == 'auto augment':
        basic_transforms.append(transforms.AutoAugment(params.get('policy', 'original')))

    elif transform.lower() == 'random erasing':
        basic_transforms.append(transforms.RandomErasing(p=params.get('p', 0.5),
                                                         scale=params.get('scale', (0.02, 0.33)),
                                                         ratio=params.get('ratio', (0.3, 3.3)),
                                                         value=params.get('value', 'random'),
                                                         inplace=params.get('inplace', False)))

    elif transform.lower() == 'gaussian noise':
        basic_transforms.append(Noise(
            params.get('mean', 0.),
            params.get('var', 0.05),
            params.get('mode', 'gaussian')
        ))

    elif transform.lower() == 'salt and pepper noise':
        basic_transforms.append(Noise(
            params.get('mean', 0.),
            params.get('var', 0.05),
            params.get('mode', 'salt')
        ))

    elif transform.lower() == 'speckle noise':
        basic_transforms.append(Noise(
            params.get('mean', 0.),
            params.get('var', 0.05),
            params.get('mode', 'speckle')
        ))
    
    # Add more transformations here
    #########################################################
    # In case of an unsupported transformation option
    else:
        print(f"Unsupported transformation: {transform}")
        return None

    return transforms.Compose(basic_transforms)

# Example of how to use the function. Input the name of the transformation and the parameters as a dictionary
# transform = create_transform('gaussian blur', {'kernel size':5, 'sigma':(0.1, 2.0)}) or
# transform = create_transform('gaussian blur') to use the default parameters
# Check if the transform is correct
# print(transform)

# Use the default transform, without any additional augmentation
train_transform = create_transform(config['data']['augmentation']['train'])

# hard coded selected classes
classes = ['ABBOTTS BABBLER', 'AFRICAN EMERALD CUCKOO', 'AFRICAN OYSTER CATCHER', 'ALBATROSS', 'ALBERTS TOWHEE', 'AMERICAN AVOCET', 'AMERICAN BITTERN', 'AMERICAN KESTREL', 'AMERICAN PIPIT', 'AMERICAN REDSTART', 'AMETHYST WOODSTAR', 'ANDEAN LAPWING', 'ANHINGA', 'ANIANIAU', 'ANNAS HUMMINGBIRD', 'ANTBIRD', 'ASIAN CRESTED IBIS', 'ASIAN DOLLARD BIRD', 'ASIAN GREEN BEE EATER', 'ASIAN OPENBILL STORK', 'AUSTRAL CANASTERO', 'AUSTRALASIAN FIGBIRD', 'AZURE JAY', 'BALD EAGLE', 'BANDED PITA', 'BLACK COCKATO', 'BLACK FRANCOLIN', 'BLACK HEADED CAIQUE', 'BLACK NECKED STILT', 'BLACK SKIMMER', 'BLACK THROATED BUSHTIT', 'BLACK THROATED HUET', 'BLACK VENTED SHEARWATER', 'BLACK VULTURE', 'BLACK-CAPPED CHICKADEE', 'BLONDE CRESTED WOODPECKER', 'BLOOD PHEASANT', 'BLUE GROSBEAK', 'BLUE MALKOHA', 'BLUE THROATED TOUCANET', 'BOBOLINK', 'BROWN HEADED COWBIRD', 'CALIFORNIA CONDOR', 'CALIFORNIA QUAIL', 'CAMPO FLICKER', 'CANARY', 'CAPE GLOSSY STARLING', 'CASSOWARY', 'CHESTNET BELLIED EUPHONIA', 'CHINESE BAMBOO PARTRIDGE', 'CHUCAO TAPACULO', 'CINNAMON TEAL', 'COCKATOO', 'COLLARED ARACARI', 'COLLARED CRESCENTCHEST', 'COMMON GRACKLE', 'COMMON HOUSE MARTIN', 'COPPERY TAILED COUCAL', 'CRAB PLOVER', 'CRANE HAWK', 'CRESTED CARACARA', 'CRESTED SHRIKETIT', 'CRESTED WOOD PARTRIDGE', 'CRIMSON CHAT', 'CURL CRESTED ARACURI', 'DALMATIAN PELICAN', 'DARJEELING WOODPECKER', 'DOUBLE BRESTED CORMARANT', 'DOUBLE EYED FIG PARROT', 'DUSKY LORY', 'EASTERN MEADOWLARK', 'EASTERN YELLOW ROBIN', 'ELEGANT TROGON', 'ELLIOTS  PHEASANT', 'EMPEROR PENGUIN', 'EURASIAN BULLFINCH', 'EURASIAN MAGPIE', 'EUROPEAN GOLDFINCH', 'FAIRY BLUEBIRD', 'FAIRY TERN', 'FASCIATED WREN', 'FLAME TANAGER', 'FOREST WAGTAIL', 'FRIGATE', 'GANG GANG COCKATOO', 'GILA WOODPECKER', 'GOLD WING WARBLER', 'GOLDEN BOWER BIRD', 'GOLDEN PHEASANT', 'GOLDEN PIPIT', 'GRANDALA', 'GRAY KINGBIRD', 'GREAT GRAY OWL', 'GREAT JACAMAR', 'GREAT XENOPS', 'GREEN JAY', 'GREEN WINGED DOVE', 'GREY HEADED FISH EAGLE', 'GROVED BILLED ANI', 'GUINEA TURACO', 'HAWAIIAN GOOSE', 'HIMALAYAN BLUETAIL', 'HOATZIN', 'HOOPOES', 'HORNED GUAN', 'HORNED SUNGEM', 'HOUSE FINCH', 'HYACINTH MACAW', 'IMPERIAL SHAQ', 'INCA TERN', 'INDIGO BUNTING', 'INLAND DOTTEREL', 'IWI', 'JACK SNIPE', 'JANDAYA PARAKEET', 'JAPANESE ROBIN', 'JAVA SPARROW', 'KILLDEAR', 'KING VULTURE', 'KNOB BILLED DUCK', 'LAZULI BUNTING', 'LILAC ROLLER', 'LIMPKIN', 'LITTLE AUK', 'LOGGERHEAD SHRIKE', 'LONG-EARED OWL', 'LOONEY BIRDS', 'MANGROVE CUCKOO', 'MASKED LAPWING', 'MOURNING DOVE', 'NICOBAR PIGEON', 'NORTHERN BEARDLESS TYRANNULET', 'NORTHERN FLICKER', 'NORTHERN FULMAR', 'NORTHERN JACANA', 'NORTHERN MOCKINGBIRD', 'OILBIRD', 'ORANGE BRESTED BUNTING', 'ORIENTAL BAY OWL', 'OYSTER CATCHER', 'PARUS MAJOR', 'PEACOCK', 'PHAINOPEPLA', 'PINK ROBIN', 'PLUSH CRESTED JAY', 'POMARINE JAEGER', 'PUFFIN', 'PURPLE SWAMPHEN', 'PYRRHULOXIA', 'RED BEARDED BEE EATER', 'RED BELLIED PITTA', 'RED BILLED TROPICBIRD', 'RED FACED CORMORANT', 'RED LEGGED HONEYCREEPER', 'RED WINGED BLACKBIRD', 'REGENT BOWERBIRD', 'ROSEATE SPOONBILL', 'RUBY CROWNED KINGLET', 'SAMATRAN THRUSH', 'SAND MARTIN', 'SATYR TRAGOPAN', 'SCARLET CROWNED FRUIT DOVE', 'SCARLET IBIS', 'SHOEBILL', 'SHORT BILLED DOWITCHER', 'SMITHS LONGSPUR', 'SNOWY SHEATHBILL', 'SPLENDID WREN', 'SPOTTED CATBIRD', 'SPOTTED WHISTLING DUCK', 'SRI LANKA BLUE MAGPIE', 'STEAMER DUCK', 'STORK BILLED KINGFISHER', 'STRIATED CARACARA', 'SURF SCOTER', 'TAILORBIRD', 'TAKAHE', 'TASMANIAN HEN', 'TAWNY FROGMOUTH', 'TEAL DUCK', 'TOUCHAN', 'TOWNSENDS WARBLER', 'TREE SWALLOW', 'TRICOLORED BLACKBIRD', 'TROPICAL KINGBIRD', 'TURKEY VULTURE', 'TURQUOISE MOTMOT', 'VARIED THRUSH', 'VEERY', 'VERMILION FLYCATHER', 'WALL CREAPER', 'WHIMBREL', 'WHITE BREASTED WATERHEN', 'WHITE CRESTED HORNBILL', 'WHITE EARED HUMMINGBIRD', 'WHITE TAILED TROPIC', 'WILD TURKEY', 'WOOD THRUSH', 'YELLOW BELLIED FLOWERPECKER', 'YELLOW CACIQUE']

# customize dataset class
class BirdDataset(torch.utils.data.Dataset):

    def __init__(self, imgs_list, class_to_int, transforms = None):

        super().__init__()
        self.imgs_list = imgs_list
        self.class_to_int = class_to_int
        self.transforms = transforms


    def __getitem__(self, index):

        image_path = self.imgs_list[index]

        #Reading image
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0

        #Retriving class label
        label = image_path.split("/")[-2]
        label = self.class_to_int[label]

        #Applying transforms on image
        if self.transforms:
            image = self.transforms(image)

        return image, label


    def __len__(self):
        return len(self.imgs_list)
    
# import os # moved to the top cell
import random
random.seed(21)

DIR_TRAIN = "./train/"
DIR_VALID = "./valid/"
DIR_TEST = "./test/"

# Exploring Dataset
# ############################
# # change N_CLASSES to set number of classes(labels) to pick
# N_CLASSES = 10
# ############################
# all_classes = os.listdir(DIR_TRAIN)
# classes = random.sample(all_classes, N_CLASSES)
# print("Total Classes: ",len(classes))
# print(classes)

# Counting total train, valid & test images
train_count = 0
valid_count = 0
test_count = 0
for _class in classes:
    train_count += len(os.listdir(DIR_TRAIN + _class))
    valid_count += len(os.listdir(DIR_VALID + _class))
    test_count += len(os.listdir(DIR_TEST + _class))

# print("Total train images: ",train_count)
# print("Total valid images: ",valid_count)
# print("Total test images: ",test_count)


# Creating a list of all images based on specified labels
train_imgs = []
valid_imgs = []
test_imgs = []

for _class in classes:

    for img in os.listdir(DIR_TRAIN + _class):
        train_imgs.append(DIR_TRAIN + _class + "/" + img)

    for img in os.listdir(DIR_VALID + _class):
        valid_imgs.append(DIR_VALID + _class + "/" + img)

    for img in os.listdir(DIR_TEST + _class):
        test_imgs.append(DIR_TEST + _class + "/" + img)

class_to_int = {classes[i] : i for i in range(len(classes))}

# print(class_to_int)

trainset = BirdDataset(train_imgs, class_to_int, transforms = train_transform)
validset = BirdDataset(valid_imgs, class_to_int, transforms = train_transform)
# put testset to the bottom of the code
#testset = BirdDataset(test_imgs, class_to_int)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, num_workers=0, shuffle=True)
validloader = torch.utils.data.DataLoader(validset, batch_size=64, num_workers=0, shuffle=False)
#testloader = torch.utils.data.DataLoader(testset, batch_size=64, num_workers=0, shuffle=False)

dataloaders = {
    "train": trainloader,
    "valid": validloader,
}
datasizes = {
    "train": len(trainset),
    "valid": len(validset),
}

CLASSES = list(class_to_int.keys())

def train_model(model, criterion, optimizer, scheduler, epochs=25):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    train_accuracy_list = []  # List to store training accuracy values per epoch
    # test_accuracy_list = []  # List to store testing accuracy values per epoch
    valid_accuracy_list = []  # List to store validation accuracy values per epoch
    train_loss_list = []  # List to store training loss values per epoch
    valid_loss_list = []  # List to store validation loss values per epoch

    for epoch in range(epochs):
        print("Epoch {}/{}".format(epoch, epochs - 1))
        print("-" * 10)
        # print elapsed time in minutes
        print("Elapsed time: ", (time.time() - since) // 60, "m " + str(round((time.time() - since) % 60, 0)) + "s")

        # for phase in ["train", "test"]:
        for phase in ["train", "valid"]:
            if phase == "train":
                model.train()
            else:
                model.eval()
            running_loss = 0.0
            running_corrects = 0

            # Iterate over data
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Zero the parameters
                optimizer.zero_grad()

                # Forward
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == "train":
                scheduler.step()

            epoch_loss = running_loss / datasizes[phase]
            epoch_acc = running_corrects.double() / datasizes[phase]
            # ## Mac does not support float 64, need to conver to float 32
            # ## run below instead:
            # running_corrects.to(torch.float32) / datasizes[phase]

            print("{} Loss: {:.4f} Acc: {:.4f}".format(phase, epoch_loss, epoch_acc))
            if phase == "train":
                train_accuracy_list.append(epoch_acc.item())  # Store training accuracy value
                train_loss_list.append(epoch_loss)  # Store training loss value

            else:
                valid_accuracy_list.append(epoch_acc.item())  # Store validation accuracy value
                valid_loss_list.append(epoch_loss)  # Store validation loss value

            # if phase == "test" and epoch_acc > best_acc:
            if phase == "valid" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        print()

    time_elapsed = time.time() - since
    print("Training complete in {:0f}m {:0f}s".format(time_elapsed // 60, time_elapsed % 60))
    print("Best val Acc: {:4f}".format(best_acc))

    # Load best model parameters
    model.load_state_dict(best_model_wts)
    # return model, train_accuracy_list, test_accuracy_list
    return model, train_accuracy_list, valid_accuracy_list, train_loss_list, valid_loss_list, time_elapsed


# Load pretrained model based on config
model_name = config["model"]["name"]
pretrained = config["model"]["pretrained"]

# optimizer
optimizer_type = config["optimizer"]["type"]
lr = config["optimizer"]["lr"]
momentum = config["optimizer"]["momentum"]

# scheduler
step_size = config["scheduler"]["step_size"]
gamma = config["scheduler"]["gamma"]

# Number of epochs
epochs = config["training"]["epochs"]

# check if model is available in pytorch
if hasattr(models, model_name):
    model_ft = getattr(models, model_name)(pretrained=pretrained)

    # Set requires_grad to False for all layers except the fully connected (fc) layer
    for param in model_ft.parameters():
        param.requires_grad = False

if model_name.startswith("resnet"):
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, len(CLASSES))

elif model_name.startswith("vgg"):
    model_ft.classifier = nn.Sequential(
    nn.Linear(25088, 4096),
    nn.ReLU(True),
    nn.Dropout(0.5),
    nn.Linear(4096, 2048),
    nn.ReLU(True),
    nn.Dropout(0.5),
    nn.Linear(2048, len(CLASSES))
    )

elif model_name.startswith("efficientnet"):
    model_ft = EfficientNet.from_pretrained('efficientnet-b0')
    model_ft._fc = nn.Linear(1280, len(CLASSES))
    

elif model_name.startswith("mobilenet"):
    model_ft.classifier = nn.Sequential(
    nn.Linear(in_features=576, out_features=1024, bias=True),
    nn.Hardswish(),
    nn.Dropout(p=0.2, inplace=True),
    nn.Linear(in_features=1024, out_features=len(CLASSES), bias=True)
    )

model_ft = model_ft.to(device)

# If multiple GPUs are available, then use DataParallel
if num_gpu > 1:
    model_ft = nn.DataParallel(model_ft)

criterion = nn.CrossEntropyLoss()
optimizer_ft = getattr(optim, optimizer_type)(model_ft.parameters(), lr=lr, momentum=momentum)
exp_lr_sc = lr_scheduler.StepLR(optimizer_ft, step_size=step_size, gamma=gamma)

model_ft, train_accuracy_list, valid_accuracy_list, train_loss_list, valid_loss_list, training_time  = train_model(
    model_ft, criterion, optimizer_ft, exp_lr_sc, epochs
)

# Plotting the accuracies
epochs = range(1, len(train_accuracy_list) + 1)
plt.plot(epochs, train_accuracy_list, label='Training Accuracy')
# plt.plot(epochs, test_accuracy_list, label='Testing Accuracy')
plt.plot(epochs, valid_accuracy_list, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
# plt.title('Training and Testing Accuracies')
plt.title('Training and Validation Accuracies')
plt.legend()
# save the figure, with the model as filename
if not os.path.exists('./figures'):
    os.makedirs('./figures')
plt.savefig('figures/' + model_name + '.png', dpi=300)
#plt.savefig('figures/resnet_default_50epoch_accuracy.png', dpi=300)


# save entire model
# create directory if not exists
if not os.path.exists('./saved_models'):
    os.makedirs('./saved_models')
save_dir_entire = './saved_models/' + model_name + '_entire.pth'
torch.save(model_ft, save_dir_entire)

criterion = nn.CrossEntropyLoss()
model_trained = torch.load(save_dir_entire)

# List of augmentation scenarios from the YAML file
test_augmentation_scenarios = config['data']['augmentation']['test'].split(', ')

# Check if the CSV file already exists
if not os.path.exists('./metrics'):
    os.makedirs('./metrics')
csv_file = './metrics/results.csv'

# Iterate over each augmentation scenario
for augmentation_scenario in test_augmentation_scenarios:
    # Set the appropriate transformation for the test dataset based on the scenario
    test_transform = create_transform(augmentation_scenario)
    
    # Create the test dataset with the specified transformation
    testset = BirdDataset(test_imgs, class_to_int, transforms=test_transform)

    # Create the test dataloader
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, num_workers=0, shuffle=False)

    # Perform testing with the current augmentation scenario
    running_tloss = 0.0
    running_tcorrects = 0
    with torch.no_grad():
        for tinputs, tlabels in testloader:
            tinputs = tinputs.to(device).float()
            tlabels = tlabels.to(device)
            toutputs = model_trained(tinputs)
            _, preds = torch.max(toutputs, 1)
            tloss = criterion(toutputs, tlabels)

            running_tloss += tloss.item() * tinputs.size(0)
            running_tcorrects += torch.sum(preds == tlabels.data)

    # Calculate the testing loss and accuracy for the current scenario
    test_loss = running_tloss / len(testloader.dataset)
    test_acc = running_tcorrects.double() / len(testloader.dataset)

    # Create a DataFrame with the new data for the current scenario
    data = {
        'Model': [model_name],
        'Augmentation Scenario': [augmentation_scenario],
        'Training Time': [training_time],
        'Testing Accuracy': [test_acc.item()],
        'Testing Loss': [test_loss]
    }
    df = pd.DataFrame(data)

    # Write the DataFrame to the CSV file for the current scenario
    if os.path.exists(csv_file):
        # Read the existing data from the CSV file
        existing_df = pd.read_csv(csv_file)
        # Concatenate the new data on top of the existing data
        updated_df = pd.concat([existing_df, df], ignore_index=True)
        # Write the updated DataFrame to the CSV file
        updated_df.to_csv(csv_file, index=False)
    else:
        # Write the new DataFrame to the CSV file
        df.to_csv(csv_file, index=False)



