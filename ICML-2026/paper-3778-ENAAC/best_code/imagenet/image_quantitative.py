import argparse
import random
import os
from tqdm import tqdm

import torch
from torch.utils.data import Subset
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from scipy.stats import ttest_ind

from utils import ModelWrapper, ReLU_inplace_to_False, get_lowest_highest_patches, get_random_patches, paste_patches


parser = argparse.ArgumentParser(description='Quantitative results for image classification')
parser.add_argument('--data_dir', metavar='DIR', default='/datasets/imagenet/',
                    help='path to dataset')
parser.add_argument('--model', required=True,
                    choices=['vgg19', 'resnet50', 'vit_b_16'],
                    help='model architecture')
parser.add_argument('--model_layer', required=True,
                    choices=['model.features[34]', 'model.layer4[2].conv3', 'model.layer3[2].conv3', 'model.layer2[2].conv3', 'model.layer1[2].conv3', 'encoder.layers.encoder_layer_11.mlp[3]'],
                    help='model architecture')
parser.add_argument('--batch_size', default=64, type=int,
                    help='batch size')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--nr_images_per_channel', default=100, type=int,
                    help='Number of images that are used to evaluate a single channel.')
parser.add_argument('--patch_size', default=64, type=int,
                    help='Size of patches.')
parser.add_argument('--patch_stride', default=16, type=int,
                    help='Size of patches.')
parser.add_argument('--nr_patches', default=8, type=int,
                    help='Number of patches.')

def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    ### Model ###
    if args.model == 'vgg19':
        model = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
    elif args.model == 'resnet50':
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    elif args.model == 'vit_b_16':
        model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
    else:
        print('Model not implemented!')
        assert False

    # remove inplace operations
    ReLU_inplace_to_False(model)

    # get model that outputs activations in layer of interest
    if args.model_layer == 'model.features[34]':
        intermediate_layer_model = ModelWrapper(model, model.features[34])
    elif args.model_layer == 'model.layer4[2].conv3':
        intermediate_layer_model = ModelWrapper(model, model.layer4[2].conv3)
    elif args.model_layer == 'model.layer3[2].conv3':
        intermediate_layer_model = ModelWrapper(model, model.layer3[2].conv3)
    elif args.model_layer == 'model.layer2[2].conv3':
        intermediate_layer_model = ModelWrapper(model, model.layer2[2].conv3)
    elif args.model_layer == 'model.layer1[2].conv3':
        intermediate_layer_model = ModelWrapper(model, model.layer1[2].conv3)
    elif args.model_layer == 'encoder.layers.encoder_layer_11.mlp[3]':
        intermediate_layer_model = ModelWrapper(model, model.encoder.layers.encoder_layer_11.mlp[3], architecture='TransformerMIN')
    else:
        print('Layer not implemented!')
        assert False

    intermediate_layer_model.eval()
    intermediate_layer_model.cuda()

    ### Dataset ###
    valdir = os.path.join(args.data_dir, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

    dataset = datasets.ImageFolder(valdir, transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]))

    loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size, shuffle=False,
            num_workers=8, pin_memory=True)

    ### Get activations for each channel for each image ###
    layer_activations_all = []

    with torch.no_grad():
        for images, labels in tqdm(loader):
            images = images.cuda()
            labels = labels.cuda()

            layer_activations = intermediate_layer_model(images)
            layer_activations = layer_activations.detach().to(torch.float16).to('cpu')
            layer_activations_all.append(layer_activations)

    layer_activations_all = torch.cat(layer_activations_all, dim=0) # nr_images(50_000) x nr_channels

    ### Get activations for top images ###
    layer_activations_topk = torch.topk(layer_activations_all, args.nr_images_per_channel, dim=0)[0] # args.nr_images_per_channel x nr_channels
    print('Top activating images (mean):', layer_activations_topk.mean())
    print('Top activating images (std):', layer_activations_topk.std(dim=0).mean())

    ### Get activations for random images ###
    nr_samples = layer_activations_all.shape[0]
    # get N unique random indices along the batch dimension
    rand_indices = torch.randperm(nr_samples)[:args.nr_images_per_channel]
    # sample those rows
    layer_activations_random = layer_activations_all[rand_indices]  # shape: [N, nr_activations]
    print('Activations for random images (mean):', layer_activations_random.mean())
    print('Activations for random images (std):', layer_activations_random.std(dim=0).mean())

    ### Get patches with lowest and highest activations for each channel ###
    loader_for_patches = torch.utils.data.DataLoader(
            dataset,
            batch_size=1, shuffle=False,
            num_workers=0, pin_memory=True)

    nr_channels = layer_activations_all.shape[1]
    if args.model == 'vgg19' or args.model == 'resnet50':
        bottom_patches, bottom_patches_scores, top_patches, top_patches_scores = get_lowest_highest_patches(loader_for_patches, intermediate_layer_model, nr_channels, args.patch_size, args.patch_stride, args.nr_patches, architecture="cnn")
    elif args.model == 'vit_b_16':
        bottom_patches, bottom_patches_scores, top_patches, top_patches_scores = get_lowest_highest_patches(loader_for_patches, intermediate_layer_model, nr_channels, args.patch_size, args.patch_stride, args.nr_patches, architecture="vit")

    mean_activations_per_channel_bottom_patches = []
    mean_activations_per_channel_top_patches = []
    mean_activations_per_channel_random_patches = []

    std_activations_per_channel_bottom_patches = []
    std_activations_per_channel_top_patches = []
    std_activations_per_channel_random_patches = []

    ### Get activations for top images with patches inserted ###
    nr_channels = layer_activations_all.shape[1]
    nr_channels_w_significant_change = 0
    nr_channels_wo_significant_change = 0

    for channel_idx in tqdm(range(nr_channels)):

        activations_for_channel_bottom_patches = []
        activations_for_channel_top_patches = []
        activations_for_channel_random_patches = []

        topk_idx = torch.topk(layer_activations_all[:,channel_idx], args.nr_images_per_channel, dim=0)[1]
        topk_dataset = Subset(dataset, topk_idx)
        topk_loader = torch.utils.data.DataLoader(
           topk_dataset,
           batch_size=args.batch_size, shuffle=False,
           num_workers=0, pin_memory=True)

        bottom_patches_for_channel = bottom_patches[channel_idx] # 16, 3, 32, 32
        top_patches_for_channel = top_patches[channel_idx] # 16, 3, 32, 32
        random_patches_for_channel = get_random_patches(loader, args.patch_size, args.patch_stride, args.nr_patches) # 16, 3, 32, 32

        with torch.no_grad():
            for images, labels in topk_loader:
                images = images.cuda()

                images_with_bottom_patches = paste_patches(images, bottom_patches_for_channel)
                images_with_top_patches = paste_patches(images, top_patches_for_channel)
                images_with_random_patches = paste_patches(images, random_patches_for_channel)

                channel_activations_bottom_patches = intermediate_layer_model(images_with_bottom_patches)[:,channel_idx]
                channel_activations_bottom_patches = channel_activations_bottom_patches.detach().to(torch.float16).to('cpu')
                activations_for_channel_bottom_patches.append(channel_activations_bottom_patches)

                channel_activations_top_patches = intermediate_layer_model(images_with_top_patches)[:,channel_idx]
                channel_activations_top_patches = channel_activations_top_patches.detach().to(torch.float16).to('cpu')
                activations_for_channel_top_patches.append(channel_activations_top_patches)

                channel_activations_random_patches = intermediate_layer_model(images_with_random_patches)[:,channel_idx]
                channel_activations_random_patches = channel_activations_random_patches.detach().to(torch.float16).to('cpu')
                activations_for_channel_random_patches.append(channel_activations_random_patches)
                    
        activations_for_channel_bottom_patches = torch.cat(activations_for_channel_bottom_patches, dim=0)
        activations_for_channel_top_patches = torch.cat(activations_for_channel_top_patches, dim=0)
        activations_for_channel_random_patches = torch.cat(activations_for_channel_random_patches, dim=0)

        mean_activations_per_channel_bottom_patches.append(torch.tensor([activations_for_channel_bottom_patches.mean()]))
        mean_activations_per_channel_top_patches.append(torch.tensor([activations_for_channel_top_patches.mean()]))
        mean_activations_per_channel_random_patches.append(torch.tensor([activations_for_channel_random_patches.mean()]))

        std_activations_per_channel_bottom_patches.append(torch.tensor([activations_for_channel_bottom_patches.std()]))
        std_activations_per_channel_top_patches.append(torch.tensor([activations_for_channel_top_patches.std()]))
        std_activations_per_channel_random_patches.append(torch.tensor([activations_for_channel_random_patches.std()]))

        stat, p_value = ttest_ind(activations_for_channel_random_patches.flatten().detach().cpu().numpy(), activations_for_channel_bottom_patches.flatten().detach().cpu().numpy())
        if p_value <= 0.05:
            nr_channels_w_significant_change += 1
        else:
            nr_channels_wo_significant_change += 1


    print('Activations for top images with bottom patches (mean):', torch.cat(mean_activations_per_channel_bottom_patches, dim=0).mean())
    print('Activations for top images with bottom patches (std):', torch.cat(std_activations_per_channel_bottom_patches, dim=0).mean())
    print('Activations for top images with top patches (mean):', torch.cat(mean_activations_per_channel_top_patches, dim=0).mean())
    print('Activations for top images with top patches (std):', torch.cat(std_activations_per_channel_top_patches, dim=0).mean())
    print('Activations for top images with random patches (mean):', torch.cat(mean_activations_per_channel_random_patches, dim=0).mean())
    print('Activations for top images with random patches (std):', torch.cat(std_activations_per_channel_random_patches, dim=0).mean())

    print('Number channels with significant change:', nr_channels_w_significant_change)
    print('Number channels without significant change:', nr_channels_wo_significant_change)

if __name__ == '__main__':
    main()