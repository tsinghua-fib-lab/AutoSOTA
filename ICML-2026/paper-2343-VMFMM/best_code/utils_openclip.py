from tqdm import tqdm

import torch
import torch.nn.functional as F
import os
# import clip
import open_clip


def cls_acc(output, target, topk=1):
    pred = output.topk(topk, 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    acc = float(correct[: topk].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
    acc = 100 * acc / target.shape[0]
    return acc


def openclip_classifier(classnames, template, model, tokenizer, reduce='mean', gpt=False, wordnet_dict=None):
    device = next(model.parameters()).device
    with torch.no_grad():
        clip_weights = []
        if wordnet_dict is not None:
            indices = []
            i = 0
            for classname in classnames:
                allnames = [classname] + wordnet_dict[classname]
                for name in allnames:
                    name = name.replace('_', ' ')
                    texts = [t.format(name) for t in template]
                    texts = tokenizer(texts).to(device)

                    class_embeddings = model.encode_text(texts)
                    class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
                    if reduce == 'mean':
                        class_embedding = class_embeddings.mean(dim=0)
                        class_embedding /= class_embedding.norm()
                        clip_weights.append(class_embedding)
                    if reduce is None:
                        class_embeddings /= class_embeddings.norm(dim=1, keepdim=True)
                        clip_weights.append(class_embeddings)
                    i += 1
                indices.append(i)

            return clip_weights, indices
        else:
            for classname in classnames:
                classname = classname.replace('_', ' ')
                if gpt:
                    texts = template[classname]
                else:
                    texts = [t.format(classname) for t in template]
                texts = tokenizer(texts).to(device)

                class_embeddings = model.encode_text(texts)
                class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
                if reduce == 'mean':
                    class_embedding = class_embeddings.mean(dim=0)
                    class_embedding /= class_embedding.norm()
                    clip_weights.append(class_embedding)
                if reduce is None:
                    class_embeddings /= class_embeddings.norm(dim=1, keepdim=True)
                    clip_weights.append(class_embeddings)

            clip_weights = torch.stack(clip_weights, dim=-1).cuda()
    return clip_weights


def get_all_features_openclip(args, test_loader, dataset, model, tokenizer):
    clip_prototypes = openclip_classifier(dataset.classnames, dataset.template, model, tokenizer, reduce=None)
    test_features, test_labels = pre_load_features_openclip(args, "test", model, test_loader)
    return test_features, test_labels, clip_prototypes


def pre_load_features_openclip(args, split, model, loader, n_views=1):
    device = next(model.parameters()).device
    if not args.load:
        features, labels = [], []
        with torch.no_grad():
            for view in range(n_views):
                length = 0
                for i, (images, target) in enumerate(tqdm(loader)):
                    if n_views == 1:
                        images, target = images.to(device), target.to(device)
                        image_features = model.encode_image(images)
                        image_features /= image_features.norm(dim=-1, keepdim=True)
                        features.append(image_features.cpu())
                        labels.append(target.cpu())
                    else:
                        images, target = images.to(device), target.to(device)
                        image_features = model.encode_image(images)
                        image_features /= image_features.norm(dim=-1, keepdim=True)
                        if view == 0:
                            labels.append(target.cpu())
                            if i == 0:
                                mean_features = image_features
                            else:
                                mean_features = torch.cat((mean_features, image_features))
                        else:
                            mean_features[length:length + image_features.size(0)] += image_features
                            length += image_features.size(0)

        if n_views > 1:
            mean_features = mean_features / n_views
            features = mean_features / mean_features.norm(dim=-1, keepdim=True)
            labels = torch.cat(labels)
        elif n_views == 1:
            features, labels = torch.cat(features), torch.cat(labels)

        torch.save(features, args.cache_dir + "/" + split + "_f.pt")
        torch.save(labels, args.cache_dir + "/" + split + "_l.pt")
    else:
        try:
            features = torch.load(args.cache_dir + "/" + split + "_f.pt")
            labels = torch.load(args.cache_dir + "/" + split + "_l.pt")
        except FileNotFoundError:
            print("Cache not found...")

    return features, labels


def build_cache_model(cfg, clip_model, train_loader_cache, n_views=0, reduce=None):
    print('... for shot samples from train split:')

    if cfg['load_cache'] == False:    
        cache_keys = []
        cache_values = []
        if n_views == 0:
            n_epochs =1
        else:
            n_epochs = n_views
        with torch.no_grad():
            # Data augmentation for the cache model
            for augment_idx in range(n_epochs):
                train_features = []
                train_labels = []
                for i, (images, target) in enumerate(tqdm(train_loader_cache)):
                    images = images.cuda()
                    image_features = clip_model.encode_image(images)
                    train_features.append(image_features)
                    
                    if augment_idx == 0:
                        target = target.cuda()
                        cache_values.append(target)
                        
                cache_keys.append(torch.cat(train_features, dim=0).unsqueeze(0))


        
        if n_views == 1:
            cache_keys = torch.cat(cache_keys, dim=0).mean(dim=0)
            cache_keys /= cache_keys.norm(dim=-1, keepdim=True)
        else:
            cache_keys = torch.cat(cache_keys, dim=0) # [n_views, n_classes, n_features]
            if reduce == 'mean':
                cache_keys = cache_keys.mean(0, keepdim=True)
                
            cache_keys /= cache_keys.norm(dim=-1, keepdim=True)
            cache_keys.permute(0, 2, 1)
            
        cache_values = F.one_hot(torch.cat(cache_values, dim=0)).half()

        torch.save(cache_keys, cfg['cache_dir'] + '/keys_' + str(cfg['shots']) + "shots.pt")
        torch.save(cache_values, cfg['cache_dir'] + '/values_' + str(cfg['shots']) + "shots.pt")

    else:
        cache_keys = torch.load(cfg['cache_dir'] + '/keys_' + str(cfg['shots']) + "shots.pt")
        cache_values = torch.load(cfg['cache_dir'] + '/values_' + str(cfg['shots']) + "shots.pt")

    return cache_keys, cache_values
