import os, sys
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HOME'] = '/autosota_cache/hf'
os.environ['HF_HUB_CACHE'] = '/autosota_cache/hf/hub'
os.environ['HF_DATASETS_CACHE'] = '/autosota_cache/hf/datasets'
import torch
sys.path.insert(0, '/repo')
import clip
from datasets import load_dataset
from tqdm import tqdm
from utils import clip_classifier
from datasets.imagenet import imagenet_classes, imagenet_templates

device = 'cuda:0'
print(f'Loading CLIP ViT-B/16 on {device}...')
clip_model, preprocess = clip.load('ViT-B/16', device=device)
clip_model.eval()

print('Computing text prototypes...')
clip_prototypes = clip_classifier(imagenet_classes, imagenet_templates, clip_model, reduce=None)

print('Loading ImageNet validation from HF...')
ds = load_dataset('ILSVRC/imagenet-1k', split='validation', trust_remote_code=True)
print(f'Dataset size: {len(ds)}')
print(f'Extracting features...')

features, labels_list = [], []
batch_size = 128
for i in tqdm(range(0, len(ds), batch_size)):
    batch_end = min(i + batch_size, len(ds))
    images_batch, lbls_batch = [], []
    for j in range(i, batch_end):
        ex = ds[j]
        images_batch.append(preprocess(ex['image'].convert('RGB')))
        lbls_batch.append(ex['label'])
    image_tensor = torch.stack(images_batch).to(device)
    with torch.no_grad():
        feats = clip_model.encode_image(image_tensor)
        feats = feats / feats.norm(dim=-1, keepdim=True)
    features.append(feats.cpu())
    labels_list.extend(lbls_batch)

features = torch.cat(features, dim=0)
labels = torch.tensor(labels_list, dtype=torch.long)
print(f'Features: {features.shape}, Labels: {labels.shape}, Unique classes: {len(torch.unique(labels))}')

cache_dir = '/repo/caches/imagenet'
os.makedirs(cache_dir, exist_ok=True)
torch.save(features, os.path.join(cache_dir, 'test_f.pt'))
torch.save(labels, os.path.join(cache_dir, 'test_l.pt'))
torch.save(clip_prototypes, os.path.join(cache_dir, 'clip_prototypes.pt'))
print('Saved features to /repo/caches/imagenet/')

zs_logits = 100 * features @ clip_prototypes.squeeze()
zs_preds = zs_logits.argmax(dim=1)
zs_acc = (zs_preds == labels).float().mean() * 100
print(f'Zero-shot accuracy: {zs_acc:.2f}%')
print('Done!')
