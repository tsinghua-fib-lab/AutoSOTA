import requests
import os
import csv
from tqdm import tqdm

# Replace with your real token
API_TOKEN = 'TODO'
HEADERS = {
    'Authorization': f'Bearer {API_TOKEN}'
}

BASE_URL = 'https://api.isic-archive.com/api/v2'
DOWNLOAD_DIR = '/datasets/ISIC2020'

os.makedirs(DOWNLOAD_DIR, exist_ok=True)
os.makedirs(DOWNLOAD_DIR + "/train", exist_ok=True)
os.makedirs(DOWNLOAD_DIR + "/train" + "/benign", exist_ok=True)
os.makedirs(DOWNLOAD_DIR + "/train" + "/malignant", exist_ok=True)

os.makedirs(DOWNLOAD_DIR + "/val", exist_ok=True)
os.makedirs(DOWNLOAD_DIR + "/val" + "/benign", exist_ok=True)
os.makedirs(DOWNLOAD_DIR + "/val" + "/malignant", exist_ok=True)

# Diagnosis -> Malignancy mapping
MALIGNANT_LABELS = {'melanoma', 'basal cell carcinoma', 'squamous cell carcinoma'}
def is_malignant(diagnosis):
    return 'malignant' if diagnosis in MALIGNANT_LABELS else 'benign'


# Get list of all images (paginated)
def get_all_image_ids(collection_id):
    image_ids = []
    labels = []
    limit = 100
    offset = 0

    request_url = f'{BASE_URL}/images/search?collections={collection_id}&limit={limit}'

    while True:
        response = requests.get(request_url, headers=HEADERS)
        data = response.json()
        for img in data['results']:
            clinical = img['metadata']['clinical']

            # Older ISIC API responses used benign_malignant. Current responses
            # for these collections expose the same binary label as diagnosis_1.
            label = clinical.get('benign_malignant')
            if label is None and clinical.get('diagnosis_1') in {'Benign', 'Malignant'}:
                label = clinical['diagnosis_1'].lower()

            if label == 'benign' or label == 'malignant':
                image_ids.append(img['isic_id'])
                labels.append(label)
        if not data['next']:
            break
        request_url = data['next']
        offset += limit

    return image_ids, labels

# Get metadata for one image
def get_diagnosis_for_image(image_id):
    url = f'{BASE_URL}/images/{image_id}'
    response = requests.get(url, headers=HEADERS)
    if response.status_code == 200:
        data = response.json()
        return data.get('metadata', {}).get('clinical', {}).get('diagnosis', 'unknown')
    return 'unknown'

def download_image_from_url(image_url, save_path):
    try:
        r = requests.get(image_url, stream=True)
        if r.status_code == 200:
            with open(save_path, 'wb') as f:
                for chunk in r.iter_content(1024):
                    f.write(chunk)
        else:
            print(f"Failed to download {image_url} (status {r.status_code})")
    except Exception as e:
        print(f"Error: {e}")

# Download images
def download_images(image_ids, labels, split='train'):
    for image_id, label in tqdm(zip(image_ids, labels)):
        save_path = os.path.join(DOWNLOAD_DIR, split, label, f"{image_id}.jpg")
        if os.path.exists(save_path):
            continue
        img_url = f'{BASE_URL}/images/{image_id}'
        response = requests.get(img_url, headers=HEADERS)
        if response.status_code == 200:
            data = response.json()
            full_image_url = data['files']['full']['url']
            download_image_from_url(full_image_url, save_path)
        else:
            print(f'Failed to download {image_id}')
            continue

# Run the whole process

image_ids, labels = get_all_image_ids(70) # ISIC 2020 train
print(len(image_ids))
download_images(image_ids, labels, split='train')

image_ids, labels = get_all_image_ids(68) # ISIC 2020 test, used as val
print(len(image_ids))
download_images(image_ids, labels, split='val')
