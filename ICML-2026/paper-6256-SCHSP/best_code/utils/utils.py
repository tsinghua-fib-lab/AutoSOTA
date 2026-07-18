import os
import random
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision
from PIL import Image
import torch.nn.functional as F
import json

from tqdm import tqdm

from config.config import DATA_DIR, WEIGHTS_DIR

# Import ConvNext from transformers
from transformers import ConvNextImageProcessor, ConvNextModel, ConvNextForImageClassification

# Import nn and optim from torch
import torch.nn as nn
import torch.optim as optim

# Import model utilities for consistent layer manipulation
from utils.model_utils import turn_final_layer_to_id


def data_path(*parts):
    return os.path.join(DATA_DIR, *parts)

def set_random_seeds(seed: int = 42) -> None:
    """
    Sets random seeds for Python's random module, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to set (default is 42).
    """
    # Set random seed for Python's built-in random module
    random.seed(seed)

    # Set random seed for NumPy
    np.random.seed(seed)

    # Set random seed for PyTorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def set_global_seed(seed):
    global GLOBAL_SEED
    GLOBAL_SEED = seed
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print(f"Global seed set to: {seed}")


def get_label_names(dataset_name):
    if dataset_name.startswith("cifar100"):
        return ["apple", "aquarium fish", "baby", "bear", "beaver", "bed", "bee", "beetle", "bicycle", "bottle", "bowl", "boy", "bridge", "bus", "butterfly", "camel", "can", "castle", "caterpillar", "cattle", "chair", "chimpanzee", "clock", "cloud", "cockroach", "couch", "crab", "crocodile", "cup", "dinosaur", "dolphin", "elephant", "flatfish", "forest", "fox", "girl", "hamster", "house", "kangaroo", "keyboard", "lamp", "lawn mower", "leopard", "lion", "lizard", "lobster", "man", "maple tree", "motorcycle", "mountain", "mouse", "mushroom", "oak tree", "orange", "orchid", "otter", "palm tree", "pear", "pickup truck", "pine tree", "plain", "plate", "poppy", "porcupine", "possum", "rabbit", "raccoon", "ray", "road", "rocket", "rose", "sea", "seal", "shark", "shrew", "skunk", "skyscraper", "snail", "snake", "spider", "squirrel", "streetcar", "sunflower", "sweet pepper", "table", "tank", "telephone", "television", "tiger", "tractor", "train", "trout", "tulip", "turtle", "wardrobe", "whale", "willow tree", "wolf", "woman", "worm"]
    if dataset_name.startswith("cifar10"):
        return ['airplane', 'automobile', 'bird', 'cat', 'deer/elk', 'dog', 'frog', 'horse', 'ship', 'truck']
    elif dataset_name in ["imagenet1kval", "imagenet1k"]:
        # Load ImageNet-1k class names from text file
        txt_path = data_path("imagenet1k", "imagenet_classes.txt")
        if not os.path.exists(txt_path):
            raise FileNotFoundError(f"ImageNet classes text file not found: {txt_path}")
        with open(txt_path, 'r') as f:
            class_names = [line.strip() for line in f.readlines()]
        return class_names
    elif dataset_name == "imagenet21k":
        # Load ImageNet-21k class names from text file
        txt_path = data_path("imagenet21k_classes.txt")
        if not os.path.exists(txt_path):
            raise FileNotFoundError(f"ImageNet-21K classes text file not found: {txt_path}")
        with open(txt_path, 'r') as f:
            class_names = [line.strip() for line in f.readlines()]
        return class_names
    elif dataset_name == "imagenet21k_2extra":
        # Load ImageNet-21k 2extra class names from text file
        txt_path = data_path("imagenet21k_2extra_classes.txt")
        if not os.path.exists(txt_path):
            raise FileNotFoundError(f"ImageNet-21K 2extra classes text file not found: {txt_path}")
        with open(txt_path, 'r') as f:
            class_names = [line.strip() for line in f.readlines()]
        return class_names
    elif dataset_name == "food101":
        # Food101 class names (101 food categories)
        food_names = ["Apple pie", "Baby back ribs", "Baklava", "Beef carpaccio", "Beef tartare", "Beet salad", "Beignets", "Bibimbap", "Bread pudding", "Breakfast burrito", "Bruschetta", "Caesar salad", "Cannoli", "Caprese salad", "Carrot cake", "Ceviche", "Cheesecake", "Cheese plate", "Chicken curry", "Chicken quesadilla", "Chicken wings", "Chocolate cake", "Chocolate mousse", "Churros", "Clam chowder", "Club sandwich", "Crab cakes", "Creme brulee", "Croque madame", "Cup cakes", "Deviled eggs", "Donuts", "Dumplings", "Edamame", "Eggs benedict", "Escargots", "Falafel", "Filet mignon", "Fish and chips", "Foie gras", "French fries", "French onion soup", "French toast", "Fried calamari", "Fried rice", "Frozen yogurt", "Garlic bread", "Gnocchi", "Greek salad", "Grilled cheese sandwich", "Grilled salmon", "Guacamole", "Gyoza", "Hamburger", "Hot and sour soup", "Hot dog", "Huevos rancheros", "Hummus", "Ice cream", "Lasagna", "Lobster bisque", "Lobster roll sandwich", "Macaroni and cheese", "Macarons", "Miso soup", "Mussels", "Nachos", "Omelette", "Onion rings", "Oysters", "Pad thai", "Paella", "Pancakes", "Panna cotta", "Peking duck", "Pho", "Pizza", "Pork chop", "Poutine", "Prime rib", "Pulled pork sandwich", "Ramen", "Ravioli", "Red velvet cake", "Risotto", "Samosa", "Sashimi", "Scallops", "Seaweed salad", "Shrimp and grits", "Spaghetti bolognese", "Spaghetti carbonara", "Spring rolls", "Steak", "Strawberry shortcake", "Sushi", "Tacos", "Takoyaki", "Tiramisu", "Tuna tartare", "Waffles"]
        return [f"{food_name}, a type of food" for food_name in food_names]
    elif dataset_name == "flowers102":
        # Flowers102 class names (102 flower categories)
        flowers_names = [
        "pink primrose flower",
        "hard-leaved pocket orchid flower",
        "canterbury bells flower",
        "sweet pea flower",
        "english marigold flower",
        "tiger lily flower",
        "moon orchid flower",
        "bird of paradise flower",
        "monkshood flower",
        "globe thistle flower",
        "snapdragon flower",
        "colt's foot flower",
        "king protea flower",
        "spear thistle flower",
        "yellow iris flower",
        "globe flower",
        "purple coneflower flower",
        "peruvian lily flower",
        "balloon flower",
        "giant white arum lily flower",
        "fire lily flower",
        "pincushion flower",
        "fritillary flower",
        "red ginger flower",
        "grape hyacinth flower",
        "corn poppy flower",
        "prince of wales feathers flower",
        "stemless gentian flower",
        "artichoke flower",
        "sweet william flower",
        "carnation flower",
        "garden phlox flower",
        "love in the mist flower",
        "mexican aster flower",
        "alpine sea holly flower",
        "ruby-lipped cattleya flower",
        "cape flower",
        "great masterwort flower",
        "siam tulip flower",
        "lenten rose flower",
        "barbeton daisy flower",
        "daffodil flower",
        "sword lily flower",
        "poinsettia flower",
        "bolero deep blue flower",
        "wallflower flower",
        "marigold flower",
        "buttercup flower",
        "oxeye daisy flower",
        "common dandelion flower",
        "petunia flower",
        "wild pansy flower",
        "primula flower",
        "sunflower flower",
        "pelargonium flower",
        "bishop of llandaff flower",
        "gaura flower",
        "geranium flower",
        "orange dahlia flower",
        "pink and yellow dahlia flower",
        "cautleya spicata flower",
        "japanese anemone flower",
        "black-eyed susan flower",
        "silverbush flower",
        "californian poppy flower",
        "osteospermum flower",
        "spring crocus flower",
        "bearded iris flower",
        "windflower flower",
        "tree poppy flower",
        "gazania flower",
        "azalea flower",
        "water lily flower",
        "rose flower",
        "thorn apple flower",
        "morning glory flower",
        "passion flower",
        "lotus flower",
        "toad lily flower",
        "anthurium flower",
        "frangipani flower",
        "clematis flower",
        "hibiscus flower",
        "columbine flower",
        "desert-rose flower",
        "tree mallow flower",
        "magnolia flower",
        "cyclamen flower",
        "watercress flower",
        "canna lily flower",
        "hippeastrum flower",
        "bee balm flower",
        "air plant flower",
        "foxglove flower",
        "bougainvillea flower",
        "camellia flower",
        "mallow flower",
        "mexican petunia flower",
        "bromelia flower",
        "blanket flower",
        "trumpet creeper flower",
        "blackberry lily flower"
        ]
        return [f"{flower_name}, a type of flower" for flower_name in flowers_names]
    elif dataset_name == "oxfordpets":
        # Oxford Pets class names (37 pet categories: 25 dog breeds + 12 cat breeds)
        # return ["abyssinian cat", "bengal cat", "birman cat", "bombay cat", "british shorthair cat", "egyptian mau cat", "maine coon cat", "persian cat", "ragdoll cat", "russian blue cat", "siamese cat", "sphynx cat", "american bulldog dog", "american pit bull terrier dog", "basset hound dog", "beagle dog", "boxer dog", "chihuahua dog", "english cocker spaniel dog", "english setter dog", "german shorthaired dog", "great pyrenees dog", "havanese dog", "japanese chin dog", "keeshond dog", "leonberger dog", "miniature pinscher dog", "newfoundland dog", "pomeranian dog", "pug dog", "saint bernard dog", "samoyed dog", "scottish terrier dog", "shiba inu dog", "staffordshire bull terrier dog", "wheaten terrier dog", "yorkshire terrier dog"]
        pet_names = ["Abyssinian",
        "American Bulldog",
        "American Pit Bull Terrier",
        "Basset Hound",
        "Beagle",
        "Bengal",
        "Birman",
        "Bombay",
        "Boxer",
        "British Shorthair",
        "Chihuahua",
        "Egyptian Mau",
        "English Cocker Spaniel",
        "English Setter",
        "German Shorthaired",
        "Great Pyrenees",
        "Havanese",
        "Japanese Chin",
        "Keeshond",
        "Leonberger",
        "Maine Coon",
        "Miniature Pinscher",
        "Newfoundland",
        "Persian",
        "Pomeranian",
        "Pug",
        "Ragdoll",
        "Russian Blue",
        "Saint Bernard",
        "Samoyed",
        "Scottish Terrier",
        "Shiba Inu",
        "Siamese",
        "Sphynx",
        "Staffordshire Bull Terrier",
        "Wheaten Terrier",
        "Yorkshire Terrier"]

        return [f"{pet_name}, a type of pet" for pet_name in pet_names]
    elif dataset_name == "eurosat":
        # EuroSAT class names (10 land use/land cover categories)
        eurosat_names = ["annual crop land",
        "forest",
        "brushland or shrubland",
        "highway or road",
        "industrial buildings or commercial buildings",
        "pasture land",
        "permanent crop land",
        "residential buildings or homes or apartments",
        "river",
        "lake or sea"]
        return [f"{name}  (aerial photo)" for name in eurosat_names]
    elif dataset_name == "resisc45":
        # RESISC45 class names (45 remote sensing scene categories)
        resisc45_names = ["airplane", "airport", "baseball diamond", "basketball court", "beach", "bridge", "chaparral", "church", "circular farmland", "cloud", "commercial area", "dense residential", "desert", "forest", "freeway", "golf course", "ground track field", "harbor", "industrial area", "intersection", "island", "lake", "meadow", "medium residential", "mobile home park", "mountain", "overpass", "palace", "parking lot", "railway", "railway station", "rectangular farmland", "river", "roundabout", "runway", "sea ice", "ship", "snowberg", "sparse residential", "stadium", "storage tank", "tennis court", "terrace", "thermal power station", "wetland"]
        return [f"{name} (aerial photo)" for name in resisc45_names]
    elif dataset_name == "mnist":
        # MNIST class names (10 digit categories)
        mnist_names = [str(i) for i in range(10)]
        return [f"the number: \"{c}\"." for c in mnist_names]
    elif dataset_name == 'dtd':
        dtd_names = [
    "banded",
    "blotchy",
    "braided",
    "bubbly",
    "bumpy",
    "chequered",
    "cobwebbed",
    "cracked",
    "crosshatched",
    "crystalline",
    "dotted",
    "fibrous",
    "flecked",
    "freckled",
    "frilly",
    "gauzy",
    "grid",
    "grooved",
    "honeycombed",
    "interlaced",
    "knitted",
    "lacelike",
    "lined",
    "marbled",
    "matted",
    "meshed",
    "paisley",
    "perforated",
    "pitted",
    "pleated",
    "polka-dotted",
    "porous",
    "potholed",
    "scaly",
    "smeared",
    "spiralled",
    "sprinkled",
    "stained",
    "stratified",
    "striped",
    "studded",
    "swirly",
    "veined",
    "waffled",
    "woven",
    "wrinkled",
    "zigzagged",
            ]
        return [f"{name} texture" for name in dtd_names]
    elif dataset_name == 'places365':
        file_path = data_path('categories_places365.txt')
        main_names = []
        with open(file_path, 'r') as f:
            for line in f:
                # Get part after the second slash and before the final digit
                name = line.split('/', 2)[2].rsplit(' ', 1)[0]
                # Replace slashes and underscores with spaces
                name = name.replace('/', ' ').replace('_', ' ')
                main_names.append(name)
        return main_names
    elif dataset_name == 'flickr30k':
        # Get Flickr30k captions
        from config.config import FLICKR30_CAPTIONS_PATH
        captions_path = FLICKR30_CAPTIONS_PATH
        captions = []
        with open(captions_path, 'r', encoding='utf-8') as f:
            # read first line to skip header
            header = f.readline().strip()
            for line in f:
                line = line.strip()
                if not line:
                    continue
                assert len(line.split(".jpg,")) == 2, f"Invalid line format in captions file: {line}"
                key, caption = line.split(".jpg,")
                # Remove leading/trailing whitespace
                caption = caption.strip()
                # Remove leading/trailing quotes if present
                caption = caption.strip('"').strip("'").strip()
                img_id = key
                captions.append((caption))
        return [caption.strip() for caption in captions]
    elif dataset_name == 'flickr30k_test':
        # Get Flickr30k test captions
        from config.config import TEST_FLICKR30_CAPTIONS_PATH
        captions_path = TEST_FLICKR30_CAPTIONS_PATH
        captions = []
        with open(captions_path, 'r', encoding='utf-8') as f:
            # read first line to skip header
            header = f.readline().strip()
            for line in f:
                line = line.strip()
                if not line:
                    continue
                assert len(line.split(".jpg,")) == 2, f"Invalid line format in captions file: {line}"
                key, caption = line.split(".jpg,")
                # Remove leading/trailing whitespace
                caption = caption.strip()
                # Remove leading/trailing quotes if present
                caption = caption.strip('"').strip("'").strip()
                img_id = key
                captions.append((caption))
        return [caption.strip() for caption in captions]
    elif dataset_name == 'inaturalist':
        # Load inat2021_common_names.csv file and return list of common names
        import csv
        common_names = []
        with open(data_path('inat2021_common_names.csv'), 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                common_names.append(row['common_name'])
        return common_names
    elif dataset_name == "ham10000":
        return [
            "Melanocytic nevi (benign moles)",
            "Melanoma (malignant skin cancer)",
            "Benign keratosis-like lesions (solar lentigines, seborrheic keratoses)",
            "Basal cell carcinoma",
            "Actinic keratoses / intraepithelial carcinoma (pre-cancerous lesions)",
            "Vascular lesions (angiomas, etc.)",
            "Dermatofibroma (benign skin growth)"
        ]
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

def get_one_caption_per_dataset(dataset_name):
    # Load utils/dataset_captions.json
    with open('./utils/dataset_captions.json', 'r') as f:
        dataset_captions = json.load(f)
    return dataset_captions[dataset_name]

def get_clip_tokenizer():
    return torch.hub.load(f"openai/CLIP", "tokenize")

def get_clip_tokenizer_decoder():
    from transformers import CLIPTokenizerFast
    return (CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch16")).decode

def get_clip_text_encoder(clip_model_name, device):
    clip_model_name_map = {
        "clip_vitb32": "ViT_B_32",
        "clip_vitb16": "ViT_B_16",
        "clip_vitl14": "ViT_L_14",
        "clip_vitl14_336px": "ViT_L_14_336px",
        "clip_resnet50": "RN50",
        "clip_resnet101": "RN101"
        }
    
    clip_model, _ = torch.hub.load('openai/CLIP', clip_model_name_map[clip_model_name])
    clip_model.to(device)
    clip_model.eval()
    clip_text_encoder = clip_model.encode_text
    return clip_text_encoder

def get_preprocess(model_name):
    """
    Get preprocessing transforms for a given model.
    
    Args:
        model_name (str): Name of the model (e.g., 'clip_vitb32', 'dinov2_vitb14', 'timm/eva02_base_patch14_224.mim_in22k', etc.)
        
    Returns:
        torchvision.transforms.Compose: Preprocessing transforms for the model
    """
    if model_name.startswith('clip'):
        # For CLIP models, load the model temporarily to get the preprocessing
        clip_model_name_map = {
            "clip_vitb32": "ViT_B_32",
            "clip_vitb16": "ViT_B_16",
            "clip_vitl14": "ViT_L_14",
            "clip_vitl14_336px": "ViT_L_14_336px",
            "clip_resnet50": "RN50",
            "clip_resnet101": "RN101"
        }
        
        assert model_name in clip_model_name_map, "Unsupported CLIP model name"
        
        # Load the CLIP model using torch.hub to get preprocessing
        _, preprocess = torch.hub.load('openai/CLIP', clip_model_name_map[model_name])
        return preprocess

    elif model_name.startswith('dinov2'):
        # DINOv2 standard preprocessing
        preprocess = transforms.Compose([
            transforms.Resize(256, interpolation=Image.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
        return preprocess

    elif model_name.startswith('timm/') or model_name.startswith('timm_'):
        # TIMM models - get preprocessing parameters from model config
        import timm
        
        # Remove 'timm/' prefix for timm.create_model
        timm_model_name = model_name[5:]  # Remove 'timm/' prefix
            
        try:   
            # Load the full model temporarily to get data config
            full_model = timm.create_model(timm_model_name, pretrained=True)

        # Else try with hf_hub:timm/ + 
        except Exception as e:
            print(f"Warning: Could not load TIMM model {timm_model_name} directly: {e}. Trying with hf_hub:timm/{timm_model_name}")
            full_model = timm.create_model(f"hf_hub:timm/{timm_model_name}", pretrained=True)        
        
        # Get the default data config for this model
        data_config = timm.data.resolve_model_data_config(full_model)
        input_size = data_config['input_size'][1]  # Height (assuming square inputs)
        mean = data_config['mean']
        std = data_config['std']
        
        # Clean up the temporary model
        del full_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        import gc
        gc.collect()
        
        # Create preprocessing with correct input size and normalization
        preprocess = transforms.Compose([
            transforms.Resize(int(input_size * 256 / 224)),  # Scale resize accordingly
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        return preprocess

    else:
        raise ValueError(f"Unsupported model name: {model_name}. Supported models: CLIP models (clip_*), DINOv2 models (dinov2_*), TIMM models (timm/*)")
        # # Torchvision models - standard ImageNet preprocessing
        # preprocess = transforms.Compose([
        #     transforms.Resize(256),
        #     transforms.CenterCrop(224),
        #     transforms.ToTensor(),
        #     transforms.Normalize(
        #         mean=[0.485, 0.456, 0.406],
        #         std=[0.229, 0.224, 0.225]
        #     ),
        # ])
        # return preprocess

def get_backbone(model_name):
    # Get preprocessing transforms using the isolated function
    preprocess = get_preprocess(model_name)
    
    if model_name.startswith('clip'):

        # Map model_name to the appropriate format expected by openai/CLIP repository
        clip_model_name_map = {
            "clip_vitb32": "ViT_B_32",
            "clip_vitb16": "ViT_B_16",
            "clip_vitl14": "ViT_L_14",
            "clip_vitl14_336px": "ViT_L_14_336px",
            "clip_resnet50": "RN50",
            "clip_resnet101": "RN101"
        }

        # Ensure the model name exists in the map
        assert model_name in clip_model_name_map, "Unsupported CLIP model name"

        # Load the CLIP model using torch.hub
        model, _ = torch.hub.load('openai/CLIP', clip_model_name_map[model_name])
        
        # Ensure the model is in float32 to avoid type mismatches
        model = model.float()

        # Define the model feature dimensions for different CLIP models
        model_features_dim_dict = {
            "clip_vitb32": 512,
            "clip_vitb16": 512,
            "clip_vitl14": 768,
            "clip_vitl14_336px": 768,
            "clip_resnet50": 1024,
            "clip_resnet101": 512
        }

        # Get the feature dimension for the specified model
        features_dim = model_features_dim_dict[model_name]

        # Assign the loaded model to the backbone variable
        backbone = model.visual
        backbone.eval()  # Set to evaluation mode

    elif model_name.startswith('dinov2'):
        # Print available backbones
        print('DINOv2 available backbones:')
        print(torch.hub.list('facebookresearch/dinov2'))

        dinov2 = torch.hub.load("facebookresearch/dinov2", model_name)

        model_features_dim_dict = {"dinov2_vits14":384,
                                   'dinov2_vitb14':768,
                                   'dinov2_vitl14':1024,
                                   'dinov2_vitg14':1536}

        features_dim = model_features_dim_dict[model_name]
        backbone = dinov2
    elif model_name.startswith('timm/'):
        # TIMM models - use timm library for feature extraction
        import timm
        
        # Remove 'timm/' prefix for timm.create_model
        timm_model_name = model_name[5:]  # Remove 'timm/' prefix
        print(f"Loading TIMM backbone: {timm_model_name}")
        
        # Load the full model with pretrained weights
        try:   
            # Load the full model temporarily to get data config
            full_model = timm.create_model(timm_model_name, pretrained=True)
        except Exception as e:
            print(f"Warning: Could not load TIMM model {timm_model_name} directly: {e}. Trying with hf_hub:timm/{timm_model_name}")
            full_model = timm.create_model(f"hf_hub:timm/{timm_model_name}", pretrained=True)
        full_model.eval()  # Set to evaluation mode 
        #### ATENTTION!! Not doing this before using a dummy input changes the parameters
        #### as batchnorm layers statistics are updated during the forward pass if not in eval mode
        
        # Get the default data config for this model to get correct input size first
        data_config = timm.data.resolve_model_data_config(full_model)
        input_size = data_config['input_size'][1]  # Height (assuming square inputs)
        mean = data_config['mean']
        std = data_config['std']
        
        print(f"   TIMM model input size: {input_size}x{input_size}")
        print(f"   TIMM model normalization - mean: {mean}, std: {std}")
        
        # Use the new utility function to replace the final layer with Identity
        try:
            backbone = turn_final_layer_to_id(full_model, model_name)
            print(f"   Successfully converted TIMM model to backbone using turn_final_layer_to_id")
        except Exception as e:
            print(f"   Warning: Could not replace final layer with Identity for {timm_model_name}: {e}")
            raise e

        # Get the feature dimension from the model by applying backbone to a dummy input
        dummy_input = torch.randn(1, 3, input_size, input_size)  # Batch size 1
        with torch.no_grad():
            features = backbone(dummy_input)
        if features.dim() == 2:
            features_dim = features.shape[1]
        else:
            # Raise error if output is not 2D
            raise ValueError(f"Backbone {timm_model_name} did not return 2D features, got {features.dim()}D tensor")
        
        print(f"   TIMM backbone loaded with feature dim: {features_dim}")
    else:
        raise ValueError(f"Unsupported model name: {model_name}. Supported models: CLIP models (clip_*), DINOv2 models (dinov2_*), TIMM models (timm/*)")
        # # Try torchvision models
        # import torchvision.models as models
        
        # # Define feature dimensions for torchvision models
        # model_features_dim_dict = {
        #     "resnet18": 512,
        #     "resnet34": 512,
        #     "resnet50": 2048,
        #     "resnet101": 2048,
        #     "resnet152": 2048,
        #     "resnext50_32x4d": 2048,
        #     "resnext101_32x8d": 2048,
        #     "densenet121": 1024,
        #     "densenet169": 1664,
        #     "densenet201": 1920,
        #     "mobilenet_v2": 1280,
        #     "efficientnet_b0": 1280,
        #     "efficientnet_b1": 1280,
        #     "efficientnet_b2": 1408,
        #     "efficientnet_b3": 1536,
        #     "efficientnet_b4": 1792,
        #     "efficientnet_b5": 2048,
        #     "efficientnet_b6": 2304,
        #     "efficientnet_b7": 2560,
        #     "regnet_x_400mf": 400,
        #     "regnet_x_1_6gf": 672,
        #     "regnet_x_3_2gf": 1008,
        # }
        
        # if model_name not in model_features_dim_dict:
        #     raise ValueError(f"Unsupported torchvision model name: {model_name}")
        
        # features_dim = model_features_dim_dict[model_name]
        
        # # Load the pre-trained model directly for feature extraction
        # model_fn = getattr(models, model_name)
        # model = model_fn(weights='DEFAULT')  # Use 'DEFAULT' for latest weights
        
        # # Remove the classification head to get feature extractor
        # if hasattr(model, 'classifier'):
        #     # Models like EfficientNet, MobileNet
        #     backbone = torch.nn.Sequential(*list(model.children())[:-1])
        # elif hasattr(model, 'fc'):
        #     # Models like ResNet, RegNet
        #     backbone = torch.nn.Sequential(*list(model.children())[:-1])
        # elif hasattr(model, 'head'):
        #     # Some other models
        #     backbone = torch.nn.Sequential(*list(model.children())[:-1])
        # else:
        #     # Fallback: try to remove last layer
        #     backbone = torch.nn.Sequential(*list(model.children())[:-1])
        
        # # Add adaptive pooling to ensure consistent output size
        # print("Adding adaptive pooling to backbone")
        # backbone.add_module('adaptive_pool', torch.nn.AdaptiveAvgPool2d((1, 1)))
        # backbone.add_module('flatten', torch.nn.Flatten())
    backbone.model_name = model_name
    return backbone, features_dim, preprocess

def generate_embeddings(embeddings_path, labels_path, backbone, dataloader, device, model_name=None):
    backbone.to(device)
    backbone.eval()
    
    # Check if the backbone uses half precision
    backbone_dtype = next(backbone.parameters()).dtype
    
    all_embeddings = []
    all_labels = []

    # Generate embeddings, concatenate all batches and save them detached in the embeddings_path
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader):
            inputs = inputs.to(device)
            # Convert input tensor to match the backbone's dtype
            inputs = inputs.to(dtype=backbone_dtype)
            outputs = backbone(inputs)
            
            # Handle different model output formats
            if model_name and model_name.startswith('convnext'):
                # ConvNext models return BaseModelOutputWithPooling
                # Use pooler_output for global representation
                outputs = outputs.pooler_output
            # For CLIP and DINOv2, outputs are already tensors
            
            outputs = outputs.detach().cpu()
            labels = labels.detach().cpu()
            all_embeddings.append(outputs)
            all_labels.append(labels)

    # Concatenate all embeddings and labels
    all_embeddings = torch.cat(all_embeddings, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    # Save the concatenated embeddings and labels
    torch.save(all_embeddings, embeddings_path)
    torch.save(all_labels, labels_path)
    
    print(f'Embeddings saved to {embeddings_path} with shape {all_embeddings.shape}')
    print(f'Labels saved to {labels_path} with shape {all_labels.shape}')

def get_weights_file_with_latest_timestamp(dir):
    weights_files = [f for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f)) and f.endswith('.pth')]
    if not weights_files:
        return None
    weights_files.sort(key=lambda x: os.path.getmtime(os.path.join(dir, x)), reverse=True)
    return os.path.join(dir, weights_files[0])

def get_text_encoder(model_name, device):
    """
    Get a complete text encoder function that includes tokenization.
    
    Args:
        model_name: Name of the model (e.g., 'clip_vitb32', 'all-roberta-large-v1', etc.)
        device: Device to load the model on
        
    Returns:
        Function that takes a list of text strings and returns embeddings
    """
    if model_name.startswith('clip'):
        # Get the raw CLIP text encoder and tokenizer
        clip_text_encoder = get_clip_text_encoder(model_name, device)
        clip_tokenizer = get_clip_tokenizer()
        
        def complete_text_encoder(texts):
            """
            Complete CLIP text encoding function with tokenization.
            
            Args:
                texts: List of text strings or single text string
                
            Returns:
                Text embeddings tensor
            """
            # Ensure texts is a list
            if isinstance(texts, str):
                texts = [texts]
            
            # Tokenize and encode
            tokens = clip_tokenizer(texts, truncate=True).to(device)
            with torch.no_grad():
                embeddings = clip_text_encoder(tokens)
            return embeddings
        
        return complete_text_encoder
    
    elif model_name == 'all-roberta-large-coco-contrastive':
        # Fine-tuned model from local path
        try:
            from sentence_transformers import SentenceTransformer
            from pathlib import Path
        except ImportError:
            raise ImportError("sentence-transformers is required for this model. Install with: pip install sentence-transformers")
        
        # Load the fine-tuned model from local path
        model_path = "finetuned_text/all-roberta-large-coco-contrastive"
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Fine-tuned model not found at {model_path}. Please ensure the model has been trained and saved.")
        
        st_model = SentenceTransformer(model_path)
        st_model.eval()  # Set to evaluation mode
        st_model.to(device)
        
        def complete_text_encoder(texts):
            """
            Complete fine-tuned Sentence Transformers text encoding function.
            
            Args:
                texts: List of text strings or single text string
                
            Returns:
                Text embeddings tensor
            """
            # Ensure texts is a list
            if isinstance(texts, str):
                texts = [texts]
            
            # Encode using fine-tuned sentence transformers
            with torch.no_grad():
                embeddings = st_model.encode(texts, convert_to_tensor=True, device=device)
            return embeddings
        
        return complete_text_encoder
    
    elif model_name.startswith('all-') or model_name in ['all-roberta-large-v1', 'all-mpnet-base-v2', 'all-MiniLM-L6-v2']:
        # Sentence Transformers models
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError("sentence-transformers is required for this model. Install with: pip install sentence-transformers")
        
        # Load the sentence transformer model
        st_model = SentenceTransformer(model_name)
        st_model.eval()  # Set to evaluation mode
        st_model.to(device)
        
        def complete_text_encoder(texts):
            """
            Complete Sentence Transformers text encoding function.
            
            Args:
                texts: List of text strings or single text string
                
            Returns:
                Text embeddings tensor
            """
            # Ensure texts is a list
            if isinstance(texts, str):
                texts = [texts]
            
            # Encode using sentence transformers (handles tokenization internally)
            with torch.no_grad():
                embeddings = st_model.encode(texts, convert_to_tensor=True, device=device)
            return embeddings
        
        return complete_text_encoder
    
    elif model_name in ['sup-simcse-bert-base-uncased', 'unsup-simcse-bert-base-uncased']:
        # SimCSE models
        try:
            from transformers import AutoTokenizer, AutoModel
        except ImportError:
            raise ImportError("transformers is required for SimCSE models. Install with: pip install transformers")
        
        # Map model names to HuggingFace model identifiers
        model_mapping = {
            'sup-simcse-bert-base-uncased': 'princeton-nlp/sup-simcse-bert-base-uncased',
            'unsup-simcse-bert-base-uncased': 'princeton-nlp/unsup-simcse-bert-base-uncased'
        }
        
        hf_model_name = model_mapping[model_name]
        
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
        model = AutoModel.from_pretrained(hf_model_name)
        model.to(device)
        model.eval()
        
        def complete_text_encoder(texts):
            """
            Complete SimCSE text encoding function with tokenization.
            
            Args:
                texts: List of text strings or single text string
                
            Returns:
                Text embeddings tensor (L2 normalized)
            """
            # Ensure texts is a list
            if isinstance(texts, str):
                texts = [texts]
            
            # Tokenize
            encoded = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
            encoded = {k: v.to(device) for k, v in encoded.items()}
            
            # Forward pass (no gradient needed)
            with torch.no_grad():
                outputs = model(**encoded, output_hidden_states=False, return_dict=True)
                # Use pooler_output as sentence embedding (BERT)
                embeddings = outputs.pooler_output  # shape: (batch_size, hidden_dim)
            
            # Normalize embeddings (L2) for cosine similarity
            embeddings = F.normalize(embeddings, p=2, dim=1)
            return embeddings
        
        return complete_text_encoder
    
    # Add other text encoders here in the future
    # elif model_name.startswith('bert'):
    #     return get_bert_complete_text_encoder(model_name, device)
    # elif model_name.startswith('t5'):
    #     return get_t5_complete_text_encoder(model_name, device)
    else:
        raise ValueError(f"Unsupported text encoder model: {model_name}. Supported models: CLIP models (clip_*), Sentence Transformers (all-*), SimCSE models (sup-simcse-bert-base-uncased, unsup-simcse-bert-base-uncased)")

def get_tokenizer(model_name):
    """
    Get a tokenizer based on the model name.
    
    Args:
        model_name: Name of the model (e.g., 'clip_vitb32', 'bert', etc.)
        
    Returns:
        Tokenizer function
    """
    if model_name.startswith('clip'):
        return get_clip_tokenizer()
    # Add other tokenizers here in the future
    # elif model_name.startswith('bert'):
    #     return get_bert_tokenizer(model_name)
    # elif model_name.startswith('t5'):
    #     return get_t5_tokenizer(model_name)
    else:
        raise ValueError(f"Unsupported tokenizer model: {model_name}")


def get_definitions_from_wnids(ids_file):
    from nltk.corpus import wordnet as wn
    import nltk
    # Download wordnet if not already present
    try:
        wn.synsets('dog')
    except LookupError:
        nltk.download('wordnet')
        nltk.download('omw-1.4')

    """
    Get definitions from WordNet IDs.
    Args:
        ids_file (str): Path to the file containing WordNet IDs (e.g., wnid_*.txt)
    Returns:
        List of definitions corresponding to the WordNet IDs
    """

    # Load the IDs and lemmas
    with open(ids_file, 'r') as f:
        wordnet_ids = [line.strip() for line in f.readlines()]

    definitions = []
    for wordnet_id in wordnet_ids:
        try:
            synset = wn.synset_from_pos_and_offset('n', int(wordnet_id[1:]))
            definitions.append(synset.definition())
        except Exception as e:
            print(f"Error processing ID {wordnet_id}: {e}")

    return definitions

def get_imagenet1k_names_hypernyms(ids_file):
    """
    Get ImageNet-1k class names with their lowest common hypernyms.
    
    Args:
        ids_file (str): Path to the file containing WordNet IDs (e.g., wnid_imagenet1k.txt)
        
    Returns:
        List of strings combining class names with their hypernyms
    """
    
    # Download wordnet if not already present
    from nltk.corpus import wordnet as wn    
    try:
        wn.synsets('dog')
    except LookupError:
        import nltk
        nltk.download('wordnet')
        nltk.download('omw-1.4')
    
    # Load the WordNet IDs
    with open(ids_file, 'r') as f:
        wordnet_ids = [line.strip() for line in f.readlines()]
    
    names_with_hypernyms = []
    
    for wordnet_id in wordnet_ids:
        try:
            # Get the synset from WordNet ID
            synset = wn.synset_from_pos_and_offset('n', int(wordnet_id[1:]))
            
            # Get the class name (lemma)
            class_name = synset.lemmas()[0].name().replace('_', ' ')
            
            # Get hypernyms
            hypernyms = synset.hypernyms()
            
            if hypernyms:
                # Get the most specific hypernym (lowest common hypernym)
                hypernym = hypernyms[0]
                hypernym_name = hypernym.lemmas()[0].name().replace('_', ' ')
                
                # Concatenate class name with hypernym
                combined_name = f"{class_name} {hypernym_name}"
            else:
                # If no hypernyms, just use the class name
                combined_name = class_name
            
            names_with_hypernyms.append(combined_name)
            
        except Exception as e:
            print(f"Error processing ID {wordnet_id}: {e}")
            # Fallback to just the ID if processing fails
            names_with_hypernyms.append(wordnet_id)
    
    return names_with_hypernyms

def get_imagenet1k_alternative_hypernym_prompts(ids_file):
    """
    Get ImageNet-1k class names with alternative hypernym-based prompts.
    This creates different prompt variations using WordNet hypernyms.
    
    Args:
        ids_file (str): Path to the file containing WordNet IDs (e.g., wnid_imagenet1k.txt)
        
    Returns:
        List of strings with alternative hypernym-based prompts
    """
    
    # Download wordnet if not already present
    from nltk.corpus import wordnet as wn    
    try:
        wn.synsets('dog')
    except LookupError:
        import nltk
        nltk.download('wordnet')
        nltk.download('omw-1.4')
    
    # Load the WordNet IDs
    with open(ids_file, 'r') as f:
        wordnet_ids = [line.strip() for line in f.readlines()]
    
    alternative_prompts = []
    
    for wordnet_id in wordnet_ids:
        try:
            # Get the synset from WordNet ID
            synset = wn.synset_from_pos_and_offset('n', int(wordnet_id[1:]))
            
            # Get the class name (lemma)
            class_name = synset.lemmas()[0].name().replace('_', ' ')
            
            # Get all hypernyms (going up the hierarchy)
            all_hypernyms = []
            current_synset = synset
            while current_synset.hypernyms():
                hypernym = current_synset.hypernyms()[0]
                hypernym_name = hypernym.lemmas()[0].name().replace('_', ' ')
                all_hypernyms.append(hypernym_name)
                current_synset = hypernym
                # Stop after getting 3 levels to avoid going too abstract
                if len(all_hypernyms) >= 3:
                    break
            
            # Create alternative prompt patterns
            if len(all_hypernyms) >= 2:
                # Use second-level hypernym for more general categorization
                prompt = f"a type of {all_hypernyms[1]}, specifically a {class_name}"
            elif len(all_hypernyms) >= 1:
                # Use first-level hypernym
                prompt = f"a {class_name}, which is a {all_hypernyms[0]}"
            else:
                # Fallback to just the class name with article
                prompt = f"a {class_name}"
            
            alternative_prompts.append(prompt)
            
        except Exception as e:
            print(f"Error processing ID {wordnet_id}: {e}")
            # Fallback to a generic prompt
            alternative_prompts.append(f"a {wordnet_id}")
    
    return alternative_prompts

def get_imagenet1k_synonym_prompts(ids_file):
    """
    Get ImageNet-1k class names with their WordNet synonyms.
    This creates prompt variations using synonyms from WordNet.
    
    Args:
        ids_file (str): Path to the file containing WordNet IDs (e.g., wnid_imagenet1k.txt)
        
    Returns:
        List of strings with synonym-based prompts
    """
    
    # Download wordnet if not already present
    from nltk.corpus import wordnet as wn    
    try:
        wn.synsets('dog')
    except LookupError:
        import nltk
        nltk.download('wordnet')
        nltk.download('omw-1.4')
    
    # Load the WordNet IDs
    with open(ids_file, 'r') as f:
        wordnet_ids = [line.strip() for line in f.readlines()]
    
    synonym_prompts = []
    
    for wordnet_id in wordnet_ids:
        try:
            # Get the synset from WordNet ID
            synset = wn.synset_from_pos_and_offset('n', int(wordnet_id[1:]))
            
            # Get all lemmas (synonyms) in this synset
            lemmas = synset.lemmas()
            
            # Convert lemmas to readable names
            synonyms = []
            for lemma in lemmas:
                synonym = lemma.name().replace('_', ' ')
                synonyms.append(synonym)
            
            # If we have multiple synonyms, pick the second one (first is usually the original)
            # Otherwise, use the original class name
            if len(synonyms) > 1:
                chosen_synonym = synonyms[1]  # Use second synonym
            else:
                chosen_synonym = synonyms[0]  # Use original if no alternatives
            
            synonym_prompts.append(chosen_synonym)
            
        except Exception as e:
            print(f"Error processing ID {wordnet_id}: {e}")
            # Fallback to the wordnet_id if processing fails
            synonym_prompts.append(wordnet_id)
    
    return synonym_prompts

def get_template_for_dataset(dataset_name):
    """
    Get the appropriate template for creating prompts based on the dataset.
    
    Args:
        dataset_name (str): Name of the dataset
        
    Returns:
        str: Template string with {} placeholder for class name
    """
    if dataset_name == "flowers102":
        return "A photo of a {}"
    elif dataset_name == "food101":
        return "A photo of a {}"
    elif dataset_name == "eurosat":
        return "A satellite photo of a {}"
    elif dataset_name == "resisc45":
        return "A satellite photo of a {}"
    elif dataset_name == "mnist":
        return "A photo of {}"
    elif dataset_name == "oxfordpets":
        # For pets, we need to determine if it's a cat or dog breed
        # This will need to be handled specially in the calling function
        return "A photo of a {}"
    elif dataset_name == "dtd":
        return "A photo of a {}"
    elif dataset_name == "places365":
        return "A scene photo of a {}"
    elif dataset_name.startswith("cifar10") or dataset_name.startswith("imagenet"):
        # Default template for general image classification datasets
        return "A photo of a {}."
    else:
        # Default template for unknown datasets
        return "A photo of a {}."

def cleanup_dataset_embeddings(dataset_name: str, model_name: str, embeddings_dir=None) -> None:
    """
    Clean up embeddings for a specific dataset from the embeddings directory.
    Args:
        dataset_name: Name of the dataset (e.g., 'cifar10', 'cifar100')
        embeddings_dir: Optionally override the embeddings directory
    """
    if embeddings_dir is None:
        from config.config import EMBEDDINGS_DIR
        embeddings_dir = EMBEDDINGS_DIR
    print(f"🧹 Cleaning up embeddings for {dataset_name} and model {model_name}...")
    if not os.path.exists(embeddings_dir):
        print(f"   Embeddings directory not found: {embeddings_dir}")
        return
    removed_files = 0
    for filename in os.listdir(embeddings_dir):
        if filename.startswith(f"{dataset_name}_{model_name}") and filename.endswith('.pt'):
            file_path = os.path.join(embeddings_dir, filename)
            try:
                os.remove(file_path)
                removed_files += 1
                print(f"   Removed: {filename}")
            except OSError as e:
                print(f"   Failed to remove {filename}: {e}")
    if removed_files == 0:
        print(f"   No embedding files found for {dataset_name}  with model {model_name}")
    else:
        print(f"   Successfully removed {removed_files} embedding files")

def cleanup_all_embeddings(embeddings_dir=None) -> None:
    """
    Clean up all embeddings from the embeddings directory.
    Args:
        embeddings_dir: Optionally override the embeddings directory
    """
    if embeddings_dir is None:
        from config.config import EMBEDDINGS_DIR
        embeddings_dir = EMBEDDINGS_DIR
    print("🧹 Cleaning up all embeddings...")
    if not os.path.exists(embeddings_dir):
        print(f"   Embeddings directory not found: {embeddings_dir}")
        return
    import shutil
    try:
        shutil.rmtree(embeddings_dir)
        os.makedirs(embeddings_dir, exist_ok=True)
        print(f"   Successfully cleaned embeddings directory: {embeddings_dir}")
    except OSError as e:
        print(f"   Failed to clean embeddings directory: {e}")

def get_imagenet21k_names_hypernyms(ids_file):
    """
    Get ImageNet-21k class names with their lowest common hypernyms.
    
    Args:
        ids_file (str): Path to the file containing WordNet IDs (e.g., wnid_imagenet21k.txt)
        
    Returns:
        List of strings combining class names with their hypernyms
    """
    
    # Download wordnet if not already present
    from nltk.corpus import wordnet as wn    
    try:
        wn.synsets('dog')
    except LookupError:
        import nltk
        nltk.download('wordnet')
        nltk.download('omw-1.4')
    
    # Load the WordNet IDs
    with open(ids_file, 'r') as f:
        wordnet_ids = [line.strip() for line in f.readlines()]
    
    names_with_hypernyms = []
    
    for wordnet_id in wordnet_ids:
        try:
            # Get the synset from WordNet ID
            synset = wn.synset_from_pos_and_offset('n', int(wordnet_id[1:]))
            
            # Get the class name (lemma)
            class_name = synset.lemmas()[0].name().replace('_', ' ')
            
            # Get hypernyms
            hypernyms = synset.hypernyms()
            
            if hypernyms:
                # Get the most specific hypernym (lowest common hypernym)
                hypernym = hypernyms[0]
                hypernym_name = hypernym.lemmas()[0].name().replace('_', ' ')
                
                # Concatenate class name with hypernym
                combined_name = f"{class_name} {hypernym_name}"
            else:
                # If no hypernyms, just use the class name
                combined_name = class_name
            
            names_with_hypernyms.append(combined_name)
            
        except Exception as e:
            print(f"Error processing ID {wordnet_id}: {e}")
            # Fallback to just the ID if processing fails
            names_with_hypernyms.append(wordnet_id)
    
    return names_with_hypernyms

def get_imagenet21k_alternative_hypernym_prompts(ids_file):
    """
    Get ImageNet-21k class names with alternative hypernym-based prompts.
    This creates different prompt variations using WordNet hypernyms.
    
    Args:
        ids_file (str): Path to the file containing WordNet IDs (e.g., wnid_imagenet21k.txt)
        
    Returns:
        List of strings with alternative hypernym-based prompts
    """
    
    # Download wordnet if not already present
    from nltk.corpus import wordnet as wn    
    try:
        wn.synsets('dog')
    except LookupError:
        import nltk
        nltk.download('wordnet')
        nltk.download('omw-1.4')
    
    # Load the WordNet IDs
    with open(ids_file, 'r') as f:
        wordnet_ids = [line.strip() for line in f.readlines()]
    
    alternative_prompts = []
    
    for wordnet_id in wordnet_ids:
        try:
            # Get the synset from WordNet ID
            synset = wn.synset_from_pos_and_offset('n', int(wordnet_id[1:]))
            
            # Get the class name (lemma)
            class_name = synset.lemmas()[0].name().replace('_', ' ')
            
            # Get all hypernyms (going up the hierarchy)
            all_hypernyms = []
            current_synset = synset
            while current_synset.hypernyms():
                hypernym = current_synset.hypernyms()[0]
                hypernym_name = hypernym.lemmas()[0].name().replace('_', ' ')
                all_hypernyms.append(hypernym_name)
                current_synset = hypernym
                # Stop after getting 3 levels to avoid going too abstract
                if len(all_hypernyms) >= 3:
                    break
            
            # Create alternative prompt patterns
            if len(all_hypernyms) >= 2:
                # Use second-level hypernym for more general categorization
                prompt = f"a type of {all_hypernyms[1]}, specifically a {class_name}"
            elif len(all_hypernyms) >= 1:
                # Use first-level hypernym
                prompt = f"a {class_name}, which is a {all_hypernyms[0]}"
            else:
                # Fallback to just the class name with article
                prompt = f"a {class_name}"
            
            alternative_prompts.append(prompt)
            
        except Exception as e:
            print(f"Error processing ID {wordnet_id}: {e}")
            # Fallback to a generic prompt
            alternative_prompts.append(f"a {wordnet_id}")
    
    return alternative_prompts

def get_imagenet21k_synonym_prompts(ids_file):
    """
    Get ImageNet-21k class names with their WordNet synonyms.
    This creates prompt variations using synonyms from WordNet.
    
    Args:
        ids_file (str): Path to the file containing WordNet IDs (e.g., wnid_imagenet21k.txt)
        
    Returns:
        List of strings with synonym-based prompts
    """
    
    # Download wordnet if not already present
    from nltk.corpus import wordnet as wn    
    try:
        wn.synsets('dog')
    except LookupError:
        import nltk
        nltk.download('wordnet')
        nltk.download('omw-1.4')
    
    # Load the WordNet IDs
    with open(ids_file, 'r') as f:
        wordnet_ids = [line.strip() for line in f.readlines()]
    
    synonym_prompts = []
    
    for wordnet_id in wordnet_ids:
        try:
            # Get the synset from WordNet ID
            synset = wn.synset_from_pos_and_offset('n', int(wordnet_id[1:]))
            
            # Get all lemmas (synonyms) in this synset
            lemmas = synset.lemmas()
            
            # Convert lemmas to readable names
            synonyms = []
            for lemma in lemmas:
                synonym = lemma.name().replace('_', ' ')
                synonyms.append(synonym)
            
            # If we have multiple synonyms, pick the second one (first is usually the original)
            # Otherwise, use the original class name
            if len(synonyms) > 1:
                chosen_synonym = synonyms[1]  # Use second synonym
            else:
                chosen_synonym = synonyms[0]  # Use original if no alternatives
            
            synonym_prompts.append(chosen_synonym)
            
        except Exception as e:
            print(f"Error processing ID {wordnet_id}: {e}")
            # Fallback to the wordnet_id if processing fails
            synonym_prompts.append(wordnet_id)
    
    return synonym_prompts

def get_imagenet21k_2extra_names_hypernyms(ids_file):
    """
    Get ImageNet-21k 2extra class names with their lowest common hypernyms.
    
    Args:
        ids_file (str): Path to the file containing WordNet IDs (e.g., wnid_imagenet21k_2extra.txt)
        
    Returns:
        List of strings combining class names with their hypernyms
    """
    
    # Download wordnet if not already present
    from nltk.corpus import wordnet as wn    
    try:
        wn.synsets('dog')
    except LookupError:
        import nltk
        nltk.download('wordnet')
        nltk.download('omw-1.4')
    
    # Load the WordNet IDs
    with open(ids_file, 'r') as f:
        wordnet_ids = [line.strip() for line in f.readlines()]
    
    names_with_hypernyms = []
    
    for wordnet_id in wordnet_ids:
        try:
            # Get the synset from WordNet ID
            synset = wn.synset_from_pos_and_offset('n', int(wordnet_id[1:]))
            
            # Get the class name (lemma)
            class_name = synset.lemmas()[0].name().replace('_', ' ')
            
            # Get hypernyms
            hypernyms = synset.hypernyms()
            
            if hypernyms:
                # Get the most specific hypernym (lowest common hypernym)
                hypernym = hypernyms[0]
                hypernym_name = hypernym.lemmas()[0].name().replace('_', ' ')
                
                # Concatenate class name with hypernym
                combined_name = f"{class_name} {hypernym_name}"
            else:
                # If no hypernyms, just use the class name
                combined_name = class_name
            
            names_with_hypernyms.append(combined_name)
            
        except Exception as e:
            print(f"Error processing ID {wordnet_id}: {e}")
            # Fallback to just the ID if processing fails
            names_with_hypernyms.append(wordnet_id)
    
    return names_with_hypernyms

def get_imagenet21k_2extra_alternative_hypernym_prompts(ids_file):
    """
    Get ImageNet-21k 2extra class names with alternative hypernym-based prompts.
    This creates different prompt variations using WordNet hypernyms.
    
    Args:
        ids_file (str): Path to the file containing WordNet IDs (e.g., wnid_imagenet21k_2extra.txt)
        
    Returns:
        List of strings with alternative hypernym-based prompts
    """
    
    # Download wordnet if not already present
    from nltk.corpus import wordnet as wn    
    try:
        wn.synsets('dog')
    except LookupError:
        import nltk
        nltk.download('wordnet')
        nltk.download('omw-1.4')
    
    # Load the WordNet IDs
    with open(ids_file, 'r') as f:
        wordnet_ids = [line.strip() for line in f.readlines()]
    
    alternative_prompts = []
    
    for wordnet_id in wordnet_ids:
        try:
            # Get the synset from WordNet ID
            synset = wn.synset_from_pos_and_offset('n', int(wordnet_id[1:]))
            
            # Get the class name (lemma)
            class_name = synset.lemmas()[0].name().replace('_', ' ')
            
            # Get all hypernyms (going up the hierarchy)
            all_hypernyms = []
            current_synset = synset
            while current_synset.hypernyms():
                hypernym = current_synset.hypernyms()[0]
                hypernym_name = hypernym.lemmas()[0].name().replace('_', ' ')
                all_hypernyms.append(hypernym_name)
                current_synset = hypernym
                # Stop after getting 3 levels to avoid going too abstract
                if len(all_hypernyms) >= 3:
                    break
            
            # Create alternative prompt patterns
            if len(all_hypernyms) >= 2:
                # Use second-level hypernym for more general categorization
                prompt = f"a type of {all_hypernyms[1]}, specifically a {class_name}"
            elif len(all_hypernyms) >= 1:
                # Use first-level hypernym
                prompt = f"a {class_name}, which is a {all_hypernyms[0]}"
            else:
                # Fallback to just the class name with article
                prompt = f"a {class_name}"
            
            alternative_prompts.append(prompt)
            
        except Exception as e:
            print(f"Error processing ID {wordnet_id}: {e}")
            # Fallback to a generic prompt
            alternative_prompts.append(f"a {wordnet_id}")
    
    return alternative_prompts

def get_imagenet21k_2extra_synonym_prompts(ids_file):
    """
    Get ImageNet-21k 2extra class names with their WordNet synonyms.
    This creates prompt variations using synonyms from WordNet.
    
    Args:
        ids_file (str): Path to the file containing WordNet IDs (e.g., wnid_imagenet21k_2extra.txt)
        
    Returns:
        List of strings with synonym-based prompts
    """
    
    # Download wordnet if not already present
    from nltk.corpus import wordnet as wn    
    try:
        wn.synsets('dog')
    except LookupError:
        import nltk
        nltk.download('wordnet')
        nltk.download('omw-1.4')
    
    # Load the WordNet IDs
    with open(ids_file, 'r') as f:
        wordnet_ids = [line.strip() for line in f.readlines()]
    
    synonym_prompts = []
    
    for wordnet_id in wordnet_ids:
        try:
            # Get the synset from WordNet ID
            synset = wn.synset_from_pos_and_offset('n', int(wordnet_id[1:]))
            
            # Get all lemmas (synonyms) in this synset
            lemmas = synset.lemmas()
            
            # Convert lemmas to readable names
            synonyms = []
            for lemma in lemmas:
                synonym = lemma.name().replace('_', ' ')
                synonyms.append(synonym)
            
            # If we have multiple synonyms, pick the second one (first is usually the original)
            # Otherwise, use the original class name
            if len(synonyms) > 1:
                chosen_synonym = synonyms[1]  # Use second synonym
            else:
                chosen_synonym = synonyms[0]  # Use original if no alternatives
            
            synonym_prompts.append(chosen_synonym)
            
        except Exception as e:
            print(f"Error processing ID {wordnet_id}: {e}")
            # Fallback to the wordnet_id if processing fails
            synonym_prompts.append(wordnet_id)
    
    return synonym_prompts

def get_augmented_class_names(dataset_name, augmentation_type="synonym", data_dir=None):
    """
    Get augmented class names for a dataset using WordNet-based augmentation.
    
    Args:
        dataset_name (str): Name of the dataset (e.g., 'imagenet1k', 'imagenet21k', 'imagenet21k_2extra')
        augmentation_type (str): Type of augmentation ('synonym', 'hypernym', 'alternative_hypernym')
        data_dir (str): Optional path to data directory. If None, uses default structure.
        
    Returns:
        List of augmented class names/prompts
    """
    
    # Set default data directory if not provided
    if data_dir is None:
        data_dir = DATA_DIR
    
    # Define the WordNet ID files for different datasets
    id_files = {
        "imagenet1k": os.path.join(data_dir, "imagenet1k", "imagenet1k_wnids.txt"),
        "imagenet21k": os.path.join(data_dir, "imagenet21k_wnids.txt"),
        "imagenet21k_2extra": os.path.join(data_dir, "imagenet21k_2extra_wnids.txt"),
    }
    
    # Check if dataset is supported
    if dataset_name not in id_files:
        raise ValueError(f"Dataset '{dataset_name}' not supported for augmentation. Supported datasets: {list(id_files.keys())}")
    
    # Check if the WordNet ID file exists
    ids_file = id_files[dataset_name]
    if not os.path.exists(ids_file):
        raise FileNotFoundError(f"WordNet IDs file not found: {ids_file}")
    
    # Choose the appropriate augmentation function based on dataset and type
    if dataset_name == "imagenet1k":
        if augmentation_type == "synonym":
            return get_imagenet1k_synonym_prompts(ids_file)
        elif augmentation_type == "hypernym":
            return get_imagenet1k_names_hypernyms(ids_file)
        elif augmentation_type == "alternative_hypernym":
            return get_imagenet1k_alternative_hypernym_prompts(ids_file)
        else:
            raise ValueError(f"Unsupported augmentation type '{augmentation_type}' for {dataset_name}")
    
    elif dataset_name == "imagenet21k":
        if augmentation_type == "synonym":
            return get_imagenet21k_synonym_prompts(ids_file)
        elif augmentation_type == "hypernym":
            return get_imagenet21k_names_hypernyms(ids_file)
        elif augmentation_type == "alternative_hypernym":
            return get_imagenet21k_alternative_hypernym_prompts(ids_file)
        else:
            raise ValueError(f"Unsupported augmentation type '{augmentation_type}' for {dataset_name}")
    
    elif dataset_name == "imagenet21k_2extra":
        if augmentation_type == "synonym":
            return get_imagenet21k_2extra_synonym_prompts(ids_file)
        elif augmentation_type == "hypernym":
            return get_imagenet21k_2extra_names_hypernyms(ids_file)
        elif augmentation_type == "alternative_hypernym":
            return get_imagenet21k_2extra_alternative_hypernym_prompts(ids_file)
        else:
            raise ValueError(f"Unsupported augmentation type '{augmentation_type}' for {dataset_name}")
    
    else:
        raise ValueError(f"Unsupported dataset '{dataset_name}' for augmentation")


def compute_and_save_mean_representation(image_model_name, device='cuda'):
    """
    Compute the mean representation of all imagenet1kval samples and save it.
    
    Args:
        image_model_name: Name of the image model
        device: Device to use for computation
        
    Returns:
        mean_repr: Mean representation tensor
        mean_repr_path: Path where the mean representation was saved
    """
    from config.config import EMBEDDINGS_DIR
    from dataloaders.datasets_and_dataloaders import get_dataloaders
    
    print("🔄 Computing mean representation for imagenet1kval dataset...")
    
    backbone, image_embedding_dim, _ = get_backbone(image_model_name)
    backbone.to(device)
    backbone.eval()
    
    preprocess = get_preprocess(image_model_name)
    _, _, loader = get_dataloaders("imagenet1kval", 128, preprocess, only_test=True)
    
    all_features = []
    with torch.no_grad():
        for images, _ in tqdm(loader, desc="Computing mean representation"):
            images = images.to(device)
            feats = backbone(images)
            feats = F.normalize(feats, p=2, dim=-1)
            all_features.append(feats.cpu())
    
    all_features = torch.cat(all_features, dim=0)
    mean_repr = all_features.mean(dim=0, keepdim=True)
    
    print(f"Mean representation computed with shape: {mean_repr.shape}")
    
    # Save the mean representation
    os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
    mean_repr_name = f"imagenet1kval_reprmean_{image_model_name.replace('/', '_')}.pt"
    mean_repr_path = os.path.join(EMBEDDINGS_DIR, mean_repr_name)
    torch.save(mean_repr, mean_repr_path)
    print(f"✅ Mean representation saved to: {mean_repr_path}")
    
    return mean_repr.to(device), mean_repr_path


def apply_attention_based_preprocessing(weight_vectors, image_reprs_avg, device='cuda'):
    """
    Apply attention-based preprocessing to weight vectors.
    
    For each weight vector:
    1. Compute attention scores: a = softmax(w · R^T) where R is average image representations
    2. Compute attended representation: w' = a · R (weighted sum of image representations)
    3. Normalize: w_final = w' / ||w'||
    
    This is done in matrix form for all vectors simultaneously.
    
    Args:
        weight_vectors: Tensor of shape (num_classes, embedding_dim)
        image_reprs_avg: Tensor of shape (num_classes, embedding_dim) with average image representations per class
        device: Device to use for computation
        
    Returns:
        preprocessed_weights: Tensor of shape (num_classes, embedding_dim), normalized to unit hypersphere
    """
    print("🔄 Applying attention-based preprocessing to weight vectors...")
    
    weight_vectors = weight_vectors.to(device).float()
    image_reprs_avg = image_reprs_avg.to(device).float()
    
    # Compute attention scores: A = softmax(W · R^T)
    # W shape: (num_classes, embedding_dim)
    # R^T shape: (embedding_dim, num_classes)
    # A shape: (num_classes, num_classes)
    attention_scores = torch.mm(weight_vectors, image_reprs_avg.t())  # (num_classes, num_classes)
    attention_scores = F.softmax(attention_scores, dim=1)  # Softmax over image representations
    
    print(f"  Attention scores shape: {attention_scores.shape}")
    print(f"  Attention scores mean: {attention_scores.mean():.6f}, std: {attention_scores.std():.6f}")
    
    # Compute attended representation: W' = A · R
    # A shape: (num_classes, num_classes)
    # R shape: (num_classes, embedding_dim)
    # W' shape: (num_classes, embedding_dim)
    preprocessed_weights = torch.mm(attention_scores, image_reprs_avg)  # (num_classes, embedding_dim)
    
    print(f"  Preprocessed weights before normalization - shape: {preprocessed_weights.shape}")
    print(f"  Preprocessed weights norm - mean: {preprocessed_weights.norm(dim=1).mean():.6f}, std: {preprocessed_weights.norm(dim=1).std():.6f}")
    
    # Normalize to unit hypersphere
    preprocessed_weights = F.normalize(preprocessed_weights, p=2, dim=-1)
    
    print(f"  Preprocessed weights after normalization - shape: {preprocessed_weights.shape}")
    print(f"  Preprocessed weights norm - mean: {preprocessed_weights.norm(dim=1).mean():.6f}, std: {preprocessed_weights.norm(dim=1).std():.6f}")
    
    print(f"  Applied transformation: softmax(W · R^T) · R")
    
    return preprocessed_weights


def train_linear_projection(weight_vectors, image_reprs_avg, num_epochs=1000, learning_rate=0.01, 
                           weight_decay=0.1, dropout_rate=0.0, device='cuda', verbose=True):
    """
    Train a lightweight linear projection to map weight vectors into image representation space.
    
    This function trains a linear layer that maps weight vectors to image representation space
    using MSE loss with regularization (weight decay and optional dropout).
    The goal is to create a learned transformation that aligns weights with image representations.
    
    Args:
        weight_vectors: Tensor of shape (num_classes, weight_dim) with weight vectors
        image_reprs_avg: Tensor of shape (num_classes, image_dim) with average image representations per class
        num_epochs: Number of training epochs (default: 1000)
        learning_rate: Learning rate for the optimizer (default: 0.01)
        weight_decay: L2 regularization coefficient for weight decay (default: 0.1)
        dropout_rate: Dropout probability for regularization (default: 0.0, no dropout)
        device: Device to use for computation (default: 'cuda')
        verbose: Whether to print training progress (default: True)
        
    Returns:
        linear_layer: Trained linear layer (nn.Linear module)
        training_loss: List of loss values per epoch
    """
    weight_vectors = weight_vectors.to(device).float()
    image_reprs_avg = image_reprs_avg.to(device).float()
    
    # Normalize targets for better training - match normalized space rather than raw magnitudes
    image_reprs_avg_normalized = F.normalize(image_reprs_avg, p=2, dim=-1)
    
    weight_dim = weight_vectors.shape[1]
    image_dim = image_reprs_avg.shape[1]
    
    # Create a simple linear layer (with optional dropout) to map from weight space to image space
    # For a single linear layer, dropout is less effective than weight decay
    # We apply it AFTER the linear layer (output) rather than before
    print(f'************** Dropout rate: {dropout_rate} ****************')
    if dropout_rate > 0:
        linear_layer = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(weight_dim, image_dim)
        ).to(device)
    else:
        linear_layer = nn.Linear(weight_dim, image_dim).to(device)
    
    # Loss function and optimizer with weight decay for regularization
    criterion = nn.MSELoss()
    optimizer = optim.Adam(linear_layer.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    if verbose:
        print(f"🔄 Training linear projection: {weight_dim} -> {image_dim}")
        print(f"  Num epochs: {num_epochs}, Learning rate: {learning_rate}, Weight decay: {weight_decay}, Dropout: {dropout_rate}")
        print(f"  Network architecture: {linear_layer}")
    
    linear_layer.train()
    training_loss = []
    best_loss = float('inf')
    best_epoch = 0
    
    # Debug: check that dropout is actually in the model
    initial_weights = None
    if dropout_rate > 0:
        has_dropout = any(isinstance(m, nn.Dropout) for m in linear_layer.modules())
        if verbose:
            print(f"  ✓ Dropout layer present in model: {has_dropout}")
        
        # Store initial weights to check if they're being updated
        if isinstance(linear_layer, nn.Sequential):
            initial_weights = linear_layer[1].weight.data.clone()  # Get the linear layer (index 1)
        else:
            initial_weights = linear_layer.weight.data.clone()
    
    for epoch in range(num_epochs):
        # Forward pass
        predicted_image_reprs = linear_layer(weight_vectors)
        
        # Compute base loss on normalized targets for better training signal
        loss = criterion(predicted_image_reprs, image_reprs_avg_normalized)
        
        # Debug output for first few epochs with high dropout
        if dropout_rate >= 0.5 and epoch < 3 and initial_weights is not None:
            mean_pred = predicted_image_reprs.abs().mean().item()
            mean_target = image_reprs_avg.abs().mean().item()
            
            # Check weight changes
            if isinstance(linear_layer, nn.Sequential):
                current_weights = linear_layer[1].weight.data  # Get the linear layer (index 1)
            else:
                current_weights = linear_layer.weight.data
            weight_change = (current_weights - initial_weights).abs().mean().item()
            
            if verbose:
                print(f"    [Debug Epoch {epoch+1}] Mean |predicted|: {mean_pred:.6f}, Mean |target|: {mean_target:.6f}, Weight change: {weight_change:.9f}, Loss: {loss.item():.6f}")
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        training_loss.append(loss.item())
        
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_epoch = epoch + 1
        
        if verbose and (epoch + 1) % 100 == 0:
            print(f"  Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.6f}")
            # print summary of weights
            if isinstance(linear_layer, nn.Sequential):
                weights = linear_layer[1].weight.data  # Get the linear layer (index 1)
            else:
                weights = linear_layer.weight.data
            print(f"    Weights - mean: {weights.mean().item():.6f}, std: {weights.std().item():.6f}")
    
    if verbose:
        print(f"  Training completed. Best loss: {best_loss:.6f} at epoch {best_epoch}")
    
    linear_layer.eval()
    return linear_layer, training_loss


def apply_linear_projection_preprocessing(weight_vectors, image_reprs_avg, linear_layer=None, 
                                          num_epochs=1000, learning_rate=0.01, weight_decay=0.1,
                                          dropout_rate=0.0, device='cuda', train_projection=True):
    """
    Apply linear projection preprocessing to weight vectors.
    
    This function either trains a new linear projection (if train_projection=True) or uses a 
    provided trained linear layer to map weight vectors into image representation space.
    
    Args:
        weight_vectors: Tensor of shape (num_classes, weight_dim) with weight vectors
        image_reprs_avg: Tensor of shape (num_classes, image_dim) with average image representations per class
        linear_layer: Pre-trained linear layer (if train_projection=False, this must be provided)
        num_epochs: Number of training epochs for the linear projection (default: 1000)
        learning_rate: Learning rate for training the linear projection (default: 0.01)
        weight_decay: L2 regularization coefficient for weight decay (default: 0.1)
        dropout_rate: Dropout probability for regularization (default: 0.0, no dropout)
        device: Device to use for computation (default: 'cuda')
        train_projection: Whether to train a new projection or use the provided one (default: True)
        
    Returns:
        preprocessed_weights: Tensor of shape (num_classes, image_dim), normalized to unit hypersphere
        linear_layer: The trained linear layer (for later reuse)
    """
    print("🔄 Applying linear projection preprocessing to weight vectors...")
    
    weight_vectors = weight_vectors.to(device).float()
    image_reprs_avg = image_reprs_avg.to(device).float()
    
    if train_projection:
        # Train a new linear projection
        linear_layer, training_loss = train_linear_projection(
            weight_vectors, image_reprs_avg,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            dropout_rate=dropout_rate,
            device=device,
            verbose=True
        )
    else:
        if linear_layer is None:
            raise ValueError("linear_layer must be provided when train_projection=False")
        linear_layer = linear_layer.to(device)
    
    # Apply the learned linear transformation
    with torch.no_grad():
        preprocessed_weights = linear_layer(weight_vectors)
    
    print(f"  Projected weights shape: {preprocessed_weights.shape}")
    print(f"  Projected weights norm - mean: {preprocessed_weights.norm(dim=1).mean():.6f}, std: {preprocessed_weights.norm(dim=1).std():.6f}")
    
    # Normalize to unit hypersphere
    preprocessed_weights = F.normalize(preprocessed_weights, p=2, dim=-1)
    
    print(f"  Projected weights after normalization - shape: {preprocessed_weights.shape}")
    print(f"  Projected weights norm - mean: {preprocessed_weights.norm(dim=1).mean():.6f}, std: {preprocessed_weights.norm(dim=1).std():.6f}")
    
    print(f"  Applied transformation: LinearLayer(W) / ||LinearLayer(W)||")
    
    return preprocessed_weights, linear_layer
