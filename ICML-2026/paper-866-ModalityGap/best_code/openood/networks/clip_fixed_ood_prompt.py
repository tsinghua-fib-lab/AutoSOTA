import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
# import clip
from openood.networks.clip import clip
from openood.networks.clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from openood.networks.clip_for_wordnet_prepare import generate_cossim_idname_wordnet_dedup
# import ipdb
# import pdb
import json, os

_tokenizer = _Tokenizer()
imagenet_classes = [
    'tench', 'goldfish', 'great white shark', 'tiger shark',
    'hammerhead shark', 'electric ray', 'stingray', 'rooster', 'hen',
    'ostrich', 'brambling', 'goldfinch', 'house finch', 'junco',
    'indigo bunting', 'American robin', 'bulbul', 'jay', 'magpie', 'chickadee',
    'American dipper', 'kite (bird of prey)', 'bald eagle', 'vulture',
    'great grey owl', 'fire salamander', 'smooth newt', 'newt',
    'spotted salamander', 'axolotl', 'American bullfrog', 'tree frog',
    'tailed frog', 'loggerhead sea turtle', 'leatherback sea turtle',
    'mud turtle', 'terrapin', 'box turtle', 'banded gecko', 'green iguana',
    'Carolina anole', 'desert grassland whiptail lizard', 'agama',
    'frilled-necked lizard', 'alligator lizard', 'Gila monster',
    'European green lizard', 'chameleon', 'Komodo dragon', 'Nile crocodile',
    'American alligator', 'triceratops', 'worm snake', 'ring-necked snake',
    'eastern hog-nosed snake', 'smooth green snake', 'kingsnake',
    'garter snake', 'water snake', 'vine snake', 'night snake',
    'boa constrictor', 'African rock python', 'Indian cobra', 'green mamba',
    'sea snake', 'Saharan horned viper', 'eastern diamondback rattlesnake',
    'sidewinder rattlesnake', 'trilobite', 'harvestman', 'scorpion',
    'yellow garden spider', 'barn spider', 'European garden spider',
    'southern black widow', 'tarantula', 'wolf spider', 'tick', 'centipede',
    'black grouse', 'ptarmigan', 'ruffed grouse', 'prairie grouse', 'peafowl',
    'quail', 'partridge', 'african grey parrot', 'macaw',
    'sulphur-crested cockatoo', 'lorikeet', 'coucal', 'bee eater', 'hornbill',
    'hummingbird', 'jacamar', 'toucan', 'duck', 'red-breasted merganser',
    'goose', 'black swan', 'tusker', 'echidna', 'platypus', 'wallaby', 'koala',
    'wombat', 'jellyfish', 'sea anemone', 'brain coral', 'flatworm',
    'nematode', 'conch', 'snail', 'slug', 'sea slug', 'chiton',
    'chambered nautilus', 'Dungeness crab', 'rock crab', 'fiddler crab',
    'red king crab', 'American lobster', 'spiny lobster', 'crayfish',
    'hermit crab', 'isopod', 'white stork', 'black stork', 'spoonbill',
    'flamingo', 'little blue heron', 'great egret', 'bittern bird',
    'crane bird', 'limpkin', 'common gallinule', 'American coot', 'bustard',
    'ruddy turnstone', 'dunlin', 'common redshank', 'dowitcher',
    'oystercatcher', 'pelican', 'king penguin', 'albatross', 'grey whale',
    'killer whale', 'dugong', 'sea lion', 'Chihuahua', 'Japanese Chin',
    'Maltese', 'Pekingese', 'Shih Tzu', 'King Charles Spaniel', 'Papillon',
    'toy terrier', 'Rhodesian Ridgeback', 'Afghan Hound', 'Basset Hound',
    'Beagle', 'Bloodhound', 'Bluetick Coonhound', 'Black and Tan Coonhound',
    'Treeing Walker Coonhound', 'English foxhound', 'Redbone Coonhound',
    'borzoi', 'Irish Wolfhound', 'Italian Greyhound', 'Whippet',
    'Ibizan Hound', 'Norwegian Elkhound', 'Otterhound', 'Saluki',
    'Scottish Deerhound', 'Weimaraner', 'Staffordshire Bull Terrier',
    'American Staffordshire Terrier', 'Bedlington Terrier', 'Border Terrier',
    'Kerry Blue Terrier', 'Irish Terrier', 'Norfolk Terrier',
    'Norwich Terrier', 'Yorkshire Terrier', 'Wire Fox Terrier',
    'Lakeland Terrier', 'Sealyham Terrier', 'Airedale Terrier',
    'Cairn Terrier', 'Australian Terrier', 'Dandie Dinmont Terrier',
    'Boston Terrier', 'Miniature Schnauzer', 'Giant Schnauzer',
    'Standard Schnauzer', 'Scottish Terrier', 'Tibetan Terrier',
    'Australian Silky Terrier', 'Soft-coated Wheaten Terrier',
    'West Highland White Terrier', 'Lhasa Apso', 'Flat-Coated Retriever',
    'Curly-coated Retriever', 'Golden Retriever', 'Labrador Retriever',
    'Chesapeake Bay Retriever', 'German Shorthaired Pointer', 'Vizsla',
    'English Setter', 'Irish Setter', 'Gordon Setter', 'Brittany dog',
    'Clumber Spaniel', 'English Springer Spaniel', 'Welsh Springer Spaniel',
    'Cocker Spaniel', 'Sussex Spaniel', 'Irish Water Spaniel', 'Kuvasz',
    'Schipperke', 'Groenendael dog', 'Malinois', 'Briard', 'Australian Kelpie',
    'Komondor', 'Old English Sheepdog', 'Shetland Sheepdog', 'collie',
    'Border Collie', 'Bouvier des Flandres dog', 'Rottweiler',
    'German Shepherd Dog', 'Dobermann', 'Miniature Pinscher',
    'Greater Swiss Mountain Dog', 'Bernese Mountain Dog',
    'Appenzeller Sennenhund', 'Entlebucher Sennenhund', 'Boxer', 'Bullmastiff',
    'Tibetan Mastiff', 'French Bulldog', 'Great Dane', 'St. Bernard', 'husky',
    'Alaskan Malamute', 'Siberian Husky', 'Dalmatian', 'Affenpinscher',
    'Basenji', 'pug', 'Leonberger', 'Newfoundland dog', 'Great Pyrenees dog',
    'Samoyed', 'Pomeranian', 'Chow Chow', 'Keeshond', 'brussels griffon',
    'Pembroke Welsh Corgi', 'Cardigan Welsh Corgi', 'Toy Poodle',
    'Miniature Poodle', 'Standard Poodle',
    'Mexican hairless dog (xoloitzcuintli)', 'grey wolf',
    'Alaskan tundra wolf', 'red wolf or maned wolf', 'coyote', 'dingo',
    'dhole', 'African wild dog', 'hyena', 'red fox', 'kit fox', 'Arctic fox',
    'grey fox', 'tabby cat', 'tiger cat', 'Persian cat', 'Siamese cat',
    'Egyptian Mau', 'cougar', 'lynx', 'leopard', 'snow leopard', 'jaguar',
    'lion', 'tiger', 'cheetah', 'brown bear', 'American black bear',
    'polar bear', 'sloth bear', 'mongoose', 'meerkat', 'tiger beetle',
    'ladybug', 'ground beetle', 'longhorn beetle', 'leaf beetle',
    'dung beetle', 'rhinoceros beetle', 'weevil', 'fly', 'bee', 'ant',
    'grasshopper', 'cricket insect', 'stick insect', 'cockroach',
    'praying mantis', 'cicada', 'leafhopper', 'lacewing', 'dragonfly',
    'damselfly', 'red admiral butterfly', 'ringlet butterfly',
    'monarch butterfly', 'small white butterfly', 'sulphur butterfly',
    'gossamer-winged butterfly', 'starfish', 'sea urchin', 'sea cucumber',
    'cottontail rabbit', 'hare', 'Angora rabbit', 'hamster', 'porcupine',
    'fox squirrel', 'marmot', 'beaver', 'guinea pig', 'common sorrel horse',
    'zebra', 'pig', 'wild boar', 'warthog', 'hippopotamus', 'ox',
    'water buffalo', 'bison', 'ram (adult male sheep)', 'bighorn sheep',
    'Alpine ibex', 'hartebeest', 'impala (antelope)', 'gazelle',
    'arabian camel', 'llama', 'weasel', 'mink', 'European polecat',
    'black-footed ferret', 'otter', 'skunk', 'badger', 'armadillo',
    'three-toed sloth', 'orangutan', 'gorilla', 'chimpanzee', 'gibbon',
    'siamang', 'guenon', 'patas monkey', 'baboon', 'macaque', 'langur',
    'black-and-white colobus', 'proboscis monkey', 'marmoset',
    'white-headed capuchin', 'howler monkey', 'titi monkey',
    "Geoffroy's spider monkey", 'common squirrel monkey', 'ring-tailed lemur',
    'indri', 'Asian elephant', 'African bush elephant', 'red panda',
    'giant panda', 'snoek fish', 'eel', 'silver salmon', 'rock beauty fish',
    'clownfish', 'sturgeon', 'gar fish', 'lionfish', 'pufferfish', 'abacus',
    'abaya', 'academic gown', 'accordion', 'acoustic guitar',
    'aircraft carrier', 'airliner', 'airship', 'altar', 'ambulance',
    'amphibious vehicle', 'analog clock', 'apiary', 'apron', 'trash can',
    'assault rifle', 'backpack', 'bakery', 'balance beam', 'balloon',
    'ballpoint pen', 'Band-Aid', 'banjo', 'baluster / handrail', 'barbell',
    'barber chair', 'barbershop', 'barn', 'barometer', 'barrel', 'wheelbarrow',
    'baseball', 'basketball', 'bassinet', 'bassoon', 'swimming cap',
    'bath towel', 'bathtub', 'station wagon', 'lighthouse', 'beaker',
    'military hat (bearskin or shako)', 'beer bottle', 'beer glass',
    'bell tower', 'baby bib', 'tandem bicycle', 'bikini', 'ring binder',
    'binoculars', 'birdhouse', 'boathouse', 'bobsleigh', 'bolo tie',
    'poke bonnet', 'bookcase', 'bookstore', 'bottle cap', 'hunting bow',
    'bow tie', 'brass memorial plaque', 'bra', 'breakwater', 'breastplate',
    'broom', 'bucket', 'buckle', 'bulletproof vest', 'high-speed train',
    'butcher shop', 'taxicab', 'cauldron', 'candle', 'cannon', 'canoe',
    'can opener', 'cardigan', 'car mirror', 'carousel', 'tool kit',
    'cardboard box / carton', 'car wheel', 'automated teller machine',
    'cassette', 'cassette player', 'castle', 'catamaran', 'CD player', 'cello',
    'mobile phone', 'chain', 'chain-link fence', 'chain mail', 'chainsaw',
    'storage chest', 'chiffonier', 'bell or wind chime', 'china cabinet',
    'Christmas stocking', 'church', 'movie theater', 'cleaver',
    'cliff dwelling', 'cloak', 'clogs', 'cocktail shaker', 'coffee mug',
    'coffeemaker', 'spiral or coil', 'combination lock', 'computer keyboard',
    'candy store', 'container ship', 'convertible', 'corkscrew', 'cornet',
    'cowboy boot', 'cowboy hat', 'cradle', 'construction crane',
    'crash helmet', 'crate', 'infant bed', 'Crock Pot', 'croquet ball',
    'crutch', 'cuirass', 'dam', 'desk', 'desktop computer',
    'rotary dial telephone', 'diaper', 'digital clock', 'digital watch',
    'dining table', 'dishcloth', 'dishwasher', 'disc brake', 'dock',
    'dog sled', 'dome', 'doormat', 'drilling rig', 'drum', 'drumstick',
    'dumbbell', 'Dutch oven', 'electric fan', 'electric guitar',
    'electric locomotive', 'entertainment center', 'envelope',
    'espresso machine', 'face powder', 'feather boa', 'filing cabinet',
    'fireboat', 'fire truck', 'fire screen', 'flagpole', 'flute',
    'folding chair', 'football helmet', 'forklift', 'fountain', 'fountain pen',
    'four-poster bed', 'freight car', 'French horn', 'frying pan', 'fur coat',
    'garbage truck', 'gas mask or respirator', 'gas pump', 'goblet', 'go-kart',
    'golf ball', 'golf cart', 'gondola', 'gong', 'gown', 'grand piano',
    'greenhouse', 'radiator grille', 'grocery store', 'guillotine',
    'hair clip', 'hair spray', 'half-track', 'hammer', 'hamper', 'hair dryer',
    'hand-held computer', 'handkerchief', 'hard disk drive', 'harmonica',
    'harp', 'combine harvester', 'hatchet', 'holster', 'home theater',
    'honeycomb', 'hook', 'hoop skirt', 'gymnastic horizontal bar',
    'horse-drawn vehicle', 'hourglass', 'iPod', 'clothes iron',
    'carved pumpkin', 'jeans', 'jeep', 'T-shirt', 'jigsaw puzzle', 'rickshaw',
    'joystick', 'kimono', 'knee pad', 'knot', 'lab coat', 'ladle', 'lampshade',
    'laptop computer', 'lawn mower', 'lens cap', 'letter opener', 'library',
    'lifeboat', 'lighter', 'limousine', 'ocean liner', 'lipstick',
    'slip-on shoe', 'lotion', 'music speaker', 'loupe magnifying glass',
    'sawmill', 'magnetic compass', 'messenger bag', 'mailbox', 'tights',
    'one-piece bathing suit', 'manhole cover', 'maraca', 'marimba', 'mask',
    'matchstick', 'maypole', 'maze', 'measuring cup', 'medicine cabinet',
    'megalith', 'microphone', 'microwave oven', 'military uniform', 'milk can',
    'minibus', 'miniskirt', 'minivan', 'missile', 'mitten', 'mixing bowl',
    'mobile home', 'ford model t', 'modem', 'monastery', 'monitor', 'moped',
    'mortar and pestle', 'graduation cap', 'mosque', 'mosquito net', 'vespa',
    'mountain bike', 'tent', 'computer mouse', 'mousetrap', 'moving van',
    'muzzle', 'metal nail', 'neck brace', 'necklace', 'baby pacifier',
    'notebook computer', 'obelisk', 'oboe', 'ocarina', 'odometer',
    'oil filter', 'pipe organ', 'oscilloscope', 'overskirt', 'bullock cart',
    'oxygen mask', 'product packet / packaging', 'paddle', 'paddle wheel',
    'padlock', 'paintbrush', 'pajamas', 'palace', 'pan flute', 'paper towel',
    'parachute', 'parallel bars', 'park bench', 'parking meter',
    'railroad car', 'patio', 'payphone', 'pedestal', 'pencil case',
    'pencil sharpener', 'perfume', 'Petri dish', 'photocopier', 'plectrum',
    'Pickelhaube', 'picket fence', 'pickup truck', 'pier', 'piggy bank',
    'pill bottle', 'pillow', 'ping-pong ball', 'pinwheel', 'pirate ship',
    'drink pitcher', 'block plane', 'planetarium', 'plastic bag', 'plate rack',
    'farm plow', 'plunger', 'Polaroid camera', 'pole', 'police van', 'poncho',
    'pool table', 'soda bottle', 'plant pot', "potter's wheel", 'power drill',
    'prayer rug', 'printer', 'prison', 'missile', 'projector', 'hockey puck',
    'punching bag', 'purse', 'quill', 'quilt', 'race car', 'racket',
    'radiator', 'radio', 'radio telescope', 'rain barrel',
    'recreational vehicle', 'fishing casting reel', 'reflex camera',
    'refrigerator', 'remote control', 'restaurant', 'revolver', 'rifle',
    'rocking chair', 'rotisserie', 'eraser', 'rugby ball',
    'ruler measuring stick', 'sneaker', 'safe', 'safety pin', 'salt shaker',
    'sandal', 'sarong', 'saxophone', 'scabbard', 'weighing scale',
    'school bus', 'schooner', 'scoreboard', 'CRT monitor', 'screw',
    'screwdriver', 'seat belt', 'sewing machine', 'shield', 'shoe store',
    'shoji screen / room divider', 'shopping basket', 'shopping cart',
    'shovel', 'shower cap', 'shower curtain', 'ski', 'balaclava ski mask',
    'sleeping bag', 'slide rule', 'sliding door', 'slot machine', 'snorkel',
    'snowmobile', 'snowplow', 'soap dispenser', 'soccer ball', 'sock',
    'solar thermal collector', 'sombrero', 'soup bowl', 'keyboard space bar',
    'space heater', 'space shuttle', 'spatula', 'motorboat', 'spider web',
    'spindle', 'sports car', 'spotlight', 'stage', 'steam locomotive',
    'through arch bridge', 'steel drum', 'stethoscope', 'scarf', 'stone wall',
    'stopwatch', 'stove', 'strainer', 'tram', 'stretcher', 'couch', 'stupa',
    'submarine', 'suit', 'sundial', 'sunglasses', 'sunglasses', 'sunscreen',
    'suspension bridge', 'mop', 'sweatshirt', 'swim trunks / shorts', 'swing',
    'electrical switch', 'syringe', 'table lamp', 'tank', 'tape player',
    'teapot', 'teddy bear', 'television', 'tennis ball', 'thatched roof',
    'front curtain', 'thimble', 'threshing machine', 'throne', 'tile roof',
    'toaster', 'tobacco shop', 'toilet seat', 'torch', 'totem pole',
    'tow truck', 'toy store', 'tractor', 'semi-trailer truck', 'tray',
    'trench coat', 'tricycle', 'trimaran', 'tripod', 'triumphal arch',
    'trolleybus', 'trombone', 'hot tub', 'turnstile', 'typewriter keyboard',
    'umbrella', 'unicycle', 'upright piano', 'vacuum cleaner', 'vase',
    'vaulted or arched ceiling', 'velvet fabric', 'vending machine',
    'vestment', 'viaduct', 'violin', 'volleyball', 'waffle iron', 'wall clock',
    'wallet', 'wardrobe', 'military aircraft', 'sink', 'washing machine',
    'water bottle', 'water jug', 'water tower', 'whiskey jug', 'whistle',
    'hair wig', 'window screen', 'window shade', 'Windsor tie', 'wine bottle',
    'airplane wing', 'wok', 'wooden spoon', 'wool', 'split-rail fence',
    'shipwreck', 'sailboat', 'yurt', 'website', 'comic book', 'crossword',
    'traffic or street sign', 'traffic light', 'dust jacket', 'menu', 'plate',
    'guacamole', 'consomme', 'hot pot', 'trifle', 'ice cream', 'popsicle',
    'baguette', 'bagel', 'pretzel', 'cheeseburger', 'hot dog',
    'mashed potatoes', 'cabbage', 'broccoli', 'cauliflower', 'zucchini',
    'spaghetti squash', 'acorn squash', 'butternut squash', 'cucumber',
    'artichoke', 'bell pepper', 'cardoon', 'mushroom', 'Granny Smith apple',
    'strawberry', 'orange', 'lemon', 'fig', 'pineapple', 'banana', 'jackfruit',
    'cherimoya (custard apple)', 'pomegranate', 'hay', 'carbonara',
    'chocolate syrup', 'dough', 'meatloaf', 'pizza', 'pot pie', 'burrito',
    'red wine', 'espresso', 'tea cup', 'eggnog', 'mountain', 'bubble', 'cliff',
    'coral reef', 'geyser', 'lakeshore', 'promontory', 'sandbar', 'beach',
    'valley', 'volcano', 'baseball player', 'bridegroom', 'scuba diver',
    'rapeseed', 'daisy', "yellow lady's slipper", 'corn', 'acorn', 'rose hip',
    'horse chestnut seed', 'coral fungus', 'agaric', 'gyromitra',
    'stinkhorn mushroom', 'earth star fungus', 'hen of the woods mushroom',
    'bolete', 'corn cob', 'toilet paper'
]
all_wnids = ['n01440764', 'n01443537', 'n01484850', 'n01491361', 'n01494475', 'n01496331', 'n01498041', 'n01514668', 'n01514859', 'n01518878', 'n01530575', 'n01531178', 'n01532829', 'n01534433', 'n01537544', 'n01558993', 'n01560419', 'n01580077', 'n01582220', 'n01592084', 'n01601694', 'n01608432', 'n01614925', 'n01616318', 'n01622779', 'n01629819', 'n01630670', 'n01631663', 'n01632458', 'n01632777', 'n01641577', 'n01644373', 'n01644900', 'n01664065', 'n01665541', 'n01667114', 'n01667778', 'n01669191', 'n01675722', 'n01677366', 'n01682714', 'n01685808', 'n01687978', 'n01688243', 'n01689811', 'n01692333', 'n01693334', 'n01694178', 'n01695060', 'n01697457', 'n01698640', 'n01704323', 'n01728572', 'n01728920', 'n01729322', 'n01729977', 'n01734418', 'n01735189', 'n01737021', 'n01739381', 'n01740131', 'n01742172', 'n01744401', 'n01748264', 'n01749939', 'n01751748', 'n01753488', 'n01755581', 'n01756291', 'n01768244', 'n01770081', 'n01770393', 'n01773157', 'n01773549', 'n01773797', 'n01774384', 'n01774750', 'n01775062', 'n01776313', 'n01784675', 'n01795545', 'n01796340', 'n01797886', 'n01798484', 'n01806143', 'n01806567', 'n01807496', 'n01817953', 'n01818515', 'n01819313', 'n01820546', 'n01824575', 'n01828970', 'n01829413', 'n01833805', 'n01843065', 'n01843383', 'n01847000', 'n01855032', 'n01855672', 'n01860187', 'n01871265', 'n01872401', 'n01873310', 'n01877812', 'n01882714', 'n01883070', 'n01910747', 'n01914609', 'n01917289', 'n01924916', 'n01930112', 'n01943899', 'n01944390', 'n01945685', 'n01950731', 'n01955084', 'n01968897', 'n01978287', 'n01978455', 'n01980166', 'n01981276', 'n01983481', 'n01984695', 'n01985128', 'n01986214', 'n01990800', 'n02002556', 'n02002724', 'n02006656', 'n02007558', 'n02009229', 'n02009912', 'n02011460', 'n02012849', 'n02013706', 'n02017213', 'n02018207', 'n02018795', 'n02025239', 'n02027492', 'n02028035', 'n02033041', 'n02037110', 'n02051845', 'n02056570', 'n02058221', 'n02066245', 'n02071294', 'n02074367', 'n02077923', 'n02085620', 'n02085782', 'n02085936', 'n02086079', 'n02086240', 'n02086646', 'n02086910', 'n02087046', 'n02087394', 'n02088094', 'n02088238', 'n02088364', 'n02088466', 'n02088632', 'n02089078', 'n02089867', 'n02089973', 'n02090379', 'n02090622', 'n02090721', 'n02091032', 'n02091134', 'n02091244', 'n02091467', 'n02091635', 'n02091831', 'n02092002', 'n02092339', 'n02093256', 'n02093428', 'n02093647', 'n02093754', 'n02093859', 'n02093991', 'n02094114', 'n02094258', 'n02094433', 'n02095314', 'n02095570', 'n02095889', 'n02096051', 'n02096177', 'n02096294', 'n02096437', 'n02096585', 'n02097047', 'n02097130', 'n02097209', 'n02097298', 'n02097474', 'n02097658', 'n02098105', 'n02098286', 'n02098413', 'n02099267', 'n02099429', 'n02099601', 'n02099712', 'n02099849', 'n02100236', 'n02100583', 'n02100735', 'n02100877', 'n02101006', 'n02101388', 'n02101556', 'n02102040', 'n02102177', 'n02102318', 'n02102480', 'n02102973', 'n02104029', 'n02104365', 'n02105056', 'n02105162', 'n02105251', 'n02105412', 'n02105505', 'n02105641', 'n02105855', 'n02106030', 'n02106166', 'n02106382', 'n02106550', 'n02106662', 'n02107142', 'n02107312', 'n02107574', 'n02107683', 'n02107908', 'n02108000', 'n02108089', 'n02108422', 'n02108551', 'n02108915', 'n02109047', 'n02109525', 'n02109961', 'n02110063', 'n02110185', 'n02110341', 'n02110627', 'n02110806', 'n02110958', 'n02111129', 'n02111277', 'n02111500', 'n02111889', 'n02112018', 'n02112137', 'n02112350', 'n02112706', 'n02113023', 'n02113186', 'n02113624', 'n02113712', 'n02113799', 'n02113978', 'n02114367', 'n02114548', 'n02114712', 'n02114855', 'n02115641', 'n02115913', 'n02116738', 'n02117135', 'n02119022', 'n02119789', 'n02120079', 'n02120505', 'n02123045', 'n02123159', 'n02123394', 'n02123597', 'n02124075', 'n02125311', 'n02127052', 'n02128385', 'n02128757', 'n02128925', 'n02129165', 'n02129604', 'n02130308', 'n02132136', 'n02133161', 'n02134084', 'n02134418', 'n02137549', 'n02138441', 'n02165105', 'n02165456', 'n02167151', 'n02168699', 'n02169497', 'n02172182', 'n02174001', 'n02177972', 'n02190166', 'n02206856', 'n02219486', 'n02226429', 'n02229544', 'n02231487', 'n02233338', 'n02236044', 'n02256656', 'n02259212', 'n02264363', 'n02268443', 'n02268853', 'n02276258', 'n02277742', 'n02279972', 'n02280649', 'n02281406', 'n02281787', 'n02317335', 'n02319095', 'n02321529', 'n02325366', 'n02326432', 'n02328150', 'n02342885', 'n02346627', 'n02356798', 'n02361337', 'n02363005', 'n02364673', 'n02389026', 'n02391049', 'n02395406', 'n02396427', 'n02397096', 'n02398521', 'n02403003', 'n02408429', 'n02410509', 'n02412080', 'n02415577', 'n02417914', 'n02422106', 'n02422699', 'n02423022', 'n02437312', 'n02437616', 'n02441942', 'n02442845', 'n02443114', 'n02443484', 'n02444819', 'n02445715', 'n02447366', 'n02454379', 'n02457408', 'n02480495', 'n02480855', 'n02481823', 'n02483362', 'n02483708', 'n02484975', 'n02486261', 'n02486410', 'n02487347', 'n02488291', 'n02488702', 'n02489166', 'n02490219', 'n02492035', 'n02492660', 'n02493509', 'n02493793', 'n02494079', 'n02497673', 'n02500267', 'n02504013', 'n02504458', 'n02509815', 'n02510455', 'n02514041', 'n02526121', 'n02536864', 'n02606052', 'n02607072', 'n02640242', 'n02641379', 'n02643566', 'n02655020', 'n02666196', 'n02667093', 'n02669723', 'n02672831', 'n02676566', 'n02687172', 'n02690373', 'n02692877', 'n02699494', 'n02701002', 'n02704792', 'n02708093', 'n02727426', 'n02730930', 'n02747177', 'n02749479', 'n02769748', 'n02776631', 'n02777292', 'n02782093', 'n02783161', 'n02786058', 'n02787622', 'n02788148', 'n02790996', 'n02791124', 'n02791270', 'n02793495', 'n02794156', 'n02795169', 'n02797295', 'n02799071', 'n02802426', 'n02804414', 'n02804610', 'n02807133', 'n02808304', 'n02808440', 'n02814533', 'n02814860', 'n02815834', 'n02817516', 'n02823428', 'n02823750', 'n02825657', 'n02834397', 'n02835271', 'n02837789', 'n02840245', 'n02841315', 'n02843684', 'n02859443', 'n02860847', 'n02865351', 'n02869837', 'n02870880', 'n02871525', 'n02877765', 'n02879718', 'n02883205', 'n02892201', 'n02892767', 'n02894605', 'n02895154', 'n02906734', 'n02909870', 'n02910353', 'n02916936', 'n02917067', 'n02927161', 'n02930766', 'n02939185', 'n02948072', 'n02950826', 'n02951358', 'n02951585', 'n02963159', 'n02965783', 'n02966193', 'n02966687', 'n02971356', 'n02974003', 'n02977058', 'n02978881', 'n02979186', 'n02980441', 'n02981792', 'n02988304', 'n02992211', 'n02992529', 'n02999410', 'n03000134', 'n03000247', 'n03000684', 'n03014705', 'n03016953', 'n03017168', 'n03018349', 'n03026506', 'n03028079', 'n03032252', 'n03041632', 'n03042490', 'n03045698', 'n03047690', 'n03062245', 'n03063599', 'n03063689', 'n03065424', 'n03075370', 'n03085013', 'n03089624', 'n03095699', 'n03100240', 'n03109150', 'n03110669', 'n03124043', 'n03124170', 'n03125729', 'n03126707', 'n03127747', 'n03127925', 'n03131574', 'n03133878', 'n03134739', 'n03141823', 'n03146219', 'n03160309', 'n03179701', 'n03180011', 'n03187595', 'n03188531', 'n03196217', 'n03197337', 'n03201208', 'n03207743', 'n03207941', 'n03208938', 'n03216828', 'n03218198', 'n03220513', 'n03223299', 'n03240683', 'n03249569', 'n03250847', 'n03255030', 'n03259280', 'n03271574', 'n03272010', 'n03272562', 'n03290653', 'n03291819', 'n03297495', 'n03314780', 'n03325584', 'n03337140', 'n03344393', 'n03345487', 'n03347037', 'n03355925', 'n03372029', 'n03376595', 'n03379051', 'n03384352', 'n03388043', 'n03388183', 'n03388549', 'n03393912', 'n03394916', 'n03400231', 'n03404251', 'n03417042', 'n03424325', 'n03425413', 'n03443371', 'n03444034', 'n03445777', 'n03445924', 'n03447447', 'n03447721', 'n03450230', 'n03452741', 'n03457902', 'n03459775', 'n03461385', 'n03467068', 'n03476684', 'n03476991', 'n03478589', 'n03481172', 'n03482405', 'n03483316', 'n03485407', 'n03485794', 'n03492542', 'n03494278', 'n03495258', 'n03496892', 'n03498962', 'n03527444', 'n03529860', 'n03530642', 'n03532672', 'n03534580', 'n03535780', 'n03538406', 'n03544143', 'n03584254', 'n03584829', 'n03590841', 'n03594734', 'n03594945', 'n03595614', 'n03598930', 'n03599486', 'n03602883', 'n03617480', 'n03623198', 'n03627232', 'n03630383', 'n03633091', 'n03637318', 'n03642806', 'n03649909', 'n03657121', 'n03658185', 'n03661043', 'n03662601', 'n03666591', 'n03670208', 'n03673027', 'n03676483', 'n03680355', 'n03690938', 'n03691459', 'n03692522', 'n03697007', 'n03706229', 'n03709823', 'n03710193', 'n03710637', 'n03710721', 'n03717622', 'n03720891', 'n03721384', 'n03724870', 'n03729826', 'n03733131', 'n03733281', 'n03733805', 'n03742115', 'n03743016', 'n03759954', 'n03761084', 'n03763968', 'n03764736', 'n03769881', 'n03770439', 'n03770679', 'n03773504', 'n03775071', 'n03775546', 'n03776460', 'n03777568', 'n03777754', 'n03781244', 'n03782006', 'n03785016', 'n03786901', 'n03787032', 'n03788195', 'n03788365', 'n03791053', 'n03792782', 'n03792972', 'n03793489', 'n03794056', 'n03796401', 'n03803284', 'n03804744', 'n03814639', 'n03814906', 'n03825788', 'n03832673', 'n03837869', 'n03838899', 'n03840681', 'n03841143', 'n03843555', 'n03854065', 'n03857828', 'n03866082', 'n03868242', 'n03868863', 'n03871628', 'n03873416', 'n03874293', 'n03874599', 'n03876231', 'n03877472', 'n03877845', 'n03884397', 'n03887697', 'n03888257', 'n03888605', 'n03891251', 'n03891332', 'n03895866', 'n03899768', 'n03902125', 'n03903868', 'n03908618', 'n03908714', 'n03916031', 'n03920288', 'n03924679', 'n03929660', 'n03929855', 'n03930313', 'n03930630', 'n03933933', 'n03935335', 'n03937543', 'n03938244', 'n03942813', 'n03944341', 'n03947888', 'n03950228', 'n03954731', 'n03956157', 'n03958227', 'n03961711', 'n03967562', 'n03970156', 'n03976467', 'n03976657', 'n03977966', 'n03980874', 'n03982430', 'n03983396', 'n03991062', 'n03992509', 'n03995372', 'n03998194', 'n04004767', 'n04005630', 'n04008634', 'n04009552', 'n04019541', 'n04023962', 'n04026417', 'n04033901', 'n04033995', 'n04037443', 'n04039381', 'n04040759', 'n04041544', 'n04044716', 'n04049303', 'n04065272', 'n04067472', 'n04069434', 'n04070727', 'n04074963', 'n04081281', 'n04086273', 'n04090263', 'n04099969', 'n04111531', 'n04116512', 'n04118538', 'n04118776', 'n04120489', 'n04125021', 'n04127249', 'n04131690', 'n04133789', 'n04136333', 'n04141076', 'n04141327', 'n04141975', 'n04146614', 'n04147183', 'n04149813', 'n04152593', 'n04153751', 'n04154565', 'n04162706', 'n04179913', 'n04192698', 'n04200800', 'n04201297', 'n04204238', 'n04204347', 'n04208210', 'n04209133', 'n04209239', 'n04228054', 'n04229816', 'n04235860', 'n04238763', 'n04239074', 'n04243546', 'n04251144', 'n04252077', 'n04252225', 'n04254120', 'n04254680', 'n04254777', 'n04258138', 'n04259630', 'n04263257', 'n04264628', 'n04265275', 'n04266014', 'n04270147', 'n04273569', 'n04275548', 'n04277352', 'n04285008', 'n04286575', 'n04296562', 'n04310018', 'n04311004', 'n04311174', 'n04317175', 'n04325704', 'n04326547', 'n04328186', 'n04330267', 'n04332243', 'n04335435', 'n04336792', 'n04344873', 'n04346328', 'n04347754', 'n04350905', 'n04355338', 'n04355933', 'n04356056', 'n04357314', 'n04366367', 'n04367480', 'n04370456', 'n04371430', 'n04371774', 'n04372370', 'n04376876', 'n04380533', 'n04389033', 'n04392985', 'n04398044', 'n04399382', 'n04404412', 'n04409515', 'n04417672', 'n04418357', 'n04423845', 'n04428191', 'n04429376', 'n04435653', 'n04442312', 'n04443257', 'n04447861', 'n04456115', 'n04458633', 'n04461696', 'n04462240', 'n04465501', 'n04467665', 'n04476259', 'n04479046', 'n04482393', 'n04483307', 'n04485082', 'n04486054', 'n04487081', 'n04487394', 'n04493381', 'n04501370', 'n04505470', 'n04507155', 'n04509417', 'n04515003', 'n04517823', 'n04522168', 'n04523525', 'n04525038', 'n04525305', 'n04532106', 'n04532670', 'n04536866', 'n04540053', 'n04542943', 'n04548280', 'n04548362', 'n04550184', 'n04552348', 'n04553703', 'n04554684', 'n04557648', 'n04560804', 'n04562935', 'n04579145', 'n04579432', 'n04584207', 'n04589890', 'n04590129', 'n04591157', 'n04591713', 'n04592741', 'n04596742', 'n04597913', 'n04599235', 'n04604644', 'n04606251', 'n04612504', 'n04613696', 'n06359193', 'n06596364', 'n06785654', 'n06794110', 'n06874185', 'n07248320', 'n07565083', 'n07579787', 'n07583066', 'n07584110', 'n07590611', 'n07613480', 'n07614500', 'n07615774', 'n07684084', 'n07693725', 'n07695742', 'n07697313', 'n07697537', 'n07711569', 'n07714571', 'n07714990', 'n07715103', 'n07716358', 'n07716906', 'n07717410', 'n07717556', 'n07718472', 'n07718747', 'n07720875', 'n07730033', 'n07734744', 'n07742313', 'n07745940', 'n07747607', 'n07749582', 'n07753113', 'n07753275', 'n07753592', 'n07754684', 'n07760859', 'n07768694', 'n07802026', 'n07831146', 'n07836838', 'n07860988', 'n07871810', 'n07873807', 'n07875152', 'n07880968', 'n07892512', 'n07920052', 'n07930864', 'n07932039', 'n09193705', 'n09229709', 'n09246464', 'n09256479', 'n09288635', 'n09332890', 'n09399592', 'n09421951', 'n09428293', 'n09468604', 'n09472597', 'n09835506', 'n10148035', 'n10565667', 'n11879895', 'n11939491', 'n12057211', 'n12144580', 'n12267677', 'n12620546', 'n12768682', 'n12985857', 'n12998815', 'n13037406', 'n13040303', 'n13044778', 'n13052670', 'n13054560', 'n13133613', 'n15075141']
imagenet_200_wnids = {'n01443537', 'n01484850', 'n01494475', 'n01498041', 'n01514859', 'n01518878', 'n01531178', 'n01534433', 'n01614925', 'n01616318', 'n01630670', 'n01632777', 'n01644373', 'n01677366', 'n01694178', 'n01748264', 'n01770393', 'n01774750', 'n01784675', 'n01806143', 'n01820546', 'n01833805', 'n01843383', 'n01847000', 'n01855672', 'n01860187', 'n01882714', 'n01910747', 'n01944390', 'n01983481', 'n01986214', 'n02007558', 'n02009912', 'n02051845', 'n02056570', 'n02066245', 'n02071294', 'n02077923', 'n02085620', 'n02086240', 'n02088094', 'n02088238', 'n02088364', 'n02088466', 'n02091032', 'n02091134', 'n02092339', 'n02094433', 'n02096585', 'n02097298', 'n02098286', 'n02099601', 'n02099712', 'n02102318', 'n02106030', 'n02106166', 'n02106550', 'n02106662', 'n02108089', 'n02108915', 'n02109525', 'n02110185', 'n02110341', 'n02110958', 'n02112018', 'n02112137', 'n02113023', 'n02113624', 'n02113799', 'n02114367', 'n02117135', 'n02119022', 'n02123045', 'n02128385', 'n02128757', 'n02129165', 'n02129604', 'n02130308', 'n02134084', 'n02138441', 'n02165456', 'n02190166', 'n02206856', 'n02219486', 'n02226429', 'n02233338', 'n02236044', 'n02268443', 'n02279972', 'n02317335', 'n02325366', 'n02346627', 'n02356798', 'n02363005', 'n02364673', 'n02391049', 'n02395406', 'n02398521', 'n02410509', 'n02423022', 'n02437616', 'n02445715', 'n02447366', 'n02480495', 'n02480855', 'n02481823', 'n02483362', 'n02486410', 'n02510455', 'n02526121', 'n02607072', 'n02655020', 'n02672831', 'n02701002', 'n02749479', 'n02769748', 'n02793495', 'n02797295', 'n02802426', 'n02808440', 'n02814860', 'n02823750', 'n02841315', 'n02843684', 'n02883205', 'n02906734', 'n02909870', 'n02939185', 'n02948072', 'n02950826', 'n02951358', 'n02966193', 'n02980441', 'n02992529', 'n03124170', 'n03272010', 'n03345487', 'n03372029', 'n03424325', 'n03452741', 'n03467068', 'n03481172', 'n03494278', 'n03495258', 'n03498962', 'n03594945', 'n03602883', 'n03630383', 'n03649909', 'n03676483', 'n03710193', 'n03773504', 'n03775071', 'n03888257', 'n03930630', 'n03947888', 'n04086273', 'n04118538', 'n04133789', 'n04141076', 'n04146614', 'n04147183', 'n04192698', 'n04254680', 'n04266014', 'n04275548', 'n04310018', 'n04325704', 'n04347754', 'n04389033', 'n04409515', 'n04465501', 'n04487394', 'n04522168', 'n04536866', 'n04552348', 'n04591713', 'n07614500', 'n07693725', 'n07695742', 'n07697313', 'n07697537', 'n07714571', 'n07714990', 'n07718472', 'n07720875', 'n07734744', 'n07742313', 'n07745940', 'n07749582', 'n07753275', 'n07753592', 'n07768694', 'n07873807', 'n07880968', 'n07920052', 'n09472597', 'n09835506', 'n10565667', 'n12267677'}
imagenet_r_mask = [wnid in imagenet_200_wnids for wnid in all_wnids] # True or False
imagenet200_classes = []
for i in range(len(imagenet_r_mask)):
    if imagenet_r_mask[i]:
        imagenet200_classes.append(imagenet_classes[i])

# ImageNet-S50 subset (50 classes)
imagenet_s50_wnids = {'n01443537', 'n01491361', 'n01531178', 'n01644373', 'n02104029', 'n02119022', 'n02123597', 'n02133161', 'n02165456', 'n02281406', 'n02325366', 'n02342885', 'n02396427', 'n02483362', 'n02504458', 'n02510455', 'n02690373', 'n02747177', 'n02783161', 'n02814533', 'n02859443', 'n02917067', 'n02992529', 'n03014705', 'n03047690', 'n03095699', 'n03197337', 'n03201208', 'n03445777', 'n03452741', 'n03584829', 'n03630383', 'n03775546', 'n03791053', 'n03874599', 'n03891251', 'n04026417', 'n04335435', 'n04380533', 'n04404412', 'n04447861', 'n04507155', 'n04522168', 'n04557648', 'n04562935', 'n04612504', 'n06794110', 'n07749582', 'n07831146', 'n12998815'}
imagenet_s50_mask = [wnid in imagenet_s50_wnids for wnid in all_wnids]
imagenet_s50_classes = []
for i in range(len(imagenet_s50_mask)):
    if imagenet_s50_mask[i]:
        imagenet_s50_classes.append(imagenet_classes[i])

# ImageNet-S (Full 919 classes)
imagenet_s_wnids = {
    'n01440764', 'n01443537', 'n01484850', 'n01491361', 'n01494475', 'n01496331', 'n01498041',
    'n01514668', 'n01514859', 'n01518878', 'n01530575', 'n01531178', 'n01532829', 'n01534433',
    'n01537544', 'n01558993', 'n01560419', 'n01580077', 'n01582220', 'n01592084', 'n01601694',
    'n01608432', 'n01614925', 'n01616318', 'n01622779', 'n01629819', 'n01630670', 'n01631663',
    'n01632458', 'n01632777', 'n01641577', 'n01644373', 'n01644900', 'n01664065', 'n01665541',
    'n01667114', 'n01667778', 'n01669191', 'n01675722', 'n01677366', 'n01682714', 'n01685808',
    'n01687978', 'n01688243', 'n01689811', 'n01692333', 'n01693334', 'n01694178', 'n01695060',
    'n01697457', 'n01698640', 'n01704323', 'n01728572', 'n01728920', 'n01729322', 'n01729977',
    'n01734418', 'n01735189', 'n01737021', 'n01739381', 'n01740131', 'n01742172', 'n01744401',
    'n01748264', 'n01749939', 'n01751748', 'n01753488', 'n01755581', 'n01756291', 'n01768244',
    'n01770081', 'n01770393', 'n01773157', 'n01773549', 'n01773797', 'n01774384', 'n01774750',
    'n01775062', 'n01776313', 'n01784675', 'n01795545', 'n01796340', 'n01797886', 'n01798484',
    'n01806143', 'n01806567', 'n01807496', 'n01817953', 'n01818515', 'n01819313', 'n01820546',
    'n01824575', 'n01828970', 'n01829413', 'n01833805', 'n01843065', 'n01843383', 'n01847000',
    'n01855032', 'n01855672', 'n01860187', 'n01871265', 'n01872401', 'n01873310', 'n01877812',
    'n01882714', 'n01883070', 'n01910747', 'n01914609', 'n01917289', 'n01924916', 'n01930112',
    'n01943899', 'n01944390', 'n01945685', 'n01950731', 'n01955084', 'n01968897', 'n01978287',
    'n01978455', 'n01980166', 'n01981276', 'n01983481', 'n01984695', 'n01985128', 'n01986214',
    'n01990800', 'n02002556', 'n02002724', 'n02006656', 'n02007558', 'n02009229', 'n02009912',
    'n02011460', 'n02012849', 'n02013706', 'n02017213', 'n02018207', 'n02018795', 'n02025239',
    'n02027492', 'n02028035', 'n02033041', 'n02037110', 'n02051845', 'n02056570', 'n02058221',
    'n02066245', 'n02071294', 'n02074367', 'n02077923', 'n02085620', 'n02085782', 'n02085936',
    'n02086079', 'n02086240', 'n02086646', 'n02086910', 'n02087046', 'n02087394', 'n02088094',
    'n02088238', 'n02088364', 'n02088466', 'n02088632', 'n02089078', 'n02089867', 'n02089973',
    'n02090379', 'n02090622', 'n02090721', 'n02091032', 'n02091134', 'n02091244', 'n02091467',
    'n02091635', 'n02091831', 'n02092002', 'n02092339', 'n02093256', 'n02093428', 'n02093647',
    'n02093754', 'n02093859', 'n02093991', 'n02094114', 'n02094258', 'n02094433', 'n02095314',
    'n02095570', 'n02095889', 'n02096051', 'n02096177', 'n02096294', 'n02096437', 'n02096585',
    'n02097047', 'n02097130', 'n02097209', 'n02097298', 'n02097474', 'n02097658', 'n02098105',
    'n02098286', 'n02098413', 'n02099267', 'n02099429', 'n02099601', 'n02099712', 'n02099849',
    'n02100236', 'n02100583', 'n02100735', 'n02100877', 'n02101006', 'n02101388', 'n02101556',
    'n02102040', 'n02102177', 'n02102318', 'n02102480', 'n02102973', 'n02104029', 'n02104365',
    'n02105056', 'n02105162', 'n02105251', 'n02105412', 'n02105505', 'n02105641', 'n02105855',
    'n02106030', 'n02106166', 'n02106382', 'n02106550', 'n02106662', 'n02107142', 'n02107312',
    'n02107574', 'n02107683', 'n02107908', 'n02108000', 'n02108089', 'n02108422', 'n02108551',
    'n02108915', 'n02109047', 'n02109525', 'n02109961', 'n02110063', 'n02110185', 'n02110341',
    'n02110627', 'n02110806', 'n02110958', 'n02111129', 'n02111277', 'n02111500', 'n02111889',
    'n02112018', 'n02112137', 'n02112350', 'n02112706', 'n02113023', 'n02113186', 'n02113624',
    'n02113712', 'n02113799', 'n02113978', 'n02114367', 'n02114548', 'n02114712', 'n02114855',
    'n02115641', 'n02115913', 'n02116738', 'n02117135', 'n02119022', 'n02119789', 'n02120079',
    'n02120505', 'n02123045', 'n02123159', 'n02123394', 'n02123597', 'n02124075', 'n02125311',
    'n02127052', 'n02128385', 'n02128757', 'n02128925', 'n02129165', 'n02129604', 'n02130308',
    'n02132136', 'n02133161', 'n02134084', 'n02134418', 'n02137549', 'n02138441', 'n02165105',
    'n02165456', 'n02167151', 'n02168699', 'n02169497', 'n02172182', 'n02174001', 'n02177972',
    'n02190166', 'n02206856', 'n02219486', 'n02226429', 'n02229544', 'n02231487', 'n02233338',
    'n02236044', 'n02256656', 'n02259212', 'n02264363', 'n02268443', 'n02268853', 'n02276258',
    'n02277742', 'n02279972', 'n02280649', 'n02281406', 'n02281787', 'n02317335', 'n02319095',
    'n02321529', 'n02325366', 'n02326432', 'n02328150', 'n02342885', 'n02346627', 'n02356798',
    'n02361337', 'n02363005', 'n02364673', 'n02389026', 'n02391049', 'n02395406', 'n02396427',
    'n02397096', 'n02398521', 'n02403003', 'n02408429', 'n02410509', 'n02412080', 'n02415577',
    'n02417914', 'n02422106', 'n02422699', 'n02423022', 'n02437312', 'n02437616', 'n02441942',
    'n02442845', 'n02443114', 'n02443484', 'n02444819', 'n02445715', 'n02447366', 'n02454379',
    'n02457408', 'n02480495', 'n02480855', 'n02481823', 'n02483362', 'n02483708', 'n02484975',
    'n02486261', 'n02486410', 'n02487347', 'n02488291', 'n02488702', 'n02489166', 'n02490219',
    'n02492035', 'n02492660', 'n02493509', 'n02493793', 'n02494079', 'n02497673', 'n02500267',
    'n02504013', 'n02504458', 'n02509815', 'n02510455', 'n02514041', 'n02526121', 'n02536864',
    'n02606052', 'n02607072', 'n02640242', 'n02641379', 'n02643566', 'n02655020', 'n02666196',
    'n02667093', 'n02669723', 'n02672831', 'n02676566', 'n02687172', 'n02690373', 'n02692877',
    'n02701002', 'n02704792', 'n02708093', 'n02727426', 'n02730930', 'n02747177', 'n02749479',
    'n02769748', 'n02782093', 'n02783161', 'n02786058', 'n02787622', 'n02790996', 'n02791124',
    'n02793495', 'n02794156', 'n02795169', 'n02797295', 'n02799071', 'n02802426', 'n02804414',
    'n02804610', 'n02808304', 'n02808440', 'n02814533', 'n02814860', 'n02815834', 'n02817516',
    'n02823428', 'n02823750', 'n02834397', 'n02835271', 'n02840245', 'n02841315', 'n02843684',
    'n02859443', 'n02860847', 'n02865351', 'n02869837', 'n02870880', 'n02879718', 'n02883205',
    'n02892201', 'n02892767', 'n02895154', 'n02906734', 'n02909870', 'n02910353', 'n02916936',
    'n02917067', 'n02930766', 'n02939185', 'n02948072', 'n02950826', 'n02951358', 'n02951585',
    'n02963159', 'n02965783', 'n02966687', 'n02971356', 'n02978881', 'n02979186', 'n02980441',
    'n02981792', 'n02992211', 'n02992529', 'n02999410', 'n03000134', 'n03000684', 'n03014705',
    'n03016953', 'n03017168', 'n03018349', 'n03026506', 'n03028079', 'n03041632', 'n03045698',
    'n03047690', 'n03062245', 'n03063599', 'n03063689', 'n03075370', 'n03095699', 'n03100240',
    'n03109150', 'n03110669', 'n03124043', 'n03124170', 'n03125729', 'n03126707', 'n03127747',
    'n03127925', 'n03131574', 'n03133878', 'n03134739', 'n03141823', 'n03146219', 'n03179701',
    'n03187595', 'n03188531', 'n03196217', 'n03197337', 'n03201208', 'n03207743', 'n03207941',
    'n03223299', 'n03240683', 'n03249569', 'n03250847', 'n03255030', 'n03259280', 'n03271574',
    'n03272010', 'n03272562', 'n03291819', 'n03297495', 'n03314780', 'n03325584', 'n03337140',
    'n03344393', 'n03345487', 'n03347037', 'n03355925', 'n03372029', 'n03376595', 'n03379051',
    'n03384352', 'n03388183', 'n03388549', 'n03393912', 'n03394916', 'n03400231', 'n03404251',
    'n03417042', 'n03424325', 'n03425413', 'n03443371', 'n03444034', 'n03445777', 'n03445924',
    'n03447447', 'n03447721', 'n03450230', 'n03452741', 'n03467068', 'n03476684', 'n03476991',
    'n03478589', 'n03481172', 'n03482405', 'n03483316', 'n03485407', 'n03485794', 'n03492542',
    'n03494278', 'n03495258', 'n03496892', 'n03498962', 'n03527444', 'n03530642', 'n03532672',
    'n03534580', 'n03535780', 'n03538406', 'n03544143', 'n03584254', 'n03584829', 'n03590841',
    'n03594734', 'n03594945', 'n03595614', 'n03598930', 'n03599486', 'n03602883', 'n03617480',
    'n03623198', 'n03627232', 'n03630383', 'n03633091', 'n03649909', 'n03657121', 'n03658185',
    'n03662601', 'n03666591', 'n03670208', 'n03673027', 'n03676483', 'n03680355', 'n03690938',
    'n03691459', 'n03692522', 'n03706229', 'n03709823', 'n03710193', 'n03717622', 'n03720891',
    'n03721384', 'n03724870', 'n03729826', 'n03733131', 'n03733805', 'n03742115', 'n03743016',
    'n03759954', 'n03761084', 'n03763968', 'n03764736', 'n03769881', 'n03770439', 'n03770679',
    'n03773504', 'n03775071', 'n03775546', 'n03776460', 'n03777568', 'n03777754', 'n03781244',
    'n03782006', 'n03785016', 'n03786901', 'n03787032', 'n03788195', 'n03788365', 'n03791053',
    'n03792782', 'n03792972', 'n03793489', 'n03794056', 'n03796401', 'n03803284', 'n03804744',
    'n03814639', 'n03814906', 'n03825788', 'n03832673', 'n03837869', 'n03838899', 'n03840681',
    'n03841143', 'n03843555', 'n03854065', 'n03857828', 'n03866082', 'n03868242', 'n03868863',
    'n03873416', 'n03874293', 'n03874599', 'n03876231', 'n03877472', 'n03877845', 'n03884397',
    'n03888257', 'n03888605', 'n03891251', 'n03891332', 'n03895866', 'n03902125', 'n03903868',
    'n03908618', 'n03908714', 'n03916031', 'n03920288', 'n03924679', 'n03929660', 'n03929855',
    'n03930313', 'n03930630', 'n03933933', 'n03935335', 'n03937543', 'n03938244', 'n03942813',
    'n03944341', 'n03947888', 'n03950228', 'n03954731', 'n03956157', 'n03958227', 'n03961711',
    'n03970156', 'n03976467', 'n03976657', 'n03977966', 'n03980874', 'n03982430', 'n03983396',
    'n03991062', 'n03992509', 'n03995372', 'n03998194', 'n04004767', 'n04005630', 'n04009552',
    'n04019541', 'n04023962', 'n04026417', 'n04033901', 'n04033995', 'n04037443', 'n04039381',
    'n04040759', 'n04041544', 'n04044716', 'n04049303', 'n04065272', 'n04067472', 'n04069434',
    'n04070727', 'n04074963', 'n04086273', 'n04090263', 'n04099969', 'n04116512', 'n04118538',
    'n04118776', 'n04120489', 'n04125021', 'n04127249', 'n04131690', 'n04133789', 'n04136333',
    'n04141076', 'n04141327', 'n04141975', 'n04146614', 'n04147183', 'n04149813', 'n04153751',
    'n04154565', 'n04162706', 'n04179913', 'n04192698', 'n04201297', 'n04204238', 'n04204347',
    'n04208210', 'n04209133', 'n04209239', 'n04228054', 'n04229816', 'n04235860', 'n04238763',
    'n04243546', 'n04252077', 'n04252225', 'n04254120', 'n04254680', 'n04254777', 'n04258138',
    'n04259630', 'n04263257', 'n04265275', 'n04266014', 'n04270147', 'n04273569', 'n04275548',
    'n04277352', 'n04285008', 'n04286575', 'n04310018', 'n04311004', 'n04311174', 'n04317175',
    'n04325704', 'n04326547', 'n04328186', 'n04330267', 'n04332243', 'n04335435', 'n04336792',
    'n04344873', 'n04346328', 'n04347754', 'n04350905', 'n04355338', 'n04355933', 'n04366367',
    'n04367480', 'n04370456', 'n04371430', 'n04371774', 'n04376876', 'n04380533', 'n04389033',
    'n04398044', 'n04399382', 'n04404412', 'n04409515', 'n04417672', 'n04418357', 'n04423845',
    'n04428191', 'n04429376', 'n04435653', 'n04442312', 'n04447861', 'n04456115', 'n04458633',
    'n04461696', 'n04465501', 'n04467665', 'n04476259', 'n04479046', 'n04482393', 'n04483307',
    'n04485082', 'n04486054', 'n04487081', 'n04487394', 'n04505470', 'n04507155', 'n04509417',
    'n04515003', 'n04517823', 'n04522168', 'n04525038', 'n04525305', 'n04532106', 'n04532670',
    'n04536866', 'n04540053', 'n04542943', 'n04548280', 'n04548362', 'n04550184', 'n04552348',
    'n04553703', 'n04554684', 'n04557648', 'n04560804', 'n04562935', 'n04579145', 'n04579432',
    'n04584207', 'n04589890', 'n04590129', 'n04591157', 'n04591713', 'n04596742', 'n04597913',
    'n04604644', 'n04606251', 'n04612504', 'n04613696', 'n06596364', 'n06794110', 'n06874185',
    'n07565083', 'n07579787', 'n07583066', 'n07584110', 'n07613480', 'n07614500', 'n07615774',
    'n07684084', 'n07693725', 'n07695742', 'n07697313', 'n07697537', 'n07711569', 'n07714571',
    'n07714990', 'n07715103', 'n07716358', 'n07716906', 'n07717410', 'n07717556', 'n07718472',
    'n07718747', 'n07720875', 'n07730033', 'n07734744', 'n07742313', 'n07745940', 'n07747607',
    'n07749582', 'n07753113', 'n07753275', 'n07753592', 'n07754684', 'n07760859', 'n07768694',
    'n07802026', 'n07831146', 'n07860988', 'n07871810', 'n07873807', 'n07875152', 'n07880968',
    'n07930864', 'n07932039', 'n09229709', 'n09246464', 'n09256479', 'n09835506', 'n10565667',
    'n11879895', 'n11939491', 'n12057211', 'n12144580', 'n12267677', 'n12620546', 'n12768682',
    'n12985857', 'n12998815', 'n13037406', 'n13040303', 'n13044778', 'n13052670', 'n13054560',
    'n13133613', 'n15075141'
}
imagenet_s_mask = [wnid in imagenet_s_wnids for wnid in all_wnids]
imagenet_s_classes = []
for i in range(len(imagenet_s_mask)):
    if imagenet_s_mask[i]:
        imagenet_s_classes.append(imagenet_classes[i])

cifar100_class_names = [
    "apple", "aquarium_fish", "baby", "bear", "beaver", "bed", "bee", "beetle",
    "bicycle", "bottle", "bowl", "boy", "bridge", "bus", "butterfly", "camel",
    "can", "castle", "caterpillar", "cattle", "chair", "chimpanzee", "clock",
    "cloud", "cockroach", "couch", "crab", "crocodile", "cup", "dinosaur",
    "dolphin", "elephant", "flatfish", "forest", "fox", "girl", "hamster", "house",
    "kangaroo", "keyboard", "lamp", "lawn_mower", "leopard", "lion", "lizard",
    "lobster", "man", "maple_tree", "motorcycle", "mountain", "mouse", "mushroom",
    "oak_tree", "orange", "orchid", "otter", "palm_tree", "pear", "pickup_truck",
    "pine_tree", "plain", "plate", "poppy", "porcupine", "possum", "rabbit",
    "raccoon", "ray", "road", "rocket", "rose", "sea", "seal", "shark", "shrew",
    "skunk", "skyscraper", "snail", "snake", "spider", "squirrel", "streetcar",
    "sunflower", "sweet_pepper", "table", "tank", "telephone", "television",
    "tiger", "tractor", "train", "trout", "tulip", "turtle", "wardrobe", "whale",
    "willow_tree", "wolf", "woman", "worm"
]
cifar10_class_names = [
    "airplane",   # 0
    "automobile", # 1
    "bird",       # 2
    "cat",        # 3
    "deer",       # 4
    "dog",        # 5
    "frog",       # 6
    "horse",      # 7
    "ship",       # 8
    "truck"       # 9
]
bimcv_class_names = [
    "a normal chest X-ray with clear lungs",
    "a chest X-ray with evident features of COVID-19, including ground-glass opacities, lung infiltrates, bilateral patchy shadows and consolidation"
]

sun397_classes = ['abbey', 'airplane_cabin', 'airport_terminal', 'alley', 'amphitheater', 'amusement_arcade', 'amusement_park', 'anechoic_chamber', 'outdoor apartment_building', 'indoor apse', 'aquarium', 'aqueduct', 'arch', 'archive', 'outdoor arrival_gate', 'art_gallery', 'art_school', 'art_studio', 'assembly_line', 'outdoor athletic_field', 'public atrium', 'attic', 'auditorium', 'auto_factory', 'badlands', 'indoor badminton_court', 'baggage_claim', 'shop bakery', 'exterior balcony', 'interior balcony', 'ball_pit', 'ballroom', 'bamboo_forest', 'banquet_hall', 'bar', 'barn', 'barndoor', 'baseball_field', 'basement', 'basilica', 'outdoor basketball_court', 'bathroom', 'batters_box', 'bayou', 'indoor bazaar', 'outdoor bazaar', 'beach', 'beauty_salon', 'bedroom', 'berth', 'biology_laboratory', 'indoor bistro', 'boardwalk', 'boat_deck', 'boathouse', 'bookstore', 'indoor booth', 'botanical_garden', 'indoor bow_window', 'outdoor bow_window', 'bowling_alley', 'boxing_ring', 'indoor brewery', 'bridge', 'building_facade', 'bullring', 'burial_chamber', 'bus_interior', 'butchers_shop', 'butte', 'outdoor cabin', 'cafeteria', 'campsite', 'campus', 'natural canal', 'urban canal', 'candy_store', 'canyon', 'backseat car_interior', 'frontseat car_interior', 'carrousel', 'indoor casino', 'castle', 'catacomb', 'indoor cathedral', 'outdoor cathedral', 'indoor cavern', 'cemetery', 'chalet', 'cheese_factory', 'chemistry_lab', 'indoor chicken_coop', 'outdoor chicken_coop', 'childs_room', 'indoor church', 'outdoor church', 'classroom', 'clean_room', 'cliff', 'indoor cloister', 'closet', 'clothing_store', 'coast', 'cockpit', 'coffee_shop', 'computer_room', 'conference_center', 'conference_room', 'construction_site', 'control_room', 'outdoor control_tower', 'corn_field', 'corral', 'corridor', 'cottage_garden', 'courthouse', 'courtroom', 'courtyard', 'exterior covered_bridge', 'creek', 'crevasse', 'crosswalk', 'office cubicle', 'dam', 'delicatessen', 'dentists_office', 'sand desert', 'vegetation desert', 'indoor diner', 'outdoor diner', 'home dinette', 'vehicle dinette', 'dining_car', 'dining_room', 'discotheque', 'dock', 'outdoor doorway', 'dorm_room', 'driveway', 'outdoor driving_range', 'drugstore', 'electrical_substation', 'door elevator', 'interior elevator', 'elevator_shaft', 'engine_room', 'indoor escalator', 'excavation', 'indoor factory', 'fairway', 'fastfood_restaurant', 'cultivated field', 'wild field', 'fire_escape', 'fire_station', 'indoor firing_range', 'fishpond', 'indoor florist_shop', 'food_court', 'broadleaf forest', 'needleleaf forest', 'forest_path', 'forest_road', 'formal_garden', 'fountain', 'galley', 'game_room', 'indoor garage', 'garbage_dump', 'gas_station', 'exterior gazebo', 'indoor general_store', 'outdoor general_store', 'gift_shop', 'golf_course', 'indoor greenhouse', 'outdoor greenhouse', 'indoor gymnasium', 'indoor hangar', 'outdoor hangar', 'harbor', 'hayfield', 'heliport', 'herb_garden', 'highway', 'hill', 'home_office', 'hospital', 'hospital_room', 'hot_spring', 'outdoor hot_tub', 'outdoor hotel', 'hotel_room', 'house', 'outdoor hunting_lodge', 'ice_cream_parlor', 'ice_floe', 'ice_shelf', 'indoor ice_skating_rink', 'outdoor ice_skating_rink', 'iceberg', 'igloo', 'industrial_area', 'outdoor inn', 'islet', 'indoor jacuzzi', 'indoor jail', 'jail_cell', 'jewelry_shop', 'kasbah', 'indoor kennel', 'outdoor kennel', 'kindergarden_classroom', 'kitchen', 'kitchenette', 'outdoor labyrinth', 'natural lake', 'landfill', 'landing_deck', 'laundromat', 'lecture_room', 'indoor library', 'outdoor library', 'outdoor lido_deck', 'lift_bridge', 'lighthouse', 'limousine_interior', 'living_room', 'lobby', 'lock_chamber', 'locker_room', 'mansion', 'manufactured_home', 'indoor market', 'outdoor market', 'marsh', 'martial_arts_gym', 'mausoleum', 'medina', 'water moat', 'outdoor monastery', 'indoor mosque', 'outdoor mosque', 'motel', 'mountain', 'mountain_snowy', 'indoor movie_theater', 'indoor museum', 'music_store', 'music_studio', 'outdoor nuclear_power_plant', 'nursery', 'oast_house', 'outdoor observatory', 'ocean', 'office', 'office_building', 'outdoor oil_refinery', 'oilrig', 'operating_room', 'orchard', 'outdoor outhouse', 'pagoda', 'palace', 'pantry', 'park', 'indoor parking_garage', 'outdoor parking_garage', 'parking_lot', 'parlor', 'pasture', 'patio', 'pavilion', 'pharmacy', 'phone_booth', 'physics_laboratory', 'picnic_area', 'indoor pilothouse', 'outdoor planetarium', 'playground', 'playroom', 'plaza', 'indoor podium', 'outdoor podium', 'pond', 'establishment poolroom', 'home poolroom', 'outdoor power_plant', 'promenade_deck', 'indoor pub', 'pulpit', 'putting_green', 'racecourse', 'raceway', 'raft', 'railroad_track', 'rainforest', 'reception', 'recreation_room', 'residential_neighborhood', 'restaurant', 'restaurant_kitchen', 'restaurant_patio', 'rice_paddy', 'riding_arena', 'river', 'rock_arch', 'rope_bridge', 'ruin', 'runway', 'sandbar', 'sandbox', 'sauna', 'schoolhouse', 'sea_cliff', 'server_room', 'shed', 'shoe_shop', 'shopfront', 'indoor shopping_mall', 'shower', 'skatepark', 'ski_lodge', 'ski_resort', 'ski_slope', 'sky', 'skyscraper', 'slum', 'snowfield', 'squash_court', 'stable', 'baseball stadium', 'football stadium', 'indoor stage', 'staircase', 'street', 'subway_interior', 'platform subway_station', 'supermarket', 'sushi_bar', 'swamp', 'indoor swimming_pool', 'outdoor swimming_pool', 'indoor synagogue', 'outdoor synagogue', 'television_studio', 'east_asia temple', 'south_asia temple', 'indoor tennis_court', 'outdoor tennis_court', 'outdoor tent', 'indoor_procenium theater', 'indoor_seats theater', 'thriftshop', 'throne_room', 'ticket_booth', 'toll_plaza', 'topiary_garden', 'tower', 'toyshop', 'outdoor track', 'train_railway', 'platform train_station', 'tree_farm', 'tree_house', 'trench', 'coral_reef underwater', 'utility_room', 'valley', 'van_interior', 'vegetable_garden', 'veranda', 'veterinarians_office', 'viaduct', 'videostore', 'village', 'vineyard', 'volcano', 'indoor volleyball_court', 'outdoor volleyball_court', 'waiting_room', 'indoor warehouse', 'water_tower', 'block waterfall', 'fan waterfall', 'plunge waterfall', 'watering_hole', 'wave', 'wet_bar', 'wheat_field', 'wind_farm', 'windmill', 'barrel_storage wine_cellar', 'bottle_storage wine_cellar', 'indoor wrestling_ring', 'yard', 'youth_hostel']

def get_class_names(dataset):
    if dataset == 'imagenet':
        return imagenet_classes
    elif dataset == 'imagenet200':
        return imagenet200_classes
    elif dataset == 'imagenet_a':  # ImageNet-A uses 200 classes
        return imagenet200_classes
    elif dataset == 'imagenet_r':  # ImageNet-R uses 200 classes
        return imagenet200_classes
    elif dataset == 'imagenet_v2':  # ImageNet-v2 uses full 1000 classes
        return imagenet_classes
    elif dataset == 'imagenet_s50':  # ImageNet-S50 uses 50 classes
        return imagenet_s50_classes
    elif dataset == 'imagenet_s':  # ImageNet-S (Full) uses 919 classes
        return imagenet_s_classes
    elif dataset == 'cifar10':
        return cifar10_class_names
    elif dataset == 'cifar100':
        return cifar100_class_names
    elif dataset == 'covid':
        return bimcv_class_names
    else:
        raise NotImplementedError

imagenet_templates = [
    'a bad photo of a {}.',
    'a photo of many {}.',
    'a sculpture of a {}.',
    'a photo of the hard to see {}.',
    'a low resolution photo of the {}.',
    'a rendering of a {}.',
    'graffiti of a {}.',
    'a bad photo of the {}.',
    'a cropped photo of the {}.',
    'a tattoo of a {}.',
    'the embroidered {}.',
    'a photo of a hard to see {}.',
    'a bright photo of a {}.',
    'a photo of a clean {}.',
    'a photo of a dirty {}.',
    'a dark photo of the {}.',
    'a drawing of a {}.',
    'a photo of my {}.',
    'the plastic {}.',
    'a photo of the cool {}.',
    'a close-up photo of a {}.',
    'a black and white photo of the {}.',
    'a painting of the {}.',
    'a painting of a {}.',
    'a pixelated photo of the {}.',
    'a sculpture of the {}.',
    'a bright photo of the {}.',
    'a cropped photo of a {}.',
    'a plastic {}.',
    'a photo of the dirty {}.',
    'a jpeg corrupted photo of a {}.',
    'a blurry photo of the {}.',
    'a photo of the {}.',
    'a good photo of the {}.',
    'a rendering of the {}.',
    'a {} in a video game.',
    'a photo of one {}.',
    'a doodle of a {}.',
    'a close-up photo of the {}.',
    'a photo of a {}.',
    'the origami {}.',
    'the {} in a video game.',
    'a sketch of a {}.',
    'a doodle of the {}.',
    'a origami {}.',
    'a low resolution photo of a {}.',
    'the toy {}.',
    'a rendition of the {}.',
    'a photo of the clean {}.',
    'a photo of a large {}.',
    'a rendition of a {}.',
    'a photo of a nice {}.',
    'a photo of a weird {}.',
    'a blurry photo of a {}.',
    'a cartoon {}.',
    'art of a {}.',
    'a sketch of the {}.',
    'a embroidered {}.',
    'a pixelated photo of a {}.',
    'itap of the {}.',
    'a jpeg corrupted photo of the {}.',
    'a good photo of a {}.',
    'a plushie {}.',
    'a photo of the nice {}.',
    'a photo of the small {}.',
    'a photo of the weird {}.',
    'the cartoon {}.',
    'art of the {}.',
    'a drawing of the {}.',
    'a photo of the large {}.',
    'a black and white photo of a {}.',
    'the plushie {}.',
    'a dark photo of a {}.',
    'itap of a {}.',
    'graffiti of the {}.',
    'a toy {}.',
    'itap of my {}.',
    'a photo of a cool {}.',
    'a photo of a small {}.',
    'a tattoo of the {}.',
]
simple_imagenet_template = ['a photo of a {}.']

############################### tip text prompts
tip_imagenet_templates = [
    'a bad photo of the {}.',
    'a {} in a video game.',
    'a origami {}.',
    'a photo of the small {}.',
    'art of the {}.',
    'a photo of the large {}.',
    'itap of a {}.',
]

def get_templates(text_prompt):
    if 'simple' == text_prompt:
        return simple_imagenet_template
    elif 'tip' == text_prompt:
        return tip_imagenet_templates
    elif 'vanilla' == text_prompt:
        return ['{}.']
    elif 'nice' == text_prompt:
        return ['The nice {}.']
    elif 'good' == text_prompt:
        return ['a good photo of the {}.']
    elif 'bad' == text_prompt:
        return ['a bad photo of the {}.']
    elif 'small' == text_prompt:
        return ['a photo of the small {}.']
    elif 'large' == text_prompt:
        return ['a photo of the large {}.']
    elif 'full' == text_prompt:
        return imagenet_templates
    else:
        raise NotImplementedError

def load_clip_to_cpu(backbone_name):
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)
    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    model = clip.build_model(state_dict or model.state_dict())

    # ipdb.set_trace()
    return model



# https://github.com/openai/CLIP/blob/main/notebooks/Prompt_Engineering_for_ImageNet.ipynb
def get_text_features(model, dataset, text_prompt):
    classnames = get_class_names(dataset) # imagenet --> imagenet class names
    templates = get_templates(text_prompt) # simple --> text prompt for each classes.
    print('adopt text prompt of', text_prompt)
    with torch.no_grad():
        text_features = []
        for classname in tqdm(classnames):
            texts = [template.format(classname) for template in templates]  # format with class
            ########################### use GPT generated text prompt
            # if 'cupl' in text_prompt:
            #     cupl_file = "CuPL_prompts_imagenet.json"
            #     f = open('./openood/networks/clip/gpt3_prompts/' + cupl_file)
            #     cupl_prompts = json.load(f)
            #     texts += cupl_prompts[classname]
            texts = clip.tokenize(texts).cuda()  # tokenize
            class_embeddings = model.encode_text(texts)  # embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            text_features.append(class_embedding)
        text_features = torch.stack(text_features, dim=1).cuda() # 512*1000
    # pdb.set_trace()
    mean_text = - text_features.mean(1) # use the mean directly | or the inverse mean.
    mean_text /= mean_text.norm()
    mean_text = mean_text.view(-1, 1)
    text_features = torch.cat((text_features, mean_text), dim=1)
    return text_features

# only for visualization, get the text features of SUN dataset
def get_text_features_sun(model, dataset, text_prompt):
    classnames = sun397_classes
    templates = get_templates(text_prompt) # simple --> text prompt for each classes.
    print('adopt text prompt of', text_prompt)
    with torch.no_grad():
        text_features = []
        for classname in tqdm(classnames):
            texts = [template.format(classname) for template in templates]  # format with class
            ########################### use GPT generated text prompt
            # if 'cupl' in text_prompt:
            #     cupl_file = "CuPL_prompts_imagenet.json"
            #     f = open('./openood/networks/clip/gpt3_prompts/' + cupl_file)
            #     cupl_prompts = json.load(f)
            #     texts += cupl_prompts[classname]
            texts = clip.tokenize(texts).cuda()  # tokenize
            class_embeddings = model.encode_text(texts)  # embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            text_features.append(class_embedding)
        text_features = torch.stack(text_features, dim=1).cuda() # 512*1000
    # pdb.set_trace()
    mean_text = - text_features.mean(1) # use the mean directly | or the inverse mean.
    mean_text /= mean_text.norm()
    mean_text = mean_text.view(-1, 1)
    text_features = torch.cat((text_features, mean_text), dim=1)
    return text_features


########################################### compared to FixedCLIP_NegOODPrompt, FixedCLIP_OODPrompt not use any negative text.
class FixedCLIP_OODPrompt(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        backbone = cfg.backbone.name # 'ViT-B/16'
        assert backbone in clip.available_models()
        classnames = get_class_names(cfg.backbone.dataset) # imagenet
        self.n_cls = len(classnames)
        # pdb.set_trace()
        self.n_output = self.n_cls + 10000
        # clip_model, self.preprocess = clip.load(backbone, device='cuda')
        print(f"Loading CLIP (backbone: {backbone})")
        clip_model = load_clip_to_cpu(backbone)

        # Use GPU if available, otherwise use CPU (CPU will be very slow!)
        if torch.cuda.is_available():
            clip_model = clip_model.cuda()
            print("Using GPU for inference")
        else:
            print("WARNING: No GPU available! Using CPU - this will be VERY slow!")
            print("Recommend running on a machine with NVIDIA GPU for practical use.")

        self.model = clip_model
        self.logit_scale = clip_model.logit_scale.data
        print("Turning off gradients in both the image and the text encoder")
        for name, param in clip_model.named_parameters():
            param.requires_grad_(False) ## fix clip model.
        self.text_features = get_text_features(self.model, cfg.backbone.dataset, cfg.backbone.text_prompt)


    def forward(self, x, return_feat=False):
        image_features = self.model.encode_image(x)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        logit_scale = self.logit_scale.exp()
        if return_feat:
            return image_features, self.text_features.t(), logit_scale
        else:
            logits = logit_scale * image_features @ self.text_features
            return logits


def get_selected_ood_text_list(
    dataset='imagenet',
    total_ood_num=1000,
    use_ood_labels=False,
    cache_root=None,
):
    """
    Select OOD negative labels from WordNet for a given ID dataset.

    Args:
        dataset       : Name of the ID dataset (e.g. 'imagenet', 'cifar10').
        total_ood_num : Total number of negative label prototypes to select.
        use_ood_labels: If True, forcibly include real OOD dataset class names
                        (SUN397, Places365, DTD/Textures) before filling the
                        remainder with WordNet words. Defaults to False, which
                        uses only WordNet-derived words.

    Returns:
        selected_adj_text   : List of selected adjective negative labels.
        selected_noun_text  : List of selected noun negative labels.
        unselected_adj_text : List of adjectives NOT selected (for ablations).
        unselected_noun_text: List of nouns NOT selected (for ablations).
    """
    print('the ID dataset is:', dataset)
    print('total_ood_num is:', total_ood_num)
    print('use_ood_labels (force-include SUN/Places/Textures class names):', use_ood_labels)

    foot_path = cache_root or os.path.join(os.getcwd(), 'txtfiles_output')
    if not os.path.isdir(foot_path):
        raise FileNotFoundError(f"txtfiles_output directory not found: {foot_path}")

    # The precomputed file name is the same regardless of use_ood_labels so
    # that a single cached file covers both experimental conditions. To switch
    # between the two conditions, delete the cached file and rerun with the
    # desired use_ood_labels setting. The new file will be regenerated and
    # saved automatically.
    precomputed_path = os.path.join(
        foot_path,
        dataset + f'_selected_adj_noun_0.95_{total_ood_num}_dedup.pth',
    )

    if os.path.exists(precomputed_path):
        print(f"\n[OK] Loading pre-generated negative labels from cache:")
        print(f"  {precomputed_path}")
        data = torch.load(precomputed_path)

        selected_adj_text  = data['adj']
        selected_noun_text = data['noun']

        # Display statistics about what is in the cached file so the user
        # can verify whether it was generated with or without forced OOD labels.
        has_ood = 'forced_adj' in data and 'forced_noun' in data
        if has_ood:
            forced_adj_count  = len(data['forced_adj'])
            forced_noun_count = len(data['forced_noun'])
            if forced_adj_count > 0 or forced_noun_count > 0:
                print(f"\n  Cache contains forced real-OOD class names:")
                print(f"    - DTD/Textures adjectives : {forced_adj_count}")
                print(f"    - SUN/Places nouns        : {forced_noun_count}")
                print(f"    - Total forced            : {forced_adj_count + forced_noun_count}")
            else:
                print(f"\n  Standard WordNet-only version (no forced OOD class names).")
        else:
            print(f"\n  Standard WordNet-only version (no forced OOD class names).")

        print(f"  Total negative labels loaded: {len(selected_adj_text) + len(selected_noun_text)}")

        # Return empty lists for the unselected split because we loaded from
        # a cached file that does not store the full unselected list.
        return selected_adj_text, selected_noun_text, [], []
    else:
        # Cache file not found -- generate it from scratch. This involves
        # computing cosine similarities between WordNet words and ID class
        # names using the CLIP text encoder, which can take several minutes
        # on a GPU (or much longer on CPU).
        print(f"\n[!] Pre-generated file not found: {precomputed_path}")
        print(f"    Generating from scratch now...\n")

    # ========== Load WordNet cosine-similarity data ==========
    wordnet_processed_path = os.path.join(foot_path, f'wordnet_{dataset}_cossim_dedup.pth')
    if os.path.exists(wordnet_processed_path):
        wordnet_dict = torch.load(wordnet_processed_path)
    else:
        classnames = get_class_names(dataset)
        generate_cossim_idname_wordnet_dedup(classnames, wordnet_processed_path, foot_path)
        wordnet_dict = torch.load(wordnet_processed_path)

    can_list_adj = wordnet_dict['text_list_adj']
    can_cos_adj = wordnet_dict['cos_sim_adj']
    can_list_noun = wordnet_dict['text_list_noun']
    can_cos_noun = wordnet_dict['cos_sim_noun']

    print('total length of WordNet adj texts', len(can_list_adj))
    print('total length of WordNet noun texts', len(can_list_noun))

    # ========== Optionally force-include real OOD dataset class names ==========
    # When use_ood_labels=True, we first add all class names from the actual
    # OOD evaluation datasets (DTD, SUN397, Places365) as forced negative
    # labels. The remaining slots (up to total_ood_num) are then filled with
    # WordNet-derived words selected by cosine similarity. This gives the model
    # prior knowledge about the evaluation OOD distributions, which is useful
    # for studying an oracle upper bound but should NOT be used when evaluating
    # a fair zero-shot baseline.
    forced_adj  = []
    forced_noun = []

    if use_ood_labels:
        print("\nLoading real OOD dataset class names to force-include as negative labels...")

        # 1. Load DTD texture categories (adjective-like words, e.g. 'banded', 'bubbly').
        dtd_file = os.path.join(foot_path, "textures_dtd_47.txt")
        if os.path.exists(dtd_file):
            with open(dtd_file, 'r') as f:
                dtd_classes = [line.strip() for line in f if line.strip()]
            forced_adj.extend(dtd_classes)
            print(f"  [OK] DTD/Textures: loaded {len(dtd_classes)} adjective-type class names.")
        else:
            print(f"  [MISS] DTD category file not found: {dtd_file}")
            print(f"         Warning: DTD class names will be skipped.")

        # 2. Load SUN397 scene categories (noun-type words, e.g. 'airport', 'alley').
        sun_file = os.path.join(foot_path, "sun397_categories.txt")
        if os.path.exists(sun_file):
            with open(sun_file, 'r') as f:
                sun_classes = [line.strip() for line in f if line.strip()]
            forced_noun.extend(sun_classes)
            print(f"  [OK] SUN397: loaded {len(sun_classes)} noun-type class names.")
        else:
            print(f"  [MISS] SUN397 category file not found: {sun_file}")

        # 3. Load Places365 scene categories (noun-type words).
        places_file = os.path.join(foot_path, "places365_categories.txt")
        if os.path.exists(places_file):
            with open(places_file, 'r') as f:
                places_classes = [line.strip() for line in f if line.strip()]
            forced_noun.extend(places_classes)
            print(f"  [OK] Places365: loaded {len(places_classes)} noun-type class names.")
        else:
            print(f"  [MISS] Places365 category file not found: {places_file}")

        # Deduplicate in case any word appears in both SUN and Places lists.
        forced_adj  = list(set(forced_adj))
        forced_noun = list(set(forced_noun))

        print(f"\n  Forced OOD class names summary:")
        print(f"    - Adjectives (DTD)        : {len(forced_adj)}")
        print(f"    - Nouns (SUN + Places)    : {len(forced_noun)}")
        print(f"    - Total forced            : {len(forced_adj) + len(forced_noun)}")

    # ========== Determine how many WordNet words to select ==========
    # After reserving slots for forced OOD class names (if any), the remaining
    # budget is filled with WordNet words selected by cosine similarity.
    forced_total  = len(forced_adj) + len(forced_noun)
    remaining_num = max(0, total_ood_num - forced_total)

    if use_ood_labels:
        print(f"\nSelecting {remaining_num} additional words from WordNet to fill the budget...")

    # Split the remaining budget between adjectives and nouns in proportion to
    # the sizes of their respective WordNet candidate pools.
    if remaining_num > 0:
        adj_ratio = len(can_list_adj) / (len(can_list_adj) + len(can_list_noun))
        adj_num   = int(remaining_num * adj_ratio)
        noun_num  = remaining_num - adj_num
    else:
        adj_num  = 0
        noun_num = 0
        if use_ood_labels:
            print("  Warning: the forced OOD class names already meet or exceed "
                  "the total_ood_num budget. No additional WordNet words will be added.")

    # ========== Select the least ID-similar words from WordNet ==========
    cate_num = can_cos_adj.size(1)
    cos_sim_indice_selected = int(cate_num * 0.95)

    value_cos_adj, _ = can_cos_adj.sort(1)
    value_cos_noun, _ = can_cos_noun.sort(1)

    selected_value_cos_adj = value_cos_adj[:, cos_sim_indice_selected]
    selected_value_cos_noun = value_cos_noun[:, cos_sim_indice_selected]

    value_sim_adj, indice_sim_adj = selected_value_cos_adj.sort(0)
    value_sim_noun, indice_sim_noun = selected_value_cos_noun.sort(0)

    # Pick the adj_num words from WordNet with the lowest cosine similarity
    # to any ID class name (i.e. the most semantically distant from ID).
    wordnet_adj_text  = [can_list_adj[i]  for i in indice_sim_adj[:adj_num]]
    wordnet_noun_text = [can_list_noun[i] for i in indice_sim_noun[:noun_num]]

    # Combine forced OOD class names (if any) with the WordNet selection.
    # Forced names come first so they are always included even if the budget
    # is tight.
    selected_adj_text  = forced_adj  + wordnet_adj_text
    selected_noun_text = forced_noun + wordnet_noun_text

    # Keep track of words that were NOT selected. These are useful for
    # ablation experiments that compare the effect of choosing more vs fewer
    # negative labels, or words with different similarity thresholds.
    unselected_adj_text  = [can_list_adj[i]  for i in indice_sim_adj[adj_num:]]
    unselected_noun_text = [can_list_noun[i] for i in indice_sim_noun[noun_num:]]

    print(f'\nFinal selected negative labels:')
    print(f'  - Adjectives : {len(selected_adj_text):5d}  '
          f'(forced OOD: {len(forced_adj)}, WordNet: {len(wordnet_adj_text)})')
    print(f'  - Nouns      : {len(selected_noun_text):5d}  '
          f'(forced OOD: {len(forced_noun)}, WordNet: {len(wordnet_noun_text)})')
    print(f'  - Total      : {len(selected_adj_text) + len(selected_noun_text)}')

    # ========== Save results to disk ==========
    # Both the selected and unselected splits are saved under a unified file
    # name that does not encode the use_ood_labels flag. To regenerate with
    # a different setting, simply delete these files and rerun.
    selected_adj_noun_path = os.path.join(
        foot_path,
        dataset + f'_selected_adj_noun_0.95_{total_ood_num}_dedup.pth',
    )
    unselected_adj_noun_path = os.path.join(
        foot_path,
        dataset + f'_unselected_adj_noun_0.95_{total_ood_num}_dedup.pth',
    )

    save_dict = {
        'adj': selected_adj_text,
        'noun': selected_noun_text,
        'forced_adj': forced_adj,
        'forced_noun': forced_noun,
        'wordnet_adj': wordnet_adj_text,
        'wordnet_noun': wordnet_noun_text,
    }
    torch.save(save_dict, selected_adj_noun_path)

    save_dict_unsel = {
        'adj': unselected_adj_text,
        'noun': unselected_noun_text
    }
    torch.save(save_dict_unsel, unselected_adj_noun_path)

    print(f'\n[OK] Saved selected   labels to: {selected_adj_noun_path}')
    print(f'[OK] Saved unselected labels to: {unselected_adj_noun_path}')

    return selected_adj_text, selected_noun_text, unselected_adj_text, unselected_noun_text


# https://github.com/openai/CLIP/blob/main/notebooks/Prompt_Engineering_for_ImageNet.ipynb
def get_text_features_neg(
    model,
    dataset,
    text_prompt,
    text_center,
    num_negative_labels,
    use_ood_labels=False,
    cache_root=None,
):
    """
    Generate a combined [D, num_id + num_ood] text feature matrix for both ID
    classes and OOD negative label prototypes using the frozen CLIP text encoder.

    Args:
        model        : A CLIP model instance (or any object with encode_text()).
        dataset      : Name of the ID dataset, e.g. 'imagenet'. Used to look up
                       class names and to find the correct pre-computed .pth files.
        text_prompt  : Prompt template style, e.g. 'nice', 'simple'. Passed to
                       get_templates() to build the list of prompt strings.
        text_center  : If True, subtract the mean feature vector from all
                       prototypes after generation (mean-centering).
        num_negative_labels: Number of OOD negative label prototypes to include.
        use_ood_labels: Passed through to get_selected_ood_text_list(). When
                        True, real OOD dataset class names (SUN, Places, DTD)
                        are force-included as negative labels.

    Returns:
        text_features: FloatTensor of shape [D, num_id + num_ood], where the
                       first num_id columns are ID prototypes and the remaining
                       num_ood columns are OOD negative label prototypes.
        None         : Second return value reserved for future use.
    """
    # Detect device from model
    device = next(model.parameters()).device

    classnames = get_class_names(dataset) # imagenet --> imagenet class names
    templates = get_templates(text_prompt) # simple --> text prompt for each classes.
    print('the noun template is set as:', templates)
    with torch.no_grad():
        text_features = []
        for classname in tqdm(classnames):
            texts = [template.format(classname) for template in templates]  # format with class
            # if 'cupl' in text_prompt:
            #     cupl_file = "CuPL_prompts_imagenet.json"
            #     f = open('./openood/networks/clip/gpt3_prompts/' + cupl_file)
            #     cupl_prompts = json.load(f)
            #     texts += cupl_prompts[classname]
            texts = clip.tokenize(texts).to(device)  # tokenize
            class_embeddings = model.encode_text(texts)  # embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True) # N*D
            if text_center:
                class_embedding = class_embeddings.mean(dim=0)
                class_embedding /= class_embedding.norm()
                text_features.append(class_embedding)
            else:
                text_features.append(class_embeddings)
        text_features = torch.stack(text_features, dim=1).to(device) # 512*1000

    # mean_text = - text_features.mean(1) # use the mean directly | or the inverse mean.
    # mean_text /= mean_text.norm()
    # mean_text = mean_text.view(-1, 1)
    selected_adj_text, selected_noun_text, unselected_adj_text, unselected_noun_text = get_selected_ood_text_list(
        dataset=dataset,
        total_ood_num=num_negative_labels,
        use_ood_labels=use_ood_labels,
        cache_root=cache_root,
    )
    adj_imagenet_template = ['This is a {} photo.']
    with torch.no_grad():
        text_features_neg = []
        print('the adj template is set as:', adj_imagenet_template)
        for classname in tqdm(selected_adj_text):
            texts = [template.format(classname) for template in adj_imagenet_template]  # format with class
            texts = clip.tokenize(texts).to(device)  # tokenize
            class_embeddings = model.encode_text(texts)  # embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            if text_center:
                class_embedding = class_embeddings.mean(dim=0)
                class_embedding /= class_embedding.norm()
                text_features_neg.append(class_embedding)
            else:
                text_features_neg.append(class_embeddings)
        print('the noun template is set as:', templates)
        for classname in tqdm(selected_noun_text):
            texts = [template.format(classname) for template in templates]  # format with class
            texts = clip.tokenize(texts).to(device)  # tokenize
            class_embeddings = model.encode_text(texts)  # embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            if text_center:
                class_embedding = class_embeddings.mean(dim=0)
                class_embedding /= class_embedding.norm()
                text_features_neg.append(class_embedding)
            else:
                text_features_neg.append(class_embeddings)

        ############################ Avoiding unnecessary computation for text_features_unselected
        text_features_unselected = text_features_neg
        # text_features_unselected = []
        # for classname in tqdm(unselected_adj_text):
        #     texts = [template.format(classname) for template in adj_imagenet_template]  # format with class
        #     texts = clip.tokenize(texts).to(device)  # tokenize
        #     class_embeddings = model.encode_text(texts)  # embed with text encoder
        #     class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
        #     if text_center:
        #         class_embedding = class_embeddings.mean(dim=0)
        #         class_embedding /= class_embedding.norm()
        #         text_features_unselected.append(class_embedding)
        #     else:
        #         text_features_unselected.append(class_embeddings)

        # for classname in tqdm(unselected_noun_text):
        #     texts = [template.format(classname) for template in templates]  # format with class
        #     texts = clip.tokenize(texts).to(device)  # tokenize
        #     class_embeddings = model.encode_text(texts)  # embed with text encoder
        #     class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
        #     if text_center:
        #         class_embedding = class_embeddings.mean(dim=0)
        #         class_embedding /= class_embedding.norm()
        #         text_features_unselected.append(class_embedding)
        #     else:
        #         text_features_unselected.append(class_embeddings)

        text_features_unselected = torch.stack(text_features_unselected, dim=1).to(device) # 512*1000  or  torch.Size([7, 10000, 512])
        text_features_neg = torch.stack(text_features_neg, dim=1).to(device) # 512*1000  or  torch.Size([7, 10000, 512])

    text_features_id_ood = torch.cat((text_features, text_features_neg), dim=1)
    # pdb.set_trace()
    return text_features_id_ood, text_features_unselected


class FixedCLIP_NegOODPrompt(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        backbone = cfg.backbone.name # 'ViT-B/16'
        classnames = get_class_names(cfg.backbone.dataset) # imagenet
        self.n_cls = len(classnames)
        # pdb.set_trace()
        self.n_output = self.n_cls + 10000
        assert backbone in clip.available_models()
        # clip_model, self.preprocess = clip.load(backbone, device='cuda')
        print(f"Loading CLIP (backbone: {backbone})")
        clip_model = load_clip_to_cpu(backbone)

        # Use GPU if available, otherwise use CPU (CPU will be very slow!)
        if torch.cuda.is_available():
            clip_model = clip_model.cuda()
            print("Using GPU for inference")
        else:
            print("WARNING: No GPU available! Using CPU - this will be VERY slow!")
            print("Recommend running on a machine with NVIDIA GPU for practical use.")

        self.model = clip_model
        self.logit_scale = clip_model.logit_scale.data
        print("Turning off gradients in both the image and the text encoder")
        for name, param in clip_model.named_parameters():
            param.requires_grad_(False) ## fix clip model.
        # text_features: feat_dim * text_num
        self.text_features, self.text_features_unselected = get_text_features_neg(
            self.model,
            cfg.backbone.dataset,
            cfg.backbone.text_prompt,
            cfg.backbone.text_center,
            cfg.backbone.num_negative_labels,
        )

        # print('get the text features of SUN classes for visualization.')
        # self.sun_features = get_text_features_sun(self.model, cfg.backbone.dataset, cfg.backbone.text_prompt)
        self.text_features_all = torch.cat((self.text_features, self.text_features_unselected), dim=1)  ## 512* (11k + 80k)


    def forward(self, x, return_feat=False):
        image_features = self.model.encode_image(x)
        image_features /= image_features.norm(dim=-1, keepdim=True)  ## batch * dim
        logit_scale = self.logit_scale.exp()
        # pdb.set_trace()

        if return_feat:
            return image_features, self.text_features.transpose(0,1), logit_scale ## 7*11k*512 --> 11K*7*512
        else:
            logits = logit_scale * image_features @ self.text_features  ## when eval acc, the forward pass is here.
            return logits
