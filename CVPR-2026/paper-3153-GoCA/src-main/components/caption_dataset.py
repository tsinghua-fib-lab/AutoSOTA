import os
import xml
import json
import glob
import pickle
import numpy as np
from PIL import Image
import scipy.io as sio
import torch.utils.data as data

from .dataset_util.coco_macro import COCO_CATEGORIES

mapping_20 = { 
	0: 0,
	1: 0,
	2: 0,
	3: 0,
	4: 0,
	5: 0,
	6: 0,
	7: 1,
	8: 2,
	9: 0,
	10: 0,
	11: 3,
	12: 4,
	13: 5,
	14: 0,
	15: 0,
	16: 0,
	17: 6,
	18: 0,
	19: 7,
	20: 8,
	21: 9,
	22: 10,
	23: 11,
	24: 12,
	25: 13,
	26: 14,
	27: 15,
	28: 16,
	29: 0,
	30: 0,
	31: 17,
	32: 18,
	33: 19,
	-1: 0
}

mapping_coco = {}
with open('fine_to_coarse_dict.pickle', 'rb') as f:
	d = pickle.load(f)
	d = d['fine_index_to_coarse_index']
	for k, v in d.items():
		mapping_coco[k+1] = v + 1
	mapping_coco[0] = 0
	mapping_coco[255] = 0

def encode_labels(mask, mapping):
	label_mask = np.zeros_like(mask)
	for k in mapping:
		label_mask[mask == k] = mapping[k]
	return label_mask

class ManualDataset(data.Dataset):
	def __init__(self, samples):
		super().__init__()

		self.samples = samples

	def __len__(self):
		return len(self.samples)

	def __getitem__(self, idx):
		sample = self.samples[idx]

		return sample, None, None, None, None, None


class CityDataset(data.Dataset):
	def __init__(self, samples, labels):
		super().__init__()

		self.samples = samples
		self.labels = labels

		self.class_name = [
			[], ['road', 'roads'], ['sidewalk', 'sidewalks'], ['building', 'buildings'], ['wall', 'walls'],
			['fence', 'fences'], ['pole', 'poles'], ['traffic light', 'traffic lights'],
			['traffic sign', 'traffic signs'], ['vegetation'], ['terrain'], ['sky'],
			['pedestrian', 'pedestrians'], ['rider', 'riders'], ['car', 'cars'], ['truck', 'trucks'],
			['bus', 'buses'], ['train', 'trains'], ['motorcycle', 'motorcycles'], ['bicycle', 'bicycles']
		]

	def __len__(self):
		return len(self.samples)

	def __getitem__(self, idx):
		sample = self.samples[idx]

		label = self.labels[idx]
		label = np.array(Image.open(label))
		label = encode_labels(label, mapping_20)
		label = label.reshape(-1)
		existent_objects_gt = set()
		for i in range(label.shape[0]):
			existent_objects_gt.add(label[i].item())
		if 0 in existent_objects_gt:
			existent_objects_gt.remove(0)
		existent_objects_gt = list(existent_objects_gt)
		return sample, existent_objects_gt


class HorseDataset(data.Dataset):
	def __init__(self, samples, labels):
		super().__init__()

		self.samples = samples
		self.labels = labels

	def __len__(self):
		return len(self.samples)

	def __getitem__(self, idx):
		sample = self.samples[idx]

		label = self.labels[idx]
		label = np.load(label)

		class_prompt = [
			'background', 'person', 'back', 'barrel', 'bridle', 'chest', 'ear', 'eye', 'forelock',
			'head', 'hoof', 'leg', 'mane', 'muzzle', 'neck', 'nostril', 'tail', 'thigh', 'saddle', 'shoulder',
			'leg protection'
		]
		base_prompt = 'a horse and a person in the wild'

		return sample, label, class_prompt, base_prompt

class PascalVOCDataset(data.Dataset):
	def __init__(self, samples, labels):
		super().__init__()

		self.samples = samples
		self.labels = labels

		self.class_name = [
			[], ['aeroplane', 'plane', 'airplane', 'airplanes', 'jet', 'jets', 'airliner', 'skyline', 'planes'],
			['bicycle', 'bicycles', 'bike', 'bikes'],
			[
				'bird', 'ostrich', 'seagull', 'pigeon', 'owl', 'penguin', 'goose', 'birds', 'hawk',
				'cranes', 'vulture', 'heron', 'duck', 'dove', 'puffins', 'parrot', 'parrots', 'quail'
			],
			['boat', 'ship', 'boats', 'ships', 'tugboat', 'tugboats', 'sailboat', 'sailboats'],
			['bottle', 'bottles', 'wine'],
			['bus', 'buses'], ['car', 'cars', 'vehicle', 'suv', 'truck', 'trucks', 'mustang', 'sedan'],
			['cat', 'cats', 'kitten'],
			['chair', 'chairs'], ['cow', 'cows'], ['diningtable', 'table', 'tables'],
			['dog', 'dogs', 'bulldog', 'pug', 'puppy'], ['horse', 'horses'],
			['motorbike', 'motorbikes', 'motorcycle', 'motorcycles'],
			[
				'person', 'man', 'woman', 'men', 'women', 'lady', 'kid', 'child', 'children', 'boy',
				'girl', 'baby', 'people', 'woodpecker'
			],
			['pottedplant', 'plant', 'plants'], ['sheep', 'goat', 'lamb'], ['sofa', 'couch'],
			['train', 'trains'],
			['tvmonitor', 'tv', 'tvs', 'monitor', 'monitors', 'television']
		]

	def __len__(self):
		return len(self.samples)

	def __getitem__(self, idx):
		sample = self.samples[idx]

		xml_path = sample.replace('JPEGImages', 'Annotations').replace('.jpg', '.xml')
		xml_content = xml.etree.ElementTree.parse(xml_path)
		objects = list(set([obj.find('name').text for obj in xml_content.findall('object')]))
		name_to_id = [
			'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
			'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
			'person', 'pottedplant',
			'sheep', 'sofa', 'train', 'tvmonitor',
		]
		name_to_id = {name: idx for idx, name in enumerate(name_to_id)}
		existent_objects_gt = [name_to_id[o] for o in objects]
		return sample, existent_objects_gt


class COCODataset(data.Dataset):
	def __init__(self, samples, labels):
		super().__init__()

		self.samples = samples
		self.labels = labels

		self.class_name = [
			[],  # 0
			['electronics', 'electronic'],  # 1
			['appliance', 'appliances'],  # 2
			['food'],  # 3
			['furniture', 'furnitures'],  # 4
			['indoor'],  # 5
			['kitchen stuff'],  # 6
			['accessory', 'accessories'],  # 7
			['animal', 'animals'],  # 8
			['outdoor'],  # 9
			['person', 'persons', 'man', 'men', 'woman', 'women', 'child', 'children', 'kid', 'kids'],  # 10
			['sport equipment', 'sport equipments'],  # 11
			['vehicle', 'vehicles'],  # 12
			['ceiling'],  # 13
			['floor'],  # 14
			['food'],  # 15
			['furniture'],  # 16
			['raw material'],  # 17
			['textile'],  # 18
			['wall', 'walls'],  # 19
			['window', 'windows'],  # 20
			['building', 'buildings'],  # 21
			['ground', 'grounds'],  # 22
			['plant', 'plants'],  # 23
			['sky'],  # 24
			['solid'],  # 25
			['structural'],  # 26
			['water'],  # 27
		]

	def __len__(self):
		return len(self.samples)

	def __getitem__(self, idx):
		sample = self.samples[idx]

		label = self.labels[idx]
		label = np.array(Image.open(label))
		label = encode_labels(label, mapping_coco)
		label[label == 15] = 3
		label[label == 16] = 4

		label = label.reshape(-1)
		existent_objects_gt = set()
		for i in range(label.shape[0]):
			existent_objects_gt.add(label[i].item())
		if 0 in existent_objects_gt:
			existent_objects_gt.remove(0)
		existent_objects_gt = list(existent_objects_gt)

		return sample, existent_objects_gt


class ADEDataset(data.Dataset):
	def __init__(self, samples, labels):
		super().__init__()

		self.samples = samples
		self.labels = labels

		self.class_name = [[], ['wall', 'walls'], ['building', 'buildings' 'edifice'], ['sky'], ['floor', 'flooring'], ['tree', 'trees'], ['ceiling'], ['road', 'roads', 'route'], ['bed', 'beds'], ['windowpane', 'window', 'windows'], ['grass'], ['cabinet', 'cabinets'], ['sidewalk', 'pavement'], ['person', 'individual', 'someone', 'somebody', 'mortal', 'soul', 'man', 'woman', 'men', 'women'], ['earth', 'ground'], ['door', 'doors', 'double door'], ['table', 'tables'], ['mountain', 'mount'], ['plant', 'flora', 'plant life'], ['curtain', 'drape', 'drapery', 'mantle', 'pall'], ['chair', 'chairs'], ['car', 'cars', 'auto', 'automobile', 'machine', 'motorcar'], ['water'], ['painting', 'paintings', 'picture'], ['sofa', 'couch', 'lounge'], ['shelf', 'shelves'], ['house', 'hourses'], ['sea'], ['mirror'], ['rug', 'carpet', 'carpeting'], ['field'], ['armchair', 'armchairs'], ['seat', 'seats'], ['fence', 'fencing'], ['desk', 'desks'], ['rock', 'rocks', 'stone', 'stones'], ['wardrobe', 'closet', 'press'], ['lamp', 'lamps'], ['bathtub', 'bathing tub', 'bath', 'tub'], ['railing', 'rail'], ['cushion'], ['base', 'pedestal', 'stand'], ['box', 'boxes'], ['column', 'columns', 'pillar', 'pillars'], ['signboard', 'sign'], ['chest of drawers', 'chest', 'chests', 'bureau', 'dresser'], ['counter'], ['sand'], ['sink'], ['skyscraper', 'skyscrapers'], ['fireplace', 'hearth', 'open fireplace'], ['refrigerator', 'refrigerators', 'icebox'], ['grandstand', 'covered stand'], ['path'], ['stairs', 'steps'], ['runway'], ['case', 'cases', 'display case', 'showcase', 'vitrine'], ['pool table', 'pool tables', 'billiard table', 'billiard tables', 'snooker table', 'snooker tables'], ['pillow'], ['screen door', 'screen', 'screens'], ['stairway', 'staircase'], ['river', 'rivers'], ['bridge', 'bridges', 'span'], ['bookcase', 'bookcases'], ['blind', 'screen'], ['coffee table', 'coffee tables', 'cocktail table', 'cocktail tables'], ['toilet', 'can', 'commode', 'crapper', 'pot', 'pots', 'potty', 'stool', 'throne'], ['flower', 'flowers'], ['book', 'books'], ['hill', 'hills'], ['bench', 'benches'], ['countertop'], ['stove', 'stoves', 'kitchen stove', 'range', 'kitchen range', 'cooking stove'], ['palm', 'palm tree', 'palm trees'], ['kitchen island'], ['computer', 'computers', 'computing machine', 'computing device', 'data processor', 'electronic computer', 'information processing system'], ['swivel chair', 'swivel chairs'], ['boat', 'boats'], ['bar'], ['arcade machine'], ['hovel', 'hut', 'huts', 'hutch', 'shack', 'shanty'], ['bus', 'buses', 'autobus', 'coach', 'coaches', 'charabanc', 'double-decker', 'double-deckers', 'jitney', 'motorbus', 'motorcoach', 'omnibus', 'passenger vehicle'], ['towel'], ['light', 'lights', 'light source'], ['truck', 'trucks', 'motortruck'], ['tower', 'towers'], ['chandelier', 'pendant', 'pendent'], ['awning', 'sunshade', 'sunblind'], ['streetlight', 'streetlights', 'street lamp', 'street lamps'], ['booth', 'cubicle', 'stall', 'kiosk'], ['television receiver', 'television', 'television set', 'tv', 'tv set', 'idiot box', 'boob tube', 'telly', 'goggle box'], ['airplane', 'airplanes', 'aeroplane', 'aeroplanes', 'plane', 'planes'], ['dirt track'], ['apparel', 'wearing apparel', 'dress', 'clothes'], ['pole', 'poles'], ['land', 'ground', 'soil'], ['bannister', 'banister', 'balustrade', 'balusters', 'handrail'], ['escalator', 'escalators', 'moving staircase', 'moving stairway'], ['ottoman', 'pouf', 'pouffe', 'puff', 'hassock'], ['bottle', 'bottles'], ['buffet', 'counter', 'sideboard'], ['poster', 'posters', 'posting', 'placard', 'notice', 'bill', 'card', 'cards'], ['stage', 'stages'], ['van'], ['ship', 'ships'], ['fountain', 'fountains'], ['conveyer belt', 'conveyor belt', 'conveyer', 'conveyor', 'transporter'], ['canopy'], ['washer', 'automatic washer', 'washing machine'], ['plaything', 'toy', 'toys'], ['swimming pool', 'swimming pools', 'swimming bath', 'natatorium'], ['stool'], ['barrel', 'cask'], ['basket', 'baskets', 'handbasket'], ['waterfall', 'falls'], ['tent', 'tents', 'collapsible shelter'], ['bag', 'bags'], ['minibike', 'motorbike'], ['cradle', 'cradles'], ['oven', 'ovens'], ['ball', 'balls'], ['food', 'solid food'], ['step', 'stair'], ['tank', 'tanks', 'storage tank'], ['trade name', 'brand name', 'brand', 'marque'], ['microwave', 'microwave oven', 'microwave ovens'], ['pot', 'flowerpot', 'flowerpots'], ['animal', 'animals', 'animate being', 'beast', 'beasts', 'brute', 'creature', 'fauna'], ['bicycle', 'bicycles', 'bike', 'bikes', 'wheel', 'cycle'], ['lake', 'lakes'], ['dishwasher', 'dish washer', 'dishwashing machine'], ['screen', 'silver screen', 'projection screen'], ['blanket', 'blankets', 'cover'], ['sculpture'], ['hood', 'exhaust hood'], ['sconce'], ['vase'], ['traffic light', 'traffic lights', 'traffic signal', 'traffic signals', 'stoplight'], ['tray'], ['ashcan', 'trash can', 'garbage can', 'wastebin', 'ash bin', 'ash-bin', 'ashbin', 'dustbin', 'trash barrel', 'trash bin'], ['fan', 'fans'], ['pier', 'wharf', 'wharfage', 'dock'], ['crt screen'], ['plate', 'plates'], ['monitor', 'monitors', 'monitoring device'], ['bulletin board', 'notice board'], ['shower'], ['radiator', 'radiators'], ['glass', 'glasses', 'drinking glass'], ['clock', 'clocks'], ['flag', 'flags']]

	def __len__(self):
		return len(self.samples)

	def __getitem__(self, idx):
		sample = self.samples[idx]

		# existent_objects_gt = list(range(len(self.class_name)))
		label = self.labels[idx]
		label = np.array(Image.open(label))
		label[label > 150] = 0
		label = label.reshape(-1)
		existent_objects_gt = set()
		for i in range(label.shape[0]):
			existent_objects_gt.add(label[i].item())
		existent_objects_gt.remove(0)
		existent_objects_gt = list(existent_objects_gt)

		return sample, existent_objects_gt


class COCOObjectDataset(data.Dataset):
	def __init__(self, samples, labels, additional_annotation):
		super().__init__()

		self.samples = samples
		self.labels = labels
		# https://github.com/facebookresearch/detectron2/blob/main/detectron2/data/datasets/builtin_meta.py

		self.class_name = [[], ['person', 'persons'], ['bicycle', 'bicycles'], ['car', 'cars'], ['motorcycle', 'motorcycles'], ['airplane', 'airplanes'], ['bus', 'buses'], ['train', 'trains'], ['truck', 'trucks'], ['boat', 'boats'], ['traffic light', 'traffic lights'], ['fire hydrant', 'fire hydrants'], ['stop sign', 'stop signs'], ['parking meter'], ['bench', 'benches'], ['bird', 'birds'], ['cat', 'cats'], ['dog', 'dogs'], ['horse', 'horses'], ['sheep'], ['cow'], ['elephant', 'elephants'], ['bear', 'bears'], ['zebra', 'zebras'], ['giraffe', 'giraffes'], ['backpack'], ['umbrella'], ['handbag', 'handbags'], ['tie', 'ties'], ['suitcase', 'suitcases'], ['frisbee'], ['skis'], ['snowboard', 'snowboards'], ['sports ball', 'sports balls', 'ball', 'balls'], ['kite', 'kites'], ['baseball bat', 'baseball bats'], ['baseball glove', 'baseball gloves'], ['skateboard', 'skateboards'], ['surfboard', 'surfboards'], ['tennis racket', 'tennis rackets'], ['bottle', 'bottles'], ['wine glass'], ['cup', 'cups'], ['fork', 'forks'], ['knife', 'knives'], ['spoon', 'spoons'], ['bowl', 'bowls'], ['banana', 'bananas'], ['apple', 'apples'], ['sandwich', 'sandwiches'], ['orange', 'oranges'], ['broccoli'], ['carrot', 'carrots'], ['hot dog', 'hot dogs'], ['pizza'], ['donut'], ['cake', 'cakes'], ['chair', 'chairs'], ['couch', 'couches'], ['potted plant', 'potted plants'], ['bed', 'beds'], ['dining table', 'dining tables'], ['toilet', 'toilets'], ['tv'], ['laptop', 'laptops'], ['mouse'], ['remote'], ['keyboard', 'keyboards'], ['cell phone', 'cell phones'], ['microwave'], ['oven', 'ovens'], ['toaster'], ['sink'], ['refrigerator'], ['book', 'books'], ['clock', 'clocks'], ['vase'], ['scissors'], ['teddy bear', 'teddy bears'], ['hair drier'], ['toothbrush']]
		class_to_id = {name[0]: idx+1 for idx, name in enumerate(self.class_name[1:])}
		self.id_to_id = {}
		for entry in COCO_CATEGORIES:
			raw_id = entry['id']
			if entry['isthing'] == 0:
				self.id_to_id[raw_id] = 0
			else:
				self.id_to_id[raw_id] = class_to_id[entry['name']]

		with open(additional_annotation) as f:
			self.annos = json.load(f)["annotations"]

	def __len__(self):
		return len(self.samples)

	def __getitem__(self, idx):
		sample = self.samples[idx]

		label = self.labels[idx]
		label = np.asarray(Image.open(label), dtype=np.uint32)
		label = label[:, :, 0] + 256 * label[:, :, 1] + 256 * 256 * label[:, :, 2]

		for anno in self.annos:
			if anno['file_name'] == self.labels[idx].split('/')[-1]:
				break

		for seg in anno['segments_info']:
			color_id = seg['id']
			raw_id = seg['category_id']
			new_id = self.id_to_id[raw_id]
			label[label == color_id] = new_id

		label = label.reshape(-1)
		existent_objects_gt = set()
		for i in range(label.shape[0]):
			existent_objects_gt.add(label[i].item())
		if 0 in existent_objects_gt:
			existent_objects_gt.remove(0)
		existent_objects_gt = list(existent_objects_gt)

		return sample, existent_objects_gt


class PascalContextDataset(data.Dataset):
	def __init__(self, samples, labels):
		super().__init__()

		self.samples = []
		self.labels = []
		for s, l in zip(samples, labels):
			if os.path.exists(l):
				self.samples.append(s)
				self.labels.append(l)

		self.class_name = [[], ['accordion'], ['aeroplane'], ['air conditioner'], ['antenna'], ['artillery'], ['ashtray'], ['atrium'], ['baby carriage'], ['bag'], ['ball'], ['balloon'], ['bamboo weaving'], ['barrel'], ['baseball bat'], ['basket'], ['basketball backboard'], ['bathtub'], ['bed'], ['bedclothes'], ['beer'], ['bell'], ['bench'], ['bicycle'], ['binoculars'], ['bird'], ['bird cage'], ['bird feeder'], ['bird nest'], ['blackboard'], ['board'], ['boat'], ['bone'], ['book'], ['bottle'], ['bottle opener'], ['bowl'], ['box'], ['bracelet'], ['brick'], ['bridge'], ['broom'], ['brush'], ['bucket'], ['building'], ['bus'], ['cabinet'], ['cabinet door'], ['cage'], ['cake'], ['calculator'], ['calendar'], ['camel'], ['camera'], ['camera lens'], ['can'], ['candle'], ['candle holder'], ['cap'], ['car'], ['card'], ['cart'], ['case'], ['casette recorder'], ['cash register'], ['cat'], ['cd'], ['cd player'], ['ceiling'], ['cell phone'], ['cello'], ['chain'], ['chair'], ['chessboard'], ['chicken'], ['chopstick'], ['clip'], ['clippers'], ['clock'], ['closet'], ['cloth'], ['clothes tree'], ['coffee'], ['coffee machine'], ['comb'], ['computer'], ['concrete'], ['cone'], ['container'], ['control booth'], ['controller'], ['cooker'], ['copying machine'], ['coral'], ['cork'], ['corkscrew'], ['counter'], ['court'], ['cow'], ['crabstick'], ['crane'], ['crate'], ['cross'], ['crutch'], ['cup'], ['curtain'], ['cushion'], ['cutting board'], ['dais'], ['disc'], ['disc case'], ['dishwasher'], ['dock'], ['dog'], ['dolphin'], ['door'], ['drainer'], ['dray'], ['drink dispenser'], ['drinking machine'], ['drop'], ['drug'], ['drum'], ['drum kit'], ['duck'], ['dumbbell'], ['earphone'], ['earrings'], ['egg'], ['electric fan'], ['electric iron'], ['electric pot'], ['electric saw'], ['electronic keyboard'], ['engine'], ['envelope'], ['equipment'], ['escalator'], ['exhibition booth'], ['extinguisher'], ['eyeglass'], ['fan'], ['faucet'], ['fax machine'], ['fence'], ['ferris wheel'], ['fire extinguisher'], ['fire hydrant'], ['fire place'], ['fish'], ['fish tank'], ['fishbowl'], ['fishing net'], ['fishing pole'], ['flag'], ['flagstaff'], ['flame'], ['flashlight'], ['floor'], ['flower'], ['fly'], ['foam'], ['food'], ['footbridge'], ['forceps'], ['fork'], ['forklift'], ['fountain'], ['fox'], ['frame'], ['fridge'], ['frog'], ['fruit'], ['funnel'], ['furnace'], ['game controller'], ['game machine'], ['gas cylinder'], ['gas hood'], ['gas stove'], ['gift box'], ['glass'], ['glass marble'], ['globe'], ['glove'], ['goal'], ['grandstand'], ['grass'], ['gravestone'], ['ground'], ['guardrail'], ['guitar'], ['gun'], ['hammer'], ['hand cart'], ['handle'], ['handrail'], ['hanger'], ['hard disk drive'], ['hat'], ['hay'], ['headphone'], ['heater'], ['helicopter'], ['helmet'], ['holder'], ['hook'], ['horse'], ['horse-drawn carriage'], ['hot-air balloon'], ['hydrovalve'], ['ice'], ['inflator pump'], ['ipod'], ['iron'], ['ironing board'], ['jar'], ['kart'], ['kettle'], ['key'], ['keyboard'], ['kitchen range'], ['kite'], ['knife'], ['knife block'], ['ladder'], ['ladder truck'], ['ladle'], ['laptop'], ['leaves'], ['lid'], ['life buoy'], ['light'], ['light bulb'], ['lighter'], ['line'], ['lion'], ['lobster'], ['lock'], ['machine'], ['mailbox'], ['mannequin'], ['map'], ['mask'], ['mat'], ['match book'], ['mattress'], ['menu'], ['metal'], ['meter box'], ['microphone'], ['microwave'], ['mirror'], ['missile'], ['model'], ['money'], ['monkey'], ['mop'], ['motorbike'], ['mountain'], ['mouse'], ['mouse pad'], ['musical instrument'], ['napkin'], ['net'], ['newspaper'], ['oar'], ['ornament'], ['outlet'], ['oven'], ['oxygen bottle'], ['pack'], ['pan'], ['paper'], ['paper box'], ['paper cutter'], ['parachute'], ['parasol'], ['parterre'], ['patio'], ['pelage'], ['pen'], ['pen container'], ['pencil'], ['person'], ['photo'], ['piano'], ['picture'], ['pig'], ['pillar'], ['pillow'], ['pipe'], ['pitcher'], ['plant'], ['plastic'], ['plate'], ['platform'], ['player'], ['playground'], ['pliers'], ['plume'], ['poker'], ['poker chip'], ['pole'], ['pool table'], ['postcard'], ['poster'], ['pot'], ['pottedplant'], ['printer'], ['projector'], ['pumpkin'], ['rabbit'], ['racket'], ['radiator'], ['radio'], ['rail'], ['rake'], ['ramp'], ['range hood'], ['receiver'], ['recorder'], ['recreational machines'], ['remote control'], ['road'], ['robot'], ['rock'], ['rocket'], ['rocking horse'], ['rope'], ['rug'], ['ruler'], ['runway'], ['saddle'], ['sand'], ['saw'], ['scale'], ['scanner'], ['scissors'], ['scoop'], ['screen'], ['screwdriver'], ['sculpture'], ['scythe'], ['sewer'], ['sewing machine'], ['shed'], ['sheep'], ['shell'], ['shelves'], ['shoe'], ['shopping cart'], ['shovel'], ['sidecar'], ['sidewalk'], ['sign'], ['signal light'], ['sink'], ['skateboard'], ['ski'], ['sky'], ['sled'], ['slippers'], ['smoke'], ['snail'], ['snake'], ['snow'], ['snowmobiles'], ['sofa'], ['spanner'], ['spatula'], ['speaker'], ['speed bump'], ['spice container'], ['spoon'], ['sprayer'], ['squirrel'], ['stage'], ['stair'], ['stapler'], ['stick'], ['sticky note'], ['stone'], ['stool'], ['stove'], ['straw'], ['stretcher'], ['sun'], ['sunglass'], ['sunshade'], ['surveillance camera'], ['swan'], ['sweeper'], ['swim ring'], ['swimming pool'], ['swing'], ['switch'], ['table'], ['tableware'], ['tank'], ['tap'], ['tape'], ['tarp'], ['telephone'], ['telephone booth'], ['tent'], ['tire'], ['toaster'], ['toilet'], ['tong'], ['tool'], ['toothbrush'], ['towel'], ['toy'], ['toy car'], ['track'], ['train'], ['trampoline'], ['trash bin'], ['tray'], ['tree'], ['tricycle'], ['tripod'], ['trophy'], ['truck'], ['tube'], ['turtle'], ['tvmonitor'], ['tweezers'], ['typewriter'], ['umbrella'], ['unknown'], ['vacuum cleaner'], ['vending machine'], ['video camera'], ['video game console'], ['video player'], ['video tape'], ['violin'], ['wakeboard'], ['wall'], ['wallet'], ['wardrobe'], ['washing machine'], ['watch'], ['water'], ['water dispenser'], ['water pipe'], ['water skate board'], ['watermelon'], ['whale'], ['wharf'], ['wheel'], ['wheelchair'], ['window'], ['window blinds'], ['wineglass'], ['wire'], ['wood'], ['wool']]

		ctx59_class = [
			0, 2, 259, 260, 415, 324, 9, 258, 144, 18, 19, 22, 23, 397, 25, 284,
			158, 159, 416, 33, 162, 420, 454, 295, 296, 427, 44, 45, 46, 308, 59,
			440, 445, 31, 232, 65, 354, 424, 68, 326, 72, 458, 34, 207, 80, 355,
			85, 347, 220, 349, 360, 98, 187, 104, 105, 366, 189, 368, 113, 115
		]
		self.class_name = [self.class_name[id] for id in ctx59_class]
		self.raw_to_59 = {c: idx for idx, c in enumerate(ctx59_class)}

	def __len__(self):
		return len(self.samples)

	def __getitem__(self, idx):
		sample = self.samples[idx]

		label = self.labels[idx]
		label = sio.loadmat(label)['LabelMap']
		for i in range(label.shape[0]):
			for j in range(label.shape[1]):
				if label[i][j] in self.raw_to_59:
					label[i][j] = self.raw_to_59[label[i][j]]
				else:
					label[i][j] = 0

		label = label.reshape(-1)
		existent_objects_gt = set()
		for i in range(label.shape[0]):
			existent_objects_gt.add(label[i].item())
		if 0 in existent_objects_gt:
			existent_objects_gt.remove(0)
		existent_objects_gt = list(existent_objects_gt)

		return sample, existent_objects_gt


def get_dataset(
		which_dataset, test_sample_path, test_label_path, limit, additional_annotation=None
	):
	if which_dataset == 'manual':
		return None, ManualDataset(test_sample_path)
	elif which_dataset == 'city':
		target = CityDataset
	elif which_dataset == 'horse':
		target = HorseDataset
	elif which_dataset == 'pascal-voc':
		target = PascalVOCDataset
		test_set = target(
			test_sample_path[:limit],
			test_label_path[:limit],
		)
		return test_set
	elif which_dataset == 'coco-stuff':
		target = COCODataset
	elif which_dataset == 'ade':
		target = ADEDataset
	elif which_dataset == 'coco-object':
		return COCOObjectDataset(
			sorted(glob.glob(test_sample_path))[:limit],
			sorted(glob.glob(test_label_path))[:limit],
			additional_annotation,
		)
	elif which_dataset == 'pascal-context':
		target = PascalContextDataset
		test_set = target(
			test_sample_path[:limit],
			test_label_path[:limit],
		)
		return test_set
	else:
		raise NotImplementedError

	test_set = target(
		sorted(glob.glob(test_sample_path))[:limit],
		sorted(glob.glob(test_label_path))[:limit],
	)

	return test_set
