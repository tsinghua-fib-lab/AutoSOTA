import os
import xml
import json
import glob
import pickle
import numpy as np
import torch.utils.data as data

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

# mapping_coco = {}
# with open('fine_to_coarse_dict.pickle', 'rb') as f:
# 	d = pickle.load(f)
# 	d = d['fine_index_to_coarse_index']
# 	for k, v in d.items():
# 		mapping_coco[k+1] = v + 1
# 	mapping_coco[0] = 0
# 	mapping_coco[255] = 0

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
			[], ['road'], ['sidewalk'], ['building'], ['wall'], ['fence'], ['pole'], ['traffic light'],
			['traffic sign'], ['vegetation'], ['terrain'], ['sky'], ['pedestrian'],
			['rider'], ['car'], ['truck'], ['bus'], ['train'],
			['motorcycle'], ['bicycle']
		]

	def __len__(self):
		return len(self.samples)

	def __getitem__(self, idx):
		sample = self.samples[idx]
		return sample


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
			[], ['aeroplane'], ['bicycle'], ['bird'], ['boat'], ['bottle'], ['bus'], ['car'], ['cat'],
			['chair'], ['cow'], ['dining table', 'table'], ['dog'], ['horse'], ['motorbike'],
			['person', 'man', 'woman', 'child', 'kid'], ['potted plant'],
			['sheep'], ['sofa'], ['train'], ['television'],
		]

	def __len__(self):
		return len(self.samples)

	def __getitem__(self, idx):
		sample = self.samples[idx]
		return sample


class COCODataset(data.Dataset):
	def __init__(self, samples, labels):
		super().__init__()

		self.samples = samples
		self.labels = labels

		self.class_name = [
			[],
			['electronics'],
			['appliance'],
			['food'],
			['furniture'],
			['indoor'],
			['kitchen stuff'],
			['accessory'],
			['animal'],
			['outdoor'],
			['person'],
			['sport equipment'],
			['vehicle'],
			['ceiling'],
			['floor'],
			['food'],
			['furniture'],
			['raw material'],
			['textile'],
			['wall'],
			['window'],
			['building'],
			['ground'],
			['plant'],
			['sky'],
			['solid'],
			['structural'],
			['water'],
		]

	def __len__(self):
		return len(self.samples)

	def __getitem__(self, idx):
		sample = self.samples[idx]
		return sample


class ADEDataset(data.Dataset):
	def __init__(self, samples, labels):
		super().__init__()

		self.samples = samples
		self.labels = labels

		self.class_name = [[], ['wall'], ['building', 'edifice'], ['sky'], ['floor', 'flooring'], ['tree'], ['ceiling'], ['road', 'route'], ['bed'], ['windowpane', 'window'], ['grass'], ['cabinet'], ['sidewalk', 'pavement'], ['person', 'individual', 'someone', 'somebody', 'mortal', 'soul'], ['earth', 'ground'], ['door', 'double door'], ['table'], ['mountain', 'mount'], ['plant', 'flora', 'plant life'], ['curtain', 'drape', 'drapery', 'mantle', 'pall'], ['chair'], ['car', 'auto', 'automobile', 'machine', 'motorcar'], ['water'], ['painting', 'picture'], ['sofa', 'couch', 'lounge'], ['shelf'], ['house'], ['sea'], ['mirror'], ['rug', 'carpet', 'carpeting'], ['field'], ['armchair'], ['seat'], ['fence', 'fencing'], ['desk'], ['rock', 'stone'], ['wardrobe', 'closet', 'press'], ['lamp'], ['bathtub', 'bathing tub', 'bath', 'tub'], ['railing', 'rail'], ['cushion'], ['base', 'pedestal', 'stand'], ['box'], ['column', 'pillar'], ['signboard', 'sign'], ['chest of drawers', 'chest', 'bureau', 'dresser'], ['counter'], ['sand'], ['sink'], ['skyscraper'], ['fireplace', 'hearth', 'open fireplace'], ['refrigerator', 'icebox'], ['grandstand', 'covered stand'], ['path'], ['stairs', 'steps'], ['runway'], ['case', 'display case', 'showcase', 'vitrine'], ['pool table', 'billiard table', 'snooker table'], ['pillow'], ['screen door', 'screen'], ['stairway', 'staircase'], ['river'], ['bridge', 'span'], ['bookcase'], ['blind', 'screen'], ['coffee table', 'cocktail table'], ['toilet', 'can', 'commode', 'crapper', 'pot', 'potty', 'stool', 'throne'], ['flower'], ['book'], ['hill'], ['bench'], ['countertop'], ['stove', 'kitchen stove', 'range', 'kitchen range', 'cooking stove'], ['palm', 'palm tree'], ['kitchen island'], ['computer', 'computing machine', 'computing device', 'data processor', 'electronic computer', 'information processing system'], ['swivel chair'], ['boat'], ['bar'], ['arcade machine'], ['hovel', 'hut', 'hutch', 'shack', 'shanty'], ['bus', 'autobus', 'coach', 'charabanc', 'double-decker', 'jitney', 'motorbus', 'motorcoach', 'omnibus', 'passenger vehicle'], ['towel'], ['light', 'light source'], ['truck', 'motortruck'], ['tower'], ['chandelier', 'pendant', 'pendent'], ['awning', 'sunshade', 'sunblind'], ['streetlight', 'street lamp'], ['booth', 'cubicle', 'stall', 'kiosk'], ['television receiver', 'television', 'television set', 'tv', 'tv set', 'idiot box', 'boob tube', 'telly', 'goggle box'], ['airplane', 'aeroplane', 'plane'], ['dirt track'], ['apparel', 'wearing apparel', 'dress', 'clothes'], ['pole'], ['land', 'ground', 'soil'], ['bannister', 'banister', 'balustrade', 'balusters', 'handrail'], ['escalator', 'moving staircase', 'moving stairway'], ['ottoman', 'pouf', 'pouffe', 'puff', 'hassock'], ['bottle'], ['buffet', 'counter', 'sideboard'], ['poster', 'posting', 'placard', 'notice', 'bill', 'card'], ['stage'], ['van'], ['ship'], ['fountain'], ['conveyer belt', 'conveyor belt', 'conveyer', 'conveyor', 'transporter'], ['canopy'], ['washer', 'automatic washer', 'washing machine'], ['plaything', 'toy'], ['swimming pool', 'swimming bath', 'natatorium'], ['stool'], ['barrel', 'cask'], ['basket', 'handbasket'], ['waterfall', 'falls'], ['tent', 'collapsible shelter'], ['bag'], ['minibike', 'motorbike'], ['cradle'], ['oven'], ['ball'], ['food', 'solid food'], ['step', 'stair'], ['tank', 'storage tank'], ['trade name', 'brand name', 'brand', 'marque'], ['microwave', 'microwave oven'], ['pot', 'flowerpot'], ['animal', 'animate being', 'beast', 'brute', 'creature', 'fauna'], ['bicycle', 'bike', 'wheel', 'cycle'], ['lake'], ['dishwasher', 'dish washer', 'dishwashing machine'], ['screen', 'silver screen', 'projection screen'], ['blanket', 'cover'], ['sculpture'], ['hood', 'exhaust hood'], ['sconce'], ['vase'], ['traffic light', 'traffic signal', 'stoplight'], ['tray'], ['ashcan', 'trash can', 'garbage can', 'wastebin', 'ash bin', 'ash-bin', 'ashbin', 'dustbin', 'trash barrel', 'trash bin'], ['fan'], ['pier', 'wharf', 'wharfage', 'dock'], ['crt screen'], ['plate'], ['monitor', 'monitoring device'], ['bulletin board', 'notice board'], ['shower'], ['radiator'], ['glass', 'drinking glass'], ['clock'], ['flag']]

	def __len__(self):
		return len(self.samples)

	def __getitem__(self, idx):
		sample = self.samples[idx]
		return sample


class COCOObjectDataset(data.Dataset):
	def __init__(self, samples, labels):
		super().__init__()

		self.samples = samples
		self.labels = labels

		self.class_name = [[], ['person'], ['bicycle'], ['car'], ['motorcycle'], ['airplane'], ['bus'], ['train'], ['truck'], ['boat'], ['traffic light'], ['fire hydrant'], ['stop sign'], ['parking meter'], ['bench'], ['bird'], ['cat'], ['dog'], ['horse'], ['sheep'], ['cow'], ['elephant'], ['bear'], ['zebra'], ['giraffe'], ['backpack'], ['umbrella'], ['handbag'], ['tie'], ['suitcase'], ['frisbee'], ['skis'], ['snowboard'], ['sports ball'], ['kite'], ['baseball bat'], ['baseball glove'], ['skateboard'], ['surfboard'], ['tennis racket'], ['bottle'], ['wine glass'], ['cup'], ['fork'], ['knife'], ['spoon'], ['bowl'], ['banana'], ['apple'], ['sandwich'], ['orange'], ['broccoli'], ['carrot'], ['hot dog'], ['pizza'], ['donut'], ['cake'], ['chair'], ['couch'], ['potted plant'], ['bed'], ['dining table'], ['toilet'], ['tv'], ['laptop'], ['mouse'], ['remote'], ['keyboard'], ['cell phone'], ['microwave'], ['oven'], ['toaster'], ['sink'], ['refrigerator'], ['book'], ['clock'], ['vase'], ['scissors'], ['teddy bear'], ['hair drier'], ['toothbrush']]

	def __len__(self):
		return len(self.samples)

	def __getitem__(self, idx):
		sample = self.samples[idx]
		return sample


class PascalContextDataset(data.Dataset):
	def __init__(self, samples, labels):
		super().__init__()

		self.samples = samples
		self.labels = labels

		self.class_name = [[], ['accordion'], ['aeroplane'], ['air conditioner'], ['antenna'], ['artillery'], ['ashtray'], ['atrium'], ['baby carriage'], ['bag'], ['ball'], ['balloon'], ['bamboo weaving'], ['barrel'], ['baseball bat'], ['basket'], ['basketball backboard'], ['bathtub'], ['bed'], ['bedclothes'], ['beer'], ['bell'], ['bench'], ['bicycle'], ['binoculars'], ['bird'], ['bird cage'], ['bird feeder'], ['bird nest'], ['blackboard'], ['board'], ['boat'], ['bone'], ['book'], ['bottle'], ['bottle opener'], ['bowl'], ['box'], ['bracelet'], ['brick'], ['bridge'], ['broom'], ['brush'], ['bucket'], ['building'], ['bus'], ['cabinet'], ['cabinet door'], ['cage'], ['cake'], ['calculator'], ['calendar'], ['camel'], ['camera'], ['camera lens'], ['can'], ['candle'], ['candle holder'], ['cap'], ['car'], ['card'], ['cart'], ['case'], ['casette recorder'], ['cash register'], ['cat'], ['cd'], ['cd player'], ['ceiling'], ['cell phone'], ['cello'], ['chain'], ['chair'], ['chessboard'], ['chicken'], ['chopstick'], ['clip'], ['clippers'], ['clock'], ['closet'], ['cloth'], ['clothes tree'], ['coffee'], ['coffee machine'], ['comb'], ['computer'], ['concrete'], ['cone'], ['container'], ['control booth'], ['controller'], ['cooker'], ['copying machine'], ['coral'], ['cork'], ['corkscrew'], ['counter'], ['court'], ['cow'], ['crabstick'], ['crane'], ['crate'], ['cross'], ['crutch'], ['cup'], ['curtain'], ['cushion'], ['cutting board'], ['dais'], ['disc'], ['disc case'], ['dishwasher'], ['dock'], ['dog'], ['dolphin'], ['door'], ['drainer'], ['dray'], ['drink dispenser'], ['drinking machine'], ['drop'], ['drug'], ['drum'], ['drum kit'], ['duck'], ['dumbbell'], ['earphone'], ['earrings'], ['egg'], ['electric fan'], ['electric iron'], ['electric pot'], ['electric saw'], ['electronic keyboard'], ['engine'], ['envelope'], ['equipment'], ['escalator'], ['exhibition booth'], ['extinguisher'], ['eyeglass'], ['fan'], ['faucet'], ['fax machine'], ['fence'], ['ferris wheel'], ['fire extinguisher'], ['fire hydrant'], ['fire place'], ['fish'], ['fish tank'], ['fishbowl'], ['fishing net'], ['fishing pole'], ['flag'], ['flagstaff'], ['flame'], ['flashlight'], ['floor'], ['flower'], ['fly'], ['foam'], ['food'], ['footbridge'], ['forceps'], ['fork'], ['forklift'], ['fountain'], ['fox'], ['frame'], ['fridge'], ['frog'], ['fruit'], ['funnel'], ['furnace'], ['game controller'], ['game machine'], ['gas cylinder'], ['gas hood'], ['gas stove'], ['gift box'], ['glass'], ['glass marble'], ['globe'], ['glove'], ['goal'], ['grandstand'], ['grass'], ['gravestone'], ['ground'], ['guardrail'], ['guitar'], ['gun'], ['hammer'], ['hand cart'], ['handle'], ['handrail'], ['hanger'], ['hard disk drive'], ['hat'], ['hay'], ['headphone'], ['heater'], ['helicopter'], ['helmet'], ['holder'], ['hook'], ['horse'], ['horse-drawn carriage'], ['hot-air balloon'], ['hydrovalve'], ['ice'], ['inflator pump'], ['ipod'], ['iron'], ['ironing board'], ['jar'], ['kart'], ['kettle'], ['key'], ['keyboard'], ['kitchen range'], ['kite'], ['knife'], ['knife block'], ['ladder'], ['ladder truck'], ['ladle'], ['laptop'], ['leaves'], ['lid'], ['life buoy'], ['light'], ['light bulb'], ['lighter'], ['line'], ['lion'], ['lobster'], ['lock'], ['machine'], ['mailbox'], ['mannequin'], ['map'], ['mask'], ['mat'], ['match book'], ['mattress'], ['menu'], ['metal'], ['meter box'], ['microphone'], ['microwave'], ['mirror'], ['missile'], ['model'], ['money'], ['monkey'], ['mop'], ['motorbike'], ['mountain'], ['mouse'], ['mouse pad'], ['musical instrument'], ['napkin'], ['net'], ['newspaper'], ['oar'], ['ornament'], ['outlet'], ['oven'], ['oxygen bottle'], ['pack'], ['pan'], ['paper'], ['paper box'], ['paper cutter'], ['parachute'], ['parasol'], ['parterre'], ['patio'], ['pelage'], ['pen'], ['pen container'], ['pencil'], ['person'], ['photo'], ['piano'], ['picture'], ['pig'], ['pillar'], ['pillow'], ['pipe'], ['pitcher'], ['plant'], ['plastic'], ['plate'], ['platform'], ['player'], ['playground'], ['pliers'], ['plume'], ['poker'], ['poker chip'], ['pole'], ['pool table'], ['postcard'], ['poster'], ['pot'], ['pottedplant'], ['printer'], ['projector'], ['pumpkin'], ['rabbit'], ['racket'], ['radiator'], ['radio'], ['rail'], ['rake'], ['ramp'], ['range hood'], ['receiver'], ['recorder'], ['recreational machines'], ['remote control'], ['road'], ['robot'], ['rock'], ['rocket'], ['rocking horse'], ['rope'], ['rug'], ['ruler'], ['runway'], ['saddle'], ['sand'], ['saw'], ['scale'], ['scanner'], ['scissors'], ['scoop'], ['screen'], ['screwdriver'], ['sculpture'], ['scythe'], ['sewer'], ['sewing machine'], ['shed'], ['sheep'], ['shell'], ['shelves'], ['shoe'], ['shopping cart'], ['shovel'], ['sidecar'], ['sidewalk'], ['sign'], ['signal light'], ['sink'], ['skateboard'], ['ski'], ['sky'], ['sled'], ['slippers'], ['smoke'], ['snail'], ['snake'], ['snow'], ['snowmobiles'], ['sofa'], ['spanner'], ['spatula'], ['speaker'], ['speed bump'], ['spice container'], ['spoon'], ['sprayer'], ['squirrel'], ['stage'], ['stair'], ['stapler'], ['stick'], ['sticky note'], ['stone'], ['stool'], ['stove'], ['straw'], ['stretcher'], ['sun'], ['sunglass'], ['sunshade'], ['surveillance camera'], ['swan'], ['sweeper'], ['swim ring'], ['swimming pool'], ['swing'], ['switch'], ['table'], ['tableware'], ['tank'], ['tap'], ['tape'], ['tarp'], ['telephone'], ['telephone booth'], ['tent'], ['tire'], ['toaster'], ['toilet'], ['tong'], ['tool'], ['toothbrush'], ['towel'], ['toy'], ['toy car'], ['track'], ['train'], ['trampoline'], ['trash bin'], ['tray'], ['tree'], ['tricycle'], ['tripod'], ['trophy'], ['truck'], ['tube'], ['turtle'], ['tvmonitor'], ['tweezers'], ['typewriter'], ['umbrella'], ['unknown'], ['vacuum cleaner'], ['vending machine'], ['video camera'], ['video game console'], ['video player'], ['video tape'], ['violin'], ['wakeboard'], ['wall'], ['wallet'], ['wardrobe'], ['washing machine'], ['watch'], ['water'], ['water dispenser'], ['water pipe'], ['water skate board'], ['watermelon'], ['whale'], ['wharf'], ['wheel'], ['wheelchair'], ['window'], ['window blinds'], ['wineglass'], ['wire'], ['wood'], ['wool']]

	def __len__(self):
		return len(self.samples)

	def __getitem__(self, idx):
		sample = self.samples[idx]
		return sample


def get_dataset(
		which_dataset, test_sample_path, test_label_path, limit
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
			test_label_path[:limit]
		)
		return test_set
	elif which_dataset == 'coco-stuff':
		target = COCODataset
	elif which_dataset == 'ade':
		target = ADEDataset
	elif which_dataset == 'coco-object':
		target = COCOObjectDataset
	elif which_dataset == 'pascal-context':
		target = PascalContextDataset
		test_set = target(
			test_sample_path[:limit],
			test_label_path[:limit]
		)
		return test_set
	else:
		raise NotImplementedError

	test_set = target(
		sorted(glob.glob(test_sample_path))[:limit],
		sorted(glob.glob(test_label_path))[:limit]
	)

	return test_set
