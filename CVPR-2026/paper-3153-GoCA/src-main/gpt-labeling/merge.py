import glob
import json

# ----- change this ----- #
input_dir = './results/voc'
output_file = './results/voc.json'

# ----- no need to change ----- #

all_captions = {}
captions = glob.glob(f'{input_dir}/*.txt')
for caption in captions:

	with open(caption) as f:
		content = f.read()
	all_captions[caption.split('/')[-1].replace('.txt', '')] = {'caption': content}

with open(output_file, 'w') as f:
	f.write(json.dumps(all_captions))
