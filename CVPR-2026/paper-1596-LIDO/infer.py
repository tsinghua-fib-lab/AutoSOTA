#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

import argparse
import subprocess
import datetime
import yaml
from shutil import copyfile
import os
import shutil

from modules.user import User

def set_torchsparse_kmap_mode():
    import torchsparse.nn.functional as F
    # F.set_kmap_mode("hashmap") # not working properly
    # print(f'Torchsparse kmap mode: {F.get_kmap_mode()}')

    # possible solution as in https://github.com/mit-han-lab/torchsparse/issues/308
    conv_config = F.conv_config.get_default_conv_config(conv_mode=F.get_conv_mode())
    conv_config.kmap_mode = 'hashmap'
    F.conv_config.set_global_conv_config(conv_config)
    print(f'Torchsparse kmap conv mode: {F.conv_config.get_global_conv_config().kmap_mode}')

if __name__ == '__main__':
	set_torchsparse_kmap_mode()

	splits = ('train', 'valid', 'test')
	parser = argparse.ArgumentParser("./infer_semantics.py")
	parser.add_argument(
		'--dataset',
		type=str,
		required=True,
		help='Dataset to train with. No Default',
	)
	parser.add_argument(
		'--config',
		type=str,
		required=False,
		default='config/MinkowskiNet-semantickitti.yaml',
		help='Architecture yaml cfg file. See /config/ for sample.',
	)
	parser.add_argument(
		'--data',
		type=str,
		required=False,
		default='config/labels/semantic-kitti.yaml',
		help='Classification yaml cfg file. See /config/labels for sample.',
	)
	parser.add_argument(
		'--log', '-l',
		type=str,
		default=os.getcwd() + '/log/preds/',
		help='Directory to put the predictions. Default: log/preds'
	)
	parser.add_argument(
		'--model', '-m',
		type=str,
		required=True,
		default=None,
		help='Directory to get the trained model.'
	)
	parser.add_argument(
        '--split',
        type=str,
        required=True,
        default='valid',
        help='Split to evaluate on. One of ' +
             str(splits) + '. Defaults to %(default)s',
    )
	parser.add_argument(
        '--save',
        action='store_true',
		default=False,
        help='Save predictions in the log directory. Default: False',
    )
	parser.add_argument(
		'--eval',
		action='store_true',
		default=False,
		help='Evaluate the predictions. Default: False',
	)
	parser.add_argument(
		'--fp16',
		action='store_true',
		default=False,
		help='Use FP16 precision for inference. Default: False',
	)
	FLAGS, unparsed = parser.parse_known_args()

	# print summary of what we will do
	print("----------")
	print("INTERFACE:")
	print("dataset", FLAGS.dataset)
	print("config", FLAGS.config)
	print("data", FLAGS.data)
	print("log", FLAGS.log)
	print("model", FLAGS.model)
	print("split", FLAGS.split)
	print("save", FLAGS.save)
	print("eval", FLAGS.eval)
	print("fp16", FLAGS.fp16)
	print("----------\n")
	# print("Commit hash (training version): ", str(
	# 	subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).strip()))
	print("----------\n")

	# open arch config file
	try:
		print("Opening arch config file from %s" % FLAGS.config)
		ARCH = yaml.safe_load(open(FLAGS.config,'r'))
	except Exception as e:
		print(e)
		print("Error opening arch yaml file.")
		quit()

	# open data config file
	try:
		print("Opening data config file from %s" % FLAGS.data)
		DATA = yaml.safe_load(open(FLAGS.data, 'r'))
	except Exception as e:
		print(e)
		print("Error opening data yaml file.")
		quit()

	# create log folder
	try:
		if os.path.isdir(FLAGS.log):
			shutil.rmtree(FLAGS.log)
		os.makedirs(FLAGS.log)
		os.makedirs(os.path.join(FLAGS.log, "sequences"))
		# Create folders based on split
		for seq in DATA["split"][FLAGS.split]:
			seq = '{0:02d}'.format(int(seq))
			os.makedirs(os.path.join(FLAGS.log, "sequences", seq))
			os.makedirs(os.path.join(FLAGS.log, "sequences", seq, "predictions"))
			os.makedirs(os.path.join(FLAGS.log, "sequences", seq, "scores")) # scores for anomaly segmentation
			# os.makedirs(os.path.join(FLAGS.log, "sequences", seq, "probs")) # per-point probabilities
	except Exception as e:
		print(e)
		print("Error creating log directory. Check permissions!")
		raise

	# does model folder exist?
	if os.path.isdir(FLAGS.model):
		print("model exists! Using model from %s" % (FLAGS.model))
	else:
		print("model does not exist! Can't infer...")
		quit()

	# create user and infer dataset
	user = User(ARCH, DATA, FLAGS.dataset, FLAGS.log, FLAGS.model, FLAGS.split, FLAGS.save, FLAGS.eval, FLAGS.fp16)
	user.infer()