import argparse

classifier = None
model = None
tokenizer = None
model_name = "None"
delta = None
vocab = None
vocab_decode = None
act_order = True

groupsize = 128
wbits = 4
model_dir = 'models'
model_type = 'llama'
sensorimotor = None
stop_everything = False
is_seq2seq = False
use_flash_attention_2 = False
no_use_fast = False
use_eager_attention = False
stop_everything = False

trust_remote_code = True
no_use_fast = False
code = ""
acrostic = 0
new_sentence = False
delta_senso = 0.0
delta_acro = 0.0
delta_redgreen = 0.0
flag = 0
secret_key = []
classes = []
nlp = None