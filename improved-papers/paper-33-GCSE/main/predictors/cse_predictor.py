import torch
from tqdm import tqdm
from transformers import AutoConfig
from main.models.simcse import SimCSE, SimCSERoberta


class Predictor():

    def __init__(self, tokenizer, from_pretrained=None, max_seq_len=256, batch_size=16, hard_negative_weight=0, temp=0.05):
        self.tokenizer = tokenizer
        self.from_pretrained = from_pretrained
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.hard_negative_weight = hard_negative_weight
        self.temp = temp

        self.model_init()

    def model_init(self):
        self.config = AutoConfig.from_pretrained(
            self.from_pretrained)
        if self.config.model_type == 'bert':
            self.model = SimCSE(from_pretrained=self.from_pretrained,
                                pooler_type='cls', hard_negative_weight=self.hard_negative_weight, temp=self.temp)
        elif self.config.model_type == 'roberta':
            self.model = SimCSERoberta(from_pretrained=self.from_pretrained,
                                       pooler_type='cls', hard_negative_weight=self.hard_negative_weight, temp=self.temp)

    def model_to_device(self, gpu=[0]):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.cuda()
        self.model = torch.nn.DataParallel(self.model, device_ids=gpu).cuda()
        self.model.to(device)

    def pred(self, input_text, gpu=[0]):
        '''
        - input_text: `list` of `str`, e.g: `[['sent1', 'sent2'],['sent3', 'sent4']]`
        - return: `tensor`
        '''
        self.model_to_device(gpu=gpu)
        self.model.eval()
        with torch.no_grad():
            num_batches = len(input_text) // self.batch_size + 1
            for batch_idx in tqdm(range(num_batches)):
                start_idx = batch_idx * self.batch_size
                end_idx = min((batch_idx + 1) * self.batch_size, len(input_text))
                input_text_batch = input_text[start_idx:end_idx]
                input_ids_list = []
                attention_mask_list = []
                token_type_ids_list = []
                for item in input_text_batch:
                    item_input = self.tokenizer(
                        item, add_special_tokens=True, return_tensors='pt', max_length=self.max_seq_len, padding='max_length', truncation=True)
                    input_ids_list.append(item_input['input_ids'])
                    attention_mask_list.append(item_input['attention_mask'])
                    if 'token_type_ids' in item_input:
                        token_type_ids_list.append(item_input['token_type_ids'])

                input_ids = torch.stack(input_ids_list)
                attention_mask = torch.stack(attention_mask_list)
                if 'token_type_ids' in item_input:
                    token_type_ids = torch.stack(token_type_ids_list)
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask,
                                        token_type_ids=token_type_ids)
                else:
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                yield outputs
