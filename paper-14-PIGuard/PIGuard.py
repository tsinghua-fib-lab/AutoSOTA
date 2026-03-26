import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoConfig

class PIGuard(nn.Module):
    def __init__(self, model_name, num_labels, device):
        super(PIGuard, self).__init__()
        self.device = device
        self.config = AutoConfig.from_pretrained(model_name, output_attentions=True)
        self.deberta = AutoModel.from_pretrained(model_name, config=self.config).to(device)
        self.classifier = nn.Linear(self.deberta.config.hidden_size, num_labels).to(device)
        # self.lm_head = nn.Linear(self.deberta.config.hidden_size, self.deberta.config.vocab_size).to(device)
        self.loss_fct = nn.CrossEntropyLoss().to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    

    def forward(self, input_ids, attention_mask, labels=None, mode="train"):
        outputs = self.deberta(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        last_hidden_state = outputs['last_hidden_state']
        # sequence_output = last_hidden_state
        pooled_output = last_hidden_state[:, 0, :]

        logits = self.classifier(pooled_output)
        
        loss = 0
        loss = self.loss_fct(logits.view(-1, self.classifier.out_features), labels.view(-1))
 
        return logits, loss


    def classify(self, input_text):
        tokenzied_text = self.tokenizer(input_text, return_tensors='pt', truncation=True, padding='max_length', max_length=256)
        input_ids = tokenzied_text['input_ids'].to(self.device)
        attention_mask = tokenzied_text['attention_mask'].to(self.device)
        outputs = self.deberta(input_ids=input_ids, attention_mask=attention_mask)['last_hidden_state']
        sequence_output = outputs
        pooled_output = outputs[:, 0, :]

        classification_logits = self.classifier(pooled_output)
        return classification_logits
