from llava.train.train import *
from tqdm.cli import tqdm
from transformers import AutoProcessor
from llava.model.multimodal_encoder.siglip_encoder import SigLipImageProcessor
def test_data():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    # breakpoint()
    conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_args.model_name_or_path, padding_side="right",trust_remote_code=True)
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    data_args.image_processor = SigLipImageProcessor()
    dataloader = torch.utils.data.DataLoader(
        data_module['train_dataset'],
        batch_size=training_args.per_device_train_batch_size,
        shuffle=False,
        collate_fn=data_module['data_collator'],
        num_workers=0,
    )
    i = 0
    all_seq_len = []
    for batch in tqdm(dataloader):
        seq_len = len(batch['input_ids'])
        i+= 1
     
if __name__ == "__main__":
    test_data()
    print("test_data.py executed successfully.")