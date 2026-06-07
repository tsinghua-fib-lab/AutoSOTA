import os
import tqdm
import base64
from openai import OpenAI

from components.labeling_dataset import get_dataset

from configs.current_dataset import DatasetConfig


# settings
dataset_config = DatasetConfig()

def class_name_to_str(class_name):
    all_names = []
    for category in class_name:
        all_names += category
    all_names = ', '.join(all_names)
    return all_names

test_set = get_dataset(
    dataset_config.which_dataset,
    dataset_config.sample,
    dataset_config.label,
    dataset_config.limit,
)

output_dir = dataset_config.gpt_output_dir
caption_len = dataset_config.caption_len

# ----- no need to change ----- #

dataset_class = class_name_to_str(test_set.class_name)

os.makedirs(output_dir, exist_ok=True)

if isinstance(dataset_class, list):
    dataset_class = ', '.join(dataset_class)

with open('.zshrc') as f:
    key = f.read().rstrip()
    os.environ['OPENAI_API_KEY'] = key

instruction = f'''You need to describe the content of the given image, but there are some additional requirements:
1. Make sure you list both foreground and background objects.
2. Use only nouns and no verbs.
3. Use no more than {caption_len} words.
4. When there are multiple instances of the same category, list only one of them. Don't repeat.
5. If you decide to describe multiple components of an object, such as "person, clothes", enclose all components of the same object, like "(person, clothes)". Other examples are "(person, clothes, hat)" and "(horse, harness)".
6. If an object can be described using the words listed below, make sure you use the exact word in the list. No morphological changes in parts of speech allowed. Don't use plural words.
{dataset_class}.
For example: (person, clothes), chair, store.'''


# ----- ready to chat ----- #

client = OpenAI()

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

for sample in tqdm.tqdm(test_set):
    save_name = sample.split('/')[-1]
    save_name = os.path.join(output_dir, f'{save_name}.txt')
    # skip existing resuls in case of interruption
    if os.path.exists(save_name):
        continue

    base64_image = encode_image(sample)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": instruction},
            {
                "role": "user",
                "content": [{
                    "type": "image_url",
                    "image_url": {
                        "url":  f"data:image/jpeg;base64,{base64_image}"
                    },
                }],
            }
        ],
    )

    failed = response.choices[0].message.refusal
    if failed:
        print('failed')
        continue
    result = response.choices[0].message.content

    with open(save_name, 'w') as f:
        f.write(result)
