from openai import OpenAI
import yaml

def test_api():
    with open("./configs/my_key_set.yaml", 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    config_model = list(config.values())[1]

    client = OpenAI(api_key=config_model["api_key"], 
                    base_url=config_model["api_base"])

    messages = [{"role": "user", "content": "9.11 and 9.8, which is greater?"}]
    response = client.chat.completions.create(
        model=config_model["model_name"],
        messages=messages
    )  

    content = response.choices[0].message.content
    assert type(content) == str 