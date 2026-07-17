import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer, AutoConfig
from huggingface_hub import PyTorchModelHubMixin


class QualityModel(nn.Module, PyTorchModelHubMixin):
    def __init__(self, config):
        super(QualityModel, self).__init__()
        self.model = AutoModel.from_pretrained(config["base_model"])
        self.dropout = nn.Dropout(config["fc_dropout"]).to(self.model.dtype)
        self.fc = nn.Linear(self.model.config.hidden_size, len(config["id2label"])).to(self.model.dtype)

    def forward(self, input_ids, attention_mask):
        features = self.model(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state
        dropped = self.dropout(features)
        outputs = self.fc(dropped.to(self.model.dtype)).to(self.model.dtype)
        return torch.softmax(outputs[:, 0, :], dim=1)

class QualityModelWrapper():
    """
        A wrapper for the QualityModel that handles tokenization and device management.
        NOTE: The tested model is nvidia/quality-classifier-deberta, which is a DeBERTa-based model for quality classification.
        It is trained to find good training data, and therefore can also be influenced by the length of the text/ Long term quality.
        A little bit unsure how it will react to short and unterminated texts.
    """
    def __init__(self, model_name_or_path):
        # super().__init__(*args, **kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = QualityModel.from_pretrained(model_name_or_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def forward(self, input_text_list):
        self.model.eval()
        with torch.no_grad():

            inputs = self.tokenizer(
                input_text_list, return_tensors="pt", padding="longest", truncation=True
            ).to(self.device)

            outputs = self.model(inputs["input_ids"], inputs["attention_mask"])

        # Class values: Low=0, Medium=0.5, High=1
        class_values = torch.tensor([1.0, 0.5, 0.]).to(self.device)

        # Calculate weighted grade
        grade = (outputs * class_values).sum(dim=-1)
        return grade


if  __name__ == "__main__":
    model = QualityModelWrapper("nvidia/quality-classifier-deberta")
    text_samples = [
        "Fbibiifiad",
        ".?@fdsa Low quality text.", 
        "This sentence is ok.", 
        "If after these checks Low is still giving the same sign as High, feel free to paste the exact code snippet you’re running and the output values; we can spot the mismatch in seconds..",
        "To create a repository for your project on GitHub, use the gh repo create subcommand. When prompted, select Push an existing local repository to GitHub and enter the desired name for your repository. If you want your project to belong to an organization instead of your user account, specify the organization name and project name with ORGANIZATION-NAME/PROJECT-NAME.",
        "The Renaissance period started in the 14th century and saw a renewed interest in schools of ancient philosophy, in particular Platonism. Humanism also emerged in this period. The modern period started in the 17th century. One of its central concerns was how philosophical and scientific knowledge are created. Specific importance was given to the role of reason and sensory experience. Many of these innovations were used in the Enlightenment movement to challenge traditional authorities. Several attempts to develop comprehensive systems of philosophy were made in the 19th century, for instance, by German idealism and Marxism. Influential developments in 20th-century philosophy were the emergence and application of formal logic, the focus on the role of language as well as pragmatism, and movements in continental philosophy like phenomenology, existentialism, and post-structuralism. The 20th century saw a rapid expansion of academic philosophy in terms of the number of philosophical publications and philosophers working at academic institutions. There was also a noticeable growth in the number of female philosophers, but they still remained underrepresented.",
    ]
    outputs = model.forward(text_samples)

    print(outputs)
