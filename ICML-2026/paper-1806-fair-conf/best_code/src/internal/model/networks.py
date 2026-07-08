import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2ForSequenceClassification, CLIPModel, CLIPProcessor
# from tabpfn import TabPFNClassifier
import xgboost as xgb
# from tabpfn_extensions.post_hoc_ensembles.sklearn_interface import AutoTabPFNClassifier
# from tabpfn_extensions.rf_pfn import RandomForestTabPFNClassifier
# from tabpfn_extensions.hpo import TunedTabPFNClassifier

# from tabdpt.classifier import TabDPTClassifier


# CNN for Fashion-MNIST
class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 16, 5, 1, 2)
        self.fc1 = nn.Linear(576, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class FACETModel(nn.Module):
    def __init__(self, device, used_labels, size="ViT-L/14"):
        super(FACETModel, self).__init__()
        self.device = device
        if size == "ViT-B/32":
            clip_link = "openai/clip-vit-base-patch32"
        elif size == "ViT-B/16":
            clip_link = "openai/clip-vit-base-patch16"
        elif size == "ViT-L/14":
            clip_link = "openai/clip-vit-large-patch14"
        else:
            raise ValueError(f"Invalid CLIP model size {size}")
        print(f"Loading CLIP model {clip_link}")
        self.model = CLIPModel.from_pretrained(clip_link).to(device)
        self.clip_processor = CLIPProcessor.from_pretrained(clip_link)

        self.used_labels = used_labels
        self.text = ["A photo of a " + k for k in used_labels]

    def forward(self, x):
        inputs = self.clip_processor(
            text=self.text, images=x, return_tensors="pt", padding=True
        ).to(self.device)
        outputs = self.model(**inputs)
        logits_per_image = outputs.logits_per_image

        return logits_per_image


class BiosBiasModel(nn.Module):
    def __init__(self, num_classes):
        super(BiosBiasModel, self).__init__()
        self.fc = nn.Linear(768, num_classes)

    def forward(self, x):
        x = self.fc(x)
        # output = F.log_softmax(x, dim=1)
        return x


class RAVDESSModel(nn.Module):
    def __init__(self, used_labels):
        super(RAVDESSModel, self).__init__()
        self.model = Wav2Vec2ForSequenceClassification.from_pretrained(
            "Wiam/wav2vec2-large-xlsr-53-english-finetuned-ravdess-v5"
        )
        self.used_labels = used_labels

    def forward(self, x):
        logits = self.model(x).logits
        return logits[:, self.used_labels]


class SimpleMLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.model(x)


# class TabPFN:
#     def __init__(self):
#         # self.clf = TabPFNClassifier()
#         # 1 Accuracy 0.30475
#         # self.clf = AutoTabPFNClassifier(max_time=1200, device="cuda")
#         # 2 Accuracy 0.275
#         # base_clf = TabPFNClassifier(
#         #     ignore_pretraining_limits=True,
#         #     inference_config={"SUBSAMPLE_SAMPLES": 10000},
#         # )
#         # self.clf = RandomForestTabPFNClassifier(
#         #     tabpfn=base_clf,  # (TabPFNClassifier) Base TabPFN model to be used within the Random Forest structure.
#         #     verbose=1,  # (int) Controls the verbosity; higher values show more details.
#         #     max_predict_time=120,  # (int) Maximum prediction time allowed in seconds.
#         #     n_estimators=10,
#         #     max_depth=3,
#         # )
#         # tabpfn_tree_clf2.fit
#         # prediction_probabilities = tabpfn_tree_clf.predict_proba(X_test_class)
#         # predictions = np.argmax(prediction_probabilities, axis=1)
#         # 3 Accuracy 0.29575 @ 20k Accuracy 0.2965 @ 50k
#         self.clf = TabPFNClassifier(
#             ignore_pretraining_limits=True,  # (bool) Allows the use of datasets larger than pretraining limits.
#             n_estimators=32,  # (int) Number of estimators for ensembling; improves accuracy with higher values.
#             inference_config={
#                 "SUBSAMPLE_SAMPLES": 10000,  # (int) Maximum number of samples per inference step to manage memory usage.
#             },
#         )
#         # 4 Accuracy 0.30525
#         # self.clf = TunedTabPFNClassifier()

#     def __call__(self, *args, **kwds):
#         return self.forward(*args, **kwds)

#     def forward(self, x):
#         # return torch.from_numpy(self.clf.predict_logits(x))
#         return torch.from_numpy(self.clf.predict_proba(x))

#     def fit(self, x, y):
#         self.clf.fit(x, y)

#     def predict(self, x):
#         return self.clf.predict(x)

#     def eval(self, *args):
#         return


class XGBoostClassifier:
    def __init__(self, **params):
        self.clf = xgb.XGBClassifier(**params)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def fit(self, x, y, xt, yt):
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        if isinstance(y, torch.Tensor):
            y = y.detach().cpu().numpy()
        if isinstance(xt, torch.Tensor):
            xt = xt.detach().cpu().numpy()
        if isinstance(yt, torch.Tensor):
            yt = yt.detach().cpu().numpy()
        self.clf.fit(x, y, eval_set=[(xt, yt)])

    def forward(self, x):
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()

        proba = self.clf.predict_proba(x)
        return torch.from_numpy(proba)

    def eval(self, *args, **kwargs):
        return

    def load_model(self, path: str):
        self.clf.load_model(path)

    def save_model(self, path: str):
        self.clf.save_model(path)


# class TabDPT:
#     def __init__(self):
#         self.clf = TabDPTClassifier()

#     def __call__(self, *args, **kwds):
#         return self.forward(*args, **kwds)

#     def forward(self, x):
#         if isinstance(x, torch.Tensor):
#             x = x.detach().cpu().numpy()
#         return torch.from_numpy(
#             self.clf.predict(
#                 x,
#                 n_ensembles=1,
#                 temperature=0.8,
#                 context_size=2048,
#                 permute_classes=True,
#                 seed=42,
#             )
#         )

#     def fit(self, x, y):
#         if isinstance(x, torch.Tensor):
#             x = x.detach().cpu().numpy()
#         if isinstance(y, torch.Tensor):
#             y = y.detach().cpu().numpy()
#         self.clf.fit(x, y)

#     def predict(self, x):
#         if isinstance(x, torch.Tensor):
#             x = x.detach().cpu().numpy()
#         return self.clf.predict(
#             x,
#             n_ensembles=8,
#             temperature=0.8,
#             context_size=2048,
#             permute_classes=True,
#             seed=42,
#         )

#     def eval(self, *args):
#         return
