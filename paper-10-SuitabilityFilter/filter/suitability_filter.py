import numpy as np
import torch
import torch.nn.functional as F
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

import filter.tests as ftests


def get_sf_features(data, model, device):
    """
    Get the features (signals) used by the suitability filter

    data: the data we want to calculate the features for (torch dataset)
    model: the model to be evaluated and used to calculate the features (torch model)
    device: the device used to calculate the features (torch device)

    return: the features (signals) used by the classifier (numpy array), an binary array representing prediction correctness (numpy array)
    """
    model.eval()
    all_features = []
    all_correct = []

    with torch.no_grad():
        for batch in data:
            inputs, labels = batch[0].to(device), batch[1].to(device)
            outputs = model(inputs)

            # PREDICTION CORRECTNESS
            predictions = torch.argmax(outputs, dim=1)
            correct = (predictions == labels).cpu().numpy()
            all_correct.extend(correct)

            # SIGNALS
            # Confidence signals
            softmax_outputs = F.softmax(outputs, dim=1)
            conf_max = softmax_outputs.max(dim=1)[0].cpu().numpy()
            conf_std = softmax_outputs.std(dim=1).cpu().numpy()
            conf_entropy = (
                (
                    -torch.sum(
                        softmax_outputs * torch.log(softmax_outputs + 1e-10), dim=1
                    )
                )
                .cpu()
                .numpy()
            )

            # Logits signals
            logit_mean = outputs.mean(dim=1).cpu().numpy()
            logit_max = outputs.max(dim=1)[0].cpu().numpy()
            logit_std = outputs.std(dim=1).cpu().numpy()
            logit_diff_top2 = (
                (
                    torch.topk(outputs, 2, dim=1).values[:, 0]
                    - torch.topk(outputs, 2, dim=1).values[:, 1]
                )
                .cpu()
                .numpy()
            )

            # Loss signals
            loss = F.cross_entropy(outputs, predictions, reduction="none").cpu().numpy()

            # Margin loss as difference in cross-entropy loss
            top_two_classes = torch.topk(
                outputs, 2, dim=1
            ).indices  # Get indices of top 2 classes
            pred_class_probs = softmax_outputs[
                range(len(labels)), top_two_classes[:, 0]
            ]
            next_best_class_probs = softmax_outputs[
                range(len(labels)), top_two_classes[:, 1]
            ]
            pred_class_loss = -torch.log(pred_class_probs + 1e-10).cpu().numpy()
            next_best_class_loss = (
                -torch.log(next_best_class_probs + 1e-10).cpu().numpy()
            )
            margin_loss = (
                pred_class_loss - next_best_class_loss
            )  # Difference in loss between top 2 classes

            # Class Probability Ratios
            top_two_probs = torch.topk(softmax_outputs, 2, dim=1).values.cpu().numpy()
            class_prob_ratio = top_two_probs[:, 0] / (top_two_probs[:, 1] + 1e-10)

            # Top-k Class Probabilities
            top_k = int(
                softmax_outputs.size(1) * 0.1
            )  # Calculate top 10% class probabilities
            top_k_probs_sum = (
                torch.topk(softmax_outputs, top_k, dim=1)
                .values.sum(dim=1)
                .cpu()
                .numpy()
            )

            # Energy signals
            energy = -torch.logsumexp(outputs, dim=1).cpu().numpy()

            # Combining all signals into a feature vector
            features = np.column_stack(
                [
                    conf_max,
                    conf_std,
                    conf_entropy,
                    logit_mean,
                    logit_max,
                    logit_std,
                    logit_diff_top2,
                    loss,
                    margin_loss,
                    class_prob_ratio,
                    top_k_probs_sum,
                    energy,
                ]
            )

            all_features.append(features)

    features = np.vstack(all_features)
    correct = np.array(all_correct, dtype=bool)

    return features, correct


class SuitabilityFilter:
    def __init__(
        self,
        test_features,
        test_corr,
        classifier_features,
        classifier_corr,
        device,
        normalize=True,
        feature_subset=None,
    ):
        """
        device: the device used to evaluate the model (torch device)
        test_features: the features (signals) used by the classifier for the test data (numpy array)
        test_corr: the binary array representing prediction correctness for the test data (numpy array)
        classifier_features: the features (signals) used by the classifier for the classifier data (numpy array)
        classifier_corr: the binary array representing prediction correctness for the classifier data (numpy array)
        normalize: whether to normalize the features (bool)
        feature_subset: the subset of features to be used (list of ints)

        """
        self.device = device

        self.test_features = test_features
        self.test_corr = test_corr

        self.classifier_features = classifier_features
        self.classifier_corr = classifier_corr
        self.classifier = None
        self.feature_subset = feature_subset

        self.normalize = normalize
        self.scaler = StandardScaler()

    def train_regressor(self, calibrated=True):
        """
        Train the regressor using the classifier data

        calibrated: whether the classifier should be calibrated or not (bool)
        """
        features, correct = self.classifier_features, self.classifier_corr

        if self.feature_subset is not None:
            features = features[:, self.feature_subset]

        if self.normalize:
            features = self.scaler.fit_transform(features)

        regression_model = LogisticRegression(max_iter=1000)

        if not calibrated:
            self.classifier = regression_model.fit(features, correct)
        else:
            self.classifier = CalibratedClassifierCV(
                estimator=regression_model, method="isotonic", cv=5
            ).fit(features, correct)

    def train_classifier(self, classifier="logistic_regression", calibrated=True):
        """
        Train the regressor using the classifier data

        classifier: the classifier to be used (str)
        calibrated: whether the classifier should be calibrated or not (bool)
        """
        features, correct = self.classifier_features, self.classifier_corr

        if self.feature_subset is not None:
            features = features[:, self.feature_subset]

        if self.normalize:
            features = self.scaler.fit_transform(features)

        if classifier == "logistic_regression":
            base_model = LogisticRegression(max_iter=1000)
        elif classifier == "svm":
            base_model = SVC(probability=True)
        elif classifier == "random_forest":
            base_model = RandomForestClassifier()
        elif classifier == "gradient_boosting":
            base_model = GradientBoostingClassifier()
        elif classifier == "mlp":
            base_model = MLPClassifier(max_iter=1000)
        elif classifier == "decision_tree":
            base_model = DecisionTreeClassifier()

        if not calibrated:
            self.classifier = base_model.fit(features, correct)
        else:
            self.classifier = CalibratedClassifierCV(
                estimator=base_model, method="isotonic", cv=5
            ).fit(features, correct)

    def suitability_test(
        self,
        user_features,
        margin=0,
        test_power=False,
        get_sample_size=False,
        return_predictions=False,
    ):
        """
        Perform the suitability test

        user_features: the features (signals) used by the classifier for the user data (numpy array)
        margin: the margin used for the non-inferiority test (float)
        test_power: whether to calculate the test power (bool)
        get_sample_size: whether to calculate the sample size required for the test (bool)
        """
        if self.classifier is None:
            raise ValueError("Classifier not trained")

        test_features = self.test_features

        if self.feature_subset is not None:
            test_features = test_features[:, self.feature_subset]
            user_features = user_features[:, self.feature_subset]

        if self.normalize:
            test_features = self.scaler.transform(test_features)
            user_features = self.scaler.transform(user_features)

        test_predictions = self.classifier.predict_proba(test_features)[:, 1]
        user_predictions = self.classifier.predict_proba(user_features)[:, 1]

        test = ftests.non_inferiority_ttest(
            test_predictions, user_predictions, margin=margin
        )

        if test_power:
            power = ftests.power_non_inferiority_ttest(
                test_predictions, user_predictions, margin=margin
            )
            test["power"] = power

        if get_sample_size:
            sample_size = ftests.sample_size_non_inferiority_ttest(
                test_predictions, user_predictions, power=0.8, margin=margin
            )
            test["sample_size_0.8_power"] = sample_size

        if return_predictions:
            test["test_predictions"] = test_predictions
            test["user_predictions"] = user_predictions

        return test
