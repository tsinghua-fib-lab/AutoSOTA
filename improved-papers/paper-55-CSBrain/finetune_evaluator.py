import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, f1_score, confusion_matrix, cohen_kappa_score, roc_auc_score, \
    precision_recall_curve, auc, r2_score, mean_squared_error
from tqdm import tqdm


class Evaluator:
    def __init__(self, params, data_loader):
        self.params = params
        self.data_loader = data_loader

    def get_metrics_for_multiclass(self, model):
        model.eval()

        truths = []
        preds = []
        for x, y in tqdm(self.data_loader, mininterval=1):
            x = x.cuda()
            y = y.cuda()

            pred = model(x)
            pred_y = torch.max(pred, dim=-1)[1]

            truths += y.cpu().squeeze().numpy().tolist()
            #preds += pred_y.cpu().squeeze().numpy().tolist()
            preds += pred_y.cpu().squeeze().numpy().tolist()

        truths = np.array(truths)
        preds = np.array(preds)
        acc = balanced_accuracy_score(truths, preds)
        f1 = f1_score(truths, preds, average='weighted')
        kappa = cohen_kappa_score(truths, preds)
        cm = confusion_matrix(truths, preds)
        return acc, kappa, f1, cm
    
    def get_metrics_for_multiclass_visual(self, model):
        model.eval()

        truths = []
        preds = []
        all_features = []
        raw_signal_list = []
        prepocessing_feature_list = []

        extracted_features = []

        def hook_fn(module, input, output):
            # output shape: [B, 200]
            extracted_features.append(output.detach().cpu())
        handle = model.classifier[4].register_forward_hook(hook_fn)

        with torch.no_grad():
            for x, y in tqdm(self.data_loader, mininterval=1):
                x = x.cuda()
                y = y.cuda()

                pred, raw_signal, prepocessing_feature = model(x)
                pred_y = pred.argmax(dim=-1)

                truths += y.cpu().squeeze().tolist()
                preds += pred_y.cpu().squeeze().tolist()

                raw_signal_list.append(raw_signal.cpu())
                prepocessing_feature_list.append(prepocessing_feature.cpu())


        truths = np.array(truths)
        preds = np.array(preds)
        raw_signal_list = torch.cat(raw_signal_list, dim=0)  # [N, D]
        prepocessing_feature_list = torch.cat(prepocessing_feature_list, dim=0)  # [N, D]
        all_features = torch.cat(extracted_features, dim=0)  # shape [N, 200]


        np.save("tsne_features_TUEV.npy", all_features.numpy())  # shape [N, D]
        np.save("tsne_raw_signal_TUEV.npy", raw_signal_list.numpy())  # shape [N, D]
        np.save("tsne_prepocessing_feature_TUEV.npy", prepocessing_feature_list.numpy())  # shape [N, D]
        np.save("tsne_labels_true_TUEV.npy", truths)             # shape [N]
        np.save("tsne_labels_pred_TUEV.npy", preds)              # shape [N]
        handle.remove()
        acc = balanced_accuracy_score(truths, preds)
        f1 = f1_score(truths, preds, average='weighted')
        kappa = cohen_kappa_score(truths, preds)
        cm = confusion_matrix(truths, preds)
        return acc, kappa, f1, cm

    def get_metrics_for_binaryclass(self, model):
        model.eval()

        truths = []
        preds = []
        scores = []
        for x, y in tqdm(self.data_loader, mininterval=1):
            x = x.cuda()
            y = y.cuda()
            pred = model(x)
            score_y = torch.sigmoid(pred)
            pred_y = torch.gt(score_y, 0.5).long()
            truths += y.long().cpu().squeeze().numpy().tolist()
            preds += pred_y.cpu().squeeze().numpy().tolist()
            scores += score_y.cpu().numpy().tolist()

        truths = np.array(truths)
        preds = np.array(preds)
        scores = np.array(scores)
        acc = balanced_accuracy_score(truths, preds)
        roc_auc = roc_auc_score(truths, scores)
        precision, recall, thresholds = precision_recall_curve(truths, scores, pos_label=1)
        pr_auc = auc(recall, precision)
        cm = confusion_matrix(truths, preds)
        return acc, pr_auc, roc_auc, cm

    def get_metrics_for_regression(self, model):
        model.eval()

        truths = []
        preds = []
        for x, y in tqdm(self.data_loader, mininterval=1):
            x = x.cuda()
            y = y.cuda()
            pred = model(x)
            truths += y.cpu().squeeze().numpy().tolist()
            preds += pred.cpu().squeeze().numpy().tolist()

        truths = np.array(truths)
        preds = np.array(preds)
        corrcoef = np.corrcoef(truths, preds)[0, 1]
        r2 = r2_score(truths, preds)
        rmse = mean_squared_error(truths, preds) ** 0.5
        return corrcoef, r2, rmse