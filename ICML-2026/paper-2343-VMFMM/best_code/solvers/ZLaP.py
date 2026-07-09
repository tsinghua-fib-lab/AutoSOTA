from argparse import ArgumentParser
import numpy as np
import torch
import torch.nn.functional as F

def ZLaP_solver(query_features, query_labels, clip_prototypes, k=5, gamma=5.0, alpha=0.3, scale_sim=False):
    
    """
    This code is a direct implementation of the Zero-shot Label Propagation (ZLaP) method,
    adapted from the original GitHub repository associated with the paper. Most of the logic
    and structure have been copy-pasted with minimal modification to ensure reproducibility
    and consistency with the original implementation.
    
    Original source: https://github.com/vladan-stojnic/ZLaP
    """
    
    try:
        import cupy as cp
        import faiss
        from cupyx.scipy.sparse import csr_matrix, diags, eye
        from cupyx.scipy.sparse import linalg as s_linalg
    except ImportError as e:
        raise ImportError("ZLaP_solver requires 'cupy' and 'faiss'. Please install them to use this method.") from e

    def search_faiss(X, Q, k):
        res = faiss.StandardGpuResources()
        res.setDefaultNullStreamAllDevices()
        res.setTempMemory(0)
        
        s, knn = faiss.knn_gpu(res, Q, X, k, metric=faiss.METRIC_INNER_PRODUCT)
    
        return knn, s
    
    
    def normalize_connection_graph(G):
        W = csr_matrix(G)
        W = W - diags(W.diagonal(), 0)
        S = W.sum(axis=1)
        S[S == 0] = 1
        D = cp.array(1.0 / cp.sqrt(S))
        D[cp.isnan(D)] = 0
        D[cp.isinf(D)] = 0
        D_mh = diags(D.reshape(-1), 0)
        Wn = D_mh * W * D_mh
        return Wn
    
    
    def knn2laplacian(knn, s, alpha=0.99):
        N = knn.shape[0]
        k = knn.shape[1]
        row_idx = np.arange(N)
        row_idx_rep = np.tile(row_idx, (k, 1)).T
        knn_flat = knn.flatten("F")
        row_idx_rep_flat = row_idx_rep.flatten("F")
        sim_flat = s.flatten("F")
        valid_knn = np.where(knn_flat != -1)[0]
        knn_flat = cp.array(knn_flat[valid_knn])
        row_idx_rep_flat = cp.array(row_idx_rep_flat[valid_knn])
        sim_flat = cp.array(sim_flat[valid_knn])
        W = csr_matrix(
            (sim_flat, (row_idx_rep_flat, knn_flat)),
            shape=(N, N),
        )
        W = W + W.T
        Wn = normalize_connection_graph(W)
        L = eye(Wn.shape[0]) - alpha * Wn
        return L
    
    
    def dfs_search(L, Y, tol=1e-6, maxiter=50, cast_to_numpy=True):
        out = s_linalg.cg(L, Y, tol=tol, maxiter=maxiter)[0]
        if cast_to_numpy:
            return cp.asnumpy(out)
        else:
            return out
    
    
    def normalize(x):
        return F.normalize(torch.tensor(x), p=2, dim=1).cpu().numpy()
    
    
    def accuracy(scores, labels):
        preds = np.argmax(scores, axis=1)
        acc = np.mean(100.0 * (preds == labels))
        return acc
    
    
    def get_data(dataset, model="RN50"):
        try:
            train_features = np.load(f"features/{dataset}/{model}_train_feats.npy")
            train_features = normalize(train_features.astype(np.float32))
            train_targets = np.load(f"features/{dataset}/{model}_train_targets.npy")
        except OSError:
            print("No train features! Inductive setting will not be possible!")
            train_features = None
            train_targets = None
    
        try:
            val_features = np.load(f"features/{dataset}/{model}_val_feats.npy")
            val_features = normalize(val_features.astype(np.float32))
            val_targets = np.load(f"features/{dataset}/{model}_val_targets.npy")
        except OSError:
            print("No val features!!!")
            val_features = None
            val_targets = None
    
        try:
            test_features = np.load(f"features/{dataset}/{model}_test_feats.npy")
            test_features = normalize(test_features.astype(np.float32))
            test_targets = np.load(f"features/{dataset}/{model}_test_targets.npy")
        except OSError:
            print("No test features! Using val features as test!")
            if val_features is None:
                raise ValueError("No val features either!")
    
            test_features = val_features
            test_targets = val_targets
            val_features = None
            val_targets = None
    
        try:
            clf_text = np.load(
                f"features/{dataset}/classifiers/{model}_text_classifier.npy"
            )
            clf_text = normalize(clf_text.T)
        except OSError:
            raise ValueError("No extracted text classifier!")
    
        try:
            clf_cupl_text = np.load(
                f"features/{dataset}/classifiers/{model}_cupl_text_classifier.npy"
            )
            clf_cupl_text = normalize(clf_cupl_text.T)
        except OSError:
            clf_cupl_text = None
    
        try:
            clf_image_train = np.load(
                f"features/{dataset}/classifiers/{model}_inmap_proxy_classifier_train.npy"
            )
            clf_image_train = normalize(clf_image_train.T)
        except OSError:
            clf_image_train = None
    
        try:
            clf_cupl_image_train = np.load(
                f"features/{dataset}/classifiers/{model}_cupl_inmap_proxy_classifier_train.npy"
            )
            clf_cupl_image_train = normalize(clf_cupl_image_train.T)
        except OSError:
            clf_cupl_image_train = None
    
        try:
            clf_image_val = np.load(
                f"features/{dataset}/classifiers/{model}_inmap_proxy_classifier_val.npy"
            )
            clf_image_val = normalize(clf_image_val.T)
        except OSError:
            clf_image_val = None
    
        try:
            clf_cupl_image_val = np.load(
                f"features/{dataset}/classifiers/{model}_cupl_inmap_proxy_classifier_val.npy"
            )
            clf_cupl_image_val = normalize(clf_cupl_image_val.T)
        except OSError:
            clf_cupl_image_val = None
    
        try:
            clf_image_test = np.load(
                f"features/{dataset}/classifiers/{model}_inmap_proxy_classifier_test.npy"
            )
            clf_image_test = normalize(clf_image_test.T)
        except OSError:
            print(
                "No InMaP classifer learned on test, so using the one trained on val instead!"
            )
            clf_image_test = clf_image_val
    
        try:
            clf_cupl_image_test = np.load(
                f"features/{dataset}/classifiers/{model}_cupl_inmap_proxy_classifier_test.npy"
            )
            clf_cupl_image_test = normalize(clf_cupl_image_test.T)
        except OSError:
            print(
                "No InMaP classifer learned on test, so using the one trained on val instead!"
            )
            clf_cupl_image_test = clf_cupl_image_val
    
        return (
            train_features,
            train_targets,
            val_features,
            val_targets,
            test_features,
            test_targets,
            clf_text,
            clf_image_train,
            clf_image_val,
            clf_image_test,
            clf_cupl_text,
            clf_cupl_image_train,
            clf_cupl_image_val,
            clf_cupl_image_test,
        )
    
    
    def voc_ap(rec, prec, true_num):
        mrec = np.concatenate(([0.0], rec, [1.0]))
        mpre = np.concatenate(([0.0], prec, [0.0]))
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
        i = np.where(mrec[1:] != mrec[:-1])[0]
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
        return ap
    
    
    def voc_mAP(imagessetfilelist, num, return_each=False):
        seg = imagessetfilelist
        gt_label = seg[:, num:].astype(np.int32)
    
        sample_num = len(gt_label)
        class_num = num
        tp = np.zeros(sample_num)
        fp = np.zeros(sample_num)
        aps = []
    
        for class_id in range(class_num):
            confidence = seg[:, class_id]
            sorted_ind = np.argsort(-confidence)
            sorted_label = [gt_label[x][class_id] for x in sorted_ind]
    
            for i in range(sample_num):
                tp[i] = sorted_label[i] > 0
                fp[i] = sorted_label[i] <= 0
            true_num = 0
            true_num = sum(tp)
            fp = np.cumsum(fp)
            tp = np.cumsum(tp)
            rec = tp / float(true_num)
            prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
            ap = voc_ap(rec, prec, true_num)
            aps += [ap]
    
        np.set_printoptions(precision=6, suppress=True)
        aps = np.array(aps) * 100
        mAP = np.mean(aps)
        if return_each:
            return mAP, aps
        return mAP
    
    
    def combine_separate_knns(
        knn_im2im,
        sim_im2im,
        knn_im2text,
        sim_im2text,
        num_classes,
    ):
        knn_im = knn_im2im + num_classes
        sim_im = sim_im2im
    
        knn = np.concatenate((knn_im, knn_im2text), axis=1)
        sim = np.concatenate((sim_im, sim_im2text), axis=1)
    
        return knn, sim
    
    
    def create_separate_graph(features, clf, k):
        num_classes = clf.shape[0]
        
        assert k > 0
        k_im2im = min(k, features.shape[0])
        knn_im2im, sim_im2im = search_faiss(
            features, features, k=k_im2im
        )  # image2image search
    
        k_im2text = min(k, num_classes)
        knn_im2text, sim_im2text = search_faiss(
            clf, features, k=k_im2text
        )  # image2text search
    
        knn, sim = combine_separate_knns(
            knn_im2im,
            sim_im2im,
            knn_im2text,
            sim_im2text,
            num_classes,
        )
    
        knn_text = -1 * np.ones((num_classes, knn.shape[1]), dtype=knn.dtype)
        sim_text = np.zeros((num_classes, sim.shape[1]), dtype=sim.dtype)
        knn = np.concatenate((knn_text, knn), axis=0)
        sim = np.concatenate((sim_text, sim), axis=0)
    
        return knn, sim
    
    
    def do_transductive_lp(features, clf, k, gamma, alpha, scale_sim=False):
        num_classes = clf.shape[0]
    
        knn, sim = create_separate_graph(features, clf, k)
    
        if scale_sim:
            xmin = np.min(sim[knn != -1])
            xmax = np.max(sim[knn != -1])
            sim = (sim - xmin) / (xmax - xmin)
        sim[sim < 0] = 0
    
        mask_knn = knn < num_classes
        sim[mask_knn] = sim[mask_knn] ** gamma
        L = knn2laplacian(knn, sim, alpha)
    
        scores = cp.zeros((features.shape[0], num_classes))
        for idx in range(num_classes):
           
            Y = cp.zeros((L.shape[0],))
            Y[idx] = 1
            out = dfs_search(L, Y, cast_to_numpy=False)
            scores[:, idx] = out[num_classes:]
        
        
        return scores.get()
    
    
    def get_neighbors_for_inductive(
        unlabeled_features,
        clf,
        test_features,
        k,
        gamma,
        scale_sim=False,
        xmin=None,
        xmax=None,
    ):
        num_classes = clf.shape[0]
        k_im2im = min(k, unlabeled_features.shape[0])
        test_knn, test_sim = search_faiss(
            unlabeled_features, test_features, k=k_im2im
        )  # image2image search
        test_sim[test_sim < 0] = 0
        test_knn += num_classes
        if scale_sim:
            test_sim = (test_sim - xmin) / (xmax - xmin)
    
        k_im2text = min(k, num_classes)
        test_knn_im2text, test_sim_im2text = search_faiss(
            clf, test_features, k=k_im2text
        )  # image2text search
        test_sim_im2text[test_sim_im2text < 0] = 0
        if scale_sim:
            test_sim_im2text = (test_sim_im2text - xmin) / (xmax - xmin)
        test_sim_im2text = test_sim_im2text**gamma
    
        test_knn = np.concatenate((test_knn, test_knn_im2text), axis=1)
        test_sim = np.concatenate((test_sim, test_sim_im2text), axis=1)
    
        return test_knn, test_sim
    
    
    def do_inductive_lp(
        unlabeled_features,
        clf,
        test_features,
        k,
        gamma,
        alpha,
        scale_sim=False,
    ):
        num_classes = clf.shape[0]
        knn, sim = create_separate_graph(unlabeled_features, clf, k)
    
        xmin = None
        xmax = None
        if scale_sim:
            xmin = np.min(sim[knn != -1])
            xmax = np.max(sim[knn != -1])
            sim = (sim - xmin) / (xmax - xmin)
        sim[sim < 0] = 0
    
        mask_knn = knn < num_classes
        sim[mask_knn] = sim[mask_knn] ** gamma
        L = knn2laplacian(knn, sim, alpha)
    
        test_knn, test_sim = get_neighbors_for_inductive(
            unlabeled_features,
            clf,
            test_features,
            k,
            gamma,
            scale_sim=scale_sim,
            xmin=xmin,
            xmax=xmax,
        )
    
        scores = cp.zeros((test_features.shape[0], num_classes))
        for idx, (k, s) in enumerate(zip(test_knn, test_sim)):
            Y = cp.zeros((L.shape[0],))
            Y[k] = s
            out = dfs_search(L, Y, cast_to_numpy=False)
            scores[idx, :] = out[:num_classes]
    
        return scores.get()
    
    
    def get_Linv(features, clf, k, gamma, alpha, scale_sim=False):
        num_classes = clf.shape[0]
        knn, sim = create_separate_graph(features, clf, k)
    
        xmin = None
        xmax = None
        if scale_sim:
            xmin = np.min(sim[knn != -1])
            xmax = np.max(sim[knn != -1])
            sim = (sim - xmin) / (xmax - xmin)
        sim[sim < 0] = 0
    
        mask_knn = knn < num_classes
        sim[mask_knn] = sim[mask_knn] ** gamma
        L = knn2laplacian(knn, sim, alpha)
    
        scores = cp.zeros((num_classes + features.shape[0], num_classes))
        for idx in range(num_classes):
            Y = cp.zeros((L.shape[0],))
            Y[idx] = 1
            out = dfs_search(L, Y, cast_to_numpy=False)
            scores[:, idx] = out.copy()
    
        return scores.get(), xmin, xmax
    
    
    def do_sparse_inductive_lp(
        unlabeled_features,
        clf,
        test_features,
        k,
        gamma,
        alpha,
        scale_sim=False,
    ):
        num_classes = clf.shape[0]
        Linv, xmin, xmax = get_Linv(
            unlabeled_features, clf, k, gamma, alpha, scale_sim=scale_sim
        )
    
        test_knn, test_sim = get_neighbors_for_inductive(
            unlabeled_features,
            clf,
            test_features,
            k,
            gamma,
            scale_sim=scale_sim,
            xmin=xmin,
            xmax=xmax,
        )
        test_knn = cp.array(test_knn)
        test_sim = cp.array(test_sim)
    
        Linv_sparse = np.zeros_like(Linv)
        top = np.argmax(Linv, axis=1, keepdims=True)
        np.put_along_axis(Linv_sparse, top, np.take_along_axis(Linv, top, axis=1), axis=1)
        Linv_sparse = csr_matrix(cp.array(Linv_sparse))
    
        scores = cp.zeros((test_features.shape[0], num_classes))
        for idx, (k, s) in enumerate(zip(test_knn, test_sim)):
            Z = (Linv_sparse[k, :]).copy()
            Z.data = Z.data * s.repeat(cp.diff(Z.indptr).get().tolist())
            scores[idx, :] = Z.sum(axis=0)
    
        return scores.get()
    
    
    clf_to_use = (clip_prototypes.squeeze().T).cpu().numpy()
    query_features = query_features.cpu().numpy()
    clip_prototypes = clip_prototypes.cpu().numpy()
    
    # Initial zero-shot predictions
    logits = 100 * query_features @ clip_prototypes 
    
    test_labels = query_labels.cpu().numpy()
    scores = do_transductive_lp(
        query_features,
        clf_to_use,
        k,
        gamma,
        alpha,
        scale_sim=scale_sim,
    )

    z = torch.tensor(scores).float()
    y_hat = F.softmax(torch.tensor(logits).float(), dim=1).squeeze()

    return y_hat, z