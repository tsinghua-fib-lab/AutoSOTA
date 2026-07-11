# lcl_full_eval.py

import os
import numpy as np
import umap
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from typing import Tuple, List, Dict

class LCL_Eval:
    """
    Given a concatenated AnnData (train+test with embedding in .obsm["LCL_embedding"]), this class provides:
      1) evaluate_adjusted_knn(...) → returns train/test KNN accuracy + containment + conditional‐rank‐stats
      2) plot_top_clones_umap(...)   → UMAP highlighting top‐N clones (train vs test)
      3) plot_test_accuracy_umap(...)→ UMAP: train=gray, test correct=green (on top), test incorrect=red
      4) plot_lineage_size_vs_accuracy(...) → scatter total‐lineage‐size vs test‐accuracy
      5) run_all(...)                → convenience wrapper to run all three at once
    """

    def __init__(
        self,
        adata,
        clone_key: str = "clone_id",
        dataset_key: str = "dataset",
        num_top: int = 5,
        palette: list = None,
        umap_kwargs: dict = None
    ):
        # copy so we don’t modify original
        self.adata = adata.copy()
        self.embedding_key = "LCL_embedding"
        self.clone_key    = clone_key
        self.dataset_key  = dataset_key
        self.num_top      = num_top
        self.umap_kwargs  = umap_kwargs or {"random_state": 42}

        sns.set_style("whitegrid")

        # 1) verify high‐D embedding
        if self.embedding_key not in self.adata.obsm:
            raise KeyError(f"Could not find embedding under adata.obsm['{self.embedding_key}'].")

        hd = self.adata.obsm[self.embedding_key]
        if hd.ndim != 2 or hd.shape[1] < 3:
            raise ValueError(f"adata.obsm['{self.embedding_key}'] must be shape [n_cells, D] with D≥3.")

        # 2) UMAP → obsm["UMAP_embedding"]
        reducer = umap.UMAP(**self.umap_kwargs)
        self.adata.obsm["UMAP_embedding"] = reducer.fit_transform(hd)

        # 3) top‐N clones
        counts = self.adata.obs[self.clone_key].value_counts()
        self.top_clones = counts.index[: self.num_top].tolist()

        # 4) clone_group column
        def _group_fn(x):
            return x if x in self.top_clones else "Other"
        self.adata.obs["clone_group"] = (
            self.adata.obs[self.clone_key]
            .apply(_group_fn)
            .astype("category")
        )
        cats = self.top_clones + ["Other"]
        self.adata.obs["clone_group"] = (
            self.adata.obs["clone_group"]
                .cat
                .reorder_categories(cats, ordered=True)
        )

        # 5) build color_map
        if palette is None:
            pal = sns.color_palette("tab10", n_colors=self.num_top)
            self.color_map = {self.top_clones[i]: pal[i] for i in range(self.num_top)}
        else:
            if len(palette) < self.num_top:
                raise ValueError(f"Palette length {len(palette)} < num_top={self.num_top}")
            self.color_map = {self.top_clones[i]: palette[i] for i in range(self.num_top)}
        self.color_map["Other"] = "lightgray"


    # ─── KNN / ranking / containment ──────────────────────────────────

    @staticmethod
    def compute_global_freq(labels: np.ndarray) -> Dict[int, float]:
        u, c = np.unique(labels, return_counts=True)
        tot  = len(labels)
        return {lab: cnt/tot for lab, cnt in zip(u, c)}

    @staticmethod
    def adjusted_knn_predict_with_rank(
        knn_model: KNeighborsClassifier,
        train_labels: np.ndarray,
        embeddings: np.ndarray,
        labels_true: np.ndarray,
        global_freq: Dict[int, float],
        k: int
    ) -> Tuple[
         np.ndarray,        # preds
         List[float],       # ranks (nan if not contained)
         List[int],         # containment_flag
         List[int],         # correct_flag
         List[int]          # unique_counts
    ]:
        neigh = knn_model.kneighbors(embeddings, return_distance=False)
        n_pts = embeddings.shape[0]

        preds, ranks, cont, corr, uniqs = (
            np.zeros(n_pts, dtype=train_labels.dtype),
            [], [], [], []
        )

        for i, nbrs in enumerate(neigh):
            nbr_lab = train_labels[nbrs]
            uqs, cts = np.unique(nbr_lab, return_counts=True)
            uniqs.append(len(uqs))

            local_freq = {lab: ct/k for lab, ct in zip(uqs, cts)}
            scores     = {lab: local_freq.get(lab,0) - global_freq.get(lab,0)
                          for lab in global_freq}
            best       = max(scores.items(), key=lambda x: x[1])[0]
            preds[i]   = best

            sorted_lbl = [lab for lab,_ in sorted(local_freq.items(),
                               key=lambda x: x[1], reverse=True)]
            true_lab   = labels_true[i]
            if true_lab in sorted_lbl:
                cont.append(1)
                ranks.append(sorted_lbl.index(true_lab)+1)
            else:
                cont.append(0)
                ranks.append(np.nan)

            corr.append(1 if best==true_lab else 0)

        return preds, ranks, cont, corr, uniqs


    def evaluate_adjusted_knn(
        self,
        train_embeddings: np.ndarray,
        train_labels: np.ndarray,
        test_embeddings: np.ndarray,
        test_labels: np.ndarray,
        k: int = 30
    ) -> Dict[str, Dict]:
        """
        Returns:
          train: { accuracy, avg_unique }
          test:  { accuracy,
                   containment_rate,
                   overall_avg_rank (cond. on containment),
                   avg_unique,
                   unique_quantiles (q0–q100),
                   accuracy_quantiles (q0–q100) }
        """
        gf   = self.compute_global_freq(train_labels)
        knn  = KNeighborsClassifier(n_neighbors=k).fit(train_embeddings, train_labels)

        # train‐side
        _, train_r, train_c, train_co, train_u = self.adjusted_knn_predict_with_rank(
            knn, train_labels, train_embeddings, train_labels, gf, k
        )
        train_acc      = float(np.mean(train_co))
        train_avg_uniq = float(np.mean(train_u))

        # test‐side
        _, test_r, test_c, test_co, test_u = self.adjusted_knn_predict_with_rank(
            knn, train_labels, test_embeddings, test_labels, gf, k
        )
        test_acc      = float(np.mean(test_co))
        cont_rate     = float(np.mean(test_c))
        avg_uniq_test = float(np.mean(test_u))

        # per‐lineage conditional avg‐rank
        mask = np.array(test_c, dtype=bool)
        per_rank = {}
        for lab in np.unique(test_labels[mask]):
            idxs = np.where((test_labels==lab)&mask)[0]
            per_rank[lab] = float(np.nanmean([test_r[i] for i in idxs]))
        overall_avg_rank = float(np.mean(list(per_rank.values())))

        # quantify over those per‐lineage ranks
        arr = np.array(list(per_rank.values()))
        def make_q(a):
            return {
              "q0":   round(float(np.quantile(a,0.0)),3),
              "q25":  round(float(np.quantile(a,0.25)),3),
              "q50":  round(float(np.quantile(a,0.50)),3),
              "q75":  round(float(np.quantile(a,0.75)),3),
              "q100": round(float(np.quantile(a,1.00)),3),
            }
        rank_q    = make_q(arr)

        # now build accuracy‐quantiles **per‐lineage** as well
        per_acc = {}
        for lab in np.unique(test_labels):
            idxs = np.where(test_labels==lab)[0]
            per_acc[lab] = float(np.mean([test_co[i] for i in idxs]))
        acc_arr = np.array(list(per_acc.values()))
        acc_q   = make_q(acc_arr)

        return {
          "train": {
            "accuracy":    round(train_acc,3),
            "avg_unique":  round(train_avg_uniq,3)
          },
          "test": {
            "accuracy":           round(test_acc,3),
            "containment_rate":   round(cont_rate,3),
            "overall_avg_rank":   round(overall_avg_rank,3),
            "avg_unique":         round(avg_uniq_test,3),
            "unique_quantiles":   rank_q,
            "accuracy_quantiles": acc_q
          }
        }


    # ─── UMAP PLOTS ────────────────────────────────────────────────────

    def plot_top_clones_umap(
        self, figsize=(8,6), title=None, savepath: str = None
    ):
        df     = self.adata.obs
        coords = self.adata.obsm["UMAP_embedding"]
        is_tr  = df[self.dataset_key]=="train"
        is_te  = df[self.dataset_key]=="test"

        fig, ax = plt.subplots(figsize=figsize, dpi=200)

        # “Other” first
        for side,mk,sz,lab in [
            ("train",".",8, "Train Other"),
            ("test","x",12,"Test Other")
        ]:
            mask = (df["clone_group"]=="Other") & (is_tr if side=="train" else is_te)
            ax.scatter(coords[mask,0], coords[mask,1],
                       c=self.color_map["Other"], marker=mk,
                       s=sz, alpha=0.2, label=lab)

        # top‐N clones
        for c in self.top_clones:
            for side,mk,sz,alp,lab in [
                ("train",".",30,0.8,f"Train {c}"),
                ("test","x",40,0.9,f"Test {c}")
            ]:
                mask = (df["clone_group"]==c) & (is_tr if side=="train" else is_te)
                ax.scatter(coords[mask,0], coords[mask,1],
                           c=self.color_map[c], marker=mk,
                           s=sz, alpha=alp, label=lab)

        ax.set_xlabel("UMAP 1");  ax.set_ylabel("UMAP 2")
        ax.set_title(title or f"UMAP – Top {self.num_top} Clones")

        h, l = ax.get_legend_handles_labels()
        by = dict(zip(l,h))
        ax.legend(by.values(),by.keys(),
                  bbox_to_anchor=(1.02,1),loc="upper left",
                  frameon=False,fontsize="small")

        plt.tight_layout()
        if savepath:
            savepath = os.path.splitext(savepath)[0] + ".png"
            fig.savefig(savepath, format="png", dpi=300, bbox_inches="tight")
        return fig, ax


    def plot_test_accuracy_umap(
        self,
        train_embeddings: np.ndarray,
        train_labels: np.ndarray,
        test_embeddings: np.ndarray,
        test_labels: np.ndarray,
        k: int = 30,
        figsize=(8,6),
        title=None,
        savepath: str = None
    ):
        # get per‐cell cont & corr
        gf  = self.compute_global_freq(train_labels)
        knn = KNeighborsClassifier(n_neighbors=k).fit(train_embeddings, train_labels)
        _, _, cont, corr, _ = self.adjusted_knn_predict_with_rank(
            knn, train_labels, test_embeddings, test_labels, gf, k
        )
        cont = np.array(cont); corr = np.array(corr)

        df     = self.adata.obs.copy()
        coords = self.adata.obsm["UMAP_embedding"]
        is_tr  = df[self.dataset_key]=="train"
        is_te  = df[self.dataset_key]=="test"

        status = np.zeros(len(df), dtype=int)
        status[is_tr] = 0
        ti          = np.where(is_te)[0]
        for idx,(cf,cc) in zip(ti, zip(cont,corr)):
            status[idx] = 1 if (cf==1 and cc==1) else 2

        fig, ax = plt.subplots(figsize=figsize, dpi=200)
        # draw train & incorrect first
        for sv,clr, mk,sz,alp,lab in [
            (0,"lightgray",".",12,0.2,"Train"),
            (2,"red",      "x",30,0.9,"Test Incorrect")
        ]:
            mask = status==sv
            ax.scatter(coords[mask,0],coords[mask,1],
                       c=clr,marker=mk,s=sz,alpha=alp,label=lab)
        # correct on top
        mask = status==1
        ax.scatter(coords[mask,0],coords[mask,1],
                   c="green",marker="x",s=30,alpha=0.9,
                   label="Test Correct")

        ax.set_xlabel("UMAP 1");  ax.set_ylabel("UMAP 2")
        ax.set_title(title or "UMAP – Test‐cell Accuracy")

        h, l = ax.get_legend_handles_labels()
        by = dict(zip(l,h))
        ax.legend(by.values(),by.keys(),
                  bbox_to_anchor=(1.02,1),loc="upper left",
                  frameon=False,fontsize="small")

        plt.tight_layout()
        if savepath:
            savepath = os.path.splitext(savepath)[0] + ".png"
            fig.savefig(savepath, format="png", dpi=300, bbox_inches="tight")
        return fig, ax


    def plot_lineage_size_vs_accuracy(
        self,
        train_embeddings: np.ndarray,
        train_labels: np.ndarray,
        test_embeddings: np.ndarray,
        test_labels: np.ndarray,
        k: int = 30,
        figsize=(6,6),
        title=None,
        savepath: str = None
    ):
        """
        Scatter each lineage:
          x = total #cells in that lineage (train+test)
          y = test‐set accuracy in that lineage
        """
        # 1) get per‐cell correctness on TEST
        gf  = self.compute_global_freq(train_labels)
        knn = KNeighborsClassifier(n_neighbors=k).fit(train_embeddings, train_labels)
        _, _, _, corr, _ = self.adjusted_knn_predict_with_rank(
            knn, train_labels, test_embeddings, test_labels, gf, k
        )
        corr = np.array(corr, dtype=int)

        # 2) lineage sizes
        all_labels     = np.concatenate([train_labels, test_labels])
        unique_labs    = np.unique(all_labels)
        lineage_size   = {lab: int(np.sum(all_labels==lab)) for lab in unique_labs}

        # 3) per‐lineage test accuracy
        per_line_acc = {}
        for lab in np.unique(test_labels):
            idxs = np.where(test_labels==lab)[0]
            per_line_acc[lab] = float(np.mean(corr[idxs]))

        # 4) build scatter arrays
        xs = [lineage_size[lab]    for lab in per_line_acc]
        ys = [per_line_acc[lab]    for lab in per_line_acc]

        # 5) plot
        fig, ax = plt.subplots(figsize=figsize, dpi=200)
        ax.scatter(xs, ys)
        ax.set_xlabel("Total lineage size")
        ax.set_ylabel("Test accuracy")
        ax.set_title(title or "Lineage size vs Test accuracy")

        if savepath:
            savepath = os.path.splitext(savepath)[0] + ".png"
            fig.savefig(savepath, format="png", dpi=300, bbox_inches="tight")
        return fig, ax


    # ─── convenience wrapper ───────────────────────────────────────

    def run_all(
        self,
        train_embeddings, train_labels,
        test_embeddings,  test_labels,
        k: int = 30
    ):
        stats = self.evaluate_adjusted_knn(
            train_embeddings, train_labels,
            test_embeddings,  test_labels,
            k
        )
        f1,a1 = self.plot_top_clones_umap()
        f2,a2 = self.plot_test_accuracy_umap(
            train_embeddings, train_labels,
            test_embeddings,  test_labels,
            k
        )
        f3,a3 = self.plot_lineage_size_vs_accuracy(
            train_embeddings, train_labels,
            test_embeddings,  test_labels,
            k
        )
        return stats, (f1,a1), (f2,a2), (f3,a3)