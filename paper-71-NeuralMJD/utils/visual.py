import matplotlib.pyplot as plt
import numpy as np
import os
import copy
 

def plot_sample_qualitative_stock(price_past_gt, price_future_gt, price_future_pred,
                                  ticker_factorized, static_num_to_str_dict, stock_key_ls, 
                                  save_dir, epoch, tag=""):
    """
    Plot the time-series data input-output for sanity check. Only a handful of sequences are plotted.
    """

    """initialize"""
    bs = len(stock_key_ls)
    os.makedirs(save_dir, exist_ok=True)

    """plot the qualitative results of some samples"""
    n_row, n_col = 5, 4
    n_plot = min(bs, n_row * n_col)
    n_plot = n_plot // 2 * 2

    price_past_gt = price_past_gt.detach().cpu().numpy()                # [B, T]
    price_future_gt = price_future_gt.detach().cpu().numpy()            # [B, Q]
    ticker_factorized = ticker_factorized.detach().cpu().numpy()        # [B]

    future_price_by_day_mean = price_future_gt.mean(axis=1)                     # [B]
    future_price_by_day_largest = future_price_by_day_mean.argsort()[::-1]      # [B]
    future_price_by_day_smallest = future_price_by_day_mean.argsort()           # [B]
    idx_selected = np.concatenate([future_price_by_day_largest[:n_plot//2], future_price_by_day_smallest[:n_plot//2]], axis=0)

    # select data samples with largest and smallest past spending
    past_price_by_day = price_past_gt[idx_selected]        # [N, T]
    future_price_by_day = price_future_gt[idx_selected]    # [N, Q]
    if price_future_pred is not None:
        future_dynamic_pred_ = price_future_pred.detach().cpu().numpy().reshape(bs, -1)               # [B, Q]
        future_price_by_day_pred = future_dynamic_pred_[idx_selected]  # [N, Q]
    else:
        future_price_by_day_pred = None
    stock_key_ = copy.deepcopy(stock_key_ls)
    stock_key_ls = [stock_key_[idx] for idx in idx_selected]          # [N]


    ticker_num_to_str = static_num_to_str_dict

    tickers_str_ = [ticker_num_to_str[num] if num in ticker_num_to_str else 'unknown' for num in ticker_factorized]  # [B], strings
    tickers_str = [tickers_str_[idx] for idx in idx_selected]  # [N]

    # style 1: show bars
    # fig, axes = plt.subplots(n_row, n_col, figsize=(n_col * 5, n_row * 5))
    # for i in range(n_plot):
    #     ax = axes[i // n_col, i % n_col]
    #     # plot two sets of bars showing past and future data
    #     ax.bar(range(past_spend_by_day.shape[1]), past_spend_by_day[i], color='b', alpha=0.5, label='past')
    #     ax.bar(range(past_spend_by_day.shape[1], past_spend_by_day.shape[1] + future_spend_by_day.shape[1]), future_spend_by_day[i], color='r', alpha=0.5, label='GT future')
    #     if future_spend_by_day_pred is not None:
    #         ax.bar(range(past_spend_by_day.shape[1], past_spend_by_day.shape[1] + future_spend_by_day.shape[1]), future_spend_by_day_pred[i], color='g', alpha=0.5, label='pred future')
    #     ax.legend()
        # ax.set_title("key: {:s}\n cat: {:s}".format(poi_key[i], top_cats_str[i].replace(",", ",\n")))
    # plt.tight_layout()
    # plt.savefig(os.path.join(save_dir, f'epoch_{epoch}_{tag}_input_output_bar.png'), dpi=300, bbox_inches='tight')
    # plt.close(fig)

    # style 2: show dot-line plots
    fig, axes = plt.subplots(n_row, n_col, figsize=(n_col * 5, n_row * 5))
    for i in range(n_plot):
        ax = axes[i // n_col, i % n_col]
        # plot dot-lines showing past and future data
        ax.plot(range(past_price_by_day.shape[1]), past_price_by_day[i], color='b', alpha=0.5, label='past', markersize=10, marker='o')
        ax.plot(range(past_price_by_day.shape[1], past_price_by_day.shape[1] + future_price_by_day.shape[1]), future_price_by_day[i], color='r', alpha=0.5, label='GT future', markersize=15, marker='o')
        if future_price_by_day_pred is not None:
            ax.plot(range(past_price_by_day.shape[1], past_price_by_day.shape[1] + future_price_by_day.shape[1]), future_price_by_day_pred[i], color='g', alpha=0.5, label='pred future', markersize=15, marker='o')
        ax.legend()
        ax.set_title("key: {:s}\n cat: {:s}".format(stock_key_ls[i], tickers_str[i].replace(",", ",\n")))
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'epoch_{:05d}_{:s}_input_output_line.png'.format(epoch, tag)), dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_results_per_label_stock(all_future_price_gt, all_future_price_pred, all_tickers_num, 
                                 static_num_to_str_dict, save_dir, epoch, tag=""):
    """
    Plot the time-series data input-output for sanity check. We select the overall results by category or brand.
    """

    """initialize"""

    # aggregate the results by category or brand
    def _aggregate_metrics_by_label(all_labels_num, all_gt, all_pred):
        """
        Aggregate the metrics by label, e.g., by category or number IDs.
        """
        unique_labels_num = np.unique(all_labels_num)
        l2_err = {}
        l1_err = {}
        r2_score = {}
        num_data = {}
        for label_num in unique_labels_num:
            mask_ = all_labels_num == label_num
            l2_err[label_num] = np.abs(all_gt[mask_] - all_pred[mask_]) ** 2
            l1_err[label_num] = np.abs(all_gt[mask_] - all_pred[mask_])
            r2_score[label_num] = 1.0 - l2_err[label_num].sum() / ((all_gt[mask_] - all_gt[mask_].mean()) ** 2).sum()
            num_data[label_num] = len(all_gt[mask_])
        return l2_err, l1_err, r2_score, num_data
    
    l2_err_by_ticker, l1_err_by_ticker, r2_score_by_ticker, num_data_by_ticker = _aggregate_metrics_by_label(all_tickers_num, all_future_price_gt, all_future_price_pred)


    """show error per label"""
    def _plot_label_data(metrics_per_label, num_data_by_label, metric_nm, label_nm, top_k=20):
        """
        Helper function to plot the categorical results.
        @param metrics_per_label: dict, key: cat_num, value: a list of metrics (numbers across all data datapoints)

        show label-wise metrics
        plot 1: general metrics per label
        plot 2: sorted metrics per label - highest top-k
        plot 3: sorted metrics per label - lowest top-k
        """
        assert label_nm.lower() in ['ticker']
        label_nm = label_nm.upper()
        num_plots = 3
        fig, axes = plt.subplots(num_plots, 1, figsize=(10, num_plots * 5))

        if metrics_per_label is not None and num_data_by_label is None:
            all_label_ids = list(metrics_per_label.keys())

            # plot 1: show general metrics per label
            metrics_mean = [np.mean(metrics_per_label[label_id]) for label_id in all_label_ids]
            metrics_std = [np.std(metrics_per_label[label_id]) for label_id in all_label_ids]
            # if metric_nm != 'R2-SCORE':
            #     axes[0].errorbar(all_label_ids, metrics_mean, yerr=metrics_std, fmt='o')
            axes[0].bar(all_label_ids, metrics_mean, color='orange')
            axes[0].set_xlabel('label id')
            axes[0].set_title('average {:s} metrics per label'.format(metric_nm))

            # plot 2: show sorted metrics per label (top-k)
            sorted_metrics_per_labels = sorted(metrics_per_label.items(), key=lambda x: np.mean(x[1]), reverse=True)
            sorted_metrics_per_labels = sorted_metrics_per_labels[:top_k]
            label_str_ls = [static_num_to_str_dict[label_id] if label_id in static_num_to_str_dict else 'unknown' for label_id, _ in sorted_metrics_per_labels]
            label_metric_ls = [label_metric for _, label_metric in sorted_metrics_per_labels]
            label_metric_mean_ls = [np.mean(label_metric) for label_metric in label_metric_ls]
            label_metric_std_ls = [np.std(label_metric) for label_metric in label_metric_ls]
            axes[1].bar(label_str_ls, label_metric_mean_ls, color='orange')            
            if metric_nm == 'R2-SCORE':
                # show values on top of the bar
                for i, v in enumerate(label_metric_mean_ls):
                    axes[1].text(i, v, "{:.2f}".format(v), ha='center', va='bottom')
            else:
                # error bar for non-R2-SCORE metrics
                axes[1].errorbar(label_str_ls, label_metric_mean_ls, yerr=label_metric_std_ls, fmt='o')
            axes[1].set_title('{:s} metrics (top-{:d})'.format(metric_nm, top_k))
            axes[1].set_xticks(range(len(label_str_ls)))
            axes[1].set_xticklabels(label_str_ls, rotation=45, fontsize=5, ha='right')

            # plot 3: show sorted metrics per label (lowest top-k)
            sorted_metrics_per_labels = sorted(metrics_per_label.items(), key=lambda x: np.mean(x[1]), reverse=False)
            sorted_metrics_per_labels = sorted_metrics_per_labels[:top_k]
            label_str_ls = [static_num_to_str_dict[label_id] if label_id in static_num_to_str_dict else 'unknown' for label_id, _ in sorted_metrics_per_labels]
            label_metric_ls = [label_metric for _, label_metric in sorted_metrics_per_labels]
            label_metric_mean_ls = [np.mean(label_metric) for label_metric in label_metric_ls]
            label_metric_std_ls = [np.std(label_metric) for label_metric in label_metric_ls]
            axes[2].bar(label_str_ls, label_metric_mean_ls, color='orange')
            if metric_nm == 'R2-SCORE':
                # show values on top of the bar
                for i, v in enumerate(label_metric_mean_ls):
                    axes[2].text(i, v, "{:.2f}".format(v), ha='center', va='bottom')
            else:
                # error bar for non-R2-SCORE metrics
                axes[2].errorbar(label_str_ls, label_metric_mean_ls, yerr=label_metric_std_ls, fmt='o')
            axes[2].set_title('{:s} metrics (lowest-{:d})'.format(metric_nm, top_k))
            axes[2].set_xticks(range(len(label_str_ls)))
            axes[2].set_xticklabels(label_str_ls, rotation=45, fontsize=5, ha='right')
        else:
            raise ValueError('Both metrics_per_label and num_data_by_label should be valid.')

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'epoch_{:05d}_{:s}_{:s}_per_{:s}.png'.format(epoch, tag, metric_nm, label_nm)), dpi=300, bbox_inches='tight')
        plt.close(fig)


    _plot_label_data(l1_err_by_ticker, None, 'L1-ERROR', 'ticker')
    _plot_label_data(r2_score_by_ticker, None, 'R2-SCORE', 'ticker')
