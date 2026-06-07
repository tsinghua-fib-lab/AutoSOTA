"""Patch: Multi-band ensemble - correct placement before feature overwrite."""
log = open('/repo/patch3_result.txt', 'w')

path = '/repo/src/retrieval.py'
with open(path) as f:
    content = f.read()

# 1. Add --iso_ensemble CLI argument
old_cli = 'parser.add_argument(\"--iso_gap_guided\", action=\"store_true\", default=False, help=\"Auto-determine ktop/kbottom from singular value gaps.\")'

if old_cli in content:
    new_cli = old_cli + '\n\n    parser.add_argument(\"--iso_ensemble\", action=\"store_true\", default=False, help=\"Multi-band ensemble: average similarities from multiple (ktop,kbottom) configs.\")'
    content = content.replace(old_cli, new_cli)
    with open(path, 'w') as f:
        f.write(content)
    log.write("CLI: iso_ensemble added\n")

# 2. Insert ensemble block right after W_text, W_image extraction, BEFORE apply_iso
old_iso_call = '''        iso_tau = kwargs.get('iso_tau', 0.0)
        W_text_iso, W_image_iso = apply_iso(W_text, W_image, iso_ktop, iso_kbottom, iso_tau)'''

ensemble_block = '''        iso_tau = kwargs.get('iso_tau', 0.0)
        use_iso_ensemble = kwargs.get('iso_ensemble', False)

        if use_iso_ensemble:
            band_configs = [(100, 25), (150, 50), (200, 75), (250, 100)]
            print(f"Multi-band ensemble with {len(band_configs)} configs: {band_configs}")

            all_similarities = []
            for bk_top, bk_bottom in band_configs:
                W_t_iso, W_i_iso = apply_iso(W_text, W_image, bk_top, bk_bottom, iso_tau)
                W_iso = W_i_iso if query_features_type == 'image' else W_t_iso
                q_proj = query_features @ W_iso
                g_proj = gallery_features @ W_iso
                q_proj = F.normalize(q_proj)
                g_proj = F.normalize(g_proj)
                sim = calculate_similarities(dataset_name, g_proj, q_proj, kwargs.get('split_size', 32))
                all_similarities.append(sim)

            similarities = torch.stack(all_similarities).mean(dim=0)

            if query_split == gallery_split and query_features_type == gallery_features_type:
                is_query_gallery_split_same = True
            else:
                is_query_gallery_split_same = False

            metrics = get_retrieval_metrics(dataset_name, similarities, query_labels, gallery_labels,
                                            is_query_gallery_split_same=is_query_gallery_split_same)
            gc.collect()
            torch.cuda.empty_cache()
            return {f'{dataset_name}_{key}': value for key, value in metrics.items()}

        W_text_iso, W_image_iso = apply_iso(W_text, W_image, iso_ktop, iso_kbottom, iso_tau)'''

if 'use_iso_ensemble' in content:
    log.write("Ensemble already in content\n")
elif old_iso_call in content:
    content = content.replace(old_iso_call, ensemble_block)
    with open(path, 'w') as f:
        f.write(content)
    log.write("ENSEMBLE: block inserted before apply_iso\n")
else:
    log.write("ERROR: old_iso_call not found\n")
    idx = content.find('apply_iso(W_text, W_image, iso_ktop, iso_kbottom, iso_tau)')
    log.write(f"apply_iso call at {idx}\n")
    if idx > 0:
        log.write(repr(content[idx-100:idx+100]) + '\n')

log.close()
