# --- Projection timing bar plot (from trainer timings) ---
import numpy as np
import torch
import time

def _total_model_time(tr):
    if tr is None:
        return np.nan
    if hasattr(tr, 'total_model_forward_sample_time'):
        try:
            return float(getattr(tr, 'total_model_forward_sample_time'))
        except Exception:
            pass
    if hasattr(tr, 'model_forward_times'):
        lst = getattr(tr, 'model_forward_times') or []
        try:
            return float(np.sum([float(x) for x in lst])) if len(lst) > 0 else np.nan
        except Exception:
            return np.nan
    return np.nan

def _total_proj_time(tr):
    if tr is None:
        return np.nan
    if hasattr(tr, 'total_projection_sample_time'):
        try:
            return float(getattr(tr, 'total_projection_sample_time'))
        except Exception:
            pass
    if hasattr(tr, 'projection_sample_times'):
        lst = getattr(tr, 'projection_sample_times') or []
        try:
            return float(np.sum([float(x) for x in lst])) if len(lst) > 0 else np.nan
        except Exception:
            return np.nan
    return np.nan

# Average timings across n_trials (keep first-trial samples for figures).
def compute_avg_stats(method_names, trainers_map, n_trials, num_samples=None, external_proj_time=None):
    stats = {}
    for name in method_names:
        tr = trainers_map.get(name)
        m0 = _total_model_time(tr)
        p0 = _total_proj_time(tr)
        s0 = getattr(tr, 'sampling_time', np.nan) if tr is not None else np.nan
        # Debug: print initial per-trainer timing summaries to help diagnose
        # missing projection times (visible when running plotting scripts).
        try:
            print(f"[timing debug] Trainer '{name}': total_model={m0}, total_proj={p0}, sampling_time={s0}")
        except Exception:
            pass
        stats[name] = {
            'm_sum': 0.0 if np.isnan(m0) else float(m0),
            'p_sum': 0.0 if np.isnan(p0) else float(p0),
            's_sum': 0.0 if np.isnan(s0) else float(s0),
            'count': 1 if not (np.isnan(m0) and np.isnan(p0) and np.isnan(s0)) else 0,
            'external_proj_list': []
        }

    # Allow caller to provide a pre-measured external projection time (e.g. timing
    # performed in the plotting script). If supplied and finite, use it as one
    # of the external projection timing samples for DDPM_projected.
    try:
        if external_proj_time is not None and not np.isnan(external_proj_time):
            stats.setdefault('DDPM_projected', {}).setdefault('external_proj_list', []).append(float(external_proj_time))
            # Also populate the alternate label mapping used by plotting scripts
            stats.setdefault('DDPM (proj.)', {}).setdefault('external_proj_list', []).append(float(external_proj_time))
    except Exception:
        pass

    sample_kwargs_map = {'PDM': {'PDM': True}}

    # If num_samples isn't provided we skip the per-trial sampling runs. When
    # num_samples is given we run (n_trials - 1) extra sampling trials to
    # average timings.
    trials_to_run = max(0, n_trials - 1) if num_samples is not None else 0

    for t in range(trials_to_run):
        for name in method_names:
            tr = trainers_map.get(name)
            if tr is None:
                continue
            for attr in ('model_forward_times', 'projection_sample_times', 'projection_times'):
                try:
                    setattr(tr, attr, [])
                except Exception:
                    pass
            try:
                tr.total_model_forward_sample_time = 0.0
            except Exception:
                pass
            try:
                tr.total_projection_sample_time = 0.0
            except Exception:
                pass

            kwargs = sample_kwargs_map.get(name, {})
            try:
                with torch.no_grad():
                    # num_samples may be None; ensure we only call sample when provided
                    if num_samples is None:
                        continue
                    res = tr.sample(num_samples=num_samples, **kwargs)
            except Exception:
                continue

            m = _total_model_time(tr)
            p = _total_proj_time(tr)
            s = getattr(tr, 'sampling_time', np.nan)
            if not np.isnan(m):
                stats[name]['m_sum'] += float(m)
            if not np.isnan(p):
                stats[name]['p_sum'] += float(p)
            if not np.isnan(s):
                stats[name]['s_sum'] += float(s)
            stats[name]['count'] += 1

            if name == 'DDPM':
                # Time an external projection of the DDPM samples using the same
                # trainer's projector (previous code referenced an undefined
                # `trainer_plain` variable). Use `tr` which is the trainer
                # instance for the 'DDPM' entry.
                try:
                    samples_out = res[0] if isinstance(res, (list, tuple)) and len(res) > 0 else res
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    samples_plain_projected_tmp, _, _ = tr.projector.project(torch.tensor(samples_out).cpu())
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    t1 = time.perf_counter()
                    stats.setdefault('DDPM_projected', {}).setdefault('external_proj_list', []).append(float(t1 - t0))
                    stats.setdefault('DDPM (proj.)', {}).setdefault('external_proj_list', []).append(float(t1 - t0))
                except Exception:
                    pass

    avg_stats = {}
    for name in method_names:
        # skip projection-only synthetic entry names when computing per-method averages
        if name == 'DDPM_projected' or name == 'DDPM (proj.)':
            continue
        st = stats.get(name, None)
        if st is None or st.get('count', 0) == 0:
            avg_stats[name] = {'m': np.nan, 'p': np.nan, 's': np.nan}
            continue
        cnt = float(st['count'])
        m_avg = float(st['m_sum']) / cnt if st['m_sum'] != 0 else (st['m_sum'] / cnt if st['m_sum'] == 0 and cnt > 0 else np.nan)
        p_avg = float(st['p_sum']) / cnt if st['p_sum'] != 0 else (st['p_sum'] / cnt if st['p_sum'] == 0 and cnt > 0 else np.nan)
        s_avg = float(st['s_sum']) / cnt if st['s_sum'] != 0 else (st['s_sum'] / cnt if st['s_sum'] == 0 and cnt > 0 else np.nan)
        avg_stats[name] = {'m': m_avg, 'p': p_avg, 's': s_avg}

    # Accept external projection samples stored under either historical key or new label
    ext_list = []
    ext_list += stats.get('DDPM_projected', {}).get('external_proj_list', []) if stats.get('DDPM_projected') is not None else []
    ext_list += stats.get('DDPM (proj.)', {}).get('external_proj_list', []) if stats.get('DDPM (proj.)') is not None else []
    ext_mean = float(np.mean(ext_list)) if len(ext_list) > 0 else np.nan
    ddpm_base = avg_stats.get('DDPM', {'m': np.nan, 'p': np.nan, 's': np.nan})
    avg_stats['DDPM_projected'] = {
        'm': ddpm_base.get('m', np.nan),
        'p': ext_mean,
        's': (ddpm_base.get('s', np.nan) + ext_mean) if np.isfinite(ddpm_base.get('s', np.nan)) and np.isfinite(ext_mean) else np.nan,
    }
    # Also provide the new labeling expected by updated plotting scripts
    if 'DDPM (proj.)' not in avg_stats:
        avg_stats['DDPM (proj.)'] = avg_stats['DDPM_projected']

    return avg_stats