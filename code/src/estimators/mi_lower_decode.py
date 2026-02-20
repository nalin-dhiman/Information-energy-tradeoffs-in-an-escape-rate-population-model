import numpy as np

def ridge_fit(X, y, alpha=1.0):
    # w = (X^T X + alpha I)^-1 X^T y
    n_feat = X.shape[1]
    # Check for ill-conditioning?
    # Usually fine with alpha > 0.
    A = X.T @ X + alpha * np.eye(n_feat)
    b = X.T @ y
    try:
        w = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        # Fallback to lstsq or pinv
        w = np.linalg.lstsq(A, b, rcond=None)[0]
    return w

def ridge_predict(X, w):
    return X @ w

def get_lagged_features(A, lags):
    """
    Creates feature matrix X from A with lags.
    X[t] = [A[t], A[t-1], ..., A[t-lags]]
    """
    n_samples = len(A)
    # Valid range starts at `lags`
    X = np.zeros((n_samples - lags, lags + 1))
    for i in range(lags + 1):
        # A[t-i] corresponds to X[:, i]
        # slice A from (lags-i) to (n_samples-i)
        X[:, i] = A[lags-i : n_samples-i]
    return X

def estimate_mi_lower_decode(S, A, dt, cfg):
    """
    Computes MI lower bound via decoding S from A with strict cross-validation.
    Supports list of trials (S=[S1, S2...], A=[A1, A2...]).
    
    Args:
        S: Stimulus array or list of arrays
        A: Activity array or list of arrays
        dt: Time step
        cfg: Estimator config
             - lags: history length (samples/steps)
             - folds: number of CV folds (if not trial split)
             - bandwidth: cutoff freq for downsampling
    """
    # 1. Handle Multi-Trial Input
    if isinstance(S, list):
        S_list = S
        A_list = A
        n_trials = len(S)
    else:
        S_list = [S]
        A_list = [A]
        n_trials = 1

    lags = cfg.get('lags', 10)
    alpha = cfg.get('parameters', {}).get('alpha', 1.0)
    
    # 2. Downsampling (Bandwidth-Aware)
    # dt_eff = 1 / (2 * cutoff)
    # Step = dt_eff / dt
    bandwidth = cfg.get('bandwidth', None)
    
    if bandwidth is not None:
        dt_eff = 1.0 / (2.0 * bandwidth)
        step = int(np.round(dt_eff / dt))
        if step < 1: step = 1
    else:
        dt_eff = dt
        step = 1

    # Prepare Data with Lags and Downsampling
    X_list = []
    y_list = []
    
    # decode config section might be flattened into cfg or passed in 'parameters'
    # The caller typically merges decode config into cfg.
    # We look for 'feature_mode'.
    # Fallback to 'features' for backward compat or 'rate_lags' default.
    feature_mode = cfg.get('feature_mode', cfg.get('features', 'rate_lags'))
    
    # Mode A: rate_lags (equivalent to 'rate' in previous versions)
    # Mode B: spikecount_lags (equivalent to 'spikecount')
    
    for i in range(n_trials):
        Si = S_list[i]
        Ai = A_list[i]
        
        if feature_mode == 'spikecount_lags':
            # Binning Logic
            # "Bin spikes into bins of width bin_dt (e.g. 10ms or dt_eff)"
            # Priority: cfg['bin_dt'] > dt_eff > 10ms
            
            # Determine bin size
            tgt_bin = cfg.get('bin_dt')
            if tgt_bin is None:
                if bandwidth is not None:
                     tgt_bin = 1.0 / (2.0 * bandwidth)
                else:
                     tgt_bin = 0.01 # 10ms default
            
            bin_steps = int(np.round(tgt_bin / dt))
            if bin_steps < 1: bin_steps = 1
            
            # Bin A (Sum) and S (Mean)
            n_bins = len(Ai) // bin_steps
            limit = n_bins * bin_steps
            
            # Check dimension of A. If 1D, basic reshape. 
            Ai_trunc = Ai[:limit]
            Si_trunc = Si[:limit]
            
            Ai_binned = Ai_trunc.reshape(-1, bin_steps).sum(axis=1) 
            Si_binned = Si_trunc.reshape(-1, bin_steps).mean(axis=1)
            
            # Create Lags
            # "Create lag taps over 0–500ms" -> lags depends on bin size.
            # If lags arg is provided (int), use it.
            # Else compute from window duration.
            if 'lags' in cfg:
                n_lags = int(cfg['lags'])
            else:
                window = cfg.get('lag_window', 0.5) # 500ms
                n_lags = int(window / tgt_bin)
            
            Xi = get_lagged_features(Ai_binned, n_lags)
            yi = Si_binned[n_lags:]
            
            # No downsampling needed as we already binned
            
            # Output dt_eff for this mode is the bin size
            dt_eff_out = tgt_bin
            
        else:
            # feature_mode == 'rate_lags' or default
            # Standard smooth rate processing
            Xi = get_lagged_features(Ai, lags) # Uses global 'lags' param
            yi = Si[lags:]
            
            # Downsample if requested / calculated
            if step > 1:
                Xi = Xi[::step]
                yi = yi[::step]
                
            dt_eff_out = dt_eff

        X_list.append(Xi)
        y_list.append(yi)
        
    # 3. Splitting Strategy
    # If trial split, we treat trials as units.
    # If block split, we concat and chunk.
    split_mode = cfg.get('split', 'block')
    
    mse_list = []
    r2_list = []
    weights_list = []
    var_list = []
    
    if split_mode == 'trial' and n_trials > 1:
        # Trial Split
        train_frac = cfg.get('train_frac', 0.7)
        rng = np.random.RandomState(cfg.get('seed', 42))
        indices = np.arange(n_trials)
        rng.shuffle(indices)
        
        n_train = int(n_trials * train_frac)
        train_idx = indices[:n_train]
        test_idx = indices[n_train:]
        
        if len(test_idx) == 0:
            return {'I_lower_bits_per_s': 0.0, 'error': 'No test trials'}
            
        X_train = np.concatenate([X_list[k] for k in train_idx])
        y_train = np.concatenate([y_list[k] for k in train_idx])
        X_test  = np.concatenate([X_list[k] for k in test_idx])
        y_test  = np.concatenate([y_list[k] for k in test_idx])
        
        w = ridge_fit(X_train, y_train, alpha)
        pred = ridge_predict(X_test, w)
        
        mse = np.mean((y_test - pred)**2)
        var_test = np.var(y_test)
        r2 = 1.0 - mse / (var_test + 1e-12)
        
        mse_list.append(mse)
        r2_list.append(r2)
        var_list.append(var_test) 
        if hasattr(w, 'coef_'):
             weights_list.append(w.coef_) 
        else:
             weights_list.append(w) 
        
    else:
        # Block CV (Legacy / Single Trial)
        X_all = np.concatenate(X_list)
        y_all = np.concatenate(y_list)
        
        folds = cfg.get('folds', 5)
        # Gap logic
        gap_seconds = cfg.get('gap_seconds', 0.1)
        gap = int(gap_seconds / dt_eff_out) 
        
        # ... (Block CV logic) ...
        n_samples = len(y_all)
        chunk = n_samples // folds
        for k in range(folds):
            start = k * chunk
            end = (k + 1) * chunk
            test_mask = np.zeros(n_samples, dtype=bool)
            test_mask[start:end] = True
            
            train_mask = np.ones(n_samples, dtype=bool)
            ex_start = max(0, start - gap)
            ex_end = min(n_samples, end + gap)
            train_mask[ex_start:ex_end] = False
            
            X_tr, y_tr = X_all[train_mask], y_all[train_mask]
            X_te, y_te = X_all[test_mask], y_all[test_mask]
            
            if len(y_tr) < 10 or len(y_te) < 10: continue
            
            w = ridge_fit(X_tr, y_tr, alpha)
            pred = ridge_predict(X_te, w)
            mse = np.mean((y_te - pred)**2)
            var_t = np.var(y_te)
            r2 = 1.0 - mse / (var_t + 1e-12)
            
            mse_list.append(mse)
            r2_list.append(r2)
            var_list.append(var_t)
            if hasattr(w, 'coef_'): weights_list.append(w.coef_)
            else: weights_list.append(w)

    # 4. Metrics
    if not mse_list:
        return {'I_lower_bits_per_s': 0.0, 'error': 'Failed', 'diagnostics': {}}
        
    avg_mse = np.mean(mse_list)
    avg_r2 = np.mean(r2_list)
    avg_var = np.mean(var_list)
    
    clipped = False
    if avg_mse >= avg_var:
        eff_mse = avg_var - 1e-15
        clipped = True
        mi_bits_eff = 0.0
    else:
        eff_mse = avg_mse
        mi_bits_eff = 0.5 * np.log2(avg_var / eff_mse)
        
    # Rate Scaling
    # Normalize result to bits/s
    # In both cases, mi_bits_eff is bits per sample (where sample is dt_eff_out duration)
    # Rate = bits/sample * samples/sec
    mi_rate = mi_bits_eff * (1.0 / dt_eff_out)
    
    return {
        'I_lower_bits_per_s': mi_rate,
        'mi_rate': mi_rate,
        'units': 'bits/s',
        'mi_eff_sample': mi_bits_eff,
        'dt_eff': dt_eff_out,
        'mse_test': avg_mse,
        'r2_test': avg_r2,
        'var_S_test': avg_var,
        'clipped': clipped,
        'diagnostics': {
            'estimator': f'mi_lower_decode_{feature_mode}',
            'split_mode': split_mode,
            'features': feature_mode,
            'bin_dt': dt_eff_out
        },
        'artifacts': {
            'weights_mean': np.mean(weights_list, axis=0) if weights_list else []
        }
    }
