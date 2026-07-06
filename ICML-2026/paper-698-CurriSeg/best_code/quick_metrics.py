import numpy as np, cv2, os, glob

pred_dir = "/repo/res/curriseg/COD10K/"
gt_dir = "/datasets/TestDataset/COD10K/GT/"

pred_files = sorted(glob.glob(pred_dir + "*.png"))
gt_dict = {}
for gf in sorted(glob.glob(gt_dir + "*.png") + glob.glob(gt_dir + "*.tif")):
    gt_dict[os.path.splitext(os.path.basename(gf))[0]] = gf

mae_total = fbeta_total = ephi_total = salpha_total = 0.0
count = 0
BETA2 = 0.3

for i, pf in enumerate(pred_files):
    base = os.path.splitext(os.path.basename(pf))[0]
    if base not in gt_dict:
        continue
    
    pred = cv2.imread(pf, cv2.IMREAD_GRAYSCALE).astype(np.float64) / 255.0
    gt = cv2.imread(gt_dict[base], cv2.IMREAD_GRAYSCALE).astype(np.float64) / 255.0
    if pred.shape != gt.shape:
        pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]))
    
    # MAE
    mae_total += np.abs(pred - gt).mean()
    
    # Adaptive F-measure (threshold = 2 * mean(pred))
    th = 2.0 * pred.mean()
    th = min(max(th, 0.0), 1.0)
    bin_pred = (pred >= th).astype(np.float64)
    tp = (bin_pred * gt).sum()
    fp = (bin_pred * (1 - gt)).sum()
    fn = ((1 - bin_pred) * gt).sum()
    prec = tp / (tp + fp + 1e-8)
    rec = tp / (tp + fn + 1e-8)
    fbeta_total += (1 + BETA2) * prec * rec / (BETA2 * prec + rec + 1e-8)
    
    # E-measure (simplified: alignment matrix)
    gm = gt.mean()
    fm = bin_pred.mean()
    ephi_total += (2 * fm * gm) / (fm + gm + 1e-8) if (fm + gm) > 0 else 0
    
    # S-measure
    mx, my = pred.mean(), gt.mean()
    sx = np.sqrt(((pred - mx)**2).mean())
    sy = np.sqrt(((gt - my)**2).mean())
    sxy = ((pred - mx)*(gt - my)).mean()
    C1, C2 = 0.01, 0.03
    Sr = ((2*mx*my+C1)*(2*sxy+C2)) / ((mx**2+my**2+C1)*(sx**2+sy**2+C2)+1e-8)
    
    y = gt.mean()
    if y < 0.001:
        Q = 1.0 - pred.mean()
    elif y > 0.999:
        Q = pred.mean()
    else:
        fg = (gt >= 0.5).astype(np.float64)
        bg = 1.0 - fg
        nfg, nbg = fg.sum(), bg.sum()
        if nfg < 1 or nbg < 1:
            Q = Sr
        else:
            muf = (pred * fg).sum() / nfg
            mub = (pred * bg).sum() / nbg
            Q = y * muf + (1 - y) * (1 - mub)
    salpha_total += 0.5 * Sr + 0.5 * Q
    
    count += 1
    if count % 500 == 0:
        print("  %d M=%.6f Fbeta=%.4f Ephi=%.4f Salpha=%.4f" % (
            count, mae_total/count, fbeta_total/count, ephi_total/count, salpha_total/count))

print("\n===== RESULTS (COD10K) =====")
print("M:       %.6f" % (mae_total / count))
print("Fbeta:   %.4f" % (fbeta_total / count))
print("Ephi:    %.4f" % (ephi_total / count))
print("Salpha:  %.4f" % (salpha_total / count))
