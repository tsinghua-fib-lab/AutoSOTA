
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt

def ellipse_points(m, S, n=200):
    w, U = np.linalg.eigh(0.5*(S+S.T))
    w = np.clip(w, 1e-12, None)
    ang = np.linspace(0, 2*np.pi, n)
    circ = np.stack([np.cos(ang), np.sin(ang)], axis=0)
    A = U @ (np.sqrt(w)[:,None] * np.eye(len(w)))
    pts = (A @ circ).T + m
    return pts

def plot_gaussian_ellipse(ax, m, S, lw=2.0, color='tab:blue', label=None, linestyle= '-'):
    pts = ellipse_points(m, S)
    ax.plot(pts[:,0], pts[:,1], lw=lw, alpha=1,color=color, label = label, linestyle = linestyle)

def plot_gmm_contours(ax, gmm, xlim=(-6,6), ylim=(-4,4), n=200, levels=12):
    xs = np.linspace(*xlim, n)
    ys = np.linspace(*ylim, n)
    XX, YY = np.meshgrid(xs, ys)
    ZZ = np.zeros_like(XX)
    for i in range(n):
        for j in range(n):
            x = np.array([XX[i,j], YY[i,j]])
            ZZ[i,j] = np.exp(gmm.logp(x))
    cs = ax.contour(XX, YY, ZZ, levels=levels,linewidths=0.7,alpha = 1)
    #cs = ax.contourf(XX, YY, ZZ, levels=levels,cmap="Blues", antialiased = True,alpha = 0.7)
    return cs
