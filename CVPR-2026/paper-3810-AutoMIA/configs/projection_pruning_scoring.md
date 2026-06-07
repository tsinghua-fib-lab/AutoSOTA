## Projection-Aligned Component Scoring

Let the voxel grid have shape $D\times H\times W$. The continuous density volume before PyTorch3D reshaping is
$$
 \text{dens}_{\text{raw}} = \operatorname{clip}\left(\sigma\big(T_1\cdot\text{log\_densities}-b\big)\cdot T_2,\ 0,\ 1\right)\in\mathbb{R}^{1\times D\times H\times W},
$$
where $T_1$ is the inner temperature, $T_2$ the outer scale, and $b$ the density bias. Applying a binary threshold $\tau$ yields
$$
B = \mathbf{1}[\text{dens}_{\text{raw},0} > \tau]\in\{0,1\}^{D\times H\times W}.
$$

We run 26-neighborhood connected-component labeling on $B$, discard components whose volume ratio is below `min_ratio`, and obtain $\mathcal{C}=\{C_i\}_{i=1}^M$.

For any candidate $C$:

1. Build a mask $M_C\in\{0,1\}^{D\times H\times W}$ and permute it to PyTorch3D format $\widetilde{M}_C\in\{0,1\}^{1\times 1\times H\times W\times D}$.
2. Extract densities $\mathcal{D}\in[0,1]^{1\times 1\times H\times W\times D}$ and dual color fields $\mathcal{C}_1,\mathcal{C}_2\in[0,1]^{1\times 3\times H\times W\times D}$ consistent with the forward pass.
3. For each view $k\in\{1,2\}$ compute the camera-dependent blend weight $w_k\in\{0,1\}$ from the dot product between the camera center direction and `axis_dir`, and obtain $\mathcal{C}^{(k)}=w_k\mathcal{C}_1+(1-w_k)\mathcal{C}_2$.
4. Mask the candidate volume and render both cameras to obtain alpha maps $\hat{\alpha}_k(C)\in[0,1]^{H\times W}$.

Denote the binary supervision masks by $\alpha_k^\*$ and the far-background weights by $W_k^{far}\in[1,w_{\max}]^{H\times W}$. The scoring terms are

- **IoU fidelity**
  $$
  \mathrm{IoU}_k = \frac{\langle\hat{\alpha}_k(C), \alpha_k^\*\rangle}{\|\hat{\alpha}_k(C)\|_1+\|\alpha_k^\*\|_1-\langle\hat{\alpha}_k(C), \alpha_k^\*\rangle+\varepsilon},\quad \overline{\mathrm{IoU}} = \tfrac12(\mathrm{IoU}_1+\mathrm{IoU}_2).
  $$
- **Background spill penalty**
  $$
  \mathrm{Out}_k = \frac{\langle \hat{\alpha}_k(C)\odot (1-\alpha_k^\*),\ W_k^{far}\rangle}{\|W_k^{far}\|_1+\varepsilon},\quad \overline{\mathrm{Out}}=\tfrac12(\mathrm{Out}_1+\mathrm{Out}_2).
  $$
- **Foreground fill loss**
  $$
  \mathrm{Fill}_k = \frac{\|\big(\hat{\alpha}_k(C)-1\big)\cdot \mathbf{1}[\alpha_k^\*>0.5]\|_2^2}{\|\mathbf{1}[\alpha_k^\*>0.5]\|_1+\varepsilon},\quad \overline{\mathrm{Fill}}=\tfrac12(\mathrm{Fill}_1+\mathrm{Fill}_2).
  $$
- **Scale prior**
  $$
  \text{size\_norm}(C)=\max\!\left(\frac{|C|}{\max_{C'\in \mathcal{C}} |C'|}, 10^{-6}\right), \quad \log\text{-term} = \log(\text{size\_norm}(C)).
  $$

The overall projection-aligned score is
$$
S(C)= 1.0\cdot\overline{\mathrm{IoU}} - 1.0\cdot\overline{\mathrm{Out}} - 0.3\cdot\overline{\mathrm{Fill}} + 0.05\cdot\log(\text{size\_norm}(C)).
$$

Intuition: the IoU reward explains the supervision foreground; the spill term suppresses floating artifacts with far-background weighting; the fill term drives interior densities toward opaque; the logarithmic size bias prevents spurious tiny blobs from surviving.

---

## Component Selection Rule

We sort $\mathcal{C}$ by $S(C)$, keep the top $K_{\max}$ entries, and union them with all components whose score exceeds the threshold $\theta$:
$$
\mathcal{K} = \left(\text{Top-}K_{\max} \text{ by } S\right) \cup \{C \mid S(C)\ge \theta\}.
$$

If $\mathcal{K}=\varnothing$, we keep the single best-scoring component to avoid an empty grid. Voxels outside $\mathcal{K}$ are clamped to the negative logit (effectively pruning them).

Implementation notes:
- All measurements run inside `torch.no_grad()` to avoid polluting gradients.
- Density/feature/mask tensors strictly follow the `(batch, channel, H, W, D)` layout required by PyTorch3D `Volumes`.
- View-dependent weights $w_k$ reuse the forward pass logic to maintain consistency between optimization and pruning. 
