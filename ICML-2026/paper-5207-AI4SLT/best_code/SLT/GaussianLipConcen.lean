/-
Copyright (c) 2026 Yuanhe Zhang. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Yuanhe Zhang, Jason D. Lee, Fanghui Liu
-/
import SLT.GaussianLSI.TensorizedGLSI
import SLT.LipschitzProperty
import SLT.MeasureInfrastructure
import SLT.GaussianSobolevDense.LipschitzMollification

/-!
# Concentration inequalities for general Lipschitz functions via Log-Sobolev Inequality

## Approach

The C¹ case uses Herbst argument with LSI. General Lipschitz
follows by mollification approximation: f_ε → f with preserved Lipschitz constant.
-/

open MeasureTheory ProbabilityTheory Real Finset BigOperators Function GaussianMeasure GaussianSobolev
open scoped ENNReal

namespace GaussianLipConcen

variable {n : ℕ}

/-! ### C¹ Lipschitz Functions and Gradient Bounds -/

/-- ∑ᵢ (∂ᵢf)² ≤ L² for C¹ L-Lipschitz f. -/
lemma lipschitz_gradNormSq_bound {f : (Fin n → ℝ) → ℝ} {L : ℝ} (hL : 0 ≤ L)
    (hf_lip : LipschitzWith (Real.toNNReal L) f)
    (hf_C1 : ContDiff ℝ 1 f) (x : Fin n → ℝ) :
    GaussianLSI.gradNormSq n f x ≤ L^2 := by
  unfold GaussianLSI.gradNormSq
  let E := EuclideanSpace ℝ (Fin n)
  let e := EuclideanSpace.equiv (Fin n) ℝ
  let f' : E → ℝ := f ∘ e
  have hf'_lip : LipschitzWith (Real.toNNReal L) f' := by
    have he_lip : LipschitzWith 1 e := by
      rw [lipschitzWith_iff_dist_le_mul]
      intro x y
      simp only [NNReal.coe_one, one_mul]
      have key : ∀ z : E, ‖e z‖ ≤ ‖z‖ := fun z => by
        rw [pi_norm_le_iff_of_nonneg (norm_nonneg z)]
        intro i
        rw [EuclideanSpace.norm_eq]
        -- ‖(e z) i‖ = |z i| ≤ sqrt(∑ⱼ |z j|²)
        have h_eq : (e z) i = z i := rfl
        rw [h_eq, Real.norm_eq_abs]
        have h_sq : |z i|^2 ≤ ∑ j, |z j|^2 := Finset.single_le_sum
          (f := fun j => |z j|^2) (fun j _ => sq_nonneg _) (Finset.mem_univ i)
        calc |z i| = Real.sqrt (|z i|^2) := by rw [Real.sqrt_sq (abs_nonneg _)]
          _ ≤ Real.sqrt (∑ j, |z j|^2) := Real.sqrt_le_sqrt h_sq
      calc dist (e x) (e y) = ‖e x - e y‖ := dist_eq_norm _ _
        _ = ‖e (x - y)‖ := by simp [map_sub]
        _ ≤ ‖x - y‖ := key (x - y)
        _ = dist x y := (dist_eq_norm _ _).symm
    have h := hf_lip.comp he_lip
    simp only [mul_one] at h
    exact h
  -- The partial derivatives of f equal those of f' at corresponding points
  have h_partial_eq : ∀ i, GaussianLSI.partialDeriv i f x = fderiv ℝ f' (e.symm x) (EuclideanSpace.single i 1) := by
    intro i
    simp only [GaussianLSI.partialDeriv, f']
    -- The derivative of f ∘ e at e.symm x applied to EuclideanSpace.single i 1
    -- equals the derivative of f at x applied to Pi.single i 1
    have h_chain := fderiv_comp (e.symm x) (hf_C1.differentiable (by norm_num) (e (e.symm x))) e.differentiable.differentiableAt
    rw [h_chain]
    simp only [ContinuousLinearMap.comp_apply]
    congr 1
    -- fderiv of ContinuousLinearEquiv is itself
    rw [ContinuousLinearEquiv.fderiv]
    simp only [ContinuousLinearEquiv.coe_coe]
    -- e (EuclideanSpace.single i 1) = Pi.single i 1
    rfl
  -- For EuclideanSpace, use the orthonormal basis to get ∑ᵢ (L eᵢ)² = ‖L‖²
  -- Combined with ‖fderiv f'‖ ≤ L (from Lipschitz), we get the bound
  have h_fderiv_bound : ‖fderiv ℝ f' (e.symm x)‖ ≤ L := by
    have := norm_fderiv_le_of_lipschitz ℝ hf'_lip (x₀ := e.symm x)
    simp only [Real.coe_toNNReal L hL] at this
    exact this
  -- On EuclideanSpace, the basis is orthonormal, so ∑ᵢ |L(eᵢ)|² = ‖L‖²
  let b := EuclideanSpace.basisFun (Fin n) ℝ
  have h_sum_sq : ∑ i, (fderiv ℝ f' (e.symm x) (b i))^2 = ‖fderiv ℝ f' (e.symm x)‖^2 := by
    -- Use Riesz representation: L = ⟨a, ·⟩ where a = toDual.symm L
    -- Then L(b i) = ⟨a, b i⟩ and ∑ᵢ |L(b i)|² = ‖a‖² = ‖L‖²
    haveI : CompleteSpace E := inferInstance
    let L := fderiv ℝ f' (e.symm x)
    -- L is continuous, so it has a Riesz representative
    let a : E := (InnerProductSpace.toDual ℝ E).symm L
    have ha : ∀ v, L v = @inner ℝ E _ a v := fun v => by
      exact InnerProductSpace.toDual_symm_apply.symm
    -- ∑ᵢ |⟨a, b i⟩|² = ‖a‖² by Parseval/orthonormal basis
    have h_parseval := b.sum_sq_norm_inner_right a
    -- For real inner products: ‖inner x y‖ = |inner x y|, and inner is symmetric
    have h_parseval' : ∑ i, (@inner ℝ E _ a (b i))^2 = ‖a‖^2 := by
      convert h_parseval using 2 with i
      rw [real_inner_comm]
      rw [Real.norm_eq_abs, ← sq_abs]
    -- ‖a‖ = ‖L‖ by Riesz isometry
    have h_norm : ‖a‖ = ‖L‖ := by
      simp only [a]
      rw [LinearIsometryEquiv.norm_map]
    calc ∑ i, (L (b i))^2 = ∑ i, (@inner ℝ E _ a (b i))^2 := by simp [ha]
      _ = ‖a‖^2 := h_parseval'
      _ = ‖L‖^2 := by rw [h_norm]
  -- Now combine: the partial derivatives equal L(b i), so their sum of squares equals ‖L‖² ≤ L²
  have h_basis_eq : ∀ i, b i = EuclideanSpace.single i 1 := fun i =>
    EuclideanSpace.basisFun_apply (Fin n) ℝ i
  calc ∑ i : Fin n, (GaussianLSI.partialDeriv i f x)^2
      = ∑ i, (fderiv ℝ f' (e.symm x) (EuclideanSpace.single i 1))^2 := by simp [h_partial_eq]
    _ = ∑ i, (fderiv ℝ f' (e.symm x) (b i))^2 := by simp [h_basis_eq]
    _ = ‖fderiv ℝ f' (e.symm x)‖^2 := h_sum_sq
    _ ≤ L^2 := sq_le_sq' (by linarith [norm_nonneg (fderiv ℝ f' (e.symm x))]) h_fderiv_bound

/-- |∇f|² is integrable for C¹ Lipschitz f. -/
lemma lipschitz_gradNormSq_integrable {f : (Fin n → ℝ) → ℝ} {L : ℝ} (hL : 0 ≤ L)
    (hf_lip : LipschitzWith (Real.toNNReal L) f)
    (hf_C1 : ContDiff ℝ 1 f) :
    Integrable (fun x => GaussianLSI.gradNormSq n f x) (stdGaussianPi n) := by
  apply Integrable.mono' (integrable_const (L^2))
  · -- Measurability
    apply Measurable.aestronglyMeasurable
    unfold GaussianLSI.gradNormSq
    apply Finset.measurable_sum
    intro i _
    apply Measurable.pow
    · unfold GaussianLSI.partialDeriv
      -- Continuous functions are measurable
      have hf_cont : Continuous (fderiv ℝ f) := (hf_C1.fderiv_right (m := 0) le_rfl).continuous
      -- Evaluation at a fixed vector is a continuous linear map
      have h_eval_cont : Continuous (fun L : (Fin n → ℝ) →L[ℝ] ℝ => L (Pi.single i 1)) :=
        ContinuousLinearMap.apply ℝ ℝ (Pi.single i 1) |>.continuous
      exact (h_eval_cont.comp hf_cont).measurable
    · exact measurable_const
  · filter_upwards with x
    rw [Real.norm_eq_abs]
    have h_nn : 0 ≤ GaussianLSI.gradNormSq n f x := by
      unfold GaussianLSI.gradNormSq
      exact Finset.sum_nonneg (fun i _ => sq_nonneg _)
    rw [abs_of_nonneg h_nn]
    exact lipschitz_gradNormSq_bound hL hf_lip hf_C1 x

/-- ∫|∇f|² ≤ L² for C¹ L-Lipschitz f. -/
lemma lipschitz_integral_gradNormSq_bound {f : (Fin n → ℝ) → ℝ} {L : ℝ} (hL : 0 ≤ L)
    (hf_lip : LipschitzWith (Real.toNNReal L) f)
    (hf_C1 : ContDiff ℝ 1 f) :
    ∫ x, GaussianLSI.gradNormSq n f x ∂(stdGaussianPi n) ≤ L^2 := by
  haveI : IsProbabilityMeasure (stdGaussianPi n) :=
    stdGaussianPi_isProbabilityMeasure
  calc ∫ x, GaussianLSI.gradNormSq n f x ∂(stdGaussianPi n)
      ≤ ∫ _, L^2 ∂(stdGaussianPi n) := by
        apply integral_mono (lipschitz_gradNormSq_integrable hL hf_lip hf_C1) (integrable_const _)
        intro x
        exact lipschitz_gradNormSq_bound hL hf_lip hf_C1 x
    _ = L^2 := by simp

/-! ### Herbst Argument: LSI → CGF Bound -/

/-- exp(t(f-Ef)) is integrable for Lipschitz f under Gaussian. -/
lemma lipschitz_exp_integrable {f : (Fin n → ℝ) → ℝ} {L : ℝ} (hL : 0 ≤ L)
    (hf_lip : LipschitzWith (Real.toNNReal L) f)
    (t : ℝ) :
    Integrable (fun x => exp (t * (f x - ∫ y, f y ∂(stdGaussianPi n))))
        (stdGaussianPi n) := by
  -- Key constant: C = |f(0)| + |E[f]|
  let Ef := ∫ y, f y ∂(stdGaussianPi n)
  let C := |f 0| + |Ef|
  -- Lipschitz linear growth: |f(x)| ≤ |f(0)| + L * ‖x‖
  have h_growth : ∀ x, |f x| ≤ |f 0| + L * ‖x‖ :=
    fun x => LipschitzConcentration.lipschitz_linear_growth f L hL hf_lip x
  -- Pi-norm bound: ‖x‖ ≤ ∑ᵢ |xᵢ|
  have h_norm_sum : ∀ x : Fin n → ℝ, ‖x‖ ≤ ∑ i, |x i| := by
    intro x
    rw [pi_norm_le_iff_of_nonneg (Finset.sum_nonneg fun i _ => abs_nonneg (x i))]
    intro i
    rw [Real.norm_eq_abs]
    exact Finset.single_le_sum (fun j _ => abs_nonneg (x j)) (Finset.mem_univ i)
  -- Bound on centered function: |f(x) - E[f]| ≤ C + L * ∑ᵢ |xᵢ|
  have h_centered_bound : ∀ x, |f x - Ef| ≤ C + L * ∑ i, |x i| := by
    intro x
    calc |f x - Ef| ≤ |f x| + |Ef| := abs_sub (f x) Ef
      _ ≤ (|f 0| + L * ‖x‖) + |Ef| := by linarith [h_growth x]
      _ = C + L * ‖x‖ := by ring
      _ ≤ C + L * ∑ i, |x i| := by
        have := h_norm_sum x
        nlinarith
  -- The dominating function: exp(|t| * (C + L * ∑ᵢ |xᵢ|))
  -- = exp(|t| * C) * ∏ᵢ exp(|t| * L * |xᵢ|)
  -- To show integrability of ∏ᵢ exp(c * |xᵢ|), use Integrable.fintype_prod
  -- Each factor exp(c * |y|) is integrable under gaussianReal 0 1
  have h_factor_int : ∀ c : ℝ, Integrable (fun y : ℝ => exp (c * |y|)) (gaussianReal 0 1) := by
    intro c
    apply integrable_exp_mul_abs
    · exact integrable_exp_mul_gaussianReal c
    · have : (fun y => exp (-c * y)) = (fun y => exp ((-c) * y)) := by ext; ring_nf
      rw [this]
      exact integrable_exp_mul_gaussianReal (-c)
  -- Product integrability: ∏ᵢ exp(c * |xᵢ|) is integrable under stdGaussianPi
  have h_prod_int : ∀ c : ℝ, Integrable (fun x : Fin n → ℝ => ∏ i, exp (c * |x i|))
      (stdGaussianPi n) := by
    intro c
    unfold stdGaussianPi
    exact Integrable.fintype_prod (fun _ => h_factor_int c)
  -- Convert product to sum in exponent
  have h_exp_sum_eq : ∀ c (y : Fin n → ℝ), exp (c * ∑ i, |y i|) = ∏ i : Fin n, exp (c * |y i|) := by
    intro c y
    rw [mul_sum, exp_sum]
  -- Integrability of exp(c * ∑ᵢ |xᵢ|)
  have h_sum_int : ∀ c : ℝ, Integrable (fun x : Fin n → ℝ => exp (c * ∑ i, |x i|))
      (stdGaussianPi n) := by
    intro c
    simp_rw [h_exp_sum_eq]
    exact h_prod_int c
  -- The full dominating function exp(|t| * C) * exp(|t| * L * ∑ᵢ |xᵢ|)
  let g : (Fin n → ℝ) → ℝ := fun x => exp (|t| * C) * exp (|t| * L * ∑ i, |x i|)
  have hg_int : Integrable g (stdGaussianPi n) :=
    (h_sum_int (|t| * L)).const_mul (exp (|t| * C))
  -- The original function is bounded by g
  have h_bound : ∀ x, |exp (t * (f x - Ef))| ≤ g x := by
    intro x
    rw [abs_exp]
    have h1 : t * (f x - Ef) ≤ |t| * |f x - Ef| := by
      calc t * (f x - Ef) ≤ |t * (f x - Ef)| := le_abs_self _
        _ = |t| * |f x - Ef| := abs_mul _ _
    have h2 : |t| * |f x - Ef| ≤ |t| * (C + L * ∑ i, |x i|) := by
      apply mul_le_mul_of_nonneg_left (h_centered_bound x) (abs_nonneg _)
    calc exp (t * (f x - Ef)) ≤ exp (|t| * |f x - Ef|) := exp_le_exp.mpr h1
      _ ≤ exp (|t| * (C + L * ∑ i, |x i|)) := exp_le_exp.mpr h2
      _ = exp (|t| * C + |t| * L * ∑ i, |x i|) := by ring_nf
      _ = exp (|t| * C) * exp (|t| * L * ∑ i, |x i|) := exp_add _ _
      _ = g x := rfl
  -- Apply Integrable.mono'
  apply Integrable.mono' hg_int
  · -- exp(t * (f x - Ef)) is continuous, hence measurable
    have h_cont : Continuous (fun x => exp (t * (f x - Ef))) :=
      continuous_exp.comp ((continuous_const.mul (hf_lip.continuous.sub continuous_const)))
    exact h_cont.aestronglyMeasurable
  · filter_upwards with x
    rw [Real.norm_eq_abs]
    exact h_bound x

/-! ### Grönwall-type Ratio Bound -/

/-- Grönwall: If φ(0)=0, φ'(0)=0, s·φ'(s)-φ(s)≤0 for s≠0, then φ(t)/t≤0. -/
lemma ratio_bound_gronwall (φ : ℝ → ℝ) (t : ℝ) (ht : 0 < t)
    (hφ_zero : φ 0 = 0)
    (hφ_deriv_zero : deriv φ 0 = 0)
    (hφ_diff : ∀ x, DifferentiableAt ℝ φ x)
    (h_diff_ineq : ∀ s, s ≠ 0 → s * deriv φ s - φ s ≤ 0) :
    φ t / t ≤ 0 := by
  by_contra h_contra
  push Not at h_contra
  -- If φ(t)/t > 0, we derive a contradiction
  have h_ratio_diff : ∀ x, x ∈ Set.Ioo (0 : ℝ) t → DifferentiableAt ℝ (fun s => φ s / s) x := by
    intro x hx
    apply DifferentiableAt.div (hφ_diff x) differentiableAt_id (ne_of_gt hx.1)
  have h_mvt : AntitoneOn (fun s => φ s / s) (Set.Ioo 0 t) :=
    antitoneOn_of_deriv_nonpos (convex_Ioo 0 t)
      (by
        apply ContinuousOn.div
        · apply continuousOn_of_forall_continuousAt
          intro x _
          exact (hφ_diff x).continuousAt
        · exact continuous_id.continuousOn
        · intro x hx; exact ne_of_gt hx.1)
      (by
        intro x hx
        rw [interior_Ioo] at hx
        apply (h_ratio_diff x hx).differentiableWithinAt)
      (by
        intro x hx
        simp only [interior_Ioo, Set.mem_Ioo] at hx
        have hx_pos : 0 < x := hx.1
        have hx_ne : x ≠ 0 := ne_of_gt hx_pos
        -- Compute derivative: d/dx(φ(x)/x) = (x*φ'(x) - φ(x))/x²
        have h_deriv_ratio : deriv (fun s => φ s / s) x =
            (x * deriv φ x - φ x) / x^2 := by
          have hf_eq : (fun s => φ s / s) = (fun s => φ s * s⁻¹) := by
            ext s; ring
          rw [hf_eq]
          have h1 : deriv (fun s => φ s * s⁻¹) x =
              deriv φ x * x⁻¹ + φ x * deriv (·⁻¹) x := by
            apply deriv_mul (hφ_diff x)
            exact differentiableAt_inv hx_ne
          rw [h1, deriv_inv]
          field_simp
          ring
        rw [h_deriv_ratio]
        have h3 := h_diff_ineq x hx_ne
        have hx_sq_pos : 0 < x^2 := sq_pos_of_pos hx_pos
        apply div_nonpos_of_nonpos_of_nonneg h3 (le_of_lt hx_sq_pos))
  -- Compute limit as s → 0+
  have h_limit : Filter.Tendsto (fun s => φ s / s) (nhdsWithin 0 (Set.Ioi 0)) (nhds 0) := by
    have h_lhopital : Filter.Tendsto (fun s => φ s / s) (nhdsWithin 0 (Set.Ioi 0))
        (nhds (deriv φ 0)) := by
      have h_slope := (hφ_diff 0).hasDerivAt.tendsto_slope_zero_right
      simp only [zero_add, hφ_zero, sub_zero, smul_eq_mul, inv_mul_eq_div] at h_slope
      exact h_slope
    rw [hφ_deriv_zero] at h_lhopital
    exact h_lhopital
  -- The function φ(s)/s approaches 0 as s → 0+ and is decreasing for s > 0
  have h_mono : ∀ s, s ∈ Set.Ioo 0 t → φ s / s ≥ φ t / t := fun s hs => by
    have h_ratio_cont : ContinuousAt (fun x => φ x / x) t :=
      ContinuousAt.div (hφ_diff t).continuousAt continuousAt_id (ne_of_gt ht)
    have h_tendsto : Filter.Tendsto (fun y => φ y / y) (nhdsWithin t (Set.Iio t)) (nhds (φ t / t)) :=
      h_ratio_cont.tendsto.mono_left nhdsWithin_le_nhds
    haveI : (nhdsWithin t (Set.Iio t)).NeBot := inferInstance
    apply le_of_tendsto h_tendsto
    filter_upwards [Ioo_mem_nhdsLT hs.2] with y hy
    exact h_mvt hs ⟨lt_trans hs.1 hy.1, hy.2⟩ (le_of_lt hy.1)
  -- Derive contradiction
  have h_eps : φ t / t / 2 > 0 := by linarith
  obtain ⟨δ, hδ_pos, hδ_bound⟩ := Metric.tendsto_nhdsWithin_nhds.mp h_limit (φ t / t / 2) h_eps
  let s := min (t / 2) (δ / 2)
  have hs_pos : 0 < s := lt_min (by linarith) (by linarith)
  have hs_lt_t : s < t := lt_of_le_of_lt (min_le_left _ _) (by linarith)
  have hs_lt_δ : s < δ := lt_of_le_of_lt (min_le_right _ _) (by linarith)
  have hs_in : s ∈ Set.Ioo 0 t := ⟨hs_pos, hs_lt_t⟩
  have h4 := h_mono s hs_in
  have hs_mem : s ∈ Set.Ioi 0 := hs_pos
  have hs_dist : dist s 0 < δ := by simp [abs_of_pos hs_pos, hs_lt_δ]
  have h5 := hδ_bound hs_mem hs_dist
  simp only [Real.dist_eq, sub_zero] at h5
  have h9 : |φ s / s| ≥ φ s / s := le_abs_self _
  linarith

set_option maxHeartbeats 400000 in
/-- Ent[exp(sf_c)] ≤ (s²σ²/2)·E[exp(sf_c)] via LSI. -/
lemma entropy_bound_exp_scaled {f : (Fin n → ℝ) → ℝ} {σ L_lip : ℝ}
    (_ : 0 < σ)
    (hf_lip : LipschitzWith (Real.toNNReal L_lip) f)
    (hL_lip : 0 < L_lip)
    (hf_C1 : ContDiff ℝ 1 f)
    (h_grad_bound : ∀ x, GaussianLSI.gradNormSq n f x ≤ σ^2)
    (s : ℝ) :
    LogSobolev.entropy (stdGaussianPi n)
      (fun x => exp (s * (f x - ∫ y, f y ∂(stdGaussianPi n)))) ≤
    (s^2 * σ^2 / 2) * ∫ x, exp (s * (f x - ∫ y, f y ∂(stdGaussianPi n)))
      ∂(stdGaussianPi n) := by
  let μ := stdGaussianPi n
  let f_c := fun x => f x - ∫ y, f y ∂μ
  -- Define g(x) = exp(s * f_c(x) / 2). Then g² = exp(s * f_c).
  let g := fun x => exp (s / 2 * f_c x)
  -- g² = exp(s * f_c)
  have h_gsq_eq : ∀ x, (g x)^2 = exp (s * f_c x) := by
    intro x; unfold g; rw [sq, ← exp_add]; congr 1; ring
  -- Gradient of g: |∇g|² = (s/2)² |∇f|² g²
  have h_grad_g : ∀ x, GaussianLSI.gradNormSq n g x = (s/2)^2 * GaussianLSI.gradNormSq n f x * (g x)^2 := by
    intro x
    unfold GaussianLSI.gradNormSq GaussianLSI.partialDeriv g
    have h_deriv : ∀ i, fderiv ℝ (fun x => exp (s / 2 * f_c x)) x (Pi.single i 1) =
        (s / 2) * fderiv ℝ f x (Pi.single i 1) * exp (s / 2 * f_c x) := by
      intro i
      have hf_diff : DifferentiableAt ℝ f x := hf_C1.differentiable (by norm_num) x
      have hfc_diff : DifferentiableAt ℝ f_c x := hf_diff.sub (differentiableAt_const _)
      have h_fc_fderiv : fderiv ℝ f_c x = fderiv ℝ f x := fderiv_sub_const _
      have hg_diff : DifferentiableAt ℝ (fun y => s / 2 * f_c y) x := hfc_diff.const_mul (s / 2)
      calc (fderiv ℝ (fun x => exp (s / 2 * f_c x)) x) (Pi.single i 1)
          = (exp (s / 2 * f_c x) • fderiv ℝ (fun y => s / 2 * f_c y) x) (Pi.single i 1) := by
            rw [fderiv_exp hg_diff]
        _ = exp (s / 2 * f_c x) * (fderiv ℝ (fun y => s / 2 * f_c y) x) (Pi.single i 1) := by
            simp only [smul_apply, smul_eq_mul]
        _ = exp (s / 2 * f_c x) * ((s / 2) • fderiv ℝ f_c x) (Pi.single i 1) := by
            rw [fderiv_const_mul hfc_diff]
        _ = exp (s / 2 * f_c x) * ((s / 2) * (fderiv ℝ f x) (Pi.single i 1)) := by
            simp only [smul_apply, smul_eq_mul, h_fc_fderiv]
        _ = s / 2 * (fderiv ℝ f x) (Pi.single i 1) * exp (s / 2 * f_c x) := by ring
    conv_lhs => arg 2; ext i; rw [h_deriv i, mul_pow, mul_pow]
    simp only [mul_comm ((s/2)^2), mul_assoc]
    rw [← Finset.sum_mul]
  -- Gradient bound: |∇g|² ≤ (s/2)² σ² g²
  have h_grad_g_bound : ∀ x, GaussianLSI.gradNormSq n g x ≤ (s/2)^2 * σ^2 * (g x)^2 := by
    intro x
    rw [h_grad_g x]
    have h_bound := h_grad_bound x
    have h1 : (s/2)^2 * GaussianLSI.gradNormSq n f x ≤ (s/2)^2 * σ^2 :=
      mul_le_mul_of_nonneg_left h_bound (sq_nonneg _)
    calc (s/2)^2 * GaussianLSI.gradNormSq n f x * (g x)^2
        = ((s/2)^2 * GaussianLSI.gradNormSq n f x) * (g x)^2 := by ring
      _ ≤ ((s/2)^2 * σ^2) * (g x)^2 := mul_le_mul_of_nonneg_right h1 (sq_nonneg _)
      _ = (s/2)^2 * σ^2 * (g x)^2 := by ring
  -- The entropy bound: Ent[exp(s*f_c)] ≤ (s²σ²/2) E[exp(s*f_c)]
  conv_lhs => arg 2; ext x; rw [← h_gsq_eq x]
  conv_rhs => arg 2; arg 2; ext x; rw [← h_gsq_eq x]
  -- Bound integral of gradient
  have h_grad_int : ∫ x, GaussianLSI.gradNormSq n g x ∂μ ≤ (s/2)^2 * σ^2 * ∫ x, (g x)^2 ∂μ := by
    calc ∫ x, GaussianLSI.gradNormSq n g x ∂μ
        ≤ ∫ x, (s/2)^2 * σ^2 * (g x)^2 ∂μ := by
          apply integral_mono
          · -- Integrability of gradNormSq
            have h_dom_int : Integrable (fun x => (s/2)^2 * σ^2 * (g x)^2) μ := by
              apply Integrable.const_mul
              have h_exp_int := lipschitz_exp_integrable (le_of_lt hL_lip) hf_lip s
              convert h_exp_int using 1; ext x; exact h_gsq_eq x
            apply Integrable.mono' h_dom_int
            · apply Measurable.aestronglyMeasurable
              unfold GaussianLSI.gradNormSq
              apply Finset.measurable_sum
              intro i _
              apply Measurable.pow _ measurable_const
              unfold GaussianLSI.partialDeriv g
              have hg_C1 : ContDiff ℝ 1 g := ContDiff.exp (ContDiff.mul contDiff_const (hf_C1.sub contDiff_const))
              have hg_cont : Continuous (fderiv ℝ g) := (hg_C1.fderiv_right (m := 0) le_rfl).continuous
              have h_eval_cont : Continuous (fun L : (Fin n → ℝ) →L[ℝ] ℝ => L (Pi.single i 1)) :=
                ContinuousLinearMap.apply ℝ ℝ (Pi.single i 1) |>.continuous
              exact (h_eval_cont.comp hg_cont).measurable
            · filter_upwards with x
              simp only [Real.norm_eq_abs]
              rw [abs_of_nonneg (Finset.sum_nonneg (fun i _ => sq_nonneg _))]
              exact h_grad_g_bound x
          · apply Integrable.const_mul
            have h_exp_int := lipschitz_exp_integrable (le_of_lt hL_lip) hf_lip s
            convert h_exp_int using 1; ext x; exact h_gsq_eq x
          · intro x; exact h_grad_g_bound x
      _ = (s/2)^2 * σ^2 * ∫ x, (g x)^2 ∂μ := by rw [integral_const_mul]
  -- Prepare LSI hypotheses for g
  have hg_C1 : ContDiff ℝ 1 g := ContDiff.exp (ContDiff.mul contDiff_const (hf_C1.sub contDiff_const))
  have hg_diff : Differentiable ℝ g := hg_C1.differentiable (by norm_num)
  have hg_grad_cont : ∀ i, Continuous (fun x => GaussianLSI.partialDeriv i g x) := by
    intro i
    unfold GaussianLSI.partialDeriv
    have hg_cont : Continuous (fderiv ℝ g) := (hg_C1.fderiv_right (m := 0) le_rfl).continuous
    exact (ContinuousLinearMap.apply ℝ ℝ (Pi.single i 1) |>.continuous).comp hg_cont
  have h_gsq_int : Integrable (fun x => (g x)^2) μ := by
    have h_exp_int := lipschitz_exp_integrable (le_of_lt hL_lip) hf_lip s
    convert h_exp_int using 1; ext x; exact h_gsq_eq x
  have hg_memLp : MemLp g 2 μ := (MeasureTheory.memLp_two_iff_integrable_sq hg_diff.continuous.aestronglyMeasurable).mpr h_gsq_int
  have hg_partial_memLp : ∀ i, MemLp (fun x => GaussianLSI.partialDeriv i g x) 2 μ := by
    intro i
    have h_bound : ∀ x, (GaussianLSI.partialDeriv i g x)^2 ≤ (s/2)^2 * σ^2 * (g x)^2 := fun x => by
      have h1 : (GaussianLSI.partialDeriv i g x)^2 ≤ GaussianLSI.gradNormSq n g x := by
        unfold GaussianLSI.gradNormSq
        exact Finset.single_le_sum (f := fun j => (GaussianLSI.partialDeriv j g x)^2)
          (fun j _ => sq_nonneg _) (Finset.mem_univ i)
      calc (GaussianLSI.partialDeriv i g x)^2
          ≤ GaussianLSI.gradNormSq n g x := h1
        _ ≤ (s/2)^2 * σ^2 * (g x)^2 := h_grad_g_bound x
    have h_partial_int : Integrable (fun x => (GaussianLSI.partialDeriv i g x)^2) μ := by
      apply Integrable.mono' (h_gsq_int.const_mul _)
      · exact (hg_grad_cont i).measurable.pow_const 2 |>.aestronglyMeasurable
      · filter_upwards with x
        simp only [Real.norm_eq_abs]
        rw [abs_of_nonneg (sq_nonneg _)]
        exact h_bound x
    exact (MeasureTheory.memLp_two_iff_integrable_sq (hg_grad_cont i).aestronglyMeasurable).mpr h_partial_int
  have hg_W12 : GaussianLSI.MemW12GaussianPi n g μ := ⟨hg_memLp, hg_partial_memLp⟩
  -- Log-integrability: g² * log(g²) = exp(s * f_c) * (s * f_c)
  have hg_log_int : Integrable (fun x => (g x)^2 * log ((g x)^2)) μ := by
    have h_eq : ∀ x, (g x)^2 * log ((g x)^2) = exp (s * f_c x) * (s * f_c x) := by
      intro x; rw [h_gsq_eq x]; congr 1; rw [log_exp]
    simp_rw [h_eq]
    have h_eq' : (fun x => exp (s * f_c x) * (s * f_c x)) = fun x => s * (f_c x * exp (s * f_c x)) := by
      ext x; ring
    rw [h_eq']
    apply Integrable.const_mul
    have hfc_cont : Continuous f_c := hf_C1.continuous.sub continuous_const
    have h_abs_le_exp : ∀ y : ℝ, |y| ≤ exp |y| := fun y => by linarith [Real.add_one_le_exp |y|]
    have h_fc_bound : ∀ x, |f_c x * exp (s * f_c x)| ≤
        exp ((1 + s) * f_c x) + exp ((s - 1) * f_c x) := by
      intro x
      rw [abs_mul, abs_of_pos (exp_pos _)]
      have h := h_abs_le_exp (f_c x)
      by_cases hfc : f_c x ≥ 0
      · rw [abs_of_nonneg hfc]
        have h' : f_c x ≤ exp (f_c x) := by rw [abs_of_nonneg hfc] at h; exact h
        calc f_c x * exp (s * f_c x) ≤ exp (f_c x) * exp (s * f_c x) := by
              apply mul_le_mul_of_nonneg_right h' (exp_pos _).le
          _ = exp ((1 + s) * f_c x) := by rw [← exp_add]; ring_nf
          _ ≤ exp ((1 + s) * f_c x) + exp ((s - 1) * f_c x) := by linarith [exp_pos ((s - 1) * f_c x)]
      · push Not at hfc; rw [abs_of_neg hfc]
        have h' : -f_c x ≤ exp (-f_c x) := by rw [abs_of_neg hfc] at h; exact h
        calc (-f_c x) * exp (s * f_c x) ≤ exp (-f_c x) * exp (s * f_c x) := by
              apply mul_le_mul_of_nonneg_right h' (exp_pos _).le
          _ = exp ((s - 1) * f_c x) := by rw [← exp_add]; ring_nf
          _ ≤ exp ((1 + s) * f_c x) + exp ((s - 1) * f_c x) := by linarith [exp_pos ((1 + s) * f_c x)]
    have h_dom_int : Integrable (fun x => exp ((1 + s) * f_c x) + exp ((s - 1) * f_c x)) μ :=
      (lipschitz_exp_integrable (le_of_lt hL_lip) hf_lip (1 + s)).add
        (lipschitz_exp_integrable (le_of_lt hL_lip) hf_lip (s - 1))
    apply Integrable.mono' h_dom_int
    · exact hfc_cont.aestronglyMeasurable.mul (continuous_exp.comp (continuous_const.mul hfc_cont)).aestronglyMeasurable
    · filter_upwards with x; rw [Real.norm_eq_abs]; exact h_fc_bound x
  -- Apply LSI and chain inequalities
  have h_lsi := GaussianLSI.gaussian_logSobolev_W12_pi hg_W12 hg_diff hg_grad_cont hg_log_int
  calc LogSobolev.entropy μ (fun x => (g x)^2)
      ≤ 2 * ∫ x, GaussianLSI.gradNormSq n g x ∂μ := h_lsi
    _ ≤ 2 * ((s/2)^2 * σ^2 * ∫ x, (g x)^2 ∂μ) := by
        apply mul_le_mul_of_nonneg_left h_grad_int (by norm_num : (0 : ℝ) ≤ 2)
    _ = s^2 * σ^2 / 2 * ∫ x, (g x)^2 ∂μ := by ring

set_option maxHeartbeats 600000 in
/-- CGF bound via Herbst: cgf(f-Ef,t) ≤ t²σ²/2 for C¹ f with |∇f|²≤σ². -/
theorem cgf_bound {f : (Fin n → ℝ) → ℝ} {σ L_lip : ℝ}
    (hσ : 0 < σ)
    (hf_lip : LipschitzWith (Real.toNNReal L_lip) f)
    (hL_lip : 0 < L_lip)
    (hf_C1 : ContDiff ℝ 1 f)
    (h_grad_bound : ∀ x, GaussianLSI.gradNormSq n f x ≤ σ^2)
    (hf_int : Integrable f (stdGaussianPi n))
    (t : ℝ) :
    cgf (fun x => f x - ∫ y, f y ∂(stdGaussianPi n))
        (stdGaussianPi n) t ≤ t^2 * σ^2 / 2 := by
  let μ := stdGaussianPi n
  let f_c := fun x => f x - ∫ y, f y ∂μ
  let g := fun x => exp (t / 2 * f_c x)
  have h_grad_g : ∀ x, GaussianLSI.gradNormSq n g x = (t/2)^2 * GaussianLSI.gradNormSq n f x * (g x)^2 := by
    intro x
    unfold GaussianLSI.gradNormSq GaussianLSI.partialDeriv g
    have h_deriv : ∀ i, fderiv ℝ (fun x => exp (t / 2 * f_c x)) x (Pi.single i 1) =
        (t / 2) * fderiv ℝ f x (Pi.single i 1) * exp (t / 2 * f_c x) := by
      intro i
      have hf_diff : DifferentiableAt ℝ f x := hf_C1.differentiable (by norm_num) x
      have hfc_diff : DifferentiableAt ℝ f_c x := by
        apply DifferentiableAt.sub hf_diff
        exact differentiableAt_const _
      have h_fc_fderiv : fderiv ℝ f_c x = fderiv ℝ f x := fderiv_sub_const _
      have hg_diff : DifferentiableAt ℝ (fun y => t / 2 * f_c y) x := hfc_diff.const_mul (t / 2)
      calc (fderiv ℝ (fun x => exp (t / 2 * f_c x)) x) (Pi.single i 1)
          = (exp (t / 2 * f_c x) • fderiv ℝ (fun y => t / 2 * f_c y) x) (Pi.single i 1) := by
            rw [fderiv_exp hg_diff]
        _ = exp (t / 2 * f_c x) * (fderiv ℝ (fun y => t / 2 * f_c y) x) (Pi.single i 1) := by
            simp only [smul_apply, smul_eq_mul]
        _ = exp (t / 2 * f_c x) * ((t / 2) • fderiv ℝ f_c x) (Pi.single i 1) := by
            rw [fderiv_const_mul hfc_diff]
        _ = exp (t / 2 * f_c x) * ((t / 2) * (fderiv ℝ f_c x) (Pi.single i 1)) := by
            simp only [smul_apply, smul_eq_mul]
        _ = exp (t / 2 * f_c x) * ((t / 2) * (fderiv ℝ f x) (Pi.single i 1)) := by
            rw [h_fc_fderiv]
        _ = t / 2 * (fderiv ℝ f x) (Pi.single i 1) * exp (t / 2 * f_c x) := by ring
    conv_lhs =>
      arg 2
      ext i
      rw [h_deriv i, mul_pow, mul_pow]
    simp only [mul_comm ((t/2)^2), mul_assoc]
    rw [← Finset.sum_mul]
  have h_grad_g_bound : ∀ x, GaussianLSI.gradNormSq n g x ≤ (t/2)^2 * σ^2 * (g x)^2 := by
    intro x
    rw [h_grad_g x]
    have h_bound := h_grad_bound x
    have h_gsq_pos : 0 ≤ (g x)^2 := sq_nonneg _
    have h_t2_pos : 0 ≤ (t/2)^2 := sq_nonneg _
    have h1 : (t/2)^2 * GaussianLSI.gradNormSq n f x ≤ (t/2)^2 * σ^2 :=
      mul_le_mul_of_nonneg_left h_bound h_t2_pos
    calc (t/2)^2 * GaussianLSI.gradNormSq n f x * (g x)^2
        = ((t/2)^2 * GaussianLSI.gradNormSq n f x) * (g x)^2 := by ring
      _ ≤ ((t/2)^2 * σ^2) * (g x)^2 := mul_le_mul_of_nonneg_right h1 h_gsq_pos
      _ = (t/2)^2 * σ^2 * (g x)^2 := by ring
  have h_entropy_bound : LogSobolev.entropy μ (fun x => exp (t * f_c x)) ≤
      (t^2 * σ^2 / 2) * ∫ x, exp (t * f_c x) ∂μ :=
    entropy_bound_exp_scaled hσ hf_lip hL_lip hf_C1 h_grad_bound t
  haveI : IsProbabilityMeasure μ := stdGaussianPi_isProbabilityMeasure
  have h_exp_int : Integrable (fun x => exp (t * f_c x)) μ :=
    lipschitz_exp_integrable (le_of_lt hL_lip) hf_lip t
  have h_mgf_def : mgf f_c μ t = ∫ x, exp (t * f_c x) ∂μ := by simp only [mgf, mul_comm t]
  have hf_c_centered : ∫ x, f_c x ∂μ = 0 := by
    have hμ : μ = stdGaussianPi n := rfl
    simp only [f_c, hμ]
    rw [integral_sub hf_int (integrable_const _)]
    simp only [integral_const, MeasureTheory.probReal_univ, smul_eq_mul, one_mul, sub_self]
  have h_mgf_pos : 0 < mgf f_c μ t := by rw [h_mgf_def]; apply integral_exp_pos h_exp_int
  have h_entropy_def : LogSobolev.entropy μ (fun x => exp (t * f_c x)) =
      ∫ x, exp (t * f_c x) * (t * f_c x) ∂μ - (mgf f_c μ t) * log (mgf f_c μ t) := by
    simp only [LogSobolev.entropy, h_mgf_def, log_exp, mul_comm t]
  have h_bound := h_entropy_bound
  rw [h_entropy_def] at h_bound
  by_cases ht : t = 0
  · subst ht; simp [cgf_zero]
  unfold cgf
  rw [h_mgf_def]
  have h_intExpSet : integrableExpSet f_c μ = Set.univ := by
    ext s
    simp only [integrableExpSet, Set.mem_setOf_eq, Set.mem_univ, iff_true]
    exact lipschitz_exp_integrable (le_of_lt hL_lip) hf_lip s

  have h_in_interior : ∀ s, s ∈ interior (integrableExpSet f_c μ) := by
    intro s; rw [h_intExpSet, interior_univ]; exact Set.mem_univ s
  have h_deriv_cgf : ∀ s, deriv (cgf f_c μ) s =
      (∫ x, f_c x * exp (s * f_c x) ∂μ) / (mgf f_c μ s) := by
    intro s; convert deriv_cgf (h_in_interior s) using 2
  have h_herbst_identity : ∀ s, LogSobolev.entropy μ (fun x => exp (s * f_c x)) =
      (mgf f_c μ s) * (s * deriv (cgf f_c μ) s - cgf f_c μ s) := by
    intro s
    simp only [LogSobolev.entropy]
    have h_mgf_s : mgf f_c μ s = ∫ x, exp (s * f_c x) ∂μ := by simp [mgf, mul_comm s]
    have h_cgf_s : cgf f_c μ s = log (mgf f_c μ s) := by simp [cgf]
    rw [h_deriv_cgf s, h_mgf_s, h_cgf_s]
    have h_mgf_pos_s : 0 < ∫ x, exp (s * f_c x) ∂μ := by
      apply integral_exp_pos
      exact lipschitz_exp_integrable (le_of_lt hL_lip) hf_lip s
    field_simp
    have h_log_eq : ∀ x, exp (s * f_c x) * log (exp (s * f_c x)) = exp (s * f_c x) * (s * f_c x) := by
      intro x; rw [log_exp]
    simp_rw [h_log_eq]
    rw [← h_mgf_s]
    congr 1
    have h_eq : ∫ x, exp (s * f_c x) * (s * f_c x) ∂μ = s * ∫ x, f_c x * exp (s * f_c x) ∂μ := by
      conv_lhs => arg 2; ext x; rw [show exp (s * f_c x) * (s * f_c x) = s * (f_c x * exp (s * f_c x)) by ring]
      exact MeasureTheory.integral_const_mul s _
    exact h_eq
  have h_diff_ineq : ∀ s, s * deriv (cgf f_c μ) s - cgf f_c μ s ≤ s^2 * σ^2 / 2 := by
    intro s
    have h_ent_s : LogSobolev.entropy μ (fun x => exp (s * f_c x)) ≤
        s^2 * σ^2 / 2 * ∫ x, exp (s * f_c x) ∂μ :=
      entropy_bound_exp_scaled hσ hf_lip hL_lip hf_C1 h_grad_bound s
    rw [h_herbst_identity s] at h_ent_s
    have h_mgf_eq_s : mgf f_c μ s = ∫ x, exp (s * f_c x) ∂μ := by simp [mgf, mul_comm s]
    rw [h_mgf_eq_s] at h_ent_s
    have h_int_pos : 0 < ∫ x, exp (s * f_c x) ∂μ := by
      apply integral_exp_pos
      exact lipschitz_exp_integrable (le_of_lt hL_lip) hf_lip s
    calc s * deriv (cgf f_c μ) s - cgf f_c μ s
        = (∫ x, exp (s * f_c x) ∂μ)⁻¹ * ((∫ x, exp (s * f_c x) ∂μ) *
            (s * deriv (cgf f_c μ) s - cgf f_c μ s)) := by
          field_simp
      _ ≤ (∫ x, exp (s * f_c x) ∂μ)⁻¹ * (s^2 * σ^2 / 2 * ∫ x, exp (s * f_c x) ∂μ) := by
          apply mul_le_mul_of_nonneg_left h_ent_s
          exact inv_nonneg.mpr (le_of_lt h_int_pos)
      _ = s^2 * σ^2 / 2 := by field_simp
  have h_cgf_zero : cgf f_c μ 0 = 0 := cgf_zero
  have h_deriv_cgf_zero : deriv (cgf f_c μ) 0 = 0 := by
    have h := deriv_cgf_zero (h_in_interior 0)
    rw [h, hf_c_centered]; simp only [MeasureTheory.probReal_univ, div_one]
  calc log (∫ x, exp (t * f_c x) ∂μ) = cgf f_c μ t := by simp [cgf, mgf, mul_comm t]
    _ ≤ t^2 * σ^2 / 2 := by
        by_cases ht_zero : t = 0
        · simp [ht_zero, h_cgf_zero]
        · by_cases ht_pos : 0 < t
          · have h_phi_bound : cgf f_c μ t - t^2 * σ^2 / 2 ≤ 0 := by
              let φ := fun s => cgf f_c μ s - s^2 * σ^2 / 2
              have hφ_zero : φ 0 = 0 := by simp [φ, h_cgf_zero]
              have h_phi_diff : ∀ x, DifferentiableAt ℝ φ x := by
                intro x
                apply DifferentiableAt.sub
                · exact (analyticAt_cgf (h_in_interior x)).differentiableAt
                · apply DifferentiableAt.div_const
                  apply DifferentiableAt.mul_const
                  exact differentiableAt_pow 2
              have h_deriv_phi_zero : deriv φ 0 = 0 := by
                have h1 : deriv φ 0 = deriv (cgf f_c μ) 0 - deriv (fun s => s^2 * σ^2 / 2) 0 := by
                  apply deriv_sub
                  · exact (analyticAt_cgf (h_in_interior 0)).differentiableAt
                  · apply DifferentiableAt.div_const
                    apply DifferentiableAt.mul_const
                    exact differentiableAt_pow 2
                rw [h1, h_deriv_cgf_zero]
                simp only [deriv_div_const]
                norm_num
              have h_diff_ineq_phi : ∀ s, s ≠ 0 → s * deriv φ s - φ s ≤ 0 := by
                intro s hs
                simp only [φ]
                have hcgf_diff : DifferentiableAt ℝ (cgf f_c μ) s :=
                  (analyticAt_cgf (h_in_interior s)).differentiableAt
                have hpoly_diff : DifferentiableAt ℝ (fun s => s^2 * σ^2 / 2) s := by
                  apply DifferentiableAt.div_const
                  apply DifferentiableAt.mul_const
                  exact differentiableAt_pow 2
                have h1 : deriv (fun s => cgf f_c μ s - s^2 * σ^2 / 2) s =
                    deriv (cgf f_c μ) s - s * σ^2 := by
                  have h_fn_eq : (fun s => cgf f_c μ s - s^2 * σ^2 / 2) =
                      (cgf f_c μ) - (fun s => s^2 * σ^2 / 2) := rfl
                  rw [h_fn_eq, deriv_sub hcgf_diff hpoly_diff, deriv_div_const, deriv_mul_const]
                  · have hd : deriv (fun s => s^2) s = 2 * s := by
                      have := (hasDerivAt_pow 2 s).deriv
                      simp only [Nat.cast_ofNat] at this
                      convert this using 2
                      ring
                    rw [hd]
                    ring
                  · exact differentiableAt_pow 2
                rw [h1]
                have h2 := h_diff_ineq s
                linarith
              have h_ratio_decreasing : φ t / t ≤ 0 :=
                ratio_bound_gronwall φ t ht_pos hφ_zero h_deriv_phi_zero h_phi_diff h_diff_ineq_phi
              have h_final : φ t ≤ 0 := by
                rw [div_nonpos_iff] at h_ratio_decreasing
                rcases h_ratio_decreasing with ⟨_, ht_le⟩ | ⟨h_phi_le, _⟩
                · linarith
                · exact h_phi_le
              exact h_final
            linarith
          · push Not at ht_pos
            have ht_neg : t < 0 := lt_of_le_of_ne ht_pos ht_zero
            have hmt_pos : 0 < -t := neg_pos.mpr ht_neg
            let ψ := fun s => cgf f_c μ (-s) - s^2 * σ^2 / 2
            have hψ_zero : ψ 0 = 0 := by simp [ψ, h_cgf_zero]
            have hψ_deriv_zero : deriv ψ 0 = 0 := by
              simp only [ψ]
              have hcgf_diff : DifferentiableAt ℝ (cgf f_c μ) 0 :=
                (analyticAt_cgf (h_in_interior 0)).differentiableAt
              have h_comp_diff : DifferentiableAt ℝ (fun s => cgf f_c μ (-s)) 0 := by
                apply DifferentiableAt.comp (g := cgf f_c μ) (f := fun x => -x)
                · simp only [neg_zero]; exact hcgf_diff
                · exact differentiableAt_neg_iff.mpr differentiableAt_id
              have hda1 : HasDerivAt (fun s => cgf f_c μ (-s)) 0 0 := by
                have h := h_comp_diff.hasDerivAt
                rw [deriv_comp_neg (cgf f_c μ) 0] at h
                simp only [neg_zero, h_deriv_cgf_zero, neg_zero] at h
                exact h
              have hda2 : HasDerivAt (fun s : ℝ => s^2 * σ^2 / 2) 0 0 := by
                have hp : HasDerivAt (fun s : ℝ => s^2) 0 (0 : ℝ) := by
                  simpa using (hasDerivAt_pow 2 (0 : ℝ))
                have h := (hp.mul_const (σ^2)).div_const 2
                simpa using h
              have := hda1.sub hda2
              simp only [sub_zero] at this
              exact this.deriv
            -- ψ satisfies the differential inequality s*ψ'(s) - ψ(s) ≤ 0 for s ≠ 0
            have h_diff_ineq_ψ : ∀ s, s ≠ 0 → s * deriv ψ s - ψ s ≤ 0 := by
              intro s hs
              simp only [ψ]
              have hcgf_diff : DifferentiableAt ℝ (cgf f_c μ) (-s) :=
                (analyticAt_cgf (h_in_interior (-s))).differentiableAt
              have h_comp_diff : DifferentiableAt ℝ (fun s => cgf f_c μ (-s)) s := by
                apply DifferentiableAt.comp (g := cgf f_c μ) (f := fun x => -x)
                · exact hcgf_diff
                · exact differentiableAt_neg_iff.mpr differentiableAt_id
              have h1 : deriv (fun s => cgf f_c μ (-s) - s^2 * σ^2 / 2) s =
                  -deriv (cgf f_c μ) (-s) - s * σ^2 := by
                have hda : HasDerivAt (fun s => cgf f_c μ (-s) - s^2 * σ^2 / 2)
                    (-deriv (cgf f_c μ) (-s) - s * σ^2) s := by
                  have hda1 : HasDerivAt (fun s => cgf f_c μ (-s)) (-deriv (cgf f_c μ) (-s)) s := by
                    have := h_comp_diff.hasDerivAt
                    rw [deriv_comp_neg (cgf f_c μ) s] at this
                    exact this
                  have hda2 : HasDerivAt (fun s => s^2 * σ^2 / 2) (s * σ^2) s := by
                    have hp : HasDerivAt (fun s => s^2) (2 * s) s := by
                      simpa [pow_one, Nat.cast_ofNat] using (hasDerivAt_pow 2 s)
                    have h := (hp.mul_const (σ^2)).div_const 2
                    simpa [mul_assoc] using h
                  exact hda1.sub hda2
                exact hda.deriv
              rw [h1]
              -- Using the original differential inequality at -s
              have h2 := h_diff_ineq (-s)
              -- h2: (-s) * deriv (cgf f_c μ) (-s) - cgf f_c μ (-s) ≤ (-s)² σ² / 2 = s² σ² / 2
              calc s * (-deriv (cgf f_c μ) (-s) - s * σ^2) - (cgf f_c μ (-s) - s^2 * σ^2 / 2)
                  = ((-s) * deriv (cgf f_c μ) (-s) - cgf f_c μ (-s)) - s^2 * σ^2 / 2 := by ring
                _ ≤ (s^2 * σ^2 / 2) - s^2 * σ^2 / 2 := by linarith [h2]
                _ = 0 := by ring
            -- ψ is differentiable
            have hψ_diff : ∀ x, DifferentiableAt ℝ ψ x := by
              intro x
              simp only [ψ]
              apply DifferentiableAt.sub
              · apply DifferentiableAt.comp (g := cgf f_c μ) (f := fun s => -s)
                · exact (analyticAt_cgf (h_in_interior (-x))).differentiableAt
                · exact differentiableAt_neg_iff.mpr differentiableAt_id
              · exact ((differentiableAt_pow 2).mul_const (σ^2)).div_const 2
            -- Apply the Grönwall helper lemma to ψ on (0, -t)
            have h_ratio_decreasing_ψ : ψ (-t) / (-t) ≤ 0 :=
              ratio_bound_gronwall ψ (-t) hmt_pos hψ_zero hψ_deriv_zero hψ_diff h_diff_ineq_ψ
            -- Convert ratio bound to direct bound
            -- ψ (-t) = cgf f_c μ t - t² σ²/2 and ψ (-t) / (-t) ≤ 0 with -t > 0
            have h_phi_t : cgf f_c μ t - t^2 * σ^2 / 2 ≤ 0 := by
              have h := mul_nonpos_of_nonneg_of_nonpos (le_of_lt hmt_pos) h_ratio_decreasing_ψ
              have h' : (-t) * (ψ (-t) / (-t)) = ψ (-t) := mul_div_cancel₀ _ (ne_of_gt hmt_pos)
              have h_eq1 : cgf f_c μ t = cgf f_c μ (-(-t)) := by rw [neg_neg]
              have h_eq2 : t^2 = (-t)^2 := by rw [neg_sq]
              rw [h_eq1, h_eq2]
              show ψ (-t) ≤ 0
              rw [← h']
              exact h
            exact sub_nonpos.mp h_phi_t

/-! ### Lipschitz Integrability on EuclideanSpace -/

/-- Lipschitz functions on EuclideanSpace are integrable under the Gaussian measure. -/
lemma lipschitzE_integrable {f : EuclideanSpace ℝ (Fin n) → ℝ} {L : ℝ} (hL : 0 ≤ L)
    (hf_lip : LipschitzWith (Real.toNNReal L) f) :
    Integrable f (stdGaussianE n) := by
  unfold stdGaussianE
  let e := (EuclideanSpace.equiv (Fin n) ℝ).symm.toHomeomorph.toMeasurableEquiv
  have h_eq : Measure.map (⇑(EuclideanSpace.equiv (Fin n) ℝ).symm) (stdGaussianPi n) =
              Measure.map e (stdGaussianPi n) := rfl
  rw [h_eq, integrable_map_equiv e]
  let f' := f ∘ (EuclideanSpace.equiv (Fin n) ℝ).symm
  have hf'_lip : LipschitzWith (Real.toNNReal (Real.sqrt n * L)) f' := by
    have he_lip : LipschitzWith (Real.toNNReal (Real.sqrt n)) (EuclideanSpace.equiv (Fin n) ℝ).symm := by
      rw [lipschitzWith_iff_dist_le_mul]
      intro w₁ w₂
      rw [Real.coe_toNNReal _ (Real.sqrt_nonneg n)]
      have h_norm : ∀ v : Fin n → ℝ, ‖(EuclideanSpace.equiv (Fin n) ℝ).symm v‖ ≤ Real.sqrt n * ‖v‖ := by
        intro v
        rw [EuclideanSpace.norm_eq]
        have h_sum : ∑ i, ‖v i‖^2 ≤ n * ‖v‖^2 := by
          have h_each : ∀ i, ‖v i‖^2 ≤ ‖v‖^2 := fun i => by
            have := norm_le_pi_norm v i
            exact sq_le_sq' (by linarith [norm_nonneg (v i)]) this
          calc ∑ i, ‖v i‖^2 ≤ ∑ _ : Fin n, ‖v‖^2 := sum_le_sum (fun i _ => h_each i)
            _ = n * ‖v‖^2 := by simp
        calc Real.sqrt (∑ i, ‖v i‖^2)
            ≤ Real.sqrt (n * ‖v‖^2) := Real.sqrt_le_sqrt h_sum
          _ = Real.sqrt n * Real.sqrt (‖v‖^2) := Real.sqrt_mul (Nat.cast_nonneg n) _
          _ = Real.sqrt n * |‖v‖| := by rw [Real.sqrt_sq_eq_abs]
          _ = Real.sqrt n * ‖v‖ := by rw [abs_of_nonneg (norm_nonneg _)]
      calc dist ((EuclideanSpace.equiv (Fin n) ℝ).symm w₁) ((EuclideanSpace.equiv (Fin n) ℝ).symm w₂)
          = ‖(EuclideanSpace.equiv (Fin n) ℝ).symm w₁ - (EuclideanSpace.equiv (Fin n) ℝ).symm w₂‖ := dist_eq_norm _ _
        _ = ‖(EuclideanSpace.equiv (Fin n) ℝ).symm (w₁ - w₂)‖ := by simp [map_sub]
        _ ≤ Real.sqrt n * ‖w₁ - w₂‖ := h_norm _
        _ = Real.sqrt n * dist w₁ w₂ := by rw [dist_eq_norm]
    have h := hf_lip.comp he_lip
    convert h using 2
    rw [mul_comm, Real.toNNReal_mul hL]
  exact LipschitzConcentration.lipschitz_integrable_stdGaussianPi f'
    (Real.sqrt n * L) (by positivity) hf'_lip

/-- For L-Lipschitz f on EuclideanSpace, |∇(f ∘ e.symm)|² ≤ L².
This is needed for the mollification argument. -/
lemma lipschitz_gradNormSq_bound_E {f : EuclideanSpace ℝ (Fin n) → ℝ} {L : ℝ} (hL : 0 ≤ L)
    (hf_lip : LipschitzWith (Real.toNNReal L) f)
    (hf_C1 : ContDiff ℝ 1 f) (x : EuclideanSpace ℝ (Fin n)) :
    GaussianLSI.gradNormSq n (f ∘ (EuclideanSpace.equiv (Fin n) ℝ).symm)
      ((EuclideanSpace.equiv (Fin n) ℝ) x) ≤ L^2 := by
  let e := EuclideanSpace.equiv (Fin n) ℝ
  let f' := f ∘ e.symm
  have h_fderiv_bound : ‖fderiv ℝ f x‖ ≤ L := by
    have := norm_fderiv_le_of_lipschitz ℝ hf_lip (x₀ := x)
    simp only [Real.coe_toNNReal L hL] at this
    exact this
  let b := EuclideanSpace.basisFun (Fin n) ℝ
  have h_sum_sq : ∑ i, (fderiv ℝ f x (b i))^2 = ‖fderiv ℝ f x‖^2 := by
    haveI : CompleteSpace (EuclideanSpace ℝ (Fin n)) := inferInstance
    let Lf := fderiv ℝ f x
    let a : EuclideanSpace ℝ (Fin n) := (InnerProductSpace.toDual ℝ _).symm Lf
    have ha : ∀ v, Lf v = @inner ℝ _ _ a v := fun v => InnerProductSpace.toDual_symm_apply.symm
    have h_parseval := b.sum_sq_norm_inner_right a
    have h_parseval' : ∑ i, (@inner ℝ _ _ a (b i))^2 = ‖a‖^2 := by
      convert h_parseval using 2 with i
      rw [real_inner_comm, Real.norm_eq_abs, ← sq_abs]
    have h_norm : ‖a‖ = ‖Lf‖ := by simp only [a, LinearIsometryEquiv.norm_map]
    calc ∑ i, (Lf (b i))^2 = ∑ i, (@inner ℝ _ _ a (b i))^2 := by simp [ha]
      _ = ‖a‖^2 := h_parseval'
      _ = ‖Lf‖^2 := by rw [h_norm]
  -- ∂ᵢf'(e x) = fderiv f x (b i)
  have h_partial_eq : ∀ i, GaussianLSI.partialDeriv i f' (e x) = fderiv ℝ f x (b i) := by
    intro i
    simp only [GaussianLSI.partialDeriv, f']
    have hf_diff : DifferentiableAt ℝ f x := hf_C1.differentiable (by norm_num) x
    have he_diff : DifferentiableAt ℝ e.symm (e x) := e.symm.differentiable.differentiableAt
    have hf_diff' : DifferentiableAt ℝ f (e.symm (e x)) := by
      simp only [ContinuousLinearEquiv.symm_apply_apply]
      exact hf_diff
    have h_fderiv_esymm : fderiv ℝ (↑e.symm) (e x) = (e.symm : (Fin n → ℝ) →L[ℝ] _) :=
      ContinuousLinearEquiv.fderiv e.symm
    have h_fderiv_comp : fderiv ℝ (f ∘ e.symm) (e x) = (fderiv ℝ f x).comp (e.symm : (Fin n → ℝ) →L[ℝ] _) := by
      have := fderiv_comp (e x) hf_diff' he_diff
      rw [this, h_fderiv_esymm]
      congr 1
    rw [h_fderiv_comp]
    simp only [ContinuousLinearMap.comp_apply, ContinuousLinearEquiv.coe_coe]
    congr 1
    rw [EuclideanSpace.basisFun_apply]
    rfl
  unfold GaussianLSI.gradNormSq
  calc ∑ i : Fin n, (GaussianLSI.partialDeriv i f' (e x))^2
      = ∑ i, (fderiv ℝ f x (b i))^2 := by simp [h_partial_eq]
    _ = ‖fderiv ℝ f x‖^2 := h_sum_sq
    _ ≤ L^2 := sq_le_sq' (by linarith [norm_nonneg (fderiv ℝ f x)]) h_fderiv_bound

/-! ### Extension to General Lipschitz via Mollification -/

open Filter Topology
open scoped NNReal Topology

local notation "𝔼" => EuclideanSpace ℝ (Fin n)

/-! ### Convergence Lemmas for Mollification under Gaussian Measure -/

/-- Pointwise convergence of exponentials: exp(t * f_ε(x)) → exp(t * f(x)). -/
lemma exp_mollify_tendsto_of_lipschitz {f : 𝔼 → ℝ} {L : ℝ≥0}
    (hf : LipschitzWith L f) (t : ℝ) (x : 𝔼) :
    Tendsto (fun ε => exp (t * mollify ε f x)) (nhdsWithin 0 (Set.Ioi 0)) (nhds (exp (t * f x))) := by
  have h_moll := mollify_tendsto_of_lipschitz hf x
  exact (continuous_exp.tendsto _).comp ((continuous_const.mul continuous_id).tendsto (f x) |>.comp h_moll)

/-- Mollified Lipschitz functions are Lipschitz with the same constant. -/
lemma mollify_lipschitzWith' {f : 𝔼 → ℝ} {L : ℝ≥0} {ε : ℝ} (hε : 0 < ε)
    (hf : LipschitzWith L f) : LipschitzWith L (mollify ε f) :=
  mollify_lipschitzWith hε hf

/-- |f_ε(x)| ≤ |f 0| + L√n + L‖x‖ for ε ∈ (0, 1]. -/
lemma mollify_linear_growth {f : 𝔼 → ℝ} {L : ℝ≥0} {ε : ℝ} (hε : 0 < ε) (hε1 : ε ≤ 1)
    (hf : LipschitzWith L f) (x : 𝔼) :
    |mollify ε f x| ≤ |f 0| + L * Real.sqrt n + L * ‖x‖ := by
  have h_lip := mollify_lipschitzWith' hε hf
  have h1 : dist (mollify ε f x) (mollify ε f 0) ≤ L * dist x 0 := h_lip.dist_le_mul x 0
  simp only [dist_zero_right] at h1
  rw [dist_eq_norm] at h1
  have h2 : dist (mollify ε f 0) (f 0) ≤ L * ε * Real.sqrt n :=
    mollify_dist_le_of_lipschitz' hε hf 0
  have h3 : L * ε * Real.sqrt n ≤ L * Real.sqrt n := by
    calc L * ε * Real.sqrt n = L * Real.sqrt n * ε := by ring
      _ ≤ L * Real.sqrt n * 1 := by apply mul_le_mul_of_nonneg_left hε1; positivity
      _ = L * Real.sqrt n := by ring
  have h4 : |mollify ε f 0 - f 0| ≤ L * Real.sqrt n := by
    calc |mollify ε f 0 - f 0| = dist (mollify ε f 0) (f 0) := by rw [Real.dist_eq]
      _ ≤ L * ε * Real.sqrt n := h2
      _ ≤ L * Real.sqrt n := h3
  calc |mollify ε f x| = |mollify ε f x - mollify ε f 0 + (mollify ε f 0 - f 0) + f 0| := by ring_nf
    _ ≤ |mollify ε f x - mollify ε f 0| + |mollify ε f 0 - f 0| + |f 0| := by
        calc |mollify ε f x - mollify ε f 0 + (mollify ε f 0 - f 0) + f 0|
            ≤ |mollify ε f x - mollify ε f 0 + (mollify ε f 0 - f 0)| + |f 0| := abs_add_le _ _
          _ ≤ |mollify ε f x - mollify ε f 0| + |mollify ε f 0 - f 0| + |f 0| := by
              linarith [abs_add_le (mollify ε f x - mollify ε f 0) (mollify ε f 0 - f 0)]
    _ ≤ L * ‖x‖ + L * Real.sqrt n + |f 0| := by
        rw [Real.norm_eq_abs] at h1
        linarith [h1, h4]
    _ = |f 0| + L * Real.sqrt n + L * ‖x‖ := by ring

/-- |f 0| + L√n + L‖x‖ is integrable under Gaussian. -/
lemma linear_growth_integrable {f : 𝔼 → ℝ} {L : ℝ≥0} :
    Integrable (fun x : 𝔼 => |f 0| + L * Real.sqrt n + L * ‖x‖)
      (stdGaussianE n) := by
  have h_const : Integrable (fun _ : 𝔼 => |f 0| + L * Real.sqrt n) (stdGaussianE n) :=
    integrable_const _
  have h_norm : Integrable (fun x : 𝔼 => L * ‖x‖) (stdGaussianE n) := by
    -- ‖·‖ is 1-Lipschitz, and Lipschitz functions are integrable under stdGaussianE
    have h_lip : LipschitzWith (Real.toNNReal 1) (fun x : 𝔼 => ‖x‖) := by
      simp only [Real.toNNReal_one]
      exact lipschitzWith_one_norm
    have h_int := lipschitzE_integrable (by norm_num : (0 : ℝ) ≤ 1) h_lip
    exact h_int.const_mul L
  exact h_const.add h_norm

/-- E[f_ε(X)] → E[f(X)] via dominated convergence. -/
lemma integral_mollify_tendsto_of_lipschitz {f : 𝔼 → ℝ} {L : ℝ≥0}
    (hf : LipschitzWith L f) :
    Tendsto (fun ε => ∫ x, mollify ε f x ∂(stdGaussianE n))
      (nhdsWithin 0 (Set.Ioi 0))
      (nhds (∫ x, f x ∂(stdGaussianE n))) := by
  let bound : 𝔼 → ℝ := fun x => |f 0| + L * Real.sqrt n + L * ‖x‖
  apply tendsto_integral_filter_of_dominated_convergence bound
  · filter_upwards [self_mem_nhdsWithin] with ε hε
    have h_smooth := mollify_smooth_of_lipschitz hf hε
    exact h_smooth.continuous.aestronglyMeasurable
  · filter_upwards [Ioo_mem_nhdsGT (zero_lt_one : (0 : ℝ) < 1)] with ε hε
    filter_upwards with x
    rw [Real.norm_eq_abs]
    exact mollify_linear_growth hε.1 (le_of_lt hε.2) hf x
  · exact linear_growth_integrable
  · filter_upwards with x
    exact mollify_tendsto_of_lipschitz hf x

/-- |t * f_ε(x)| ≤ |t| * (|f 0| + L√n + L‖x‖). -/
lemma mollify_exp_bound {f : 𝔼 → ℝ} {L : ℝ≥0} {ε : ℝ} (hε : 0 < ε) (hε1 : ε ≤ 1)
    (hf : LipschitzWith L f) (t : ℝ) (x : 𝔼) :
    |t * mollify ε f x| ≤ |t| * (|f 0| + L * Real.sqrt n + L * ‖x‖) := by
  rw [abs_mul]
  apply mul_le_mul_of_nonneg_left _ (abs_nonneg t)
  exact mollify_linear_growth hε hε1 hf x

/-- exp(|t| * (|f 0| + L√n + L‖x‖)) is integrable under Gaussian. -/
lemma exp_linear_growth_integrable {f : 𝔼 → ℝ} {L : ℝ≥0} (t : ℝ) :
    Integrable (fun x : 𝔼 => exp (|t| * (|f 0| + L * Real.sqrt n + L * ‖x‖)))
      (stdGaussianE n) := by
  let a := |f 0| + L * Real.sqrt n
  let b := (L : ℝ)
  have h_rewrite : ∀ x : 𝔼, exp (|t| * (|f 0| + L * Real.sqrt n + L * ‖x‖)) =
      exp (|t| * a) * exp ((|t| * b) * ‖x‖) := by
    intro x
    rw [show |f 0| + L * Real.sqrt n + L * ‖x‖ = a + b * ‖x‖ from by ring]
    rw [mul_add, exp_add]
    ring_nf
  simp_rw [h_rewrite]
  apply Integrable.const_mul
  let c := |t| * b
  have hc_nonneg : 0 ≤ c := by positivity
  unfold stdGaussianE
  let e := EuclideanSpace.equiv (Fin n) ℝ
  let eM := e.symm.toHomeomorph.toMeasurableEquiv
  have h_map_eq : Measure.map e.symm (stdGaussianPi n) =
                  Measure.map eM (stdGaussianPi n) := rfl
  rw [h_map_eq, integrable_map_equiv eM]
  have h_norm_bound : ∀ w : Fin n → ℝ, ‖e.symm w‖ ≤ Real.sqrt n * ‖w‖ := by
    intro w
    rw [EuclideanSpace.norm_eq]
    have h_sum : ∑ i, ‖w i‖^2 ≤ n * ‖w‖^2 := by
      have h_each : ∀ i, ‖w i‖^2 ≤ ‖w‖^2 := fun i => by
        have := norm_le_pi_norm w i
        exact sq_le_sq' (by linarith [norm_nonneg (w i)]) this
      calc ∑ i, ‖w i‖^2 ≤ ∑ _ : Fin n, ‖w‖^2 := Finset.sum_le_sum (fun i _ => h_each i)
        _ = n * ‖w‖^2 := by simp
    calc Real.sqrt (∑ i, ‖w i‖^2)
        ≤ Real.sqrt (n * ‖w‖^2) := Real.sqrt_le_sqrt h_sum
      _ = Real.sqrt n * Real.sqrt (‖w‖^2) := Real.sqrt_mul (Nat.cast_nonneg n) _
      _ = Real.sqrt n * |‖w‖| := by rw [Real.sqrt_sq_eq_abs]
      _ = Real.sqrt n * ‖w‖ := by rw [abs_of_nonneg (norm_nonneg _)]
  have h_norm_sum : ∀ w : Fin n → ℝ, ‖w‖ ≤ ∑ i, |w i| := by
    intro w
    rw [pi_norm_le_iff_of_nonneg (Finset.sum_nonneg fun i _ => abs_nonneg (w i))]
    intro i
    rw [Real.norm_eq_abs]
    exact Finset.single_le_sum (fun j _ => abs_nonneg (w j)) (Finset.mem_univ i)
  let d := c * Real.sqrt n
  have h_prod_int : Integrable (fun w : Fin n → ℝ => ∏ i : Fin n, exp (d * |w i|))
      (stdGaussianPi n) := by
    unfold stdGaussianPi
    have h_factor : ∀ i : Fin n, Integrable (fun y : ℝ => exp (d * |y|)) (gaussianReal 0 1) := by
      intro i
      have h1 : Integrable (fun x => exp (d * x)) (gaussianReal 0 1) :=
        integrable_exp_mul_gaussianReal (μ := 0) (v := 1) d
      have h2 : Integrable (fun x => exp ((-d) * x)) (gaussianReal 0 1) :=
        integrable_exp_mul_gaussianReal (μ := 0) (v := 1) (-d)
      have h2' : Integrable (fun ω => exp (-d * ω)) (gaussianReal 0 1) := by
        convert h2 using 2
      exact integrable_exp_mul_abs h1 h2'
    have := Integrable.fintype_prod (f := fun _ : Fin n => fun y : ℝ => exp (d * |y|))
      (μ := fun _ => gaussianReal 0 1) h_factor
    convert this using 1
  apply h_prod_int.mono'
  · exact (continuous_exp.comp
      ((continuous_const.mul continuous_norm).comp e.symm.continuous)).aestronglyMeasurable
  · filter_upwards with w
    simp only [Function.comp_apply]
    rw [Real.norm_eq_abs, abs_of_nonneg (exp_pos _).le]
    have h1 : c * ‖eM w‖ ≤ c * (Real.sqrt n * ‖w‖) := by
      apply mul_le_mul_of_nonneg_left _ hc_nonneg
      have : eM w = e.symm w := rfl
      rw [this]
      exact h_norm_bound w
    have h2 : c * (Real.sqrt n * ‖w‖) = d * ‖w‖ := by ring
    have h3 : d * ‖w‖ ≤ d * ∑ i, |w i| := by
      apply mul_le_mul_of_nonneg_left (h_norm_sum w)
      positivity
    have h4 : exp (d * ∑ i, |w i|) = ∏ i : Fin n, exp (d * |w i|) := by
      rw [mul_sum, exp_sum]
    have h_eq : |t| * b * ‖eM w‖ = c * ‖eM w‖ := by ring
    calc exp (|t| * b * ‖eM w‖) = exp (c * ‖eM w‖) := by rw [h_eq]
      _ ≤ exp (c * (Real.sqrt n * ‖w‖)) := exp_le_exp.mpr h1
      _ = exp (d * ‖w‖) := by rw [h2]
      _ ≤ exp (d * ∑ i, |w i|) := exp_le_exp.mpr h3
      _ = ∏ i : Fin n, exp (d * |w i|) := h4

/-- E[exp(t*f_ε(X))] → E[exp(t*f(X))] via dominated convergence. -/
lemma mgf_mollify_tendsto_of_lipschitz {f : 𝔼 → ℝ} {L : ℝ≥0}
    (hf : LipschitzWith L f) (t : ℝ) :
    Tendsto (fun ε => ∫ x, exp (t * mollify ε f x) ∂(stdGaussianE n))
      (nhdsWithin 0 (Set.Ioi 0))
      (nhds (∫ x, exp (t * f x) ∂(stdGaussianE n))) := by
  let bound : 𝔼 → ℝ := fun x => exp (|t| * (|f 0| + L * Real.sqrt n + L * ‖x‖))
  apply tendsto_integral_filter_of_dominated_convergence bound
  · filter_upwards [self_mem_nhdsWithin] with ε hε
    have h_smooth := mollify_smooth_of_lipschitz hf hε
    exact (continuous_exp.comp (continuous_const.mul h_smooth.continuous)).aestronglyMeasurable
  · filter_upwards [Ioo_mem_nhdsGT (zero_lt_one : (0 : ℝ) < 1)] with ε hε
    filter_upwards with x
    rw [Real.norm_eq_abs, abs_of_nonneg (exp_pos _).le]
    have h := mollify_exp_bound hε.1 (le_of_lt hε.2) hf t x
    calc exp (t * mollify ε f x) ≤ exp (|t * mollify ε f x|) := exp_le_exp_of_le (le_abs_self _)
      _ ≤ exp (|t| * (|f 0| + L * Real.sqrt n + L * ‖x‖)) := exp_le_exp.mpr h
  · exact exp_linear_growth_integrable t
  · filter_upwards with x
    exact exp_mollify_tendsto_of_lipschitz hf t x

/-- E[exp(t*(f_ε - E[f_ε]))] → E[exp(t*(f - E[f]))] as ε → 0. -/
lemma centered_mgf_mollify_tendsto_of_lipschitz {f : 𝔼 → ℝ} {L : ℝ≥0}
    (hf : LipschitzWith L f) (t : ℝ) :
    Tendsto (fun ε => ∫ x, exp (t * (mollify ε f x - ∫ y, mollify ε f y ∂(stdGaussianE n)))
        ∂(stdGaussianE n))
      (nhdsWithin 0 (Set.Ioi 0))
      (nhds (∫ x, exp (t * (f x - ∫ y, f y ∂(stdGaussianE n)))
        ∂(stdGaussianE n))) := by
  let μ := stdGaussianE n
  have h_factor : ∀ (g : 𝔼 → ℝ) (x : 𝔼),
      exp (t * (g x - ∫ y, g y ∂μ)) = exp (-t * ∫ y, g y ∂μ) * exp (t * g x) := by
    intro g x
    have h_eq : t * (g x - ∫ y, g y ∂μ) = -t * ∫ y, g y ∂μ + t * g x := by ring
    rw [h_eq, exp_add]
  have h_factor' : ∀ (g : 𝔼 → ℝ),
      (∫ x, exp (t * (g x - ∫ y, g y ∂μ)) ∂μ) =
        exp (-t * ∫ y, g y ∂μ) * (∫ x, exp (t * g x) ∂μ) := by
    intro g
    simp_rw [h_factor]
    exact MeasureTheory.integral_const_mul _ _
  have h_goal : Tendsto (fun ε => exp (-t * ∫ y, mollify ε f y ∂μ) * (∫ x, exp (t * mollify ε f x) ∂μ))
        (nhdsWithin 0 (Set.Ioi 0))
        (nhds (exp (-t * ∫ y, f y ∂μ) * (∫ x, exp (t * f x) ∂μ))) := by
    apply Tendsto.mul
    · have h_int_conv := integral_mollify_tendsto_of_lipschitz (n := n) hf
      have h_scaled : Tendsto (fun ε => -t * ∫ y, mollify ε f y ∂μ)
          (nhdsWithin 0 (Set.Ioi 0)) (nhds (-t * ∫ y, f y ∂μ)) := Tendsto.const_mul _ h_int_conv
      exact (continuous_exp.tendsto _).comp h_scaled
    · exact mgf_mollify_tendsto_of_lipschitz hf t
  simp only [μ] at h_factor' h_goal ⊢
  convert h_goal using 1
  · ext ε
    exact h_factor' (mollify ε f)
  · rw [h_factor' f]

/-- cgf(f_ε - E[f_ε], t) → cgf(f - E[f], t) as ε → 0. -/
lemma cgf_mollify_tendsto_of_lipschitz {f : 𝔼 → ℝ} {L : ℝ≥0}
    (hf : LipschitzWith L f) (t : ℝ) :
    Tendsto (fun ε => cgf (fun x => mollify ε f x -
        ∫ y, mollify ε f y ∂(stdGaussianE n))
        (stdGaussianE n) t)
      (nhdsWithin 0 (Set.Ioi 0))
      (nhds (cgf (fun x => f x - ∫ y, f y ∂(stdGaussianE n))
        (stdGaussianE n) t)) := by
  let μ := stdGaussianE n
  simp only [cgf, mgf]
  have h_mgf_conv := centered_mgf_mollify_tendsto_of_lipschitz (n := n) hf t
  haveI : IsProbabilityMeasure μ := stdGaussianE_isProbabilityMeasure
  have h_centered_int : Integrable (fun x => exp (t * (f x - ∫ y, f y ∂μ))) μ := by
    have h_eq : ∀ x, exp (t * (f x - ∫ y, f y ∂μ)) =
        exp (-t * ∫ y, f y ∂μ) * exp (t * f x) := by
      intro x
      have h : t * (f x - ∫ y, f y ∂μ) = -t * ∫ y, f y ∂μ + t * f x := by ring
      rw [h, exp_add]
    simp_rw [h_eq]
    apply Integrable.const_mul
    apply Integrable.mono' (exp_linear_growth_integrable (f := f) (L := L) t)
    · apply Continuous.aestronglyMeasurable
      exact continuous_exp.comp (continuous_const.mul hf.continuous)
    · filter_upwards with x
      rw [Real.norm_eq_abs, abs_of_nonneg (exp_pos _).le]
      have hL_pos : (0 : ℝ) ≤ L := L.coe_nonneg
      have h_bd : |f x| ≤ |f 0| + L * ‖x‖ := by
        have h_lip := hf.dist_le_mul x 0
        simp only [dist_zero_right, Real.dist_eq] at h_lip
        calc |f x| = |f x - f 0 + f 0| := by ring_nf
          _ ≤ |f x - f 0| + |f 0| := abs_add_le _ _
          _ ≤ L * ‖x‖ + |f 0| := by linarith
          _ = |f 0| + L * ‖x‖ := by ring
      calc exp (t * f x)
          ≤ exp (|t| * |f x|) := by
            apply exp_le_exp.mpr
            calc t * f x ≤ |t * f x| := le_abs_self _
              _ = |t| * |f x| := abs_mul t (f x)
        _ ≤ exp (|t| * (|f 0| + L * ‖x‖)) := by
            apply exp_le_exp.mpr
            apply mul_le_mul_of_nonneg_left h_bd (abs_nonneg _)
        _ ≤ exp (|t| * (|f 0| + L * Real.sqrt n + L * ‖x‖)) := by
            apply exp_le_exp.mpr
            apply mul_le_mul_of_nonneg_left _ (abs_nonneg _)
            linarith [mul_nonneg hL_pos (Real.sqrt_nonneg n)]
  have h_target_pos : 0 < ∫ x, exp (t * (f x - ∫ y, f y ∂μ)) ∂μ := by
    apply integral_exp_pos h_centered_int
  have h_target_ne : (∫ x, exp (t * (f x - ∫ y, f y ∂μ)) ∂μ) ≠ 0 := ne_of_gt h_target_pos
  simp only [μ] at h_mgf_conv h_target_ne ⊢
  exact (Real.continuousAt_log h_target_ne).tendsto.comp h_mgf_conv

/-! ### CGF Bounds for Lipschitz Functions -/

/-- CGF bound for C¹ Lipschitz on EuclideanSpace: cgf ≤ t²L²/2. -/
lemma lipschitz_cgf_bound_C1 {f : 𝔼 → ℝ} {L : ℝ}
    (hn : 0 < n) (hL : 0 < L)
    (hf_lip : LipschitzWith (Real.toNNReal L) f)
    (hf_C1 : ContDiff ℝ 1 f) (t : ℝ) :
    let μ := stdGaussianE n
    cgf (fun x => f x - ∫ y, f y ∂μ) μ t ≤ t^2 * L^2 / 2 := by
  let μ := stdGaussianE n
  let e := EuclideanSpace.equiv (Fin n) ℝ
  let f' := f ∘ e.symm
  let μ' := stdGaussianPi n
  have h_int_eq : ∫ y, f y ∂μ = ∫ w, f' w ∂μ' := by
    rw [integral_stdGaussianE_eq]; rfl
  have hf'_C1 : ContDiff ℝ 1 f' := hf_C1.comp e.symm.contDiff
  have hf'_lip : LipschitzWith (Real.toNNReal (Real.sqrt n * L)) f' := by
    have he_lip : LipschitzWith (Real.toNNReal (Real.sqrt n)) e.symm := by
      rw [lipschitzWith_iff_dist_le_mul]
      intro w₁ w₂
      rw [Real.coe_toNNReal _ (Real.sqrt_nonneg n)]
      have h_norm : ∀ v : Fin n → ℝ, ‖e.symm v‖ ≤ Real.sqrt n * ‖v‖ := by
        intro v
        rw [EuclideanSpace.norm_eq]
        have h_sum : ∑ i, ‖v i‖^2 ≤ n * ‖v‖^2 := by
          have h_each : ∀ i, ‖v i‖^2 ≤ ‖v‖^2 := fun i => by
            have := norm_le_pi_norm v i
            exact sq_le_sq' (by linarith [norm_nonneg (v i)]) this
          calc ∑ i, ‖v i‖^2 ≤ ∑ _ : Fin n, ‖v‖^2 := sum_le_sum (fun i _ => h_each i)
            _ = n * ‖v‖^2 := by simp
        calc Real.sqrt (∑ i, ‖v i‖^2)
            ≤ Real.sqrt (n * ‖v‖^2) := Real.sqrt_le_sqrt h_sum
          _ = Real.sqrt n * Real.sqrt (‖v‖^2) := Real.sqrt_mul (Nat.cast_nonneg n) _
          _ = Real.sqrt n * |‖v‖| := by rw [Real.sqrt_sq_eq_abs]
          _ = Real.sqrt n * ‖v‖ := by rw [abs_of_nonneg (norm_nonneg _)]
      calc dist (e.symm w₁) (e.symm w₂) = ‖e.symm w₁ - e.symm w₂‖ := dist_eq_norm _ _
        _ = ‖e.symm (w₁ - w₂)‖ := by simp [map_sub]
        _ ≤ Real.sqrt n * ‖w₁ - w₂‖ := h_norm _
        _ = Real.sqrt n * dist w₁ w₂ := by rw [dist_eq_norm]
    have h := hf_lip.comp he_lip
    convert h using 2
    rw [mul_comm, Real.toNNReal_mul (le_of_lt hL)]
  have hf'_int : Integrable f' μ' :=
    LipschitzConcentration.lipschitz_integrable_stdGaussianPi f' (Real.sqrt n * L)
      (by positivity) hf'_lip
  have h_gradbound : ∀ w, GaussianLSI.gradNormSq n f' w ≤ L^2 := fun w =>
    lipschitz_gradNormSq_bound_E (le_of_lt hL) hf_lip hf_C1 (e.symm w)
  have h_cgf' : cgf (fun w => f' w - ∫ z, f' z ∂μ') μ' t ≤ t^2 * L^2 / 2 :=
    cgf_bound hL hf'_lip (mul_pos (Real.sqrt_pos.mpr (Nat.cast_pos.mpr hn)) hL) hf'_C1 h_gradbound hf'_int t
  simp only [cgf, mgf]
  have h_eq : ∫ x, exp (t * (f x - ∫ y, f y ∂μ)) ∂μ =
              ∫ w, exp (t * (f' w - ∫ z, f' z ∂μ')) ∂μ' := by
    rw [h_int_eq]
    rw [integral_stdGaussianE_eq]
    simp only [f', e, μ', Function.comp_apply]
  rw [h_eq]
  exact h_cgf'

/-- CGF bound for Lipschitz via mollification: cgf(f-E[f], t) ≤ t²L²/2. -/
theorem lipschitz_cgf_bound {f : 𝔼 → ℝ} {L : ℝ≥0}
    (hn : 0 < n) (hL : 0 < L) (hf : LipschitzWith L f) (t : ℝ) :
    let μ := stdGaussianE n
    cgf (fun x => f x - ∫ y, f y ∂μ) μ t ≤ t^2 * (L : ℝ)^2 / 2 := by
  let μ := stdGaussianE n
  have h_cgf_conv := cgf_mollify_tendsto_of_lipschitz (n := n) hf t
  have h_bound_mollify : ∀ᶠ ε in nhdsWithin (0 : ℝ) (Set.Ioi 0),
      cgf (fun x => mollify ε f x - ∫ y, mollify ε f y ∂μ) μ t ≤ t^2 * (L : ℝ)^2 / 2 := by
    filter_upwards [Ioo_mem_nhdsGT (zero_lt_one : (0 : ℝ) < 1)] with ε hε
    have hε_pos : 0 < ε := hε.1
    have h_smooth := mollify_smooth_of_lipschitz hf hε_pos
    have h_C1 : ContDiff ℝ 1 (mollify ε f) := contDiff_infty.mp h_smooth 1
    have h_lip := mollify_lipschitzWith hε_pos hf
    have h_lip' : LipschitzWith (Real.toNNReal (L : ℝ)) (mollify ε f) := by
      convert h_lip using 2
      exact Real.toNNReal_coe
    exact lipschitz_cgf_bound_C1 hn (by positivity : 0 < (L : ℝ)) h_lip' h_C1 t
  exact le_of_tendsto h_cgf_conv h_bound_mollify

/-- exp(t*(f-E[f])) is integrable for Lipschitz f. -/
lemma lipschitz_exp_centered_integrable_E {f : 𝔼 → ℝ} {L : ℝ≥0}
    (hf : LipschitzWith L f) (t : ℝ) :
    let μ := stdGaussianE n
    Integrable (fun x => exp (t * (f x - ∫ y, f y ∂μ))) μ := by
  let μ := stdGaussianE n
  let e := EuclideanSpace.equiv (Fin n) ℝ
  let f' := f ∘ e.symm
  let μ' := stdGaussianPi n
  have hf'_lip : LipschitzWith (Real.toNNReal (Real.sqrt n * (L : ℝ))) f' := by
    have he_lip : LipschitzWith (Real.toNNReal (Real.sqrt n)) e.symm := by
      rw [lipschitzWith_iff_dist_le_mul]
      intro w₁ w₂
      rw [Real.coe_toNNReal _ (Real.sqrt_nonneg n)]
      have h_norm : ∀ v : Fin n → ℝ, ‖e.symm v‖ ≤ Real.sqrt n * ‖v‖ := by
        intro v
        rw [EuclideanSpace.norm_eq]
        have h_sum : ∑ i, ‖v i‖^2 ≤ n * ‖v‖^2 := by
          have h_each : ∀ i, ‖v i‖^2 ≤ ‖v‖^2 := fun i => by
            have := norm_le_pi_norm v i
            exact sq_le_sq' (by linarith [norm_nonneg (v i)]) this
          calc ∑ i, ‖v i‖^2 ≤ ∑ _ : Fin n, ‖v‖^2 := Finset.sum_le_sum (fun i _ => h_each i)
            _ = n * ‖v‖^2 := by simp
        calc Real.sqrt (∑ i, ‖v i‖^2)
            ≤ Real.sqrt (n * ‖v‖^2) := Real.sqrt_le_sqrt h_sum
          _ = Real.sqrt n * Real.sqrt (‖v‖^2) := Real.sqrt_mul (Nat.cast_nonneg n) _
          _ = Real.sqrt n * |‖v‖| := by rw [Real.sqrt_sq_eq_abs]
          _ = Real.sqrt n * ‖v‖ := by rw [abs_of_nonneg (norm_nonneg _)]
      calc dist (e.symm w₁) (e.symm w₂) = ‖e.symm w₁ - e.symm w₂‖ := dist_eq_norm _ _
        _ = ‖e.symm (w₁ - w₂)‖ := by simp [map_sub]
        _ ≤ Real.sqrt n * ‖w₁ - w₂‖ := h_norm _
        _ = Real.sqrt n * dist w₁ w₂ := by rw [dist_eq_norm]
    have hf_lip' : LipschitzWith (Real.toNNReal (L : ℝ)) f := by
      simp only [Real.toNNReal_coe]; exact hf
    have h := hf_lip'.comp he_lip
    simp only [mul_comm (Real.sqrt n) (L : ℝ)] at h ⊢
    convert h using 2
    rw [Real.toNNReal_mul (NNReal.coe_nonneg L), Real.toNNReal_coe]
  have h_int' := lipschitz_exp_integrable (by positivity : 0 ≤ Real.sqrt n * (L : ℝ)) hf'_lip t
  have h_mean : ∫ y, f y ∂μ = ∫ w, f' w ∂μ' := by
    rw [integral_stdGaussianE_eq]; rfl
  unfold stdGaussianE
  have h_aesm : AEStronglyMeasurable
      (fun x => exp (t * (f x - ∫ y, f y ∂Measure.map e.symm μ'))) (Measure.map e.symm μ') := by
    apply Continuous.aestronglyMeasurable
    apply continuous_exp.comp
    apply Continuous.mul continuous_const
    apply Continuous.sub hf.continuous continuous_const
  have h_aem : AEMeasurable e.symm μ' := e.symm.continuous.measurable.aemeasurable
  rw [integrable_map_measure h_aesm h_aem]
  convert h_int' using 1
  ext w
  simp only [Function.comp_apply, f']
  simp only [μ, μ', f', stdGaussianE, e] at h_mean ⊢
  rw [h_mean]
  rfl

/-! ### Main Concentration Theorems -/

/-- P(f(X) - E[f(X)] ≥ t) ≤ exp(-t²/(2L²)) for L-Lipschitz f. -/
theorem gaussian_lipschitz_concentration_one_sided {f : 𝔼 → ℝ} {L : ℝ≥0}
    (hn : 0 < n) (hL : 0 < L) (hf : LipschitzWith L f) (t : ℝ) (ht : 0 < t) :
    let μ := stdGaussianE n
    (μ {x | t ≤ f x - ∫ y, f y ∂μ}).toReal ≤ exp (-t^2 / (2 * (L : ℝ)^2)) := by
  let μ := stdGaussianE n
  haveI : IsProbabilityMeasure μ := stdGaussianE_isProbabilityMeasure
  have h_cgf : ∀ s, cgf (fun x => f x - ∫ y, f y ∂μ) μ s ≤ s^2 * (L : ℝ)^2 / 2 :=
    fun s => lipschitz_cgf_bound hn hL hf s
  have h_int : ∀ s, Integrable (fun x => exp (s * (f x - ∫ y, f y ∂μ))) μ :=
    fun s => lipschitz_exp_centered_integrable_E hf s
  exact chernoff_bound_subGaussian (by positivity : 0 < (L : ℝ)) ht h_cgf h_int

/-- P(|f(X) - E[f(X)]| ≥ t) ≤ 2exp(-t²/(2L²)) for L-Lipschitz f. -/
theorem gaussian_lipschitz_concentration {f : 𝔼 → ℝ} {L : ℝ≥0}
    (hn : 0 < n) (hL : 0 < L) (hf : LipschitzWith L f) (t : ℝ) (ht : 0 < t) :
    let μ := stdGaussianE n
    (μ {x | t ≤ |f x - ∫ y, f y ∂μ|}).toReal ≤ 2 * exp (-t^2 / (2 * (L : ℝ)^2)) := by
  let μ := stdGaussianE n
  haveI : IsProbabilityMeasure μ := stdGaussianE_isProbabilityMeasure
  have h_neg_lip : LipschitzWith L (fun x => -f x) := hf.neg
  have h_union : {x : 𝔼 | t ≤ |f x - ∫ y, f y ∂μ|} ⊆
      {x | t ≤ f x - ∫ y, f y ∂μ} ∪ {x | t ≤ -(f x - ∫ y, f y ∂μ)} := by
    intro x hx
    simp only [Set.mem_setOf_eq] at hx ⊢
    simp only [Set.mem_union, Set.mem_setOf_eq]
    rcases le_or_gt 0 (f x - ∫ y, f y ∂μ) with hpos | hneg
    · left; rwa [abs_of_nonneg hpos] at hx
    · right; rw [abs_of_neg hneg] at hx; linarith
  have h1 : (μ {x | t ≤ |f x - ∫ y, f y ∂μ|}).toReal ≤
      (μ ({x | t ≤ f x - ∫ y, f y ∂μ} ∪ {x | t ≤ -(f x - ∫ y, f y ∂μ)})).toReal :=
    ENNReal.toReal_mono (measure_ne_top _ _) (measure_mono h_union)
  have h2 : (μ ({x | t ≤ f x - ∫ y, f y ∂μ} ∪ {x | t ≤ -(f x - ∫ y, f y ∂μ)})).toReal ≤
      (μ {x | t ≤ f x - ∫ y, f y ∂μ}).toReal + (μ {x | t ≤ -(f x - ∫ y, f y ∂μ)}).toReal := by
    let A := {x : 𝔼 | t ≤ f x - ∫ y, f y ∂μ}
    let B := {x : 𝔼 | t ≤ -(f x - ∫ y, f y ∂μ)}
    rw [← ENNReal.toReal_add (measure_ne_top μ A) (measure_ne_top μ B)]
    apply ENNReal.toReal_mono
    · exact ENNReal.add_ne_top.mpr ⟨measure_ne_top μ A, measure_ne_top μ B⟩
    · exact measure_union_le A B
  have h_pos := gaussian_lipschitz_concentration_one_sided hn hL hf t ht
  have h_neg_mean : ∫ y, -f y ∂μ = -∫ y, f y ∂μ := integral_neg f
  have h_neg : (μ {x | t ≤ -(f x - ∫ y, f y ∂μ)}).toReal ≤ exp (-t^2 / (2 * (L : ℝ)^2)) := by
    have h := gaussian_lipschitz_concentration_one_sided hn hL h_neg_lip t ht
    have h_eq : {x : 𝔼 | t ≤ -f x - ∫ y, -f y ∂μ} = {x | t ≤ -(f x - ∫ y, f y ∂μ)} := by
      ext x; simp only [Set.mem_setOf_eq, h_neg_mean]; ring_nf
    simp only [μ] at h h_eq ⊢
    convert h using 1
    exact congrArg (fun s => (stdGaussianE n s).toReal) h_eq.symm
  have h3 : (μ {x | t ≤ f x - ∫ y, f y ∂μ}).toReal + (μ {x | t ≤ -(f x - ∫ y, f y ∂μ)}).toReal ≤
      exp (-t^2 / (2 * (L : ℝ)^2)) + exp (-t^2 / (2 * (L : ℝ)^2)) := add_le_add h_pos h_neg
  calc (μ {x | t ≤ |f x - ∫ y, f y ∂μ|}).toReal
      ≤ (μ ({x | t ≤ f x - ∫ y, f y ∂μ} ∪ {x | t ≤ -(f x - ∫ y, f y ∂μ)})).toReal := h1
    _ ≤ (μ {x | t ≤ f x - ∫ y, f y ∂μ}).toReal + (μ {x | t ≤ -(f x - ∫ y, f y ∂μ)}).toReal := h2
    _ ≤ exp (-t^2 / (2 * (L : ℝ)^2)) + exp (-t^2 / (2 * (L : ℝ)^2)) := h3
    _ = 2 * exp (-t^2 / (2 * (L : ℝ)^2)) := by ring

end GaussianLipConcen
