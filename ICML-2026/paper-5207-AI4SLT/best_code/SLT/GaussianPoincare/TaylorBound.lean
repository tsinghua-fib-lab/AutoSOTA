/-
Copyright (c) 2026 Yuanhe Zhang. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Yuanhe Zhang, Jason D. Lee, Fanghui Liu
-/
import SLT.GaussianPoincare.EfronSteinApp
import Mathlib.Analysis.Calculus.Taylor
import Mathlib.Analysis.Calculus.IteratedDeriv.Defs

/-!
# Taylor Expansion Analysis for Rademacher Sum Variance Bounds

This file formalizes the Taylor expansion analysis that refines the Efron-Stein
variance bound for smooth compactly supported functions composed with normalized
Rademacher sums.

## Main results

* `aPlusProd_sub_aMinusProd`: The difference `a⁺ - a⁻ = 2/√n` is constant.
* `sq_deviation_sum`: `(a⁺ - Sₙ)² + (a⁻ - Sₙ)² = 4/n` for Rademacher variables.
* `deriv2_bounded_of_compactlySupported`: For `f ∈ C_c²(ℝ)`, `sup|f''| < ∞`.
* `taylor_diff_bound`: The Taylor expansion bound on `|f(a⁺) - f(a⁻)|`.
* `variance_rademacherSum_taylor_bound`: The refined variance bound using Taylor's theorem, `Var(f(Sₙ)) ≤ E[f'(Sₙ)²] + (4K/√n)E[|f'(Sₙ)|] + 4K²/n`.

-/

noncomputable section

open MeasureTheory ProbabilityTheory Real Filter Set Function
open scoped ENNReal Topology

namespace TaylorBound

-- Use definitions from EfronSteinApp
open EfronSteinApp RademacherApprox

variable (n : ℕ) [NeZero n]

/-! ### Properties of a⁺ and a⁻ -/

/-- The difference `a⁺ - a⁻ = 2/√n` is constant (independent of x). -/
theorem aPlusProd_sub_aMinusProd (x : RademacherSpace n) :
    aPlusProd n 0 x - aMinusProd n 0 x = 2 / Real.sqrt n := by
  unfold aPlusProd aMinusProd
  ring

/-- For a Rademacher variable (taking values ±1), the sum of squared deviations
    `(a⁺ - Sₙ)² + (a⁻ - Sₙ)² = 4/n`. -/
theorem sq_deviation_sum (x : RademacherSpace n) (hx : x 0 = 1 ∨ x 0 = -1) :
    (aPlusProd n 0 x - rademacherSumProd n x)^2 +
    (aMinusProd n 0 x - rademacherSumProd n x)^2 = 4 / n := by
  unfold aPlusProd aMinusProd
  simp only [add_sub_cancel_left, sub_sub_cancel_left]
  have hn : (n : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr (NeZero.ne n)
  have hn_pos : (0 : ℝ) < n := Nat.cast_pos.mpr (Nat.pos_of_ne_zero (NeZero.ne n))
  have hsqrt : Real.sqrt n ≠ 0 := Real.sqrt_ne_zero'.mpr hn_pos
  have hsqrt_sq : (Real.sqrt n)^2 = n := Real.sq_sqrt (le_of_lt hn_pos)
  rcases hx with h | h
  · -- x 0 = 1
    simp only [h, sub_self, zero_div, ne_eq, OfNat.ofNat_ne_zero, not_false_eq_true, zero_pow]
    field_simp
    rw [hsqrt_sq]
    ring
  · -- x 0 = -1
    simp only [h, sub_neg_eq_add, one_add_one_eq_two, add_neg_cancel, zero_div]
    field_simp
    rw [hsqrt_sq]
    ring

/-- Almost everywhere version: the sum of squared deviations equals 4/n. -/
theorem sq_deviation_sum_ae :
    ∀ᵐ x ∂(rademacherProductMeasure n),
    (aPlusProd n 0 x - rademacherSumProd n x)^2 +
    (aMinusProd n 0 x - rademacherSumProd n x)^2 = 4 / n := by
  filter_upwards [coord_values_ae n 0] with x hx
  exact sq_deviation_sum n x hx

/-! ### Boundedness of Second Derivative -/

/-- The second derivative of a C² compactly supported function is continuous. -/
lemma deriv2_continuous {f : ℝ → ℝ} (hf : CompactlySupportedSmooth f) :
    Continuous (deriv (deriv f)) := by
  have h2 : ContDiff ℝ 2 f := hf.1
  have hc0 : ContDiff ℝ 0 (deriv^[2] f) := h2.iterate_deriv' 0 2
  simp only [Function.iterate_succ, Function.iterate_zero, Function.comp_apply] at hc0
  exact hc0.continuous

/-- The second derivative of a C² compactly supported function has compact support. -/
lemma deriv2_hasCompactSupport {f : ℝ → ℝ} (hf : CompactlySupportedSmooth f) :
    HasCompactSupport (deriv (deriv f)) := by
  have hsupp := hf.2
  apply HasCompactSupport.of_support_subset_isCompact hsupp.isCompact
  intro x hx
  by_contra h
  have hout : x ∉ tsupport f := h
  -- x is in the complement of the closed set tsupport f
  have hopen : IsOpen (tsupport f)ᶜ := (isClosed_tsupport f).isOpen_compl
  have hmem : (tsupport f)ᶜ ∈ 𝓝 x := hopen.mem_nhds hout
  -- On the complement, f = 0 because tsupport f ⊇ support f
  have hzero : f =ᶠ[𝓝 x] 0 := by
    filter_upwards [hmem] with y hy
    simp only [Pi.zero_apply]
    have hnotsupp : y ∉ tsupport f := hy
    rw [tsupport] at hnotsupp
    simp only [mem_closure_iff_nhds, not_forall, not_nonempty_iff_eq_empty] at hnotsupp
    obtain ⟨U, hUmem, hUempty⟩ := hnotsupp
    have hy_in_U : y ∈ U := mem_of_mem_nhds hUmem
    have hynot : y ∉ support f := by
      intro hyin
      have : y ∈ U ∩ support f := ⟨hy_in_U, hyin⟩
      rw [hUempty] at this
      exact this
    exact notMem_support.mp hynot
  have hderiv1 : deriv f =ᶠ[𝓝 x] 0 := by
    have h1 := hzero.deriv
    have hzero_eq : (0 : ℝ → ℝ) = fun _ => 0 := rfl
    rw [hzero_eq] at h1
    simp only [deriv_const'] at h1
    exact h1
  have hderiv2 : deriv (deriv f) =ᶠ[𝓝 x] 0 := by
    have h2 := hderiv1.deriv
    have hzero_eq : (0 : ℝ → ℝ) = fun _ => 0 := rfl
    rw [hzero_eq] at h2
    simp only [deriv_const'] at h2
    exact h2
  have heq : deriv (deriv f) x = 0 := hderiv2.self_of_nhds
  rw [mem_support] at hx
  exact hx heq

/-- For `f ∈ C_c²(ℝ)`, the supremum `K := sup|f''|` is finite.
This follows from the Extreme Value Theorem: f'' is continuous and compactly supported,
hence bounded. -/
theorem deriv2_bounded_of_compactlySupported {f : ℝ → ℝ} (hf : CompactlySupportedSmooth f) :
    ∃ K : ℝ, 0 ≤ K ∧ ∀ x : ℝ, |deriv (deriv f) x| ≤ K := by
  have hcont : Continuous (deriv (deriv f)) := deriv2_continuous hf
  have hsupp : HasCompactSupport (deriv (deriv f)) := deriv2_hasCompactSupport hf
  obtain ⟨C, hC⟩ := hsupp.exists_bound_of_continuous hcont
  use max 0 C
  constructor
  · exact le_max_left 0 C
  · intro x
    calc |deriv (deriv f) x| = ‖deriv (deriv f) x‖ := (Real.norm_eq_abs _).symm
      _ ≤ C := hC x
      _ ≤ max 0 C := le_max_right 0 C

/-- The bound K on the second derivative. -/
def secondDerivBound (f : ℝ → ℝ) (hf : CompactlySupportedSmooth f) : ℝ :=
  (deriv2_bounded_of_compactlySupported hf).choose

lemma secondDerivBound_nonneg {f : ℝ → ℝ} (hf : CompactlySupportedSmooth f) :
    0 ≤ secondDerivBound f hf :=
  (deriv2_bounded_of_compactlySupported hf).choose_spec.1

lemma secondDerivBound_spec {f : ℝ → ℝ} (hf : CompactlySupportedSmooth f) (x : ℝ) :
    |deriv (deriv f) x| ≤ secondDerivBound f hf :=
  (deriv2_bounded_of_compactlySupported hf).choose_spec.2 x

/-! ### The derivative of a compactly supported C² function -/

/-- The derivative of a C² function is continuous. -/
lemma deriv_continuous_of_compactlySupported {f : ℝ → ℝ} (hf : CompactlySupportedSmooth f) :
    Continuous (deriv f) := by
  have : ContDiff ℝ 1 (deriv f) := hf.1.iterate_deriv' 1 1
  exact this.continuous

/-- The derivative of a C² compactly supported function has compact support. -/
lemma deriv_hasCompactSupport {f : ℝ → ℝ} (hf : CompactlySupportedSmooth f) :
    HasCompactSupport (deriv f) := by
  apply HasCompactSupport.of_support_subset_isCompact hf.2.isCompact
  intro x hx
  by_contra h
  have hout : x ∉ tsupport f := h
  have hopen : IsOpen (tsupport f)ᶜ := (isClosed_tsupport f).isOpen_compl
  have hmem : (tsupport f)ᶜ ∈ 𝓝 x := hopen.mem_nhds hout
  have hzero : f =ᶠ[𝓝 x] 0 := by
    filter_upwards [hmem] with y hy
    simp only [Pi.zero_apply]
    have hnotsupp : y ∉ tsupport f := hy
    rw [tsupport] at hnotsupp
    simp only [mem_closure_iff_nhds, not_forall, not_nonempty_iff_eq_empty] at hnotsupp
    obtain ⟨U, hUmem, hUempty⟩ := hnotsupp
    have hy_in_U : y ∈ U := mem_of_mem_nhds hUmem
    have hynot : y ∉ support f := by
      intro hyin
      have : y ∈ U ∩ support f := ⟨hy_in_U, hyin⟩
      rw [hUempty] at this
      exact this
    exact notMem_support.mp hynot
  have hderiv1 : deriv f =ᶠ[𝓝 x] 0 := by
    have h1 := hzero.deriv
    have hzero_eq : (0 : ℝ → ℝ) = fun _ => 0 := rfl
    rw [hzero_eq] at h1
    simp only [deriv_const'] at h1
    exact h1
  have heq : deriv f x = 0 := hderiv1.self_of_nhds
  rw [mem_support] at hx
  exact hx heq

/-- The derivative of a C² compactly supported function is bounded. -/
theorem deriv_bounded_of_compactlySupported {f : ℝ → ℝ} (hf : CompactlySupportedSmooth f) :
    ∃ C : ℝ, 0 ≤ C ∧ ∀ x : ℝ, |deriv f x| ≤ C := by
  have hcont := deriv_continuous_of_compactlySupported hf
  have hsupp := deriv_hasCompactSupport hf
  obtain ⟨C, hC⟩ := hsupp.exists_bound_of_continuous hcont
  use max 0 C, le_max_left 0 C
  intro x
  calc |deriv f x| = ‖deriv f x‖ := (Real.norm_eq_abs _).symm
    _ ≤ C := hC x
    _ ≤ max 0 C := le_max_right 0 C

/-- The bound on the first derivative. -/
def firstDerivBound (f : ℝ → ℝ) (hf : CompactlySupportedSmooth f) : ℝ :=
  (deriv_bounded_of_compactlySupported hf).choose

lemma firstDerivBound_nonneg {f : ℝ → ℝ} (hf : CompactlySupportedSmooth f) :
    0 ≤ firstDerivBound f hf :=
  (deriv_bounded_of_compactlySupported hf).choose_spec.1

lemma firstDerivBound_spec {f : ℝ → ℝ} (hf : CompactlySupportedSmooth f) (x : ℝ) :
    |deriv f x| ≤ firstDerivBound f hf :=
  (deriv_bounded_of_compactlySupported hf).choose_spec.2 x

/-! ### Taylor's Theorem and Difference Bound -/

/-- iteratedDeriv 2 f = deriv (deriv f) -/
lemma iteratedDeriv_two (f : ℝ → ℝ) : iteratedDeriv 2 f = deriv (deriv f) := by
  rw [iteratedDeriv_eq_iterate]
  rfl

/-- A C² function is differentiable. -/
lemma differentiable_of_contDiff2 {f : ℝ → ℝ} (hf : ContDiff ℝ 2 f) :
    Differentiable ℝ f :=
  hf.differentiable two_ne_zero

/-- The derivative of a C² function is differentiable. -/
lemma deriv_differentiable_of_contDiff2 {f : ℝ → ℝ} (hf : ContDiff ℝ 2 f) :
    Differentiable ℝ (deriv f) := by
  have : ContDiff ℝ 1 (deriv f) := hf.iterate_deriv' 1 1
  exact this.differentiable one_ne_zero

/-- Taylor's theorem with Lagrange remainder for order 1 (giving second derivative term).
    For a C² function and a < b:
    f(b) = f(a) + f'(a)(b-a) + (1/2)f''(ξ)(b-a)² for some ξ ∈ (a, b). -/
theorem taylor_order_one {f : ℝ → ℝ} (hf : ContDiff ℝ 2 f) {a b : ℝ} (hab : a < b) :
    ∃ ξ ∈ Ioo a b,
    f b = f a + deriv f a * (b - a) + (1/2) * deriv (deriv f) ξ * (b - a)^2 := by
  -- Use taylor_mean_remainder_lagrange_iteratedDeriv with n = 1
  have hf' : ContDiffOn ℝ (1 + 1) f (uIcc a b) := by
    rw [uIcc_of_lt hab]
    norm_num
    exact hf.contDiffOn
  obtain ⟨ξ, hξ_mem, hξ_eq⟩ := taylor_mean_remainder_lagrange_iteratedDeriv hab.ne hf'
  use ξ
  constructor
  · simpa [uIoo_of_lt hab] using hξ_mem
  -- taylorWithinEval f 1 (uIcc a b) a b = f(a) + f'(a)(b-a)
  have htaylor : taylorWithinEval f 1 (uIcc a b) a b = f a + deriv f a * (b - a) := by
    rw [uIcc_of_lt hab]
    rw [taylorWithinEval_succ, taylor_within_zero_eval]
    simp only [Nat.factorial_zero, Nat.cast_one]
    rw [iteratedDerivWithin_one]
    have hdiff : DifferentiableAt ℝ f a := (differentiable_of_contDiff2 hf).differentiableAt
    have huniq : UniqueDiffWithinAt ℝ (Icc a b) a :=
      uniqueDiffOn_Icc hab a (left_mem_Icc.mpr (le_of_lt hab))
    rw [hdiff.derivWithin huniq]
    simp only [smul_eq_mul]
    ring
  rw [htaylor] at hξ_eq
  -- The remainder term: iteratedDeriv 2 f = deriv (deriv f)
  have hremainder : iteratedDeriv 2 f ξ = deriv (deriv f) ξ := congrFun (iteratedDeriv_two f) ξ
  rw [hremainder] at hξ_eq
  -- Now simplify
  have hfact : (2 : ℕ).factorial = 2 := rfl
  simp only [hfact, Nat.cast_ofNat] at hξ_eq
  linarith [hξ_eq]

/-- Taylor's theorem for b < a: there exists ξ in (b, a) such that
    f(b) = f(a) + f'(a)(b-a) + (1/2)f''(ξ)(b-a)². -/
theorem taylor_order_one_neg {f : ℝ → ℝ} (hf : ContDiff ℝ 2 f) {a b : ℝ} (hba : b < a) :
    ∃ ξ ∈ Ioo b a,
    f b = f a + deriv f a * (b - a) + (1/2) * deriv (deriv f) ξ * (b - a)^2 := by
  -- Define g(t) = f(a - t) and apply taylor_order_one to g
  let g : ℝ → ℝ := fun t => f (a - t)
  have hg : ContDiff ℝ 2 g := hf.comp (contDiff_const.sub contDiff_id)
  have hab : (0 : ℝ) < a - b := sub_pos.mpr hba
  obtain ⟨ξ', hξ'_mem, hξ'_eq⟩ := taylor_order_one hg hab
  use a - ξ'
  constructor
  · -- Show a - ξ' ∈ Ioo b a
    simp only [Set.mem_Ioo] at hξ'_mem ⊢
    constructor <;> linarith [hξ'_mem.1, hξ'_mem.2]
  · -- Show the Taylor formula
    have hg0 : g 0 = f a := by simp [g]
    have hgab : g (a - b) = f b := by simp [g]
    -- Compute g'(0) = -f'(a)
    have hg' : ∀ t, deriv g t = -deriv f (a - t) := by
      intro t
      have hd : DifferentiableAt ℝ f (a - t) := (differentiable_of_contDiff2 hf).differentiableAt
      have hinner : DifferentiableAt ℝ (fun s => a - s) t :=
        (differentiableAt_const a).sub differentiableAt_id
      have hcomp := deriv_comp t hd hinner
      -- deriv (fun s => a - s) t = -(deriv id t) = -1
      have hinner_deriv : deriv (fun s => a - s) t = -1 := by
        have h1 := deriv_const_sub (f := id) a (x := t)
        simp only [id_eq, deriv_id] at h1
        exact h1
      simp only [Function.comp_def] at hcomp
      rw [hcomp, hinner_deriv]; ring
    have hg'0 : deriv g 0 = -deriv f a := by rw [hg']; simp
    -- Compute g''(ξ') = f''(a - ξ')
    have hg'' : ∀ t, deriv (deriv g) t = deriv (deriv f) (a - t) := by
      intro t
      have heq : deriv g = fun s => -deriv f (a - s) := funext hg'
      rw [heq]
      have hd : DifferentiableAt ℝ (deriv f) (a - t) :=
        (deriv_differentiable_of_contDiff2 hf).differentiableAt
      have hinner : DifferentiableAt ℝ (fun s => a - s) t :=
        (differentiableAt_const a).sub differentiableAt_id
      -- deriv (fun s => -deriv f (a - s)) t = -(deriv (fun s => deriv f (a - s)) t)
      have hcomp := deriv_comp t hd hinner
      have hinner_deriv : deriv (fun s => a - s) t = -1 := by
        have h1 := deriv_const_sub (f := id) a (x := t)
        simp only [id_eq, deriv_id] at h1
        exact h1
      simp only [Function.comp_def] at hcomp
      have hneg : deriv (fun s => -deriv f (a - s)) t = -deriv (fun s => deriv f (a - s)) t := by
        exact deriv.neg
      rw [hneg, hcomp, hinner_deriv]
      ring
    have hg''ξ : deriv (deriv g) ξ' = deriv (deriv f) (a - ξ') := hg'' ξ'
    -- Now substitute into hξ'_eq
    have hsq : (a - b)^2 = (b - a)^2 := by ring
    calc f b = g (a - b) := hgab.symm
      _ = g 0 + deriv g 0 * ((a - b) - 0) + (1/2) * deriv (deriv g) ξ' * ((a - b) - 0)^2 := hξ'_eq
      _ = f a + (-deriv f a) * (a - b) + (1/2) * deriv (deriv f) (a - ξ') * (a - b)^2 := by
          rw [hg0, hg'0, hg''ξ]; ring_nf
      _ = f a + deriv f a * (b - a) + (1/2) * deriv (deriv f) (a - ξ') * (b - a)^2 := by
          rw [hsq]; ring

/-- Taylor's theorem for any two distinct points (using mean value bound). -/
theorem taylor_mean_value_bound {f : ℝ → ℝ} (hf : ContDiff ℝ 2 f) (K : ℝ)
    (hK : ∀ x, |deriv (deriv f) x| ≤ K) (a b : ℝ) :
    |f b - f a - deriv f a * (b - a)| ≤ K / 2 * (b - a)^2 := by
  by_cases hab : a = b
  · simp [hab]
  · rcases Ne.lt_or_gt hab with h | h
    · -- a < b: use taylor_order_one
      obtain ⟨ξ, hξ_mem, hξ_eq⟩ := taylor_order_one hf h
      calc |f b - f a - deriv f a * (b - a)|
          = |(1/2) * deriv (deriv f) ξ * (b - a)^2| := by rw [hξ_eq]; ring_nf
        _ = |1/2| * |deriv (deriv f) ξ| * |(b - a)^2| := by rw [abs_mul, abs_mul]
        _ = 1/2 * |deriv (deriv f) ξ| * (b - a)^2 := by
            rw [abs_of_pos (by norm_num : (1:ℝ)/2 > 0), abs_sq]
        _ ≤ 1/2 * K * (b - a)^2 := by
            have hK' := hK ξ
            nlinarith [hK', sq_nonneg (b - a)]
        _ = K / 2 * (b - a)^2 := by ring
    · -- b < a: use taylor_order_one_neg
      obtain ⟨ξ, hξ_mem, hξ_eq⟩ := taylor_order_one_neg hf h
      calc |f b - f a - deriv f a * (b - a)|
          = |(1/2) * deriv (deriv f) ξ * (b - a)^2| := by rw [hξ_eq]; ring_nf
        _ = |1/2| * |deriv (deriv f) ξ| * |(b - a)^2| := by rw [abs_mul, abs_mul]
        _ = 1/2 * |deriv (deriv f) ξ| * (b - a)^2 := by
            rw [abs_of_pos (by norm_num : (1:ℝ)/2 > 0), abs_sq]
        _ ≤ 1/2 * K * (b - a)^2 := by
            have hK' := hK ξ
            nlinarith [hK', sq_nonneg (b - a)]
        _ = K / 2 * (b - a)^2 := by ring

/-! ### The Key Bound -/

/-- The key bound: |f(a⁺) - f(a⁻)| ≤ (2/√n)|f'(Sₙ)| + 4K/n

This is a simplified version using direct estimates.
-/
theorem taylor_diff_abs_bound {f : ℝ → ℝ} (hf : CompactlySupportedSmooth f)
    (x : RademacherSpace n) (hx : x 0 = 1 ∨ x 0 = -1) :
    |f (aPlusProd n 0 x) - f (aMinusProd n 0 x)| ≤
    2 / Real.sqrt n * |deriv f (rademacherSumProd n x)| +
    4 * secondDerivBound f hf / n := by
  let S := rademacherSumProd n x
  let aplus := aPlusProd n 0 x
  let aminus := aMinusProd n 0 x
  let K := secondDerivBound f hf
  have hn_pos : (0 : ℝ) < n := Nat.cast_pos.mpr (Nat.pos_of_ne_zero (NeZero.ne n))
  have hsqrt_pos : 0 < Real.sqrt n := Real.sqrt_pos.mpr hn_pos
  have hsqrt_sq : (Real.sqrt n)^2 = n := Real.sq_sqrt (le_of_lt hn_pos)

  -- The difference a⁺ - a⁻ = 2/√n is constant
  have hdiff : aplus - aminus = 2 / Real.sqrt n := aPlusProd_sub_aMinusProd n x

  -- Use Mean Value Theorem: f(aplus) - f(aminus) = f'(ξ)(aplus - aminus) for some ξ
  have hdiff_f := differentiable_of_contDiff2 hf.1
  by_cases heq : aplus = aminus
  · -- aplus = aminus case: contradiction, since aplus - aminus = 2/√n > 0
    exfalso
    have hpos : (2 : ℝ) / Real.sqrt n > 0 := div_pos (by norm_num) hsqrt_pos
    rw [heq, sub_self] at hdiff
    linarith
  · -- aplus ≠ aminus case
    -- Use Mean Value Theorem: f(aplus) - f(aminus) = f'(ξ)(aplus - aminus) for some ξ
    -- For simplicity, we show aplus > aminus (since aplus - aminus = 2/√n > 0)
    have hlt_correct : aminus < aplus := by
      have hpos : (2 : ℝ) / Real.sqrt n > 0 := div_pos (by norm_num) hsqrt_pos
      linarith [hdiff]
    obtain ⟨ξ, hξ_mem, hξ_eq⟩ := exists_deriv_eq_slope f hlt_correct
      hdiff_f.continuous.continuousOn (hdiff_f.differentiableOn.mono Ioo_subset_Icc_self)
    -- f(aplus) - f(aminus) = f'(ξ) * (aplus - aminus)
    have heq_diff : f aplus - f aminus = deriv f ξ * (aplus - aminus) := by
      have hne : aplus - aminus ≠ 0 := sub_ne_zero.mpr (ne_of_gt hlt_correct)
      field_simp at hξ_eq ⊢
      linarith

    -- Now bound |f'(ξ) - f'(S)| using f''
    have hmvt_deriv : |deriv f ξ - deriv f S| ≤ K * |ξ - S| := by
      by_cases hξS : ξ = S
      · simp [hξS]
      · -- Use MVT: need to handle S < ξ or ξ < S
        rcases Ne.lt_or_gt hξS with hlt | hlt
        · -- ξ < S case
          obtain ⟨η, hη_mem, hη_eq⟩ := exists_deriv_eq_slope (deriv f) hlt
            (deriv_differentiable_of_contDiff2 hf.1).continuous.continuousOn
            ((deriv_differentiable_of_contDiff2 hf.1).differentiableOn.mono Ioo_subset_Icc_self)
          calc |deriv f ξ - deriv f S|
              = |deriv f S - deriv f ξ| := abs_sub_comm _ _
            _ = |deriv (deriv f) η * (S - ξ)| := by
                have hne : S - ξ ≠ 0 := sub_ne_zero.mpr (ne_of_gt hlt)
                rw [hη_eq]; field_simp
            _ = |deriv (deriv f) η| * |S - ξ| := abs_mul _ _
            _ ≤ K * |S - ξ| := by
                apply mul_le_mul_of_nonneg_right (secondDerivBound_spec hf η) (abs_nonneg _)
            _ = K * |ξ - S| := by rw [abs_sub_comm]
        · -- S < ξ case
          obtain ⟨η, hη_mem, hη_eq⟩ := exists_deriv_eq_slope (deriv f) hlt
            (deriv_differentiable_of_contDiff2 hf.1).continuous.continuousOn
            ((deriv_differentiable_of_contDiff2 hf.1).differentiableOn.mono Ioo_subset_Icc_self)
          calc |deriv f ξ - deriv f S|
              = |deriv (deriv f) η * (ξ - S)| := by
                  have hne : ξ - S ≠ 0 := sub_ne_zero.mpr (ne_of_gt hlt)
                  rw [hη_eq]; field_simp
            _ = |deriv (deriv f) η| * |ξ - S| := abs_mul _ _
            _ ≤ K * |ξ - S| := by
                apply mul_le_mul_of_nonneg_right (secondDerivBound_spec hf η) (abs_nonneg _)

    -- Bound |ξ - S|: ξ is between aminus and aplus, so |ξ - S| ≤ max|aplus - S|, |aminus - S|
    have hξ_bound : |ξ - S| ≤ 2 / Real.sqrt n := by
      -- ξ ∈ Ioo aminus aplus means aminus < ξ < aplus
      -- In both cases, |ξ - S| ≤ 2/√n
      rcases hx with hpos | hneg
      · -- x 0 = 1: aplus = S, aminus = S - 2/√n
        have haplus_eq : aplus = S := by
          simp only [aplus, S, aPlusProd, rademacherSumProd, hpos]
          ring
        have haminus_eq : aminus = S - 2 / Real.sqrt n := by
          simp only [aminus, S, aMinusProd, rademacherSumProd, hpos]
          ring
        -- Rewrite hξ_mem using the equalities
        rw [haplus_eq, haminus_eq] at hξ_mem
        -- Now hξ_mem : ξ ∈ Ioo (S - 2/√n) S, so S - 2/√n < ξ < S
        have h1 : S - ξ > 0 := sub_pos.mpr hξ_mem.2
        have h2 : S - ξ < 2 / Real.sqrt n := by linarith [hξ_mem.1]
        calc |ξ - S| = |-(S - ξ)| := by ring_nf
          _ = S - ξ := by rw [abs_neg, abs_of_pos h1]
          _ ≤ 2 / Real.sqrt n := le_of_lt h2
      · -- x 0 = -1: aplus = S + 2/√n, aminus = S
        have haplus_eq : aplus = S + 2 / Real.sqrt n := by
          simp only [aplus, S, aPlusProd, rademacherSumProd, hneg]
          ring
        have haminus_eq : aminus = S := by
          simp only [aminus, S, aMinusProd, rademacherSumProd, hneg]
          ring
        rw [haplus_eq, haminus_eq] at hξ_mem
        -- Now hξ_mem : ξ ∈ Ioo S (S + 2/√n), so S < ξ < S + 2/√n
        have h1 : ξ - S > 0 := sub_pos.mpr hξ_mem.1
        have h2 : ξ - S < 2 / Real.sqrt n := by linarith [hξ_mem.2]
        calc |ξ - S| = ξ - S := abs_of_pos h1
          _ ≤ 2 / Real.sqrt n := le_of_lt h2

    -- Now put it together
    have habs_add : |deriv f ξ - deriv f S + deriv f S| ≤ |deriv f ξ - deriv f S| + |deriv f S| :=
      abs_add_le _ _
    have h2pos : (0 : ℝ) ≤ 2 / Real.sqrt n := div_nonneg (by norm_num) (le_of_lt hsqrt_pos)
    have hK_nonneg := secondDerivBound_nonneg hf
    have hKξ_bound : K * |ξ - S| ≤ K * (2 / Real.sqrt n) :=
      mul_le_mul_of_nonneg_left hξ_bound hK_nonneg
    calc |f aplus - f aminus|
        = |deriv f ξ * (aplus - aminus)| := by rw [heq_diff]
      _ = |deriv f ξ| * |aplus - aminus| := abs_mul _ _
      _ = |deriv f ξ| * (2 / Real.sqrt n) := by
          rw [hdiff, abs_of_pos (div_pos (by norm_num) hsqrt_pos)]
      _ = |(deriv f ξ - deriv f S) + deriv f S| * (2 / Real.sqrt n) := by ring_nf
      _ ≤ (|deriv f ξ - deriv f S| + |deriv f S|) * (2 / Real.sqrt n) := by
          apply mul_le_mul_of_nonneg_right habs_add h2pos
      _ ≤ (K * |ξ - S| + |deriv f S|) * (2 / Real.sqrt n) := by
          apply mul_le_mul_of_nonneg_right _ h2pos
          linarith [hmvt_deriv]
      _ ≤ (K * (2 / Real.sqrt n) + |deriv f S|) * (2 / Real.sqrt n) := by
          apply mul_le_mul_of_nonneg_right _ h2pos
          linarith [hKξ_bound]
      _ = 2 / Real.sqrt n * |deriv f S| + K * (2 / Real.sqrt n) * (2 / Real.sqrt n) := by ring
      _ = 2 / Real.sqrt n * |deriv f S| + 4 * K / n := by
          congr 1
          field_simp
          rw [hsqrt_sq]
          ring

-- For now, let's state simplified versions that we can prove

/-- Simplified bound using only the derivative bound. -/
theorem taylor_diff_abs_bound_simple {f : ℝ → ℝ} (hf : CompactlySupportedSmooth f)
    (x : RademacherSpace n) :
    |f (aPlusProd n 0 x) - f (aMinusProd n 0 x)| ≤
    2 * firstDerivBound f hf / Real.sqrt n := by
  let aplus := aPlusProd n 0 x
  let aminus := aMinusProd n 0 x
  have hn_pos : (0 : ℝ) < n := Nat.cast_pos.mpr (Nat.pos_of_ne_zero (NeZero.ne n))
  have hsqrt_pos : 0 < Real.sqrt n := Real.sqrt_pos.mpr hn_pos
  have hdiff : aplus - aminus = 2 / Real.sqrt n := aPlusProd_sub_aMinusProd n x
  have hdiff_f := differentiable_of_contDiff2 hf.1

  by_cases heq : aplus = aminus
  · -- aplus = aminus case (edge case)
    simp only [aplus, aminus] at heq
    rw [heq, sub_self, abs_zero]
    apply div_nonneg
    apply mul_nonneg (by norm_num) (firstDerivBound_nonneg hf)
    exact le_of_lt hsqrt_pos
  · -- General case: use MVT
    have hlt_correct : aminus < aplus := by
      have h2pos : (0 : ℝ) < 2 / Real.sqrt n := div_pos (by norm_num) hsqrt_pos
      linarith [hdiff.symm ▸ h2pos]
    obtain ⟨ξ, hξ_mem, hξ_eq⟩ := exists_deriv_eq_slope f hlt_correct
      hdiff_f.continuous.continuousOn (hdiff_f.differentiableOn.mono Ioo_subset_Icc_self)
    have heq_diff : f aplus - f aminus = deriv f ξ * (aplus - aminus) := by
      have h := hξ_eq
      have hne : aplus - aminus ≠ 0 := sub_ne_zero.mpr heq
      field_simp at h ⊢
      linarith

    calc |f aplus - f aminus|
        = |deriv f ξ * (aplus - aminus)| := by rw [heq_diff]
      _ = |deriv f ξ| * |aplus - aminus| := abs_mul _ _
      _ = |deriv f ξ| * (2 / Real.sqrt n) := by
          rw [hdiff, abs_of_pos (div_pos (by norm_num) hsqrt_pos)]
      _ ≤ firstDerivBound f hf * (2 / Real.sqrt n) := by
          apply mul_le_mul_of_nonneg_right (firstDerivBound_spec hf ξ)
          apply div_nonneg (by norm_num) (le_of_lt hsqrt_pos)
      _ = 2 * firstDerivBound f hf / Real.sqrt n := by ring

/-! ### The Final Variance Bound (Simplified Version) -/

/-- Integrability of the derivative squared. -/
lemma integrable_deriv_sq (n : ℕ) {f : ℝ → ℝ} (hf : CompactlySupportedSmooth f) :
    Integrable (fun x : RademacherSpace n => (deriv f (rademacherSumProd n x))^2)
      (rademacherProductMeasure n) := by
  have hcont := deriv_continuous_of_compactlySupported hf
  have hsupp := deriv_hasCompactSupport hf
  obtain ⟨C, hC⟩ := hsupp.exists_bound_of_continuous hcont
  apply integrable_sq_of_bounded (C := max 0 C)
  · exact hcont.aestronglyMeasurable.comp_aemeasurable
      (measurable_rademacherSumProd n).aemeasurable
  · exact le_max_left 0 C
  · filter_upwards with x
    exact (hC _).trans (le_max_right 0 C)

/-- Integrability of the absolute value of the derivative. -/
lemma integrable_deriv_abs (n : ℕ) {f : ℝ → ℝ} (hf : CompactlySupportedSmooth f) :
    Integrable (fun x : RademacherSpace n => |deriv f (rademacherSumProd n x)|)
      (rademacherProductMeasure n) := by
  have hcont := deriv_continuous_of_compactlySupported hf
  have hsupp := deriv_hasCompactSupport hf
  obtain ⟨C, hC⟩ := hsupp.exists_bound_of_continuous hcont
  apply Integrable.abs
  apply Integrable.of_bound
    (hcont.aestronglyMeasurable.comp_aemeasurable
      (measurable_rademacherSumProd n).aemeasurable)
    C (ae_of_all _ (fun x => hC _))

/-- **Main Theorem (Simplified)**: A variance bound using the first derivative bound.

For `f ∈ C_c²(ℝ)` and C' = sup|f'|:
  Var(f(Sₙ)) ≤ C'²
-/
theorem variance_rademacherSum_simple_bound {f : ℝ → ℝ} (hf : CompactlySupportedSmooth f) :
    variance (fun x : RademacherSpace n => f (rademacherSumProd n x)) (rademacherProductMeasure n) ≤
    (firstDerivBound f hf)^2 := by
  let C' := firstDerivBound f hf
  let μ := rademacherProductMeasure n
  have hn_pos : (0 : ℝ) < n := Nat.cast_pos.mpr (Nat.pos_of_ne_zero (NeZero.ne n))
  have hsqrt_sq : (Real.sqrt n)^2 = n := Real.sq_sqrt (le_of_lt hn_pos)

  -- Start with the Efron-Stein bound
  have hES := variance_rademacherSum_efronStein n f hf

  calc variance (fun x => f (rademacherSumProd n x)) μ
      ≤ (n : ℝ) / 4 * ∫ x, (f (aPlusProd n 0 x) - f (aMinusProd n 0 x))^2 ∂μ := hES
    _ ≤ (n : ℝ) / 4 * ∫ _, (2 * C' / Real.sqrt n)^2 ∂μ := by
        apply mul_le_mul_of_nonneg_left _ (div_nonneg (Nat.cast_nonneg n) (by norm_num))
        apply integral_mono_ae
        · -- Integrability of LHS
          obtain ⟨C, hC⟩ := hf.bounded
          apply integrable_sq_of_bounded (C := 2 * C)
          · apply AEStronglyMeasurable.sub
            · exact hf.continuous.aestronglyMeasurable.comp_aemeasurable
                ((measurable_rademacherSumProd n).add
                  ((measurable_const.sub (measurable_pi_apply 0)).div measurable_const)).aemeasurable
            · exact hf.continuous.aestronglyMeasurable.comp_aemeasurable
                ((measurable_rademacherSumProd n).sub
                  ((measurable_const.add (measurable_pi_apply 0)).div measurable_const)).aemeasurable
          · linarith [hC 0, norm_nonneg (f 0)]
          · filter_upwards with x
            calc ‖f (aPlusProd n 0 x) - f (aMinusProd n 0 x)‖
                ≤ ‖f (aPlusProd n 0 x)‖ + ‖f (aMinusProd n 0 x)‖ := norm_sub_le _ _
              _ ≤ C + C := add_le_add (hC _) (hC _)
              _ = 2 * C := by ring
        · exact integrable_const _
        · filter_upwards with x
          have h := taylor_diff_abs_bound_simple n hf x
          have h_nonneg : (0 : ℝ) ≤ |f (aPlusProd n 0 x) - f (aMinusProd n 0 x)| := abs_nonneg _
          have h_bound : |f (aPlusProd n 0 x) - f (aMinusProd n 0 x)| ≤ 2 * C' / Real.sqrt n := h
          calc (f (aPlusProd n 0 x) - f (aMinusProd n 0 x))^2
              = |f (aPlusProd n 0 x) - f (aMinusProd n 0 x)|^2 := (sq_abs _).symm
            _ ≤ (2 * C' / Real.sqrt n)^2 := by
                apply sq_le_sq' _ h_bound
                linarith [h_nonneg]
    _ = (n : ℝ) / 4 * ((2 * C')^2 / n) := by
        rw [integral_const]
        have hμ_real_univ : μ.real univ = 1 := by
          rw [Measure.real_def, measure_univ, ENNReal.toReal_one]
        simp only [smul_eq_mul, hμ_real_univ, one_mul, div_pow, hsqrt_sq]
    _ = C'^2 := by field_simp; ring

/-- **Main Theorem**: The variance bound using Taylor expansion analysis.

For `f ∈ C_c²(ℝ)`, K = sup|f''|, and the normalized Rademacher sum Sₙ:
  Var(f(Sₙ)) ≤ E[f'(Sₙ)²] + (4K/√n)·E[|f'(Sₙ)|] + 4K²/n

This follows the structure from the documentation. The constants (4 instead of 2)
are slightly different due to the MVT-based approach used in the proof.
-/
theorem variance_rademacherSum_taylor_bound {f : ℝ → ℝ} (hf : CompactlySupportedSmooth f) :
    variance (fun x : RademacherSpace n => f (rademacherSumProd n x)) (rademacherProductMeasure n) ≤
    ∫ x, (deriv f (rademacherSumProd n x))^2 ∂(rademacherProductMeasure n) +
    4 * secondDerivBound f hf / Real.sqrt n *
      ∫ x, |deriv f (rademacherSumProd n x)| ∂(rademacherProductMeasure n) +
    4 * (secondDerivBound f hf)^2 / n := by
  let K := secondDerivBound f hf
  let μ := rademacherProductMeasure n
  have hn_pos : (0 : ℝ) < n := Nat.cast_pos.mpr (Nat.pos_of_ne_zero (NeZero.ne n))
  have hsqrt_pos : 0 < Real.sqrt n := Real.sqrt_pos.mpr hn_pos
  have hsqrt_sq : (Real.sqrt n)^2 = n := Real.sq_sqrt (le_of_lt hn_pos)
  have hK_nonneg := secondDerivBound_nonneg hf

  -- Start with the Efron-Stein bound
  have hES := variance_rademacherSum_efronStein n f hf

  -- Use taylor_diff_abs_bound to bound the difference squared
  -- |f(a⁺) - f(a⁻)| ≤ (2/√n)|f'(S)| + 4K/n
  -- So (f(a⁺) - f(a⁻))² ≤ ((2/√n)|f'(S)| + 4K/n)²

  -- For any x with x 0 ∈ {1, -1}, we have the bound
  have hbound : ∀ᵐ x ∂μ, (f (aPlusProd n 0 x) - f (aMinusProd n 0 x))^2 ≤
      (2 / Real.sqrt n * |deriv f (rademacherSumProd n x)| + 4 * K / n)^2 := by
    have hae := coord_values_ae n 0
    filter_upwards [hae] with x hx
    have h := taylor_diff_abs_bound n hf x hx
    have h_nonneg : (0 : ℝ) ≤ |f (aPlusProd n 0 x) - f (aMinusProd n 0 x)| := abs_nonneg _
    calc (f (aPlusProd n 0 x) - f (aMinusProd n 0 x))^2
        = |f (aPlusProd n 0 x) - f (aMinusProd n 0 x)|^2 := (sq_abs _).symm
      _ ≤ (2 / Real.sqrt n * |deriv f (rademacherSumProd n x)| + 4 * K / n)^2 := by
          apply sq_le_sq' _ h
          linarith [h_nonneg]

  -- Bound the integral
  calc variance (fun x => f (rademacherSumProd n x)) μ
      ≤ (n : ℝ) / 4 * ∫ x, (f (aPlusProd n 0 x) - f (aMinusProd n 0 x))^2 ∂μ := hES
    _ ≤ (n : ℝ) / 4 * ∫ x, (2 / Real.sqrt n * |deriv f (rademacherSumProd n x)| + 4 * K / n)^2 ∂μ := by
        apply mul_le_mul_of_nonneg_left _ (div_nonneg (Nat.cast_nonneg n) (by norm_num))
        apply integral_mono_ae
        · -- Integrability of LHS
          obtain ⟨C, hC⟩ := hf.bounded
          apply integrable_sq_of_bounded (C := 2 * C)
          · apply AEStronglyMeasurable.sub
            · exact hf.continuous.aestronglyMeasurable.comp_aemeasurable
                ((measurable_rademacherSumProd n).add
                  ((measurable_const.sub (measurable_pi_apply 0)).div measurable_const)).aemeasurable
            · exact hf.continuous.aestronglyMeasurable.comp_aemeasurable
                ((measurable_rademacherSumProd n).sub
                  ((measurable_const.add (measurable_pi_apply 0)).div measurable_const)).aemeasurable
          · linarith [hC 0, norm_nonneg (f 0)]
          · filter_upwards with x
            calc ‖f (aPlusProd n 0 x) - f (aMinusProd n 0 x)‖
                ≤ ‖f (aPlusProd n 0 x)‖ + ‖f (aMinusProd n 0 x)‖ := norm_sub_le _ _
              _ ≤ C + C := add_le_add (hC _) (hC _)
              _ = 2 * C := by ring
        · -- Integrability of RHS: (a + b)² where a, b are bounded
          let C' := firstDerivBound f hf
          have hC' := firstDerivBound_spec hf
          have hC'_nonneg := firstDerivBound_nonneg hf
          apply integrable_sq_of_bounded (C := 2 * C' / Real.sqrt n + 4 * K / n)
          · apply AEStronglyMeasurable.add
            · apply AEStronglyMeasurable.mul aestronglyMeasurable_const
              exact (continuous_abs.comp (deriv_continuous_of_compactlySupported hf)).aestronglyMeasurable.comp_aemeasurable
                (measurable_rademacherSumProd n).aemeasurable
            · exact aestronglyMeasurable_const
          · apply add_nonneg
            · apply div_nonneg (mul_nonneg (by norm_num) hC'_nonneg) (le_of_lt hsqrt_pos)
            · apply div_nonneg (mul_nonneg (by norm_num) hK_nonneg) (Nat.cast_nonneg n)
          · filter_upwards with x
            have h1 : |deriv f (rademacherSumProd n x)| ≤ C' := hC' _
            calc ‖2 / Real.sqrt n * |deriv f (rademacherSumProd n x)| + 4 * K / n‖
                = |2 / Real.sqrt n * |deriv f (rademacherSumProd n x)| + 4 * K / n| := Real.norm_eq_abs _
              _ = 2 / Real.sqrt n * |deriv f (rademacherSumProd n x)| + 4 * K / n := by
                  apply abs_of_nonneg
                  apply add_nonneg
                  · apply mul_nonneg (div_nonneg (by norm_num) (le_of_lt hsqrt_pos)) (abs_nonneg _)
                  · apply div_nonneg (mul_nonneg (by norm_num) hK_nonneg) (Nat.cast_nonneg n)
              _ ≤ 2 / Real.sqrt n * C' + 4 * K / n := by
                  apply add_le_add
                  · apply mul_le_mul_of_nonneg_left h1 (div_nonneg (by norm_num) (le_of_lt hsqrt_pos))
                  · exact le_refl _
              _ = 2 * C' / Real.sqrt n + 4 * K / n := by ring
        · exact hbound
    _ = (n : ℝ) / 4 * ∫ x, (4 / n * (deriv f (rademacherSumProd n x))^2 +
          2 * (2 / Real.sqrt n) * |deriv f (rademacherSumProd n x)| * (4 * K / n) +
          (4 * K / n)^2) ∂μ := by
        congr 1
        apply integral_congr_ae
        filter_upwards with x
        have h_deriv_sq : |deriv f (rademacherSumProd n x)|^2 = (deriv f (rademacherSumProd n x))^2 :=
          sq_abs _
        have hsqrt_inv_sq : (Real.sqrt n)⁻¹ ^ 2 = (n : ℝ)⁻¹ := by
          rw [inv_pow, hsqrt_sq]
        ring_nf
        rw [← h_deriv_sq, hsqrt_inv_sq]
        ring
    _ = (n : ℝ) / 4 * (∫ x, 4 / n * (deriv f (rademacherSumProd n x))^2 ∂μ +
          ∫ x, 16 * K / (Real.sqrt n * n) * |deriv f (rademacherSumProd n x)| ∂μ +
          ∫ x, 16 * K^2 / n^2 ∂μ) := by
        congr 1
        rw [← integral_add, ← integral_add]
        · apply integral_congr_ae
          filter_upwards with x
          ring
        · apply Integrable.add
          · apply Integrable.const_mul (integrable_deriv_sq n hf)
          · apply Integrable.const_mul (integrable_deriv_abs n hf)
        · exact integrable_const _
        · apply Integrable.const_mul (integrable_deriv_sq n hf)
        · apply Integrable.const_mul (integrable_deriv_abs n hf)
    _ = (n : ℝ) / 4 * (4 / n * ∫ x, (deriv f (rademacherSumProd n x))^2 ∂μ +
          16 * K / (Real.sqrt n * n) * ∫ x, |deriv f (rademacherSumProd n x)| ∂μ +
          16 * K^2 / n^2) := by
        congr 1
        have h1 : ∫ x, 4 / ↑n * (deriv f (rademacherSumProd n x))^2 ∂μ =
            4 / ↑n * ∫ x, (deriv f (rademacherSumProd n x))^2 ∂μ := by
          simp_rw [← smul_eq_mul, MeasureTheory.integral_smul]
        have h2 : ∫ x, 16 * K / (Real.sqrt n * ↑n) * |deriv f (rademacherSumProd n x)| ∂μ =
            16 * K / (Real.sqrt n * ↑n) * ∫ x, |deriv f (rademacherSumProd n x)| ∂μ := by
          simp_rw [← smul_eq_mul, MeasureTheory.integral_smul]
        have h3 : ∫ (x : RademacherSpace n), 16 * K ^ 2 / ↑n ^ 2 ∂μ = 16 * K ^ 2 / ↑n ^ 2 := by
          rw [integral_const]
          simp [Measure.real_def, measure_univ, ENNReal.toReal_one]
        rw [h1, h2, h3]
    _ = ∫ x, (deriv f (rademacherSumProd n x))^2 ∂μ +
          4 * K / Real.sqrt n * ∫ x, |deriv f (rademacherSumProd n x)| ∂μ +
          4 * K^2 / n := by
        field_simp
        ring

end TaylorBound

end
