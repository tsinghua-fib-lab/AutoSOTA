/-
Copyright (c) 2026 Yuanhe Zhang. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Yuanhe Zhang, Jason D. Lee, Fanghui Liu
-/
import SLT.SubGaussian
import Mathlib.Analysis.CStarAlgebra.Matrix
import Mathlib.Analysis.InnerProductSpace.SingularValues
import Mathlib.Analysis.InnerProductSpace.Trace
import Mathlib.Probability.Distributions.Gaussian.Multivariate
import Mathlib.Analysis.Complex.ExponentialBounds
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Series
import Mathlib.MeasureTheory.Function.SpecialFunctions.Inner

/-!
# Hanson-Wright Inequality

This file formalizes a finite-dimensional Hanson-Wright tail bound for real
quadratic forms.  The public theorem proves the required Hanson-Wright MGF
certificate from independent sub-Gaussian coordinates, using local
diagonal/off-diagonal MGF infrastructure and then optimizing the resulting
two-scale Chernoff bound.

## Main definitions

* `HansonWright.quadraticForm`: the deterministic quadratic form `xᵀ A x`.
* `HansonWright.centeredQuadraticForm`: the centered random quadratic form.
* `HansonWright.frobeniusNorm`: the Frobenius norm of a finite real matrix.
* `HansonWright.operatorNorm`: the `ℓ²` operator norm of a finite real matrix.
* `HansonWright.entrywiseL1Norm`: the entrywise `ℓ¹` norm of a finite real matrix.
* `HansonWright.HasHansonWrightMGF`: the local quadratic CGF hypothesis used
  internally by the tail optimization step.

## Main results

* `HansonWright.hasSubgaussianMGF_of_abs_le_of_integral_eq_zero`: local
  symmetric bounded-variable MGF estimate.
* `HansonWright.two_sided_tail_of_cgf_bound`: Chernoff optimization for a
  two-scale CGF bound.
* `HansonWright.hasHansonWrightMGF_of_subgaussian`: a proved Hanson-Wright MGF
  certificate for independent sub-Gaussian coordinates.
* `HansonWright.hasHansonWrightMGF_of_bounded`: a proved Hanson-Wright MGF
  certificate for bounded coordinates.
* `HansonWright.hanson_wright_inequality`: Hanson-Wright tail bound after
  deriving the MGF certificate from independent sub-Gaussian coordinates.
* `HansonWright.hanson_wright_inequality_hdp`: HDP-style Hanson-Wright tail bound
  using the maximum coordinate ψ₂ scale.
-/

open MeasureTheory ProbabilityTheory Real
open scoped BigOperators NNReal

noncomputable section

namespace HansonWright

variable {Ω : Type*} [MeasurableSpace Ω]

lemma mgf_innerSL_stdGaussian {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]
    [FiniteDimensional ℝ E] [MeasurableSpace E] [BorelSpace E] (v : E) (t : ℝ) :
    mgf (fun x : E => inner ℝ v x) (stdGaussian E) t =
      exp (‖v‖ ^ 2 * t ^ 2 / 2) := by
  let L : StrongDual ℝ E := innerSL ℝ v
  have hmap : (stdGaussian E).map L =
      gaussianReal ((stdGaussian E)[L]) Var[L; stdGaussian E].toNNReal :=
    IsGaussian.map_eq_gaussianReal L
  have hmgf := mgf_gaussianReal hmap t
  have hmean : (stdGaussian E)[L] = 0 := integral_strongDual_stdGaussian L
  have hvar : (Var[L; stdGaussian E].toNNReal : ℝ) = ‖v‖ ^ 2 := by
    rw [variance_dual_stdGaussian L]
    simp [L, innerSL_apply_norm]
  change mgf (⇑L) (stdGaussian E) t = exp (‖v‖ ^ 2 * t ^ 2 / 2)
  rw [hmgf, hmean, hvar]
  ring_nf

lemma integrable_exp_innerSL_stdGaussian {E : Type*} [NormedAddCommGroup E]
    [InnerProductSpace ℝ E] [FiniteDimensional ℝ E] [MeasurableSpace E] [BorelSpace E]
    (v : E) (t : ℝ) :
    Integrable (fun x : E => exp (t * inner ℝ v x)) (stdGaussian E) := by
  rw [← mgf_pos_iff]
  rw [mgf_innerSL_stdGaussian v t]
  exact exp_pos _

lemma integral_exp_innerSL_stdGaussian {E : Type*} [NormedAddCommGroup E]
    [InnerProductSpace ℝ E] [FiniteDimensional ℝ E] [MeasurableSpace E] [BorelSpace E]
    (v : E) (t : ℝ) :
    ∫ x : E, exp (t * inner ℝ v x) ∂(stdGaussian E) =
      exp (‖v‖ ^ 2 * t ^ 2 / 2) := by
  exact mgf_innerSL_stdGaussian v t

lemma hasSubgaussianMGF_id_gaussianReal_zero_one :
    HasSubgaussianMGF id (1 : ℝ≥0) (gaussianReal 0 1) where
  integrable_exp_mul t := by
    simpa [id_eq] using integrable_exp_mul_gaussianReal (μ := 0) (v := 1) t
  mgf_le t := by
    rw [mgf_id_gaussianReal]
    simp only [zero_mul, NNReal.coe_one, one_mul, zero_add]
    rfl

lemma symmetric_inner_apply_eq_sum_eigenvalues_repr {E : Type*} [NormedAddCommGroup E]
    [InnerProductSpace ℝ E] [FiniteDimensional ℝ E]
    (T : E →ₗ[ℝ] E) (hT : T.IsSymmetric) (x : E) :
    inner ℝ (T x) x =
      ∑ i : Fin (Module.finrank ℝ E),
        hT.eigenvalues rfl i * ((hT.eigenvectorBasis rfl).repr x i) ^ 2 := by
  let b := hT.eigenvectorBasis (by rfl : Module.finrank ℝ E = Module.finrank ℝ E)
  calc
    inner ℝ (T x) x = inner ℝ (b.repr (T x)) (b.repr x) := by
      rw [b.repr.inner_map_map]
    _ = ∑ i : Fin (Module.finrank ℝ E),
        hT.eigenvalues rfl i * ((hT.eigenvectorBasis rfl).repr x i) ^ 2 := by
      have hcoord :
          ∀ i : Fin (Module.finrank ℝ E),
            b.repr (T x) i = hT.eigenvalues rfl i * b.repr x i := by
        intro i
        simpa [b] using hT.eigenvectorBasis_apply_self_apply
          (by rfl : Module.finrank ℝ E = Module.finrank ℝ E) x i
      rw [PiLp.inner_apply]
      apply Finset.sum_congr rfl
      intro i _
      rw [hcoord i]
      rw [Real.inner_apply]
      simp [b]
      ring_nf

lemma orthonormalBasis_repr_sum_smul {E : Type*} [NormedAddCommGroup E]
    [InnerProductSpace ℝ E] [FiniteDimensional ℝ E]
    (b : OrthonormalBasis (Fin (Module.finrank ℝ E)) ℝ E)
    (z : Fin (Module.finrank ℝ E) → ℝ) (i : Fin (Module.finrank ℝ E)) :
    b.repr (∑ i, z i • b i) i = z i := by
  simp only [OrthonormalBasis.repr_apply_apply, inner_sum, inner_smul_right]
  rw [Finset.sum_eq_single i]
  · simp
  · intro j _ hji
    rw [b.inner_eq_zero hji.symm]
    simp
  · intro hi
    exact (hi (Finset.mem_univ i)).elim

/-- The deterministic quadratic form `xᵀ A x`. -/
def quadraticForm {n : ℕ} (A : Matrix (Fin n) (Fin n) ℝ) (x : Fin n → ℝ) : ℝ :=
  ∑ i, ∑ j, A i j * x i * x j

/-- The random quadratic form associated to a random vector `X`. -/
def randomQuadraticForm {n : ℕ} (A : Matrix (Fin n) (Fin n) ℝ)
    (X : Fin n → Ω → ℝ) : Ω → ℝ :=
  fun ω => quadraticForm A fun i => X i ω

/-- The coordinate random vector as an element of Euclidean space. -/
def randomVector {n : ℕ} (X : Fin n → Ω → ℝ) : Ω → EuclideanSpace ℝ (Fin n) :=
  fun ω => WithLp.toLp 2 fun i => X i ω

omit [MeasurableSpace Ω] in
@[simp]
lemma randomVector_apply {n : ℕ} (X : Fin n → Ω → ℝ) (ω : Ω) (i : Fin n) :
    randomVector X ω i = X i ω := rfl

lemma randomVector_aemeasurable {μ : Measure Ω} {n : ℕ} {X : Fin n → Ω → ℝ}
    (hX_meas : ∀ i, AEMeasurable (X i) μ) :
    AEMeasurable (randomVector X) μ := by
  exact (MeasurableEquiv.toLp 2 (Fin n → ℝ)).measurable.comp_aemeasurable
    (aemeasurable_pi_lambda _ hX_meas)

omit [MeasurableSpace Ω] in
lemma measure_map_prod_map_of_aemeasurable {α β γ δ : Type*}
    [MeasurableSpace α] [MeasurableSpace β] [MeasurableSpace γ] [MeasurableSpace δ]
    {μ : Measure α} {ν : Measure β} [SFinite μ] [SFinite ν]
    {f : α → γ} {g : β → δ} (hf : AEMeasurable f μ) (hg : AEMeasurable g ν) :
    (μ.map f).prod (ν.map g) =
      (μ.prod ν).map (fun p : α × β => (f p.1, g p.2)) := by
  let f' := hf.mk f
  let g' := hg.mk g
  have hf_map : μ.map f = μ.map f' := Measure.map_congr hf.ae_eq_mk
  have hg_map : ν.map g = ν.map g' := Measure.map_congr hg.ae_eq_mk
  have hpair :
      (fun p : α × β => (f p.1, g p.2)) =ᵐ[μ.prod ν]
        Prod.map f' g' := by
    exact
      ((Measure.quasiMeasurePreserving_fst (μ := μ) (ν := ν)).ae_eq_comp hf.ae_eq_mk).prodMk
        ((Measure.quasiMeasurePreserving_snd (μ := μ) (ν := ν)).ae_eq_comp hg.ae_eq_mk)
  calc
    (μ.map f).prod (ν.map g)
        = (μ.map f').prod (ν.map g') := by rw [hf_map, hg_map]
    _ = (μ.prod ν).map (Prod.map f' g') :=
        Measure.map_prod_map μ ν hf.measurable_mk hg.measurable_mk
    _ = (μ.prod ν).map (fun p : α × β => (f p.1, g p.2)) :=
        Measure.map_congr hpair.symm

/-- The centered random quadratic form `Xᵀ A X - E Xᵀ A X`. -/
def centeredQuadraticForm {n : ℕ} (μ : Measure Ω) (A : Matrix (Fin n) (Fin n) ℝ)
    (X : Fin n → Ω → ℝ) : Ω → ℝ :=
  fun ω => randomQuadraticForm A X ω - ∫ ω, randomQuadraticForm A X ω ∂μ

/-- The squared Frobenius norm of a finite real matrix. -/
def frobeniusNormSq {n : ℕ} (A : Matrix (Fin n) (Fin n) ℝ) : ℝ :=
  ∑ i, ∑ j, (A i j) ^ 2

/-- The Frobenius norm of a finite real matrix. -/
def frobeniusNorm {n : ℕ} (A : Matrix (Fin n) (Fin n) ℝ) : ℝ :=
  sqrt (frobeniusNormSq A)

/-- The `ℓ²` operator norm of a finite real matrix. -/
def operatorNorm {n : ℕ} (A : Matrix (Fin n) (Fin n) ℝ) : ℝ :=
  ‖Matrix.toEuclideanCLM (𝕜 := ℝ) A‖

/-- The entrywise `ℓ¹` norm of a finite real matrix. -/
def entrywiseL1Norm {n : ℕ} (A : Matrix (Fin n) (Fin n) ℝ) : ℝ :=
  ∑ i, ∑ j, |A i j|

/-- The matrix obtained by deleting the diagonal entries. -/
def offDiagonalMatrix {n : ℕ} (A : Matrix (Fin n) (Fin n) ℝ) :
    Matrix (Fin n) (Fin n) ℝ :=
  fun i j => if i = j then 0 else A i j

/-- The matrix keeping only rows in `s` and columns outside `s`. -/
def cutMatrix {n : ℕ} (A : Matrix (Fin n) (Fin n) ℝ) (s : Finset (Fin n)) :
    Matrix (Fin n) (Fin n) ℝ :=
  fun i j => if i ∈ s ∧ j ∉ s then A i j else 0

/-- Coordinate projection onto a finite set of coordinates. -/
def coordinateMask {n : ℕ} (s : Finset (Fin n))
    (x : EuclideanSpace ℝ (Fin n)) : EuclideanSpace ℝ (Fin n) :=
  WithLp.toLp 2 fun i => if i ∈ s then x i else 0

@[simp]
lemma coordinateMask_apply {n : ℕ} (s : Finset (Fin n))
    (x : EuclideanSpace ℝ (Fin n)) (i : Fin n) :
    coordinateMask s x i = if i ∈ s then x i else 0 := rfl

/-- Embed a tuple indexed by a finite set into Euclidean space, filling other coordinates by zero. -/
def subtypeMask {n : ℕ} (s : Finset (Fin n)) (x : s → ℝ) :
    EuclideanSpace ℝ (Fin n) :=
  WithLp.toLp 2 fun i => if h : i ∈ s then x ⟨i, h⟩ else 0

@[simp]
lemma subtypeMask_apply {n : ℕ} (s : Finset (Fin n)) (x : s → ℝ) (i : Fin n) :
    subtypeMask s x i = if h : i ∈ s then x ⟨i, h⟩ else 0 := rfl

lemma measurable_subtypeMask {n : ℕ} (s : Finset (Fin n)) :
    Measurable (subtypeMask (n := n) s) := by
  exact (MeasurableEquiv.toLp 2 (Fin n → ℝ)).measurable.comp
    (measurable_pi_lambda _ fun i => by
      by_cases hi : i ∈ s
      · simpa [hi] using
          (measurable_pi_apply (⟨i, hi⟩ : s) : Measurable fun x : s → ℝ => x ⟨i, hi⟩)
      · simp [hi])

omit [MeasurableSpace Ω] in
lemma subtypeMask_subtype_randomVector {n : ℕ} (s : Finset (Fin n))
    (X : Fin n → Ω → ℝ) :
    (fun ω => subtypeMask s (fun i : s => X i ω)) =
      fun ω => coordinateMask s (randomVector X ω) := by
  ext ω i
  by_cases hi : i ∈ s
  · simp [hi]
  · simp [hi]

lemma coordinateMask_aemeasurable {μ : Measure Ω} {n : ℕ} (s : Finset (Fin n))
    {X : Fin n → Ω → ℝ} (hX_meas : ∀ i, AEMeasurable (X i) μ) :
    AEMeasurable (fun ω => coordinateMask s (randomVector X ω)) μ := by
  rw [← subtypeMask_subtype_randomVector s X]
  exact (measurable_subtypeMask s).aemeasurable.comp_aemeasurable
    (aemeasurable_pi_lambda _ fun i => hX_meas i)

lemma coordinateMask_indepFun_compl {μ : Measure Ω} {n : ℕ} {X : Fin n → Ω → ℝ}
    (h_indep : iIndepFun X μ) (hX_meas : ∀ i, AEMeasurable (X i) μ)
    (s : Finset (Fin n)) :
    (fun ω => coordinateMask ((Finset.univ : Finset (Fin n)) \ s) (randomVector X ω))
      ⟂ᵢ[μ]
    (fun ω => coordinateMask s (randomVector X ω)) := by
  classical
  let t : Finset (Fin n) := (Finset.univ : Finset (Fin n)) \ s
  have hdisj : Disjoint t s := by
    dsimp [t]
    exact Finset.sdiff_disjoint
  have hblocks :
      (fun ω (i : t) => X i ω) ⟂ᵢ[μ] (fun ω (i : s) => X i ω) :=
    h_indep.indepFun_finset₀ t s hdisj hX_meas
  have hcomp :
      ((subtypeMask t) ∘ fun ω (i : t) => X i ω) ⟂ᵢ[μ]
        ((subtypeMask s) ∘ fun ω (i : s) => X i ω) :=
    hblocks.comp (measurable_subtypeMask t) (measurable_subtypeMask s)
  refine hcomp.congr ?_ ?_
  · filter_upwards with ω
    ext i
    by_cases hi : i ∈ t
    · simp [t, coordinateMask, subtypeMask]
    · simp [t, coordinateMask, subtypeMask]
  · filter_upwards with ω
    ext i
    by_cases hi : i ∈ s
    · simp [coordinateMask, subtypeMask, hi]
    · simp [coordinateMask, subtypeMask, hi]

lemma frobeniusNormSq_nonneg {n : ℕ} (A : Matrix (Fin n) (Fin n) ℝ) :
    0 ≤ frobeniusNormSq A := by
  unfold frobeniusNormSq
  exact Finset.sum_nonneg fun _ _ => Finset.sum_nonneg fun _ _ => sq_nonneg _

lemma frobeniusNorm_nonneg {n : ℕ} (A : Matrix (Fin n) (Fin n) ℝ) :
    0 ≤ frobeniusNorm A :=
  sqrt_nonneg _

lemma operatorNorm_nonneg {n : ℕ} (A : Matrix (Fin n) (Fin n) ℝ) :
    0 ≤ operatorNorm A :=
  norm_nonneg _

lemma entrywiseL1Norm_nonneg {n : ℕ} (A : Matrix (Fin n) (Fin n) ℝ) :
    0 ≤ entrywiseL1Norm A := by
  unfold entrywiseL1Norm
  exact Finset.sum_nonneg fun _ _ => Finset.sum_nonneg fun _ _ => abs_nonneg _

lemma coordinateMask_norm_le {n : ℕ} (s : Finset (Fin n))
    (x : EuclideanSpace ℝ (Fin n)) :
    ‖coordinateMask s x‖ ≤ ‖x‖ := by
  refine le_of_sq_le_sq ?_ (norm_nonneg _)
  calc
    ‖coordinateMask s x‖ ^ 2
        = ∑ i : Fin n, (coordinateMask s x i) ^ 2 := by
            rw [EuclideanSpace.real_norm_sq_eq]
    _ ≤ ∑ i : Fin n, x i ^ 2 := by
        refine Finset.sum_le_sum fun i _ => ?_
        by_cases hi : i ∈ s
        · simp [hi]
        · simp [hi, sq_nonneg]
    _ = ‖x‖ ^ 2 := by
        rw [EuclideanSpace.real_norm_sq_eq]

lemma toEuclideanCLM_cutMatrix_apply {n : ℕ} (A : Matrix (Fin n) (Fin n) ℝ)
    (s : Finset (Fin n)) (x : EuclideanSpace ℝ (Fin n)) :
    Matrix.toEuclideanCLM (𝕜 := ℝ) (cutMatrix A s) x =
      coordinateMask s
        (Matrix.toEuclideanCLM (𝕜 := ℝ) A
          (coordinateMask ((Finset.univ : Finset (Fin n)) \ s) x)) := by
  ext i
  by_cases hi : i ∈ s
  · simp [Matrix.toEuclideanCLM_toLp, coordinateMask, cutMatrix, Matrix.mulVec, dotProduct, hi]
  · simp [Matrix.toEuclideanCLM_toLp, coordinateMask, cutMatrix, Matrix.mulVec, dotProduct, hi]

lemma inner_toEuclideanCLM_cutMatrix_eq_inner_masks {n : ℕ}
    (A : Matrix (Fin n) (Fin n) ℝ) (s : Finset (Fin n))
    (x y : EuclideanSpace ℝ (Fin n)) :
    inner ℝ (Matrix.toEuclideanCLM (𝕜 := ℝ) (cutMatrix A s) x) y =
      inner ℝ
        (Matrix.toEuclideanCLM (𝕜 := ℝ) A
          (coordinateMask ((Finset.univ : Finset (Fin n)) \ s) x))
        (coordinateMask s y) := by
  rw [toEuclideanCLM_cutMatrix_apply]
  rw [PiLp.inner_apply, PiLp.inner_apply]
  simp only [coordinateMask_apply, RCLike.inner_apply, conj_trivial]
  apply Finset.sum_congr rfl
  intro i _
  by_cases hi : i ∈ s
  · simp [hi]
  · simp [hi]

lemma operatorNorm_cutMatrix_le {n : ℕ} (A : Matrix (Fin n) (Fin n) ℝ)
    (s : Finset (Fin n)) :
    operatorNorm (cutMatrix A s) ≤ operatorNorm A := by
  unfold operatorNorm
  let T : EuclideanSpace ℝ (Fin n) →L[ℝ] EuclideanSpace ℝ (Fin n) :=
    Matrix.toEuclideanCLM (𝕜 := ℝ) A
  refine ContinuousLinearMap.opNorm_le_bound _ (norm_nonneg T) fun x => ?_
  rw [toEuclideanCLM_cutMatrix_apply]
  calc
    ‖coordinateMask s (T (coordinateMask ((Finset.univ : Finset (Fin n)) \ s) x))‖
        ≤ ‖T (coordinateMask ((Finset.univ : Finset (Fin n)) \ s) x)‖ :=
          coordinateMask_norm_le s _
    _ ≤ ‖T‖ * ‖coordinateMask ((Finset.univ : Finset (Fin n)) \ s) x‖ :=
          T.le_opNorm _
    _ ≤ ‖T‖ * ‖x‖ :=
          mul_le_mul_of_nonneg_left
            (coordinateMask_norm_le ((Finset.univ : Finset (Fin n)) \ s) x) (norm_nonneg T)

lemma frobeniusNormSq_cutMatrix_le {n : ℕ} (A : Matrix (Fin n) (Fin n) ℝ)
    (s : Finset (Fin n)) :
    frobeniusNormSq (cutMatrix A s) ≤ frobeniusNormSq A := by
  unfold frobeniusNormSq cutMatrix
  refine Finset.sum_le_sum fun i _ => Finset.sum_le_sum fun j _ => ?_
  by_cases h : i ∈ s ∧ j ∉ s
  · simp [h]
  · simp [h, sq_nonneg]

lemma frobeniusNorm_sq {n : ℕ} (A : Matrix (Fin n) (Fin n) ℝ) :
    frobeniusNorm A ^ 2 = frobeniusNormSq A := by
  unfold frobeniusNorm
  rw [sq_sqrt (frobeniusNormSq_nonneg A)]

lemma frobeniusNorm_cutMatrix_sq_le {n : ℕ} (A : Matrix (Fin n) (Fin n) ℝ)
    (s : Finset (Fin n)) :
    frobeniusNorm (cutMatrix A s) ^ 2 ≤ frobeniusNorm A ^ 2 := by
  rw [frobeniusNorm_sq, frobeniusNorm_sq]
  exact frobeniusNormSq_cutMatrix_le A s

lemma card_powerset_filter_mem_notMem_eq {n : ℕ} {i j : Fin n} (hij : i ≠ j) :
    (((Finset.univ : Finset (Fin n)).powerset.filter
      (fun s => i ∈ s ∧ j ∉ s)).card) =
      (((((Finset.univ : Finset (Fin n)).erase i).erase j).powerset).card) := by
  classical
  let S : Finset (Finset (Fin n)) :=
    (Finset.univ : Finset (Fin n)).powerset.filter (fun s => i ∈ s ∧ j ∉ s)
  let T : Finset (Finset (Fin n)) :=
    (((Finset.univ : Finset (Fin n)).erase i).erase j).powerset
  change S.card = T.card
  refine Finset.card_nbij' (fun u => u.erase i) (fun u => insert i u) ?_ ?_ ?_ ?_
  · intro u hu
    have hu' :
        u ∈ (Finset.univ : Finset (Fin n)).powerset.filter
          (fun s => i ∈ s ∧ j ∉ s) := by
      simpa [S] using hu
    dsimp [T]
    change u.erase i ∈ ((((Finset.univ : Finset (Fin n)).erase i).erase j).powerset)
    rw [Finset.mem_filter, Finset.mem_powerset] at hu'
    rw [Finset.mem_powerset]
    intro k hk
    rw [Finset.mem_erase] at hk ⊢
    refine ⟨?_, ?_⟩
    · intro hkj
      exact hu'.2.2 (by simpa [hkj] using hk.2)
    · rw [Finset.mem_erase]
      exact ⟨hk.1, Finset.mem_univ k⟩
  · intro u hu
    have hu_set : (u : Set (Fin n)) ⊆ (Set.univ \ {i}) \ {j} := by
      simpa [T, Finset.mem_powerset] using hu
    have hu_sub : u ⊆ (((Finset.univ : Finset (Fin n)).erase i).erase j) := by
      intro k hk
      have hkset := hu_set hk
      rw [Finset.mem_erase, Finset.mem_erase]
      simp at hkset
      exact ⟨hkset.2, hkset.1, Finset.mem_univ k⟩
    dsimp [S]
    change insert i u ∈
      ((Finset.univ : Finset (Fin n)).powerset.filter (fun s => i ∈ s ∧ j ∉ s))
    rw [Finset.mem_filter, Finset.mem_powerset]
    constructor
    · intro k hk
      exact Finset.mem_univ k
    · constructor
      · exact Finset.mem_insert_self i u
      · intro hju
        rw [Finset.mem_insert] at hju
        rcases hju with hji | hju
        · exact hij hji.symm
        · have hj_erase := hu_sub hju
          exact (Finset.mem_erase.mp hj_erase).1 rfl
  · intro u hu
    have hu' :
        u ∈ (Finset.univ : Finset (Fin n)).powerset.filter
          (fun s => i ∈ s ∧ j ∉ s) := by
      simpa [S] using hu
    rw [Finset.mem_filter] at hu'
    exact Finset.insert_erase hu'.2.1
  · intro u hu
    dsimp [T] at hu
    have hi_not : i ∉ u := by
      intro hiu
      have hi_erase := (Finset.mem_powerset.mp hu) hiu
      exact (Finset.mem_erase.mp (Finset.mem_erase.mp hi_erase).2).1 rfl
    exact Finset.erase_insert hi_not

lemma four_mul_card_powerset_filter_mem_notMem {n : ℕ} {i j : Fin n} (hij : i ≠ j) :
    4 * (((Finset.univ : Finset (Fin n)).powerset.filter
      (fun s => i ∈ s ∧ j ∉ s)).card) =
      ((Finset.univ : Finset (Fin n)).powerset).card := by
  classical
  let T : Finset (Fin n) := ((Finset.univ : Finset (Fin n)).erase i).erase j
  have hcard_filter :
      (((Finset.univ : Finset (Fin n)).powerset.filter
        (fun s => i ∈ s ∧ j ∉ s)).card) = T.powerset.card := by
    simpa [T] using card_powerset_filter_mem_notMem_eq (n := n) hij
  have hj_mem : j ∈ (Finset.univ : Finset (Fin n)).erase i := by
    rw [Finset.mem_erase]
    exact ⟨hij.symm, Finset.mem_univ j⟩
  have hT_card_add :
      T.card + 2 = (Finset.univ : Finset (Fin n)).card := by
    have h1 :
        ((Finset.univ : Finset (Fin n)).erase i).card + 1 =
          (Finset.univ : Finset (Fin n)).card :=
      Finset.card_erase_add_one (Finset.mem_univ i)
    have h2 :
        T.card + 1 = ((Finset.univ : Finset (Fin n)).erase i).card := by
      dsimp [T]
      exact Finset.card_erase_add_one hj_mem
    omega
  rw [hcard_filter, Finset.card_powerset, Finset.card_powerset]
  rw [← hT_card_add]
  rw [Nat.pow_add]
  norm_num
  rw [Nat.mul_comm]

lemma sum_powerset_cut_indicator {n : ℕ} (i j : Fin n) (a : ℝ) :
    (∑ s ∈ (Finset.univ : Finset (Fin n)).powerset,
      (if i ∈ s ∧ j ∉ s then a else 0)) =
      (((Finset.univ : Finset (Fin n)).powerset.filter
        (fun s => i ∈ s ∧ j ∉ s)).card : ℝ) * a := by
  classical
  rw [← Finset.sum_filter]
  simp

lemma four_mul_sum_cut_quadraticForm_eq_card_mul_offDiagonal {n : ℕ}
    (A : Matrix (Fin n) (Fin n) ℝ) (x : Fin n → ℝ) :
    4 * (∑ s ∈ (Finset.univ : Finset (Fin n)).powerset,
      quadraticForm (cutMatrix A s) x) =
      (((Finset.univ : Finset (Fin n)).powerset).card : ℝ) *
        quadraticForm (offDiagonalMatrix A) x := by
  classical
  let P : Finset (Finset (Fin n)) := (Finset.univ : Finset (Fin n)).powerset
  have hpair : ∀ i j : Fin n,
      4 * (∑ s ∈ P, (if i ∈ s ∧ j ∉ s then A i j * x i * x j else 0)) =
        (P.card : ℝ) * ((if i = j then 0 else A i j) * x i * x j) := by
    intro i j
    by_cases hij : i = j
    · subst j
      simp [P]
    · have hsum :=
        sum_powerset_cut_indicator (n := n) i j (A i j * x i * x j)
      have hcount_nat :
          4 * ((P.filter (fun s => i ∈ s ∧ j ∉ s)).card) = P.card := by
        simpa [P] using four_mul_card_powerset_filter_mem_notMem (n := n) hij
      have hcount_real :
          (4 : ℝ) * ((P.filter (fun s => i ∈ s ∧ j ∉ s)).card : ℝ) =
            (P.card : ℝ) := by
        exact_mod_cast hcount_nat
      calc
        4 * (∑ s ∈ P, (if i ∈ s ∧ j ∉ s then A i j * x i * x j else 0))
            = 4 * (((P.filter (fun s => i ∈ s ∧ j ∉ s)).card : ℝ) *
                (A i j * x i * x j)) := by
                rw [hsum]
        _ = ((4 : ℝ) * ((P.filter (fun s => i ∈ s ∧ j ∉ s)).card : ℝ)) *
              (A i j * x i * x j) := by ring
        _ = (P.card : ℝ) * (A i j * x i * x j) := by rw [hcount_real]
        _ = (P.card : ℝ) * ((if i = j then 0 else A i j) * x i * x j) := by
              simp [hij]
  calc
    4 * (∑ s ∈ P, quadraticForm (cutMatrix A s) x)
        = 4 * (∑ i, ∑ j, ∑ s ∈ P,
            (if i ∈ s ∧ j ∉ s then A i j * x i * x j else 0)) := by
            congr 1
            unfold quadraticForm cutMatrix
            rw [Finset.sum_comm]
            apply Finset.sum_congr rfl
            intro i _
            rw [Finset.sum_comm]
            apply Finset.sum_congr rfl
            intro j _
            apply Finset.sum_congr rfl
            intro s _
            by_cases hs : i ∈ s ∧ j ∉ s <;> simp [hs]
    _ = ∑ i, ∑ j,
          4 * (∑ s ∈ P, (if i ∈ s ∧ j ∉ s then A i j * x i * x j else 0)) := by
            rw [Finset.mul_sum]
            apply Finset.sum_congr rfl
            intro i _
            rw [Finset.mul_sum]
    _ = ∑ i, ∑ j,
          (P.card : ℝ) * ((if i = j then 0 else A i j) * x i * x j) := by
            apply Finset.sum_congr rfl
            intro i _
            apply Finset.sum_congr rfl
            intro j _
            exact hpair i j
    _ = (P.card : ℝ) * quadraticForm (offDiagonalMatrix A) x := by
            unfold quadraticForm offDiagonalMatrix
            rw [Finset.mul_sum]
            apply Finset.sum_congr rfl
            intro i _
            rw [Finset.mul_sum]

lemma quadraticForm_offDiagonal_eq_average_cut {n : ℕ}
    (A : Matrix (Fin n) (Fin n) ℝ) (x : Fin n → ℝ) :
    quadraticForm (offDiagonalMatrix A) x =
      (4 / (((Finset.univ : Finset (Fin n)).powerset).card : ℝ)) *
        ∑ s ∈ (Finset.univ : Finset (Fin n)).powerset,
          quadraticForm (cutMatrix A s) x := by
  classical
  let P : Finset (Finset (Fin n)) := (Finset.univ : Finset (Fin n)).powerset
  have hP_pos : 0 < (P.card : ℝ) := by
    exact_mod_cast Finset.card_pos.mpr ⟨∅, Finset.empty_mem_powerset _⟩
  have hmul := four_mul_sum_cut_quadraticForm_eq_card_mul_offDiagonal A x
  change quadraticForm (offDiagonalMatrix A) x =
    (4 / (P.card : ℝ)) * ∑ s ∈ P, quadraticForm (cutMatrix A s) x
  calc
    quadraticForm (offDiagonalMatrix A) x
        = (P.card : ℝ)⁻¹ * ((P.card : ℝ) *
            quadraticForm (offDiagonalMatrix A) x) := by
            field_simp [ne_of_gt hP_pos]
    _ = (P.card : ℝ)⁻¹ *
          (4 * ∑ s ∈ P, quadraticForm (cutMatrix A s) x) := by
            rw [← hmul]
    _ = (4 / (P.card : ℝ)) * ∑ s ∈ P, quadraticForm (cutMatrix A s) x := by
            field_simp [ne_of_gt hP_pos]

lemma exp_mul_quadraticForm_offDiagonal_le_average_cut {n : ℕ}
    (A : Matrix (Fin n) (Fin n) ℝ) (x : Fin n → ℝ) (l : ℝ) :
    exp (l * quadraticForm (offDiagonalMatrix A) x) ≤
      (((Finset.univ : Finset (Fin n)).powerset).card : ℝ)⁻¹ *
        ∑ s ∈ (Finset.univ : Finset (Fin n)).powerset,
          exp (4 * l * quadraticForm (cutMatrix A s) x) := by
  classical
  let P : Finset (Finset (Fin n)) := (Finset.univ : Finset (Fin n)).powerset
  let w : Finset (Fin n) → ℝ := fun _ => (P.card : ℝ)⁻¹
  let p : Finset (Fin n) → ℝ := fun s => 4 * l * quadraticForm (cutMatrix A s) x
  have hP_pos : 0 < (P.card : ℝ) := by
    exact_mod_cast Finset.card_pos.mpr ⟨∅, Finset.empty_mem_powerset _⟩
  have hw_nonneg : ∀ s ∈ P, 0 ≤ w s := by
    intro s hs
    dsimp [w]
    exact inv_nonneg.mpr hP_pos.le
  have hw_sum : ∑ s ∈ P, w s = 1 := by
    simp [w]
    field_simp [ne_of_gt hP_pos]
  have hp_mem : ∀ s ∈ P, p s ∈ (Set.univ : Set ℝ) := by
    simp
  have hconv := convexOn_exp.map_sum_le (s := Set.univ) (t := P)
    (w := w) (p := p) hw_nonneg hw_sum hp_mem
  have harg :
      (∑ s ∈ P, w s • p s) =
        l * quadraticForm (offDiagonalMatrix A) x := by
    rw [quadraticForm_offDiagonal_eq_average_cut A x]
    dsimp only [w, p]
    simp only [smul_eq_mul]
    rw [← Finset.mul_sum]
    rw [← Finset.mul_sum]
    field_simp [ne_of_gt hP_pos]
    ring
  have hrhs :
      (∑ s ∈ P, w s • exp (p s)) =
        (P.card : ℝ)⁻¹ * ∑ s ∈ P,
          exp (4 * l * quadraticForm (cutMatrix A s) x) := by
    simp [w, p, smul_eq_mul, Finset.mul_sum]
  rw [harg, hrhs] at hconv
  simpa [P] using hconv

lemma quadraticForm_eq_inner_toEuclideanCLM {n : ℕ}
    (A : Matrix (Fin n) (Fin n) ℝ) (x : Fin n → ℝ) :
    quadraticForm A x =
      inner ℝ (Matrix.toEuclideanCLM (𝕜 := ℝ) A (WithLp.toLp 2 x)) (WithLp.toLp 2 x) := by
  unfold quadraticForm
  rw [Matrix.toEuclideanCLM_toLp, PiLp.inner_apply]
  simp only [RCLike.inner_apply, conj_trivial, Matrix.mulVec, dotProduct]
  apply Finset.sum_congr rfl
  intro i _
  rw [Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro j _
  ring

lemma sum_diag_sq_le_frobeniusNormSq {n : ℕ} (A : Matrix (Fin n) (Fin n) ℝ) :
    (∑ i, A i i ^ 2) ≤ frobeniusNormSq A := by
  unfold frobeniusNormSq
  exact Finset.sum_le_sum fun i _ => by
    exact Finset.single_le_sum (fun j _ => sq_nonneg (A i j)) (Finset.mem_univ i)

lemma sum_diag_sq_le_frobeniusNorm_sq {n : ℕ} (A : Matrix (Fin n) (Fin n) ℝ) :
    (∑ i, A i i ^ 2) ≤ frobeniusNorm A ^ 2 := by
  rw [frobeniusNorm_sq]
  exact sum_diag_sq_le_frobeniusNormSq A

lemma abs_diag_le_operatorNorm {n : ℕ} (A : Matrix (Fin n) (Fin n) ℝ) (i : Fin n) :
    |A i i| ≤ operatorNorm A := by
  let e : EuclideanSpace ℝ (Fin n) := EuclideanSpace.single i (1 : ℝ)
  have hcoord :
      |A i i| ≤ ‖Matrix.toEuclideanCLM (n := Fin n) (𝕜 := ℝ) A e‖ := by
    have h :=
      PiLp.norm_apply_le
        (x := Matrix.toEuclideanCLM (n := Fin n) (𝕜 := ℝ) A e) i
    simpa [e, Matrix.toEuclideanCLM_toLp, Matrix.mulVec_single_one, Real.norm_eq_abs]
      using h
  have hop :=
    (Matrix.toEuclideanCLM (n := Fin n) (𝕜 := ℝ) A).le_opNorm e
  have he_norm : ‖e‖ = 1 := by
    simp [e]
  calc
    |A i i| ≤ ‖Matrix.toEuclideanCLM (n := Fin n) (𝕜 := ℝ) A e‖ := hcoord
    _ ≤ ‖Matrix.toEuclideanCLM (n := Fin n) (𝕜 := ℝ) A‖ * ‖e‖ := hop
    _ = operatorNorm A := by
      rw [he_norm, mul_one]
      rfl

lemma abs_quadraticForm_le {n : ℕ} (A : Matrix (Fin n) (Fin n) ℝ)
    {x : Fin n → ℝ} {K : ℝ} (hK : 0 ≤ K) (hx : ∀ i, |x i| ≤ K) :
    |quadraticForm A x| ≤ K ^ 2 * entrywiseL1Norm A := by
  unfold quadraticForm entrywiseL1Norm
  calc |∑ i, ∑ j, A i j * x i * x j|
      ≤ ∑ i, |∑ j, A i j * x i * x j| := Finset.abs_sum_le_sum_abs _ _
    _ ≤ ∑ i, ∑ j, |A i j * x i * x j| := by
        exact Finset.sum_le_sum fun i _ => Finset.abs_sum_le_sum_abs _ _
    _ ≤ ∑ i, ∑ j, |A i j| * K ^ 2 := by
        refine Finset.sum_le_sum fun i _ => Finset.sum_le_sum fun j _ => ?_
        calc |A i j * x i * x j|
            = |A i j| * |x i| * |x j| := by rw [abs_mul, abs_mul]
          _ ≤ |A i j| * K * K := by
              have hprod : |x i| * |x j| ≤ K * K :=
                mul_le_mul (hx i) (hx j) (abs_nonneg _) hK
              calc |A i j| * |x i| * |x j|
                  = |A i j| * (|x i| * |x j|) := by ring
                _ ≤ |A i j| * (K * K) :=
                    mul_le_mul_of_nonneg_left hprod (abs_nonneg _)
                _ = |A i j| * K * K := by ring
          _ = |A i j| * K ^ 2 := by ring
    _ = K ^ 2 * ∑ i, ∑ j, |A i j| := by
        rw [Finset.mul_sum]
        apply Finset.sum_congr rfl
        intro i _
        rw [Finset.mul_sum]
        apply Finset.sum_congr rfl
        intro j _
        ring

lemma randomQuadraticForm_aemeasurable {μ : Measure Ω} {n : ℕ}
    (A : Matrix (Fin n) (Fin n) ℝ) {X : Fin n → Ω → ℝ}
    (hX_meas : ∀ i, AEMeasurable (X i) μ) :
    AEMeasurable (randomQuadraticForm A X) μ := by
  unfold randomQuadraticForm quadraticForm
  exact Finset.aemeasurable_fun_sum _ fun i _ =>
    Finset.aemeasurable_fun_sum _ fun j _ =>
      (((hX_meas i).mul (hX_meas j)).const_mul (A i j)).congr
        (ae_of_all _ fun ω => by ring)

lemma randomQuadraticForm_mem_Icc_of_ae_coord_bound {μ : Measure Ω} {n : ℕ}
    (A : Matrix (Fin n) (Fin n) ℝ) {X : Fin n → Ω → ℝ} {K : ℝ}
    (hK : 0 ≤ K) (hX_bound : ∀ i, ∀ᵐ ω ∂μ, |X i ω| ≤ K) :
    ∀ᵐ ω ∂μ, randomQuadraticForm A X ω ∈
      Set.Icc (-(K ^ 2 * entrywiseL1Norm A)) (K ^ 2 * entrywiseL1Norm A) := by
  rw [← ae_all_iff] at hX_bound
  filter_upwards [hX_bound] with ω hω
  have h_abs := abs_quadraticForm_le A hK (fun i => hω i)
  exact abs_le.mp h_abs

lemma bounded_hansonWright_cgf_constant_le {n : ℕ}
    (A : Matrix (Fin n) (Fin n) ℝ) {K C : ℝ}
    (hC_bound : 2 * entrywiseL1Norm A ^ 2 ≤ C * frobeniusNorm A ^ 2) (l : ℝ) :
    2 * (K ^ 2 * entrywiseL1Norm A) ^ 2 * l ^ 2 ≤
      C * l ^ 2 * K ^ 4 * frobeniusNorm A ^ 2 := by
  have hmul : 0 ≤ K ^ 4 * l ^ 2 := by positivity
  calc 2 * (K ^ 2 * entrywiseL1Norm A) ^ 2 * l ^ 2
      = (2 * entrywiseL1Norm A ^ 2) * (K ^ 4 * l ^ 2) := by ring
    _ ≤ (C * frobeniusNorm A ^ 2) * (K ^ 4 * l ^ 2) := by
        exact mul_le_mul_of_nonneg_right hC_bound hmul
    _ = C * l ^ 2 * K ^ 4 * frobeniusNorm A ^ 2 := by ring

/-- The local quadratic CGF estimate used in the Hanson-Wright proof.

For `Y = Xᵀ A X - E Xᵀ A X`, this records
`cgf Y λ ≤ C λ² K⁴ ‖A‖_F²` for
`|λ| ≤ (2 C K² ‖A‖)⁻¹`, together with local exponential integrability. -/
def HasHansonWrightMGF {n : ℕ} (μ : Measure Ω) (A : Matrix (Fin n) (Fin n) ℝ)
    (X : Fin n → Ω → ℝ) (K C : ℝ) : Prop :=
  (∀ l : ℝ, |l| ≤ (2 * C * K ^ 2 * operatorNorm A)⁻¹ →
    cgf (centeredQuadraticForm μ A X) μ l ≤
      C * l ^ 2 * K ^ 4 * frobeniusNorm A ^ 2) ∧
  ∀ l : ℝ, |l| ≤ (2 * C * K ^ 2 * operatorNorm A)⁻¹ →
    Integrable (fun ω => exp (l * centeredQuadraticForm μ A X ω)) μ

/-- Convexity of `exp` gives the chord bound on a compact interval. -/
lemma exp_le_chord {R t x : ℝ} (hR : 0 < R) (hx : x ∈ Set.Icc (-R) R) :
    exp (t * x) ≤ ((R - x) / (2 * R)) * exp (t * (-R)) +
      ((x + R) / (2 * R)) * exp (t * R) := by
  let w : Fin 2 → ℝ :=
    fun i => if i = 0 then (R - x) / (2 * R) else (x + R) / (2 * R)
  let p : Fin 2 → ℝ := fun i => if i = 0 then t * (-R) else t * R
  have hden : 0 < 2 * R := by positivity
  have hw0 : 0 ≤ (R - x) / (2 * R) :=
    div_nonneg (sub_nonneg.mpr hx.2) hden.le
  have hw1 : 0 ≤ (x + R) / (2 * R) := by
    refine div_nonneg ?_ hden.le
    linarith [hx.1]
  have hw_nonneg : ∀ i ∈ (Finset.univ : Finset (Fin 2)), 0 ≤ w i := by
    intro i _
    fin_cases i
    · simpa [w] using hw0
    · simpa [w] using hw1
  have hw_sum : ∑ i : Fin 2, w i = 1 := by
    simp [w]
    field_simp [hR.ne']
    ring
  have hp_mem : ∀ i ∈ (Finset.univ : Finset (Fin 2)), p i ∈ (Set.univ : Set ℝ) := by
    simp
  have hconv := convexOn_exp.map_sum_le (s := Set.univ) (t := Finset.univ)
    (w := w) (p := p) hw_nonneg hw_sum hp_mem
  have harg : (∑ i : Fin 2, w i • p i) = t * x := by
    simp [w, p]
    field_simp [hR.ne']
    ring
  have hrhs : (∑ i : Fin 2, w i • exp (p i)) =
      ((R - x) / (2 * R)) * exp (t * (-R)) +
        ((x + R) / (2 * R)) * exp (t * R) := by
    simp [w, p]
  rw [harg, hrhs] at hconv
  simpa [smul_eq_mul] using hconv

/-- A self-contained bounded, centered MGF estimate.

This is the symmetric bounded-variable form of Hoeffding's lemma, proved locally
from the chord bound for `exp` and `cosh x ≤ exp (x² / 2)`. -/
lemma hasSubgaussianMGF_of_abs_le_of_integral_eq_zero {μ : Measure Ω}
    [IsProbabilityMeasure μ] {Y : Ω → ℝ} {R : ℝ}
    (hY_meas : AEMeasurable Y μ) (hR : 0 ≤ R)
    (hY_bound : ∀ᵐ ω ∂μ, |Y ω| ≤ R) (hY_center : ∫ ω, Y ω ∂μ = 0) :
    HasSubgaussianMGF Y ⟨R ^ 2, sq_nonneg R⟩ μ := by
  have hY_range : ∀ᵐ ω ∂μ, Y ω ∈ Set.Icc (-R) R := by
    filter_upwards [hY_bound] with ω hω
    exact abs_le.mp hω
  have hY_int : Integrable Y μ := Integrable.of_mem_Icc (-R) R hY_meas hY_range
  refine ⟨?_, ?_⟩
  · intro t
    exact integrable_exp_mul_of_mem_Icc hY_meas hY_range
  · intro t
    by_cases hR0 : R = 0
    · have hY_zero : Y =ᵐ[μ] 0 := by
        filter_upwards [hY_bound] with ω hω
        rw [hR0] at hω
        exact abs_eq_zero.mp (le_antisymm hω (abs_nonneg _))
      have hmgf : mgf Y μ t = 1 := by
        unfold mgf
        calc ∫ ω, exp (t * Y ω) ∂μ
            = ∫ _ω, (1 : ℝ) ∂μ := by
                apply integral_congr_ae
                filter_upwards [hY_zero] with ω hω
                simp [hω]
          _ = 1 := by simp
      rw [hmgf, hR0]
      exact one_le_exp (by positivity)
    · have hR_pos : 0 < R := lt_of_le_of_ne hR (Ne.symm hR0)
      let chord : Ω → ℝ := fun ω =>
        ((R - Y ω) / (2 * R)) * exp (t * (-R)) +
          ((Y ω + R) / (2 * R)) * exp (t * R)
      have hchord_int : Integrable chord μ := by
        dsimp [chord]
        exact ((((integrable_const R).sub hY_int).div_const (2 * R)).mul_const _).add
          (((hY_int.add (integrable_const R)).div_const (2 * R)).mul_const _)
      have hpoint : (fun ω => exp (t * Y ω)) ≤ᵐ[μ] chord := by
        filter_upwards [hY_range] with ω hω
        exact exp_le_chord hR_pos hω
      have hmono : mgf Y μ t ≤ ∫ ω, chord ω ∂μ := by
        unfold mgf
        exact integral_mono_ae (integrable_exp_mul_of_mem_Icc hY_meas hY_range)
          hchord_int hpoint
      have hleft_int :
          Integrable (fun ω => (R - Y ω) / (2 * R) * exp (t * (-R))) μ :=
        (((integrable_const R).sub hY_int).div_const (2 * R)).mul_const _
      have hright_int :
          Integrable (fun ω => (Y ω + R) / (2 * R) * exp (t * R)) μ :=
        ((hY_int.add (integrable_const R)).div_const (2 * R)).mul_const _
      have hchord_eq : ∫ ω, chord ω ∂μ = cosh (t * R) := by
        dsimp [chord]
        rw [integral_add hleft_int hright_int]
        rw [integral_mul_const, integral_mul_const]
        rw [integral_div, integral_div]
        rw [integral_sub (integrable_const R) hY_int,
          integral_add hY_int (integrable_const R)]
        simp [hY_center, Real.cosh_eq, mul_comm]
        field_simp [hR_pos.ne']
        ring
      calc mgf Y μ t
          ≤ ∫ ω, chord ω ∂μ := hmono
        _ = cosh (t * R) := hchord_eq
        _ ≤ exp ((t * R) ^ 2 / 2) := Real.cosh_le_exp_half_sq _
        _ = exp ((R ^ 2) * t ^ 2 / 2) := by ring_nf

/-- A two-sided exponential consequence of a sub-Gaussian MGF bound. -/
lemma integral_cosh_mul_le_of_hasSubgaussianMGF {μ : Measure Ω} [IsProbabilityMeasure μ]
    {X : Ω → ℝ} {c : ℝ≥0} (h : HasSubgaussianMGF X c μ) (t : ℝ) :
    ∫ ω, cosh (t * X ω) ∂μ ≤ exp ((c : ℝ) * t ^ 2 / 2) := by
  have hint_pos : Integrable (fun ω => exp (t * X ω)) μ :=
    h.integrable_exp_mul t
  have hint_neg : Integrable (fun ω => exp (-(t * X ω))) μ := by
    convert h.integrable_exp_mul (-t) using 1
    ext ω
    ring_nf
  have h_eq :
      ∫ ω, cosh (t * X ω) ∂μ =
        (mgf X μ t + mgf X μ (-t)) / 2 := by
    calc ∫ ω, cosh (t * X ω) ∂μ
        = ∫ ω, (exp (t * X ω) + exp (-(t * X ω))) / 2 ∂μ := by
          apply integral_congr_ae
          filter_upwards with ω
          rw [Real.cosh_eq]
      _ = (∫ ω, exp (t * X ω) + exp (-(t * X ω)) ∂μ) / 2 := by
          rw [integral_div]
      _ = (mgf X μ t + mgf X μ (-t)) / 2 := by
          rw [integral_add hint_pos hint_neg]
          simp [mgf]
  rw [h_eq]
  have hpos_le : mgf X μ t ≤ exp ((c : ℝ) * t ^ 2 / 2) :=
    h.mgf_le t
  have hneg_le : mgf X μ (-t) ≤ exp ((c : ℝ) * t ^ 2 / 2) := by
    simpa [sq] using h.mgf_le (-t)
  nlinarith [hpos_le, hneg_le]

/-- Each nonnegative Taylor term of `cosh` is bounded by `cosh` itself. -/
lemma cosh_taylor_term_le (y : ℝ) (m : ℕ) :
    y ^ (2 * m) / (Nat.factorial (2 * m) : ℝ) ≤ cosh y := by
  rw [Real.cosh_eq_tsum]
  have hs := (Real.hasSum_cosh y).summable
  have hnonneg : ∀ n : ℕ, 0 ≤ y ^ (2 * n) / (Nat.factorial (2 * n) : ℝ) := by
    intro n
    exact div_nonneg (Even.pow_nonneg (even_two.mul_right n) y)
      (Nat.cast_nonneg (Nat.factorial (2 * n)))
  have hsingle :=
    hs.sum_le_tsum ({m} : Finset ℕ) (fun n _ => hnonneg n)
  simpa using hsingle

/-- Sub-Gaussian MGF control bounds every integrated even Taylor term. -/
lemma integral_even_taylor_term_le_of_hasSubgaussianMGF {μ : Measure Ω}
    [IsProbabilityMeasure μ] {X : Ω → ℝ} {c : ℝ≥0}
    (h : HasSubgaussianMGF X c μ) (t : ℝ) (m : ℕ) :
    ∫ ω, (t * X ω) ^ (2 * m) / (Nat.factorial (2 * m) : ℝ) ∂μ
      ≤ exp ((c : ℝ) * t ^ 2 / 2) := by
  have hint_pos : Integrable (fun ω => exp (t * X ω)) μ :=
    h.integrable_exp_mul t
  have hint_neg : Integrable (fun ω => exp (-(t * X ω))) μ := by
    convert h.integrable_exp_mul (-t) using 1
    ext ω
    ring_nf
  have hterm_int :
      Integrable (fun ω => (t * X ω) ^ (2 * m) / (Nat.factorial (2 * m) : ℝ)) μ := by
    have hpow :
        Integrable (fun ω => (t * X ω) ^ (2 * m)) μ := by
      exact integrable_pow_of_integrable_exp_mul
        (X := fun ω => t * X ω) (t := 1) one_ne_zero
        (by
          convert hint_pos using 1
          ext ω
          ring_nf)
        (by
          convert hint_neg using 1
          ext ω
          ring_nf)
        (2 * m)
    exact hpow.div_const _
  have hcosh_int : Integrable (fun ω => cosh (t * X ω)) μ := by
    have hsum : Integrable (fun ω => exp (t * X ω) + exp (-(t * X ω))) μ :=
      hint_pos.add hint_neg
    convert hsum.div_const 2 using 1
    ext ω
    rw [Real.cosh_eq]
  calc ∫ ω, (t * X ω) ^ (2 * m) / (Nat.factorial (2 * m) : ℝ) ∂μ
      ≤ ∫ ω, cosh (t * X ω) ∂μ := by
        exact integral_mono_ae hterm_int hcosh_int
          (ae_of_all _ fun ω => cosh_taylor_term_le (t * X ω) m)
    _ ≤ exp ((c : ℝ) * t ^ 2 / 2) :=
        integral_cosh_mul_le_of_hasSubgaussianMGF h t

/-- Even moment bound obtained from sub-Gaussian MGF control at an arbitrary scale. -/
lemma integral_even_power_le_of_hasSubgaussianMGF {μ : Measure Ω}
    [IsProbabilityMeasure μ] {X : Ω → ℝ} {c : ℝ≥0}
    (h : HasSubgaussianMGF X c μ) {s : ℝ} (hs : 0 < s) (m : ℕ) :
    ∫ ω, X ω ^ (2 * m) ∂μ
      ≤ (Nat.factorial (2 * m) : ℝ) / s ^ (2 * m) *
        exp ((c : ℝ) * s ^ 2 / 2) := by
  have hle := integral_even_taylor_term_le_of_hasSubgaussianMGF h s m
  have hscale :
      ∫ ω, (s * X ω) ^ (2 * m) / (Nat.factorial (2 * m) : ℝ) ∂μ =
        (s ^ (2 * m) / (Nat.factorial (2 * m) : ℝ)) *
          ∫ ω, X ω ^ (2 * m) ∂μ := by
    calc ∫ ω, (s * X ω) ^ (2 * m) / (Nat.factorial (2 * m) : ℝ) ∂μ
        = ∫ ω, (s ^ (2 * m) / (Nat.factorial (2 * m) : ℝ)) *
            X ω ^ (2 * m) ∂μ := by
          apply integral_congr_ae
          filter_upwards with ω
          rw [mul_pow]
          ring
      _ = (s ^ (2 * m) / (Nat.factorial (2 * m) : ℝ)) *
          ∫ ω, X ω ^ (2 * m) ∂μ := by
          rw [integral_const_mul]
  rw [hscale] at hle
  have hcoef_pos :
      0 < s ^ (2 * m) / (Nat.factorial (2 * m) : ℝ) := by
    positivity
  have hbound :
      ∫ ω, X ω ^ (2 * m) ∂μ
        ≤ exp ((c : ℝ) * s ^ 2 / 2) /
          (s ^ (2 * m) / (Nat.factorial (2 * m) : ℝ)) := by
    rw [le_div_iff₀ hcoef_pos]
    nlinarith [hle]
  calc ∫ ω, X ω ^ (2 * m) ∂μ
      ≤ exp ((c : ℝ) * s ^ 2 / 2) /
          (s ^ (2 * m) / (Nat.factorial (2 * m) : ℝ)) := hbound
    _ = (Nat.factorial (2 * m) : ℝ) / s ^ (2 * m) *
        exp ((c : ℝ) * s ^ 2 / 2) := by
        field_simp [ne_of_gt hs, Nat.cast_ne_zero.mpr (Nat.factorial_ne_zero (2 * m))]

/-- Pointwise power-series expansion of `exp (θ x²)`. -/
lemma exp_mul_sq_eq_tsum (θ x : ℝ) :
    exp (θ * x ^ 2) =
      ∑' m : ℕ, θ ^ m * x ^ (2 * m) / (Nat.factorial m : ℝ) := by
  rw [show exp (θ * x ^ 2) = ∑' m : ℕ, (θ * x ^ 2) ^ m / (Nat.factorial m : ℝ) by
    rw [Real.exp_eq_exp_ℝ, NormedSpace.exp_eq_tsum_div]]
  congr with m
  rw [mul_pow, ← pow_mul]

/-- A factorial estimate used to sum the square-exponential moment series. -/
lemma factorial_two_mul_div_factorial_le_succ_pow (m : ℕ) :
    (Nat.factorial (2 * m) : ℝ) / (Nat.factorial m : ℝ) ≤
      ((2 * m + 1 : ℕ) : ℝ) ^ m := by
  have hnat : Nat.factorial (2 * m) ≤ (2 * m + 1) ^ m * Nat.factorial m := by
    calc Nat.factorial (2 * m)
        ≤ (2 * m) ^ m * Nat.factorial m := Nat.factorial_two_mul_le m
      _ ≤ (2 * m + 1) ^ m * Nat.factorial m := by
          exact Nat.mul_le_mul_right _ (Nat.pow_le_pow_left (Nat.le_succ _) m)
  have hfac_pos : 0 < (Nat.factorial m : ℝ) := by positivity
  rw [div_le_iff₀ hfac_pos]
  exact_mod_cast hnat

/-- A geometric bound for each integrated square-exponential Taylor term. -/
lemma integral_exp_sq_series_term_le_of_hasSubgaussianMGF {μ : Measure Ω}
    [IsProbabilityMeasure μ] {X : Ω → ℝ} {c : ℝ≥0}
    (h : HasSubgaussianMGF X c μ) {θ : ℝ} (hθ : 0 ≤ θ) (m : ℕ) :
    ∫ ω, θ ^ m * X ω ^ (2 * m) / (Nat.factorial m : ℝ) ∂μ
      ≤ exp 1 * (θ * ((c : ℝ) + 1) * exp 1) ^ m := by
  let q : ℝ := ((2 * m + 1 : ℕ) : ℝ) / ((c : ℝ) + 1)
  let s : ℝ := sqrt q
  have hc1_pos : 0 < (c : ℝ) + 1 := by positivity
  have hq_pos : 0 < q := by
    dsimp [q]
    positivity
  have hq_nonneg : 0 ≤ q := hq_pos.le
  have hs_pos : 0 < s := by
    dsimp [s]
    exact sqrt_pos.mpr hq_pos
  have hs_pow : s ^ (2 * m) = q ^ m := by
    dsimp [s]
    rw [pow_mul, sq_sqrt hq_nonneg]
  have hs_sq : s ^ 2 = q := by
    dsimp [s]
    rw [sq_sqrt hq_nonneg]
  have hmoment := integral_even_power_le_of_hasSubgaussianMGF h hs_pos m
  have hterm_eq :
      ∫ ω, θ ^ m * X ω ^ (2 * m) / (Nat.factorial m : ℝ) ∂μ =
        (θ ^ m / (Nat.factorial m : ℝ)) * ∫ ω, X ω ^ (2 * m) ∂μ := by
    calc ∫ ω, θ ^ m * X ω ^ (2 * m) / (Nat.factorial m : ℝ) ∂μ
        = ∫ ω, (θ ^ m / (Nat.factorial m : ℝ)) * X ω ^ (2 * m) ∂μ := by
          apply integral_congr_ae
          filter_upwards with ω
          field_simp [Nat.cast_ne_zero.mpr (Nat.factorial_ne_zero m)]
      _ = (θ ^ m / (Nat.factorial m : ℝ)) * ∫ ω, X ω ^ (2 * m) ∂μ := by
          rw [integral_const_mul]
  rw [hterm_eq]
  have hcoef_nonneg : 0 ≤ θ ^ m / (Nat.factorial m : ℝ) := by
    positivity
  calc (θ ^ m / (Nat.factorial m : ℝ)) * ∫ ω, X ω ^ (2 * m) ∂μ
      ≤ (θ ^ m / (Nat.factorial m : ℝ)) *
          ((Nat.factorial (2 * m) : ℝ) / s ^ (2 * m) *
            exp ((c : ℝ) * s ^ 2 / 2)) := by
        exact mul_le_mul_of_nonneg_left hmoment hcoef_nonneg
    _ = θ ^ m *
          (((Nat.factorial (2 * m) : ℝ) / (Nat.factorial m : ℝ)) / q ^ m) *
            exp ((c : ℝ) * q / 2) := by
        rw [hs_pow, hs_sq]
        field_simp [Nat.cast_ne_zero.mpr (Nat.factorial_ne_zero m)]
    _ ≤ θ ^ m * (((c : ℝ) + 1) ^ m) * exp (m + 1 : ℝ) := by
        have hfac := factorial_two_mul_div_factorial_le_succ_pow m
        have hratio :
            ((Nat.factorial (2 * m) : ℝ) / (Nat.factorial m : ℝ)) / q ^ m
              ≤ ((c : ℝ) + 1) ^ m := by
          have hnum_pos : 0 < (((2 * m + 1 : ℕ) : ℝ) ^ m) := by positivity
          have hc1pow_nonneg : 0 ≤ ((c : ℝ) + 1) ^ m := by positivity
          calc
            ((Nat.factorial (2 * m) : ℝ) / (Nat.factorial m : ℝ)) / q ^ m
                = ((Nat.factorial (2 * m) : ℝ) / (Nat.factorial m : ℝ)) *
                  (((c : ℝ) + 1) ^ m / (((2 * m + 1 : ℕ) : ℝ) ^ m)) := by
                    dsimp [q]
                    rw [div_pow]
                    field_simp [hc1_pos.ne']
            _ ≤ (((2 * m + 1 : ℕ) : ℝ) ^ m) *
                  (((c : ℝ) + 1) ^ m / (((2 * m + 1 : ℕ) : ℝ) ^ m)) := by
                    exact mul_le_mul_of_nonneg_right hfac
                      (div_nonneg hc1pow_nonneg hnum_pos.le)
            _ = ((c : ℝ) + 1) ^ m := by
                    field_simp [ne_of_gt hnum_pos]
        have hexp_le : exp ((c : ℝ) * q / 2) ≤ exp (m + 1 : ℝ) := by
          apply exp_le_exp.mpr
          have hc_le : (c : ℝ) / ((c : ℝ) + 1) ≤ 1 := by
            rw [div_le_one hc1_pos]
            linarith
          have hnon : 0 ≤ ((2 * m + 1 : ℕ) : ℝ) / 2 := by positivity
          calc
            (c : ℝ) * q / 2 =
                ((c : ℝ) / ((c : ℝ) + 1)) * (((2 * m + 1 : ℕ) : ℝ) / 2) := by
                  dsimp [q]
                  field_simp [hc1_pos.ne']
            _ ≤ 1 * (((2 * m + 1 : ℕ) : ℝ) / 2) :=
                mul_le_mul_of_nonneg_right hc_le hnon
            _ ≤ (m + 1 : ℝ) := by
                norm_num
                nlinarith
        calc
          θ ^ m * (((Nat.factorial (2 * m) : ℝ) / (Nat.factorial m : ℝ)) / q ^ m) *
              exp ((c : ℝ) * q / 2)
              ≤ θ ^ m * ((c : ℝ) + 1) ^ m * exp ((c : ℝ) * q / 2) := by
                gcongr
          _ ≤ θ ^ m * ((c : ℝ) + 1) ^ m * exp (m + 1 : ℝ) := by
                gcongr
    _ = exp 1 * (θ * ((c : ℝ) + 1) * exp 1) ^ m := by
        rw [show (m + 1 : ℝ) = (m : ℝ) + 1 by norm_num, exp_add]
        rw [show exp (m : ℝ) = exp 1 ^ m by
          rw [show (m : ℝ) = (m : ℝ) * 1 by ring, Real.exp_nat_mul]]
        rw [mul_pow, mul_pow]
        ring

/-- A geometric Taylor-term bound using any positive real proxy above the sub-Gaussian parameter. -/
lemma integral_exp_sq_series_term_le_of_hasSubgaussianMGF_of_le {μ : Measure Ω}
    [IsProbabilityMeasure μ] {X : Ω → ℝ} {c : ℝ≥0}
    (h : HasSubgaussianMGF X c μ) {θ C0 : ℝ} (hθ : 0 ≤ θ)
    (hC0 : 0 < C0) (hc_le : (c : ℝ) ≤ C0) (m : ℕ) :
    ∫ ω, θ ^ m * X ω ^ (2 * m) / (Nat.factorial m : ℝ) ∂μ
      ≤ exp 1 * (θ * C0 * exp 1) ^ m := by
  let q : ℝ := ((2 * m + 1 : ℕ) : ℝ) / C0
  let s : ℝ := sqrt q
  have hq_pos : 0 < q := by
    dsimp [q]
    positivity
  have hq_nonneg : 0 ≤ q := hq_pos.le
  have hs_pos : 0 < s := by
    dsimp [s]
    exact sqrt_pos.mpr hq_pos
  have hs_pow : s ^ (2 * m) = q ^ m := by
    dsimp [s]
    rw [pow_mul, sq_sqrt hq_nonneg]
  have hs_sq : s ^ 2 = q := by
    dsimp [s]
    rw [sq_sqrt hq_nonneg]
  have hmoment := integral_even_power_le_of_hasSubgaussianMGF h hs_pos m
  have hterm_eq :
      ∫ ω, θ ^ m * X ω ^ (2 * m) / (Nat.factorial m : ℝ) ∂μ =
        (θ ^ m / (Nat.factorial m : ℝ)) * ∫ ω, X ω ^ (2 * m) ∂μ := by
    calc ∫ ω, θ ^ m * X ω ^ (2 * m) / (Nat.factorial m : ℝ) ∂μ
        = ∫ ω, (θ ^ m / (Nat.factorial m : ℝ)) * X ω ^ (2 * m) ∂μ := by
          apply integral_congr_ae
          filter_upwards with ω
          field_simp [Nat.cast_ne_zero.mpr (Nat.factorial_ne_zero m)]
      _ = (θ ^ m / (Nat.factorial m : ℝ)) * ∫ ω, X ω ^ (2 * m) ∂μ := by
          rw [integral_const_mul]
  rw [hterm_eq]
  have hcoef_nonneg : 0 ≤ θ ^ m / (Nat.factorial m : ℝ) := by
    positivity
  calc (θ ^ m / (Nat.factorial m : ℝ)) * ∫ ω, X ω ^ (2 * m) ∂μ
      ≤ (θ ^ m / (Nat.factorial m : ℝ)) *
          ((Nat.factorial (2 * m) : ℝ) / s ^ (2 * m) *
            exp ((c : ℝ) * s ^ 2 / 2)) := by
        exact mul_le_mul_of_nonneg_left hmoment hcoef_nonneg
    _ = θ ^ m *
          (((Nat.factorial (2 * m) : ℝ) / (Nat.factorial m : ℝ)) / q ^ m) *
            exp ((c : ℝ) * q / 2) := by
        rw [hs_pow, hs_sq]
        field_simp [Nat.cast_ne_zero.mpr (Nat.factorial_ne_zero m)]
    _ ≤ θ ^ m * C0 ^ m * exp (m + 1 : ℝ) := by
        have hfac := factorial_two_mul_div_factorial_le_succ_pow m
        have hratio :
            ((Nat.factorial (2 * m) : ℝ) / (Nat.factorial m : ℝ)) / q ^ m
              ≤ C0 ^ m := by
          have hnum_pos : 0 < (((2 * m + 1 : ℕ) : ℝ) ^ m) := by positivity
          have hC0pow_nonneg : 0 ≤ C0 ^ m := by positivity
          calc
            ((Nat.factorial (2 * m) : ℝ) / (Nat.factorial m : ℝ)) / q ^ m
                = ((Nat.factorial (2 * m) : ℝ) / (Nat.factorial m : ℝ)) *
                  (C0 ^ m / (((2 * m + 1 : ℕ) : ℝ) ^ m)) := by
                    dsimp [q]
                    rw [div_pow]
                    field_simp [hC0.ne']
            _ ≤ (((2 * m + 1 : ℕ) : ℝ) ^ m) *
                  (C0 ^ m / (((2 * m + 1 : ℕ) : ℝ) ^ m)) := by
                    exact mul_le_mul_of_nonneg_right hfac
                      (div_nonneg hC0pow_nonneg hnum_pos.le)
            _ = C0 ^ m := by
                    field_simp [ne_of_gt hnum_pos]
        have hexp_le : exp ((c : ℝ) * q / 2) ≤ exp (m + 1 : ℝ) := by
          apply exp_le_exp.mpr
          have hc_div_le : (c : ℝ) / C0 ≤ 1 := by
            rw [div_le_one hC0]
            exact hc_le
          have hnon : 0 ≤ ((2 * m + 1 : ℕ) : ℝ) / 2 := by positivity
          calc
            (c : ℝ) * q / 2 =
                ((c : ℝ) / C0) * (((2 * m + 1 : ℕ) : ℝ) / 2) := by
                  dsimp [q]
                  field_simp [hC0.ne']
            _ ≤ 1 * (((2 * m + 1 : ℕ) : ℝ) / 2) :=
                mul_le_mul_of_nonneg_right hc_div_le hnon
            _ ≤ (m + 1 : ℝ) := by
                norm_num
                nlinarith
        calc
          θ ^ m * (((Nat.factorial (2 * m) : ℝ) / (Nat.factorial m : ℝ)) / q ^ m) *
              exp ((c : ℝ) * q / 2)
              ≤ θ ^ m * C0 ^ m * exp ((c : ℝ) * q / 2) := by
                gcongr
          _ ≤ θ ^ m * C0 ^ m * exp (m + 1 : ℝ) := by
                gcongr
    _ = exp 1 * (θ * C0 * exp 1) ^ m := by
        rw [show (m + 1 : ℝ) = (m : ℝ) + 1 by norm_num, exp_add]
        rw [show exp (m : ℝ) = exp 1 ^ m by
          rw [show (m : ℝ) = (m : ℝ) * 1 by ring, Real.exp_nat_mul]]
        rw [mul_pow, mul_pow]
        ring

/-- The square-exponential Taylor integrals are summable below the explicit radius. -/
lemma summable_integral_exp_sq_series_of_hasSubgaussianMGF {μ : Measure Ω}
    [IsProbabilityMeasure μ] {X : Ω → ℝ} {c : ℝ≥0}
    (h : HasSubgaussianMGF X c μ) {θ : ℝ} (hθ : 0 ≤ θ)
    (hθ_small : θ * ((c : ℝ) + 1) * exp 1 < 1) :
    Summable fun m : ℕ =>
      ∫ ω, θ ^ m * X ω ^ (2 * m) / (Nat.factorial m : ℝ) ∂μ := by
  set r : ℝ := θ * ((c : ℝ) + 1) * exp 1 with hr_def
  have hr_nonneg : 0 ≤ r := by
    rw [hr_def]
    positivity
  have hgeom : Summable fun m : ℕ => exp 1 * r ^ m :=
    (summable_geometric_of_lt_one hr_nonneg (by simpa [r, hr_def] using hθ_small)).mul_left
      (exp 1)
  refine Summable.of_nonneg_of_le ?_ ?_ hgeom
  · intro m
    exact integral_nonneg_of_ae (ae_of_all _ fun ω => by
      exact div_nonneg
        (mul_nonneg (pow_nonneg hθ m) (Even.pow_nonneg (even_two.mul_right m) (X ω)))
        (Nat.cast_nonneg (Nat.factorial m)))
  · intro m
    simpa [r, hr_def] using integral_exp_sq_series_term_le_of_hasSubgaussianMGF h hθ m

/-- Summability of square-exponential Taylor integrals using a positive proxy `C0 ≥ c`. -/
lemma summable_integral_exp_sq_series_of_hasSubgaussianMGF_of_le {μ : Measure Ω}
    [IsProbabilityMeasure μ] {X : Ω → ℝ} {c : ℝ≥0}
    (h : HasSubgaussianMGF X c μ) {θ C0 : ℝ} (hθ : 0 ≤ θ)
    (hC0 : 0 < C0) (hc_le : (c : ℝ) ≤ C0)
    (hθ_small : θ * C0 * exp 1 < 1) :
    Summable fun m : ℕ =>
      ∫ ω, θ ^ m * X ω ^ (2 * m) / (Nat.factorial m : ℝ) ∂μ := by
  set r : ℝ := θ * C0 * exp 1 with hr_def
  have hr_nonneg : 0 ≤ r := by
    rw [hr_def]
    positivity
  have hgeom : Summable fun m : ℕ => exp 1 * r ^ m :=
    (summable_geometric_of_lt_one hr_nonneg (by simpa [r, hr_def] using hθ_small)).mul_left
      (exp 1)
  refine Summable.of_nonneg_of_le ?_ ?_ hgeom
  · intro m
    exact integral_nonneg_of_ae (ae_of_all _ fun ω => by
      exact div_nonneg
        (mul_nonneg (pow_nonneg hθ m) (Even.pow_nonneg (even_two.mul_right m) (X ω)))
        (Nat.cast_nonneg (Nat.factorial m)))
  · intro m
    simpa [r, hr_def] using
      integral_exp_sq_series_term_le_of_hasSubgaussianMGF_of_le h hθ hC0 hc_le m

/-- Integral form of the square-exponential Taylor expansion. -/
lemma integral_exp_mul_sq_eq_tsum_integrals {μ : Measure Ω}
    [IsProbabilityMeasure μ] {X : Ω → ℝ} {c : ℝ≥0}
    (h : HasSubgaussianMGF X c μ) {θ : ℝ} (hθ : 0 ≤ θ)
    (hsum : Summable fun m : ℕ =>
      ∫ ω, θ ^ m * X ω ^ (2 * m) / (Nat.factorial m : ℝ) ∂μ) :
    ∫ ω, exp (θ * X ω ^ 2) ∂μ =
      ∑' m : ℕ, ∫ ω, θ ^ m * X ω ^ (2 * m) / (Nat.factorial m : ℝ) ∂μ := by
  let F : ℕ → Ω → ℝ :=
    fun m ω => θ ^ m * X ω ^ (2 * m) / (Nat.factorial m : ℝ)
  have hF_nonneg : ∀ m : ℕ, 0 ≤ᵐ[μ] F m := by
    intro m
    exact ae_of_all _ fun ω => by
      dsimp [F]
      exact div_nonneg
        (mul_nonneg (pow_nonneg hθ m) (Even.pow_nonneg (even_two.mul_right m) (X ω)))
        (Nat.cast_nonneg (Nat.factorial m))
  have hF_int : ∀ m : ℕ, Integrable (F m) μ := by
    intro m
    have hpow : Integrable (fun ω => X ω ^ (2 * m)) μ := by
      exact integrable_pow_of_integrable_exp_mul
        (X := X) (t := 1) one_ne_zero
        (by simpa using h.integrable_exp_mul 1)
        (by simpa using h.integrable_exp_mul (-1))
        (2 * m)
    dsimp [F]
    exact (hpow.const_mul (θ ^ m)).div_const _
  have hF_sum_norm : Summable fun m : ℕ => ∫ ω, ‖F m ω‖ ∂μ := by
    convert hsum using 1
    ext m
    apply integral_congr_ae
    filter_upwards [hF_nonneg m] with ω hω
    rw [Real.norm_of_nonneg hω]
  have htsum := integral_tsum_of_summable_integral_norm hF_int hF_sum_norm
  change ∫ ω, exp (θ * X ω ^ 2) ∂μ = ∑' m : ℕ, ∫ ω, F m ω ∂μ
  rw [htsum]
  apply integral_congr_ae
  filter_upwards with ω
  dsimp [F]
  rw [exp_mul_sq_eq_tsum θ (X ω)]

/-- A quantitative square-exponential integral bound below the proxy radius. -/
lemma integral_exp_mul_sq_le_inv_of_hasSubgaussianMGF_of_le {μ : Measure Ω}
    [IsProbabilityMeasure μ] {X : Ω → ℝ} {c : ℝ≥0}
    (h : HasSubgaussianMGF X c μ) {θ C0 : ℝ} (hθ : 0 ≤ θ)
    (hC0 : 0 < C0) (hc_le : (c : ℝ) ≤ C0)
    (hθ_small : θ * C0 * exp 1 < 1) :
    ∫ ω, exp (θ * X ω ^ 2) ∂μ ≤ exp 1 * (1 - θ * C0 * exp 1)⁻¹ := by
  set r : ℝ := θ * C0 * exp 1 with hr_def
  have hr_nonneg : 0 ≤ r := by rw [hr_def]; positivity
  have hr_lt : r < 1 := by simpa [r, hr_def] using hθ_small
  have hsum :=
    summable_integral_exp_sq_series_of_hasSubgaussianMGF_of_le h hθ hC0 hc_le hθ_small
  have hgeom : Summable fun m : ℕ => exp 1 * r ^ m :=
    (summable_geometric_of_lt_one hr_nonneg hr_lt).mul_left (exp 1)
  rw [integral_exp_mul_sq_eq_tsum_integrals h hθ hsum]
  calc
    (∑' m : ℕ, ∫ ω, θ ^ m * X ω ^ (2 * m) / (Nat.factorial m : ℝ) ∂μ)
        ≤ ∑' m : ℕ, exp 1 * r ^ m := by
          refine Summable.tsum_le_tsum ?_ hsum hgeom
          intro m
          simpa [r, hr_def] using
            integral_exp_sq_series_term_le_of_hasSubgaussianMGF_of_le h hθ hC0 hc_le m
    _ = exp 1 * (1 - r)⁻¹ := by
          rw [tsum_mul_left]
          rw [tsum_geometric_of_lt_one hr_nonneg hr_lt]
    _ = exp 1 * (1 - θ * C0 * exp 1)⁻¹ := by rw [hr_def]

/-- A sharper square-exponential bound with the exact zeroth Taylor term isolated. -/
lemma integral_exp_mul_sq_le_one_add_tail_of_hasSubgaussianMGF_of_le {μ : Measure Ω}
    [IsProbabilityMeasure μ] {X : Ω → ℝ} {c : ℝ≥0}
    (h : HasSubgaussianMGF X c μ) {θ C0 : ℝ} (hθ : 0 ≤ θ)
    (hC0 : 0 < C0) (hc_le : (c : ℝ) ≤ C0)
    (hθ_small : θ * C0 * exp 1 < 1) :
    ∫ ω, exp (θ * X ω ^ 2) ∂μ ≤
      1 + exp 1 * ((θ * C0 * exp 1) * (1 - θ * C0 * exp 1)⁻¹) := by
  set r : ℝ := θ * C0 * exp 1 with hr_def
  let term : ℕ → ℝ :=
    fun m => ∫ ω, θ ^ m * X ω ^ (2 * m) / (Nat.factorial m : ℝ) ∂μ
  have hr_nonneg : 0 ≤ r := by rw [hr_def]; positivity
  have hr_lt : r < 1 := by simpa [r, hr_def] using hθ_small
  have hsum : Summable term := by
    simpa [term] using
      summable_integral_exp_sq_series_of_hasSubgaussianMGF_of_le h hθ hC0 hc_le hθ_small
  have hterm0 : term 0 = 1 := by
    dsimp [term]
    simp
  have htail_sum : Summable fun n : ℕ => term (n + 1) := by
    exact (summable_nat_add_iff (f := term) 1).mpr hsum
  have hgeom_tail : Summable fun n : ℕ => exp 1 * r ^ (n + 1) := by
    have hgeom_base : Summable fun n : ℕ => r ^ n :=
      summable_geometric_of_lt_one hr_nonneg hr_lt
    have hgeom_shift : Summable fun n : ℕ => r ^ (n + 1) :=
      (summable_nat_add_iff (f := fun n : ℕ => r ^ n) 1).mpr hgeom_base
    exact hgeom_shift.mul_left (exp 1)
  rw [integral_exp_mul_sq_eq_tsum_integrals h hθ (by simpa [term] using hsum)]
  change (∑' m : ℕ, term m) ≤ 1 + exp 1 * (r * (1 - r)⁻¹)
  rw [hsum.tsum_eq_zero_add, hterm0]
  gcongr
  calc
    (∑' n : ℕ, term (n + 1)) ≤ ∑' n : ℕ, exp 1 * r ^ (n + 1) := by
      refine Summable.tsum_le_tsum ?_ htail_sum hgeom_tail
      intro n
      simpa [term, r, hr_def] using
        integral_exp_sq_series_term_le_of_hasSubgaussianMGF_of_le h hθ hC0 hc_le (n + 1)
    _ = exp 1 * (r * (1 - r)⁻¹) := by
      rw [tsum_mul_left]
      rw [show (∑' n : ℕ, r ^ (n + 1)) = r * (1 - r)⁻¹ by
        have hgeom_base : Summable fun n : ℕ => r ^ n :=
          summable_geometric_of_lt_one hr_nonneg hr_lt
        have hsplit := hgeom_base.sum_add_tsum_nat_add 1
        rw [Finset.sum_range_one, tsum_geometric_of_lt_one hr_nonneg hr_lt] at hsplit
        have hshift : (∑' n : ℕ, r ^ (n + 1)) = (1 - r)⁻¹ - 1 := by
          linarith
        rw [hshift]
        have hden : 1 - r ≠ 0 := by linarith
        field_simp [hden]
        ring]

/-- A square-exponential bound with the zeroth and first Taylor terms isolated. -/
lemma integral_exp_mul_sq_le_one_add_linear_add_tail_of_hasSubgaussianMGF_of_le
    {μ : Measure Ω} [IsProbabilityMeasure μ] {X : Ω → ℝ} {c : ℝ≥0}
    (h : HasSubgaussianMGF X c μ) {θ C0 : ℝ} (hθ : 0 ≤ θ)
    (hC0 : 0 < C0) (hc_le : (c : ℝ) ≤ C0)
    (hθ_small : θ * C0 * exp 1 < 1) :
    ∫ ω, exp (θ * X ω ^ 2) ∂μ ≤
      1 + θ * ∫ ω, X ω ^ 2 ∂μ +
        exp 1 * (((θ * C0 * exp 1) ^ 2) * (1 - θ * C0 * exp 1)⁻¹) := by
  set r : ℝ := θ * C0 * exp 1 with hr_def
  let term : ℕ → ℝ :=
    fun m => ∫ ω, θ ^ m * X ω ^ (2 * m) / (Nat.factorial m : ℝ) ∂μ
  have hr_nonneg : 0 ≤ r := by rw [hr_def]; positivity
  have hr_lt : r < 1 := by simpa [r, hr_def] using hθ_small
  have hsum : Summable term := by
    simpa [term] using
      summable_integral_exp_sq_series_of_hasSubgaussianMGF_of_le h hθ hC0 hc_le hθ_small
  have hterm0 : term 0 = 1 := by
    dsimp [term]
    simp
  have hterm1 : term 1 = θ * ∫ ω, X ω ^ 2 ∂μ := by
    dsimp [term]
    simp
    rw [integral_const_mul]
  have htail_sum : Summable fun n : ℕ => term (n + 2) := by
    exact (summable_nat_add_iff (f := term) 2).mpr hsum
  have hgeom_tail : Summable fun n : ℕ => exp 1 * r ^ (n + 2) := by
    have hgeom_base : Summable fun n : ℕ => r ^ n :=
      summable_geometric_of_lt_one hr_nonneg hr_lt
    have hgeom_shift : Summable fun n : ℕ => r ^ (n + 2) :=
      (summable_nat_add_iff (f := fun n : ℕ => r ^ n) 2).mpr hgeom_base
    exact hgeom_shift.mul_left (exp 1)
  rw [integral_exp_mul_sq_eq_tsum_integrals h hθ (by simpa [term] using hsum)]
  change (∑' m : ℕ, term m) ≤
    1 + θ * ∫ ω, X ω ^ 2 ∂μ + exp 1 * (r ^ 2 * (1 - r)⁻¹)
  have hsplit := hsum.sum_add_tsum_nat_add 2
  rw [Finset.sum_range_succ, Finset.sum_range_one, hterm0, hterm1] at hsplit
  rw [← hsplit]
  gcongr
  calc
    (∑' n : ℕ, term (n + 2)) ≤ ∑' n : ℕ, exp 1 * r ^ (n + 2) := by
      refine Summable.tsum_le_tsum ?_ htail_sum hgeom_tail
      intro n
      simpa [term, r, hr_def] using
        integral_exp_sq_series_term_le_of_hasSubgaussianMGF_of_le h hθ hC0 hc_le (n + 2)
    _ = exp 1 * (r ^ 2 * (1 - r)⁻¹) := by
      rw [tsum_mul_left]
      rw [show (∑' n : ℕ, r ^ (n + 2)) = r ^ 2 * (1 - r)⁻¹ by
        have hgeom_base : Summable fun n : ℕ => r ^ n :=
          summable_geometric_of_lt_one hr_nonneg hr_lt
        have hsplit2 := hgeom_base.sum_add_tsum_nat_add 2
        rw [Finset.sum_range_succ, Finset.sum_range_one,
          tsum_geometric_of_lt_one hr_nonneg hr_lt] at hsplit2
        have hshift : (∑' n : ℕ, r ^ (n + 2)) = (1 - r)⁻¹ - (1 + r) := by
          linarith
        rw [hshift]
        have hden : 1 - r ≠ 0 := by linarith
        field_simp [hden]
        ring]

/-- A linear-in-`θ` square-exponential bound at half the explicit radius. -/
lemma integral_exp_mul_sq_le_exp_linear_of_hasSubgaussianMGF_of_le {μ : Measure Ω}
    [IsProbabilityMeasure μ] {X : Ω → ℝ} {c : ℝ≥0}
    (h : HasSubgaussianMGF X c μ) {θ C0 : ℝ} (hθ : 0 ≤ θ)
    (hC0 : 0 < C0) (hc_le : (c : ℝ) ≤ C0)
    (hθ_half : θ * C0 * exp 1 ≤ 1 / 2) :
    ∫ ω, exp (θ * X ω ^ 2) ∂μ ≤ exp (2 * exp 1 * (θ * C0 * exp 1)) := by
  set r : ℝ := θ * C0 * exp 1 with hr_def
  have hr_nonneg : 0 ≤ r := by rw [hr_def]; positivity
  have hr_half : r ≤ 1 / 2 := by simpa [r, hr_def] using hθ_half
  have hr_lt : r < 1 := by linarith
  have hbase :=
    integral_exp_mul_sq_le_one_add_tail_of_hasSubgaussianMGF_of_le h hθ hC0 hc_le
      (by simpa [r, hr_def] using hr_lt)
  have hden_pos : 0 < 1 - r := by linarith
  have hinv_le : (1 - r)⁻¹ ≤ 2 := by
    rw [inv_le_comm₀ hden_pos two_pos]
    linarith
  have htail :
      exp 1 * (r * (1 - r)⁻¹) ≤ 2 * exp 1 * r := by
    have hmul : r * (1 - r)⁻¹ ≤ r * 2 :=
      mul_le_mul_of_nonneg_left hinv_le hr_nonneg
    calc
      exp 1 * (r * (1 - r)⁻¹) ≤ exp 1 * (r * 2) :=
        mul_le_mul_of_nonneg_left hmul (exp_pos 1).le
      _ = 2 * exp 1 * r := by ring
  calc
    ∫ ω, exp (θ * X ω ^ 2) ∂μ
        ≤ 1 + exp 1 * (r * (1 - r)⁻¹) := by
          simpa [r, hr_def] using hbase
    _ ≤ 1 + 2 * exp 1 * r := by linarith
    _ ≤ exp (2 * exp 1 * r) := by
          simpa [add_comm] using Real.add_one_le_exp (2 * exp 1 * r)
    _ = exp (2 * exp 1 * (θ * C0 * exp 1)) := by rw [hr_def]

/-- Gaussian square-form bound for a positive symmetric operator, proved by spectral
diagonalization and one-dimensional square-exponential estimates. -/
lemma integral_exp_quadratic_stdGaussian_le {E : Type*} [NormedAddCommGroup E]
    [InnerProductSpace ℝ E] [FiniteDimensional ℝ E] [MeasurableSpace E] [BorelSpace E]
    (T : E →L[ℝ] E) (hTpos : T.toLinearMap.IsPositive) {θ : ℝ} (hθ : 0 ≤ θ)
    (hsmall : ∀ i : Fin (Module.finrank ℝ E),
      θ * hTpos.isSymmetric.eigenvalues rfl i * exp 1 ≤ 1 / 2) :
    ∫ x : E, exp (θ * inner ℝ (T x) x) ∂(stdGaussian E) ≤
      exp (2 * exp 1 ^ 2 * θ *
        (∑ i : Fin (Module.finrank ℝ E), hTpos.isSymmetric.eigenvalues rfl i)) := by
  let hsym := hTpos.isSymmetric
  let b := hsym.eigenvectorBasis (by rfl : Module.finrank ℝ E = Module.finrank ℝ E)
  let lam : Fin (Module.finrank ℝ E) → ℝ := fun i => hsym.eigenvalues rfl i
  have hnonneg : ∀ i : Fin (Module.finrank ℝ E), 0 ≤ lam i := by
    intro i
    simpa [lam, hsym] using hTpos.nonneg_eigenvalues
      (by rfl : Module.finrank ℝ E = Module.finrank ℝ E) i
  have hsynth_aemeas :
      AEMeasurable (fun z : Fin (Module.finrank ℝ E) → ℝ => ∑ i, z i • b i)
        (Measure.pi fun _ : Fin (Module.finrank ℝ E) => gaussianReal 0 1) := by
    exact (continuous_finsetSum _ fun i _ =>
      (continuous_apply i).smul continuous_const).aemeasurable
  have hf_aesm :
      AEStronglyMeasurable (fun x : E => exp (θ * inner ℝ (T x) x))
        (Measure.map (fun z : Fin (Module.finrank ℝ E) → ℝ => ∑ i, z i • b i)
          (Measure.pi fun _ : Fin (Module.finrank ℝ E) => gaussianReal 0 1)) := by
    fun_prop
  rw [stdGaussian_eq_map_pi_orthonormalBasis b]
  rw [integral_map hsynth_aemeas hf_aesm]
  have hpoint : ∀ z : Fin (Module.finrank ℝ E) → ℝ,
      exp (θ * inner ℝ (T (∑ i, z i • b i)) (∑ i, z i • b i)) =
        ∏ i : Fin (Module.finrank ℝ E), exp ((θ * lam i) * z i ^ 2) := by
    intro z
    change exp (θ * inner ℝ (T.toLinearMap (∑ i, z i • b i)) (∑ i, z i • b i)) =
      ∏ i : Fin (Module.finrank ℝ E), exp ((θ * lam i) * z i ^ 2)
    have hquad :=
      symmetric_inner_apply_eq_sum_eigenvalues_repr T.toLinearMap hsym (∑ i, z i • b i)
    rw [hquad]
    have hsum_eq :
        (∑ i : Fin (Module.finrank ℝ E),
            hsym.eigenvalues rfl i * (hsym.eigenvectorBasis rfl).repr
              (∑ i, z i • b i) i ^ 2) =
          ∑ i : Fin (Module.finrank ℝ E), lam i * z i ^ 2 := by
      apply Finset.sum_congr rfl
      intro i _
      have hcoord :
          (hsym.eigenvectorBasis rfl).repr (∑ i, z i • b i) i = z i := by
        change b.repr (∑ i, z i • b i) i = z i
        exact orthonormalBasis_repr_sum_smul b z i
      rw [hcoord]
    rw [hsum_eq]
    rw [← Real.exp_sum]
    congr 1
    rw [Finset.mul_sum]
    apply Finset.sum_congr rfl
    intro i _
    ring
  calc
    ∫ (x : Fin (Module.finrank ℝ E) → ℝ),
        exp (θ * inner ℝ (T (∑ i, x i • b i)) (∑ i, x i • b i))
        ∂Measure.pi (fun x => gaussianReal 0 1)
        = ∫ (x : Fin (Module.finrank ℝ E) → ℝ),
          ∏ i : Fin (Module.finrank ℝ E), exp ((θ * lam i) * x i ^ 2)
        ∂Measure.pi (fun x => gaussianReal 0 1) := by
          apply integral_congr_ae
          filter_upwards with z
          exact hpoint z
    _ = ∏ i : Fin (Module.finrank ℝ E),
          ∫ x : ℝ, exp ((θ * lam i) * x ^ 2) ∂gaussianReal 0 1 := by
          simpa using (integral_fintype_prod_eq_prod (𝕜 := ℝ)
            (ι := Fin (Module.finrank ℝ E)) (E := fun _ => ℝ)
            (f := fun i x => exp ((θ * lam i) * x ^ 2))
            (μ := fun _ => gaussianReal 0 1))
    _ ≤ ∏ i : Fin (Module.finrank ℝ E),
          exp (2 * exp 1 * ((θ * lam i) * 1 * exp 1)) := by
          apply Finset.prod_le_prod
          · intro i _
            exact integral_nonneg_of_ae (ae_of_all _ fun x => exp_nonneg _)
          · intro i _
            have hθi : 0 ≤ θ * lam i := mul_nonneg hθ (hnonneg i)
            have hle := integral_exp_mul_sq_le_exp_linear_of_hasSubgaussianMGF_of_le
              (μ := gaussianReal 0 1) hasSubgaussianMGF_id_gaussianReal_zero_one
              (θ := θ * lam i) (C0 := 1) hθi (by norm_num) (by norm_num) ?_
            · simpa [id_eq] using hle
            · simpa [mul_assoc] using hsmall i
    _ = exp (2 * exp 1 ^ 2 * θ *
          (∑ i : Fin (Module.finrank ℝ E), lam i)) := by
          rw [← Real.exp_sum]
          congr 1
          rw [Finset.mul_sum]
          apply Finset.sum_congr rfl
          intro i _
          ring

lemma matrix_trace_conjTranspose_mul_self_eq_frobeniusNormSq {n : ℕ}
    (A : Matrix (Fin n) (Fin n) ℝ) :
    (A.conjTranspose * A).trace = frobeniusNormSq A := by
  unfold Matrix.trace frobeniusNormSq Matrix.diag
  simp only [Matrix.mul_apply, Matrix.conjTranspose_apply]
  simp only [star_trivial]
  rw [Finset.sum_comm]
  apply Finset.sum_congr rfl
  intro i _
  apply Finset.sum_congr rfl
  intro j _
  ring

lemma toEuclideanCLM_adjoint {n : ℕ} (A : Matrix (Fin n) (Fin n) ℝ) :
    ContinuousLinearMap.adjoint (Matrix.toEuclideanCLM (𝕜 := ℝ) A) =
      Matrix.toEuclideanCLM (𝕜 := ℝ) A.conjTranspose := by
  rw [show Matrix.toEuclideanCLM (𝕜 := ℝ) A =
    LinearMap.toContinuousLinearMap (Matrix.toEuclideanLin A) by rfl]
  rw [← LinearMap.adjoint_toContinuousLinearMap]
  rw [← Matrix.toEuclideanLin_conjTranspose_eq_adjoint]
  rfl

lemma operatorNorm_conjTranspose {n : ℕ} (A : Matrix (Fin n) (Fin n) ℝ) :
    operatorNorm A.conjTranspose = operatorNorm A := by
  unfold operatorNorm
  rw [← toEuclideanCLM_adjoint A]
  exact ContinuousLinearMap.adjoint.norm_map _

lemma frobeniusNormSq_conjTranspose {n : ℕ} (A : Matrix (Fin n) (Fin n) ℝ) :
    frobeniusNormSq A.conjTranspose = frobeniusNormSq A := by
  unfold frobeniusNormSq
  simp only [Matrix.conjTranspose_apply, star_trivial]
  rw [Finset.sum_comm]

lemma frobeniusNorm_conjTranspose_sq {n : ℕ} (A : Matrix (Fin n) (Fin n) ℝ) :
    frobeniusNorm A.conjTranspose ^ 2 = frobeniusNorm A ^ 2 := by
  rw [frobeniusNorm_sq, frobeniusNorm_sq, frobeniusNormSq_conjTranspose]

lemma trace_adjoint_comp_toEuclideanCLM_eq_frobeniusNormSq {n : ℕ}
    (A : Matrix (Fin n) (Fin n) ℝ) :
    (LinearMap.trace ℝ (EuclideanSpace ℝ (Fin n)))
      ((ContinuousLinearMap.adjoint (Matrix.toEuclideanCLM (𝕜 := ℝ) A) ∘L
        Matrix.toEuclideanCLM (𝕜 := ℝ) A).toLinearMap) = frobeniusNormSq A := by
  let T : EuclideanSpace ℝ (Fin n) →L[ℝ] EuclideanSpace ℝ (Fin n) :=
    Matrix.toEuclideanCLM (𝕜 := ℝ) A
  let S : EuclideanSpace ℝ (Fin n) →L[ℝ] EuclideanSpace ℝ (Fin n) :=
    ContinuousLinearMap.adjoint T ∘L T
  change (LinearMap.trace ℝ (EuclideanSpace ℝ (Fin n))) S.toLinearMap = frobeniusNormSq A
  rw [LinearMap.trace_eq_sum_inner S.toLinearMap (EuclideanSpace.basisFun (Fin n) ℝ)]
  unfold frobeniusNormSq
  rw [Finset.sum_comm]
  apply Finset.sum_congr rfl
  intro j _
  dsimp [S, T]
  rw [EuclideanSpace.basisFun_apply]
  rw [Matrix.toEuclideanCLM_toLp]
  rw [toEuclideanCLM_adjoint]
  rw [Matrix.toEuclideanCLM_toLp]
  rw [PiLp.inner_apply]
  simp only [PiLp.single_apply, Matrix.conjTranspose_apply, star_trivial, Matrix.mulVec,
    dotProduct]
  rw [Finset.sum_eq_single j]
  · simp
    apply Finset.sum_congr rfl
    intro i _
    ring
  · intro i _ hij
    simp [hij]
  · intro hj
    exact (hj (Finset.mem_univ j)).elim

lemma eigenvalue_adjoint_comp_le_opNorm_sq {E : Type*} [NormedAddCommGroup E]
    [InnerProductSpace ℝ E] [FiniteDimensional ℝ E]
    (T : E →L[ℝ] E) (i : Fin (Module.finrank ℝ E)) :
    let S : E →L[ℝ] E := ContinuousLinearMap.adjoint T ∘L T
    let hSpos : S.toLinearMap.IsPositive :=
      (ContinuousLinearMap.isPositive_adjoint_comp_self T).toLinearMap
    hSpos.isSymmetric.eigenvalues rfl i ≤ ‖T‖ ^ 2 := by
  intro S hSpos
  let hsym := hSpos.isSymmetric
  let b := hsym.eigenvectorBasis (by rfl : Module.finrank ℝ E = Module.finrank ℝ E)
  let e := b i
  let lam := hsym.eigenvalues rfl i
  have heig : S.toLinearMap e = lam • e := by
    exact hsym.apply_eigenvectorBasis (by rfl : Module.finrank ℝ E = Module.finrank ℝ E) i
  have hinner : inner ℝ (S e) e = lam := by
    rw [show S e = S.toLinearMap e by rfl, heig]
    rw [inner_smul_left]
    rw [inner_self_eq_norm_sq_to_K]
    simp [e, b]
  have hnorm_sq : ‖T e‖ ^ 2 = inner ℝ (S e) e := by
    simpa [S] using ContinuousLinearMap.apply_norm_sq_eq_inner_adjoint_left T e
  have hnorm_le : ‖T e‖ ≤ ‖T‖ := by
    have h := T.le_opNorm e
    simpa [e, b, OrthonormalBasis.norm_eq_one] using h
  have hnorm_nonneg : 0 ≤ ‖T e‖ := norm_nonneg _
  have hop_nonneg : 0 ≤ ‖T‖ := norm_nonneg _
  calc
    lam = ‖T e‖ ^ 2 := by rw [← hinner, ← hnorm_sq]
    _ ≤ ‖T‖ ^ 2 := by nlinarith

/-- A second-moment bound from the square-exponential Taylor-term estimate. -/
lemma integral_sq_le_of_hasSubgaussianMGF_of_le {μ : Measure Ω}
    [IsProbabilityMeasure μ] {X : Ω → ℝ} {c : ℝ≥0}
    (h : HasSubgaussianMGF X c μ) {C0 : ℝ}
    (hC0 : 0 < C0) (hc_le : (c : ℝ) ≤ C0) :
    ∫ ω, X ω ^ 2 ∂μ ≤ exp 1 * (C0 * exp 1) := by
  have hterm :=
    integral_exp_sq_series_term_le_of_hasSubgaussianMGF_of_le h
      (θ := 1) (C0 := C0) (by norm_num) hC0 hc_le 1
  simpa using hterm

/-- A fourth-moment bound from the square-exponential Taylor-term estimate. -/
lemma integral_fourth_le_of_hasSubgaussianMGF_of_le {μ : Measure Ω}
    [IsProbabilityMeasure μ] {X : Ω → ℝ} {c : ℝ≥0}
    (h : HasSubgaussianMGF X c μ) {C0 : ℝ}
    (hC0 : 0 < C0) (hc_le : (c : ℝ) ≤ C0) :
    ∫ ω, X ω ^ 4 ∂μ ≤ 2 * exp 1 * (C0 * exp 1) ^ 2 := by
  have hterm :=
    integral_exp_sq_series_term_le_of_hasSubgaussianMGF_of_le h
      (θ := 1) (C0 := C0) (by norm_num) hC0 hc_le 2
  have hfac : (Nat.factorial 2 : ℝ) = 2 := by norm_num
  have hterm' : ∫ ω, X ω ^ 4 / 2 ∂μ ≤ exp 1 * (C0 * exp 1) ^ 2 := by
    simpa [hfac, pow_succ] using hterm
  have hscale : ∫ ω, X ω ^ 4 / 2 ∂μ = (1 / 2 : ℝ) * ∫ ω, X ω ^ 4 ∂μ := by
    calc
      ∫ ω, X ω ^ 4 / 2 ∂μ = ∫ ω, (1 / 2 : ℝ) * X ω ^ 4 ∂μ := by
        apply integral_congr_ae
        filter_upwards with ω
        ring
      _ = (1 / 2 : ℝ) * ∫ ω, X ω ^ 4 ∂μ := by rw [integral_const_mul]
  rw [hscale] at hterm'
  nlinarith

/-- A global quadratic upper bound for the negative exponential on the nonnegative half-line. -/
lemma exp_neg_le_one_sub_add_sq {u : ℝ} (hu : 0 ≤ u) :
    exp (-u) ≤ 1 - u + u ^ 2 := by
  have hden_pos : 0 < 1 + u := by linarith
  have hpoly_pos : 0 < 1 - u + u ^ 2 := by nlinarith [sq_nonneg (u - 1 / 2)]
  have h_exp_ge : 1 + u ≤ exp u := by
    simpa [add_comm] using Real.add_one_le_exp u
  have hinv : (exp u)⁻¹ ≤ (1 + u)⁻¹ :=
    (inv_le_inv₀ (exp_pos u) hden_pos).2 h_exp_ge
  have hrat : (1 + u)⁻¹ ≤ 1 - u + u ^ 2 := by
    rw [inv_le_iff_one_le_mul₀ hden_pos]
    ring_nf
    nlinarith [pow_nonneg hu 3]
  calc
    exp (-u) = (exp u)⁻¹ := by rw [exp_neg]
    _ ≤ (1 + u)⁻¹ := hinv
    _ ≤ 1 - u + u ^ 2 := hrat

/-- The linear Taylor term cancels after multiplying by `exp (-u)`. -/
lemma exp_neg_mul_one_add_le_one (u : ℝ) :
    exp (-u) * (1 + u) ≤ 1 := by
  have h_exp_ge : 1 + u ≤ exp u := by
    simpa [add_comm] using Real.add_one_le_exp u
  calc
    exp (-u) * (1 + u) ≤ exp (-u) * exp u := by
      exact mul_le_mul_of_nonneg_left h_exp_ge (exp_nonneg _)
    _ = 1 := by
      rw [← exp_add]
      ring_nf
      simp

/-- Positive-parameter MGF bound for a centered square of a sub-Gaussian variable. -/
lemma integral_exp_centered_sq_le_exp_quadratic_nonneg {μ : Measure Ω}
    [IsProbabilityMeasure μ] {X : Ω → ℝ} {c : ℝ≥0}
    (h : HasSubgaussianMGF X c μ) {θ C0 : ℝ} (hθ : 0 ≤ θ)
    (hC0 : 0 < C0) (hc_le : (c : ℝ) ≤ C0)
    (hθ_half : θ * C0 * exp 1 ≤ 1 / 2) :
    ∫ ω, exp (θ * (X ω ^ 2 - ∫ ω, X ω ^ 2 ∂μ)) ∂μ ≤
      exp (2 * exp 1 * (θ * C0 * exp 1) ^ 2) := by
  set m : ℝ := ∫ ω, X ω ^ 2 ∂μ with hm_def
  set r : ℝ := θ * C0 * exp 1 with hr_def
  have hm_nonneg : 0 ≤ m := by
    rw [hm_def]
    exact integral_nonneg_of_ae (ae_of_all _ fun ω => sq_nonneg (X ω))
  have hr_nonneg : 0 ≤ r := by rw [hr_def]; positivity
  have hr_half : r ≤ 1 / 2 := by simpa [r, hr_def] using hθ_half
  have hr_lt : r < 1 := by linarith
  have hsmall : θ * C0 * exp 1 < 1 := by simpa [r, hr_def] using hr_lt
  have hbase :=
    integral_exp_mul_sq_le_one_add_linear_add_tail_of_hasSubgaussianMGF_of_le h hθ
      hC0 hc_le hsmall
  have hcenter :
      ∫ ω, exp (θ * (X ω ^ 2 - m)) ∂μ =
        exp (-θ * m) * ∫ ω, exp (θ * X ω ^ 2) ∂μ := by
    calc
      ∫ ω, exp (θ * (X ω ^ 2 - m)) ∂μ
          = ∫ ω, exp (-θ * m) * exp (θ * X ω ^ 2) ∂μ := by
            apply integral_congr_ae
            filter_upwards with ω
            rw [← exp_add]
            congr 1
            ring
      _ = exp (-θ * m) * ∫ ω, exp (θ * X ω ^ 2) ∂μ := by
            rw [integral_const_mul]
  have htail_nonneg :
      0 ≤ exp 1 * (r ^ 2 * (1 - r)⁻¹) := by
    have hden_pos : 0 < 1 - r := by linarith
    positivity
  have hmul_base :
      exp (-θ * m) * ∫ ω, exp (θ * X ω ^ 2) ∂μ ≤
        exp (-θ * m) *
          (1 + θ * m + exp 1 * (r ^ 2 * (1 - r)⁻¹)) := by
    rw [← hm_def] at hbase
    simpa [m, r, hr_def, add_assoc] using
      mul_le_mul_of_nonneg_left hbase (exp_nonneg _)
  have hmain :
      exp (-θ * m) *
          (1 + θ * m + exp 1 * (r ^ 2 * (1 - r)⁻¹))
        ≤ 1 + exp 1 * (r ^ 2 * (1 - r)⁻¹) := by
    have hexp_le_one : exp (-θ * m) ≤ 1 := by
      rw [Real.exp_le_one_iff]
      nlinarith [hθ, hm_nonneg]
    have htail_mul :
        exp (-θ * m) * (exp 1 * (r ^ 2 * (1 - r)⁻¹)) ≤
          exp 1 * (r ^ 2 * (1 - r)⁻¹) := by
      calc
        exp (-θ * m) * (exp 1 * (r ^ 2 * (1 - r)⁻¹))
            ≤ 1 * (exp 1 * (r ^ 2 * (1 - r)⁻¹)) := by
              exact mul_le_mul_of_nonneg_right hexp_le_one htail_nonneg
        _ = exp 1 * (r ^ 2 * (1 - r)⁻¹) := by ring
    calc
      exp (-θ * m) *
          (1 + θ * m + exp 1 * (r ^ 2 * (1 - r)⁻¹))
          = exp (-(θ * m)) * (1 + θ * m) +
              exp (-θ * m) * (exp 1 * (r ^ 2 * (1 - r)⁻¹)) := by ring_nf
      _ ≤ 1 + exp 1 * (r ^ 2 * (1 - r)⁻¹) :=
          add_le_add (exp_neg_mul_one_add_le_one (θ * m)) htail_mul
  have htail_le :
      exp 1 * (r ^ 2 * (1 - r)⁻¹) ≤ 2 * exp 1 * r ^ 2 := by
    have hden_pos : 0 < 1 - r := by linarith
    have hinv_le : (1 - r)⁻¹ ≤ 2 := by
      rw [inv_le_comm₀ hden_pos two_pos]
      linarith
    have hmul : r ^ 2 * (1 - r)⁻¹ ≤ r ^ 2 * 2 :=
      mul_le_mul_of_nonneg_left hinv_le (sq_nonneg r)
    calc
      exp 1 * (r ^ 2 * (1 - r)⁻¹) ≤ exp 1 * (r ^ 2 * 2) :=
        mul_le_mul_of_nonneg_left hmul (exp_pos 1).le
      _ = 2 * exp 1 * r ^ 2 := by ring
  calc
    ∫ ω, exp (θ * (X ω ^ 2 - ∫ ω, X ω ^ 2 ∂μ)) ∂μ
        = ∫ ω, exp (θ * (X ω ^ 2 - m)) ∂μ := by rw [hm_def]
    _ = exp (-θ * m) * ∫ ω, exp (θ * X ω ^ 2) ∂μ := hcenter
    _ ≤ exp (-θ * m) *
          (1 + θ * m + exp 1 * (r ^ 2 * (1 - r)⁻¹)) := hmul_base
    _ ≤ 1 + exp 1 * (r ^ 2 * (1 - r)⁻¹) := hmain
    _ ≤ 1 + 2 * exp 1 * r ^ 2 := by linarith
    _ ≤ exp (2 * exp 1 * r ^ 2) := by
      simpa [add_comm] using Real.add_one_le_exp (2 * exp 1 * r ^ 2)
    _ = exp (2 * exp 1 * (θ * C0 * exp 1) ^ 2) := by rw [hr_def]

/-- Negative-parameter MGF bound for a centered square of a sub-Gaussian variable. -/
lemma integral_exp_centered_sq_le_exp_quadratic_nonpos {μ : Measure Ω}
    [IsProbabilityMeasure μ] {X : Ω → ℝ} {c : ℝ≥0}
    (h : HasSubgaussianMGF X c μ) {θ C0 : ℝ} (hθ : θ ≤ 0)
    (hC0 : 0 < C0) (hc_le : (c : ℝ) ≤ C0) :
    ∫ ω, exp (θ * (X ω ^ 2 - ∫ ω, X ω ^ 2 ∂μ)) ∂μ ≤
      exp (2 * exp 1 * ((-θ) * C0 * exp 1) ^ 2) := by
  set a : ℝ := -θ with ha_def
  set m : ℝ := ∫ ω, X ω ^ 2 ∂μ with hm_def
  set M4 : ℝ := ∫ ω, X ω ^ 4 ∂μ with hM4_def
  have ha_nonneg : 0 ≤ a := by rw [ha_def]; linarith
  have hX2_int : Integrable (fun ω => X ω ^ 2) μ := by
    exact integrable_pow_of_integrable_exp_mul
      (X := X) (t := 1) one_ne_zero
      (by simpa using h.integrable_exp_mul 1)
      (by simpa using h.integrable_exp_mul (-1)) 2
  have hX4_int : Integrable (fun ω => X ω ^ 4) μ := by
    exact integrable_pow_of_integrable_exp_mul
      (X := X) (t := 1) one_ne_zero
      (by simpa using h.integrable_exp_mul 1)
      (by simpa using h.integrable_exp_mul (-1)) 4
  have hneg_int : Integrable (fun ω => exp (-a * X ω ^ 2)) μ := by
    exact Integrable.of_bound
      ((hX2_int.aemeasurable.const_mul (-a)).exp.aestronglyMeasurable) 1 (by
        filter_upwards with ω
        rw [Real.norm_eq_abs, abs_of_nonneg (exp_nonneg _)]
        rw [Real.exp_le_one_iff]
        exact mul_nonpos_of_nonpos_of_nonneg (neg_nonpos.mpr ha_nonneg) (sq_nonneg (X ω)))
  have hpoly_int :
      Integrable (fun ω => 1 - a * X ω ^ 2 + a ^ 2 * X ω ^ 4) μ :=
    ((integrable_const 1).sub (hX2_int.const_mul a)).add (hX4_int.const_mul (a ^ 2))
  have hneg_bound :
      ∫ ω, exp (-a * X ω ^ 2) ∂μ ≤ 1 - a * m + a ^ 2 * M4 := by
    have hpoint :
        (fun ω => exp (-a * X ω ^ 2)) ≤ᵐ[μ]
          fun ω => 1 - a * X ω ^ 2 + a ^ 2 * X ω ^ 4 := by
      filter_upwards with ω
      have hbasic := exp_neg_le_one_sub_add_sq
        (mul_nonneg ha_nonneg (sq_nonneg (X ω)))
      calc
        exp (-a * X ω ^ 2) = exp (-(a * X ω ^ 2)) := by ring_nf
        _ ≤ 1 - (a * X ω ^ 2) + (a * X ω ^ 2) ^ 2 := hbasic
        _ = 1 - a * X ω ^ 2 + a ^ 2 * X ω ^ 4 := by ring_nf
    calc
      ∫ ω, exp (-a * X ω ^ 2) ∂μ
          ≤ ∫ ω, 1 - a * X ω ^ 2 + a ^ 2 * X ω ^ 4 ∂μ :=
            integral_mono_ae hneg_int hpoly_int hpoint
      _ = 1 - a * m + a ^ 2 * M4 := by
        calc
          ∫ ω, 1 - a * X ω ^ 2 + a ^ 2 * X ω ^ 4 ∂μ
              = ∫ ω, (1 - a * X ω ^ 2) + a ^ 2 * X ω ^ 4 ∂μ := by rfl
          _ = ∫ ω, 1 - a * X ω ^ 2 ∂μ + ∫ ω, a ^ 2 * X ω ^ 4 ∂μ := by
              rw [show (fun ω => (1 - a * X ω ^ 2) + a ^ 2 * X ω ^ 4) =
                (fun ω => 1 - a * X ω ^ 2) + (fun ω => a ^ 2 * X ω ^ 4) by
                rfl]
              exact integral_add ((integrable_const 1).sub (hX2_int.const_mul a))
                (hX4_int.const_mul (a ^ 2))
          _ = (1 - a * m) + a ^ 2 * M4 := by
              rw [integral_sub (integrable_const 1) (hX2_int.const_mul a)]
              rw [integral_const_mul, integral_const_mul]
              simp [hm_def, hM4_def]
          _ = 1 - a * m + a ^ 2 * M4 := by ring_nf
  have hcenter :
      ∫ ω, exp (θ * (X ω ^ 2 - m)) ∂μ =
        exp (a * m) * ∫ ω, exp (-a * X ω ^ 2) ∂μ := by
    calc
      ∫ ω, exp (θ * (X ω ^ 2 - m)) ∂μ
          = ∫ ω, exp (a * m) * exp (-a * X ω ^ 2) ∂μ := by
            apply integral_congr_ae
            filter_upwards with ω
            rw [← exp_add]
            congr 1
            rw [ha_def]
            ring_nf
      _ = exp (a * m) * ∫ ω, exp (-a * X ω ^ 2) ∂μ := by
            rw [integral_const_mul]
  have hM4_bound : M4 ≤ 2 * exp 1 * (C0 * exp 1) ^ 2 := by
    rw [hM4_def]
    exact integral_fourth_le_of_hasSubgaussianMGF_of_le h hC0 hc_le
  have hmgf_le_exp_M4 :
      exp (a * m) * ∫ ω, exp (-a * X ω ^ 2) ∂μ ≤ exp (a ^ 2 * M4) := by
    calc
      exp (a * m) * ∫ ω, exp (-a * X ω ^ 2) ∂μ
          ≤ exp (a * m) * (1 - a * m + a ^ 2 * M4) := by
            exact mul_le_mul_of_nonneg_left hneg_bound (exp_nonneg _)
      _ ≤ exp (a * m) * exp (-a * m + a ^ 2 * M4) := by
            exact mul_le_mul_of_nonneg_left
              (by linarith [Real.add_one_le_exp (-a * m + a ^ 2 * M4)])
              (exp_nonneg _)
      _ = exp (a ^ 2 * M4) := by
            rw [← exp_add]
            ring_nf
  have hquad_le :
      a ^ 2 * M4 ≤ a ^ 2 * (2 * exp 1 * (C0 * exp 1) ^ 2) :=
    mul_le_mul_of_nonneg_left hM4_bound (sq_nonneg a)
  calc
    ∫ ω, exp (θ * (X ω ^ 2 - ∫ ω, X ω ^ 2 ∂μ)) ∂μ
        = ∫ ω, exp (θ * (X ω ^ 2 - m)) ∂μ := by rw [hm_def]
    _ = exp (a * m) * ∫ ω, exp (-a * X ω ^ 2) ∂μ := hcenter
    _ ≤ exp (a ^ 2 * M4) := hmgf_le_exp_M4
    _ ≤ exp (a ^ 2 * (2 * exp 1 * (C0 * exp 1) ^ 2)) :=
        exp_le_exp.mpr hquad_le
    _ = exp (2 * exp 1 * ((-θ) * C0 * exp 1) ^ 2) := by
        rw [ha_def]
        ring_nf

/-- Square-exponential integrability from summability of the nonnegative moment series. -/
lemma integrable_exp_mul_sq_of_summable_integrals {μ : Measure Ω}
    [IsProbabilityMeasure μ] {X : Ω → ℝ} {c : ℝ≥0}
    (h : HasSubgaussianMGF X c μ) {θ : ℝ} (hθ : 0 ≤ θ)
    (hsum : Summable fun m : ℕ =>
      ∫ ω, θ ^ m * X ω ^ (2 * m) / (Nat.factorial m : ℝ) ∂μ) :
    Integrable (fun ω => exp (θ * X ω ^ 2)) μ := by
  let F : ℕ → Ω → ℝ :=
    fun m ω => θ ^ m * X ω ^ (2 * m) / (Nat.factorial m : ℝ)
  have hF_nonneg : ∀ m : ℕ, 0 ≤ᵐ[μ] F m := by
    intro m
    exact ae_of_all _ fun ω => by
      dsimp [F]
      exact div_nonneg
        (mul_nonneg (pow_nonneg hθ m) (Even.pow_nonneg (even_two.mul_right m) (X ω)))
        (Nat.cast_nonneg (Nat.factorial m))
  have hF_int : ∀ m : ℕ, Integrable (F m) μ := by
    intro m
    have hpow : Integrable (fun ω => X ω ^ (2 * m)) μ := by
      exact integrable_pow_of_integrable_exp_mul
        (X := X) (t := 1) one_ne_zero
        (by simpa using h.integrable_exp_mul 1)
        (by simpa using h.integrable_exp_mul (-1))
        (2 * m)
    dsimp [F]
    exact (hpow.const_mul (θ ^ m)).div_const _
  have hF_integral_nonneg : ∀ m : ℕ, 0 ≤ ∫ ω, F m ω ∂μ := by
    intro m
    exact integral_nonneg_of_ae (hF_nonneg m)
  have hF_lintegral_eq :
      (∑' m : ℕ, ∫⁻ ω, ENNReal.ofReal (F m ω) ∂μ) =
        ENNReal.ofReal (∑' m : ℕ, ∫ ω, F m ω ∂μ) := by
    calc (∑' m : ℕ, ∫⁻ ω, ENNReal.ofReal (F m ω) ∂μ)
        = ∑' m : ℕ, ENNReal.ofReal (∫ ω, F m ω ∂μ) := by
          congr with m
          exact (ofReal_integral_eq_lintegral_ofReal (hF_int m) (hF_nonneg m)).symm
      _ = ENNReal.ofReal (∑' m : ℕ, ∫ ω, F m ω ∂μ) := by
          exact (ENNReal.ofReal_tsum_of_nonneg hF_integral_nonneg hsum).symm
  have h_exp_lintegral_eq :
      ∫⁻ ω, ENNReal.ofReal (exp (θ * X ω ^ 2)) ∂μ =
        ∑' m : ℕ, ∫⁻ ω, ENNReal.ofReal (F m ω) ∂μ := by
    calc ∫⁻ ω, ENNReal.ofReal (exp (θ * X ω ^ 2)) ∂μ
        = ∫⁻ ω, ∑' m : ℕ, ENNReal.ofReal (F m ω) ∂μ := by
          apply lintegral_congr_ae
          filter_upwards with ω
          have hsummable :
              Summable fun m : ℕ => F m ω := by
            convert Real.summable_pow_div_factorial (θ * X ω ^ 2) using 1
            ext m
            dsimp [F]
            rw [mul_pow, ← pow_mul]
          rw [exp_mul_sq_eq_tsum θ (X ω)]
          exact ENNReal.ofReal_tsum_of_nonneg
            (fun m => by
              show 0 ≤ θ ^ m * X ω ^ (2 * m) / (Nat.factorial m : ℝ)
              exact div_nonneg
                (mul_nonneg (pow_nonneg hθ m)
                  (Even.pow_nonneg (even_two.mul_right m) (X ω)))
                (Nat.cast_nonneg (Nat.factorial m))) hsummable
      _ = ∑' m : ℕ, ∫⁻ ω, ENNReal.ofReal (F m ω) ∂μ := by
          rw [lintegral_tsum]
          intro m
          exact (hF_int m).aemeasurable.ennreal_ofReal
  have h_exp_lintegral_ne_top :
      ∫⁻ ω, ENNReal.ofReal (exp (θ * X ω ^ 2)) ∂μ ≠ ⊤ := by
    rw [h_exp_lintegral_eq, hF_lintegral_eq]
    exact ENNReal.ofReal_ne_top
  exact (lintegral_ofReal_ne_top_iff_integrable
    ((h.aemeasurable.pow_const 2).const_mul θ).exp.aestronglyMeasurable
    (ae_of_all _ fun ω => (exp_pos _).le)).mp h_exp_lintegral_ne_top

/-- Square-exponential integrability for a sub-Gaussian variable at small positive parameter. -/
lemma integrable_exp_mul_sq_of_hasSubgaussianMGF {μ : Measure Ω}
    [IsProbabilityMeasure μ] {X : Ω → ℝ} {c : ℝ≥0}
    (h : HasSubgaussianMGF X c μ) {θ : ℝ} (hθ : 0 ≤ θ)
    (hθ_small : θ * ((c : ℝ) + 1) * exp 1 < 1) :
    Integrable (fun ω => exp (θ * X ω ^ 2)) μ :=
  integrable_exp_mul_sq_of_summable_integrals h hθ
    (summable_integral_exp_sq_series_of_hasSubgaussianMGF h hθ hθ_small)

/-- Small-parameter square-exponential integrability using a positive proxy `C0 ≥ c`. -/
lemma integrable_exp_mul_sq_of_hasSubgaussianMGF_of_le {μ : Measure Ω}
    [IsProbabilityMeasure μ] {X : Ω → ℝ} {c : ℝ≥0}
    (h : HasSubgaussianMGF X c μ) {θ C0 : ℝ} (hθ : 0 ≤ θ)
    (hC0 : 0 < C0) (hc_le : (c : ℝ) ≤ C0)
    (hθ_small : θ * C0 * exp 1 < 1) :
    Integrable (fun ω => exp (θ * X ω ^ 2)) μ :=
  integrable_exp_mul_sq_of_summable_integrals h hθ
    (summable_integral_exp_sq_series_of_hasSubgaussianMGF_of_le h hθ hC0 hc_le hθ_small)

/-- Gaussian square-form integrability for a positive symmetric operator, under the
strict version of the same coordinate smallness condition. -/
lemma integrable_exp_quadratic_stdGaussian {E : Type*} [NormedAddCommGroup E]
    [InnerProductSpace ℝ E] [FiniteDimensional ℝ E] [MeasurableSpace E] [BorelSpace E]
    (T : E →L[ℝ] E) (hTpos : T.toLinearMap.IsPositive) {θ : ℝ} (hθ : 0 ≤ θ)
    (hsmall : ∀ i : Fin (Module.finrank ℝ E),
      θ * hTpos.isSymmetric.eigenvalues rfl i * exp 1 < 1) :
    Integrable (fun x : E => exp (θ * inner ℝ (T x) x)) (stdGaussian E) := by
  let hsym := hTpos.isSymmetric
  let b := hsym.eigenvectorBasis (by rfl : Module.finrank ℝ E = Module.finrank ℝ E)
  let lam : Fin (Module.finrank ℝ E) → ℝ := fun i => hsym.eigenvalues rfl i
  have hnonneg : ∀ i : Fin (Module.finrank ℝ E), 0 ≤ lam i := by
    intro i
    simpa [lam, hsym] using hTpos.nonneg_eigenvalues
      (by rfl : Module.finrank ℝ E = Module.finrank ℝ E) i
  have hsynth_aemeas :
      AEMeasurable (fun z : Fin (Module.finrank ℝ E) → ℝ => ∑ i, z i • b i)
        (Measure.pi fun _ : Fin (Module.finrank ℝ E) => gaussianReal 0 1) := by
    exact (continuous_finsetSum _ fun i _ =>
      (continuous_apply i).smul continuous_const).aemeasurable
  have hf_aesm :
      AEStronglyMeasurable (fun x : E => exp (θ * inner ℝ (T x) x))
        (Measure.map (fun z : Fin (Module.finrank ℝ E) → ℝ => ∑ i, z i • b i)
          (Measure.pi fun _ : Fin (Module.finrank ℝ E) => gaussianReal 0 1)) := by
    fun_prop
  have hpoint : ∀ z : Fin (Module.finrank ℝ E) → ℝ,
      exp (θ * inner ℝ (T (∑ i, z i • b i)) (∑ i, z i • b i)) =
        ∏ i : Fin (Module.finrank ℝ E), exp ((θ * lam i) * z i ^ 2) := by
    intro z
    change exp (θ * inner ℝ (T.toLinearMap (∑ i, z i • b i)) (∑ i, z i • b i)) =
      ∏ i : Fin (Module.finrank ℝ E), exp ((θ * lam i) * z i ^ 2)
    have hquad :=
      symmetric_inner_apply_eq_sum_eigenvalues_repr T.toLinearMap hsym (∑ i, z i • b i)
    rw [hquad]
    have hsum_eq :
        (∑ i : Fin (Module.finrank ℝ E),
            hsym.eigenvalues rfl i * (hsym.eigenvectorBasis rfl).repr
              (∑ i, z i • b i) i ^ 2) =
          ∑ i : Fin (Module.finrank ℝ E), lam i * z i ^ 2 := by
      apply Finset.sum_congr rfl
      intro i _
      have hcoord :
          (hsym.eigenvectorBasis rfl).repr (∑ i, z i • b i) i = z i := by
        change b.repr (∑ i, z i • b i) i = z i
        exact orthonormalBasis_repr_sum_smul b z i
      rw [hcoord]
    rw [hsum_eq]
    rw [← Real.exp_sum]
    congr 1
    rw [Finset.mul_sum]
    apply Finset.sum_congr rfl
    intro i _
    ring
  rw [stdGaussian_eq_map_pi_orthonormalBasis b]
  rw [integrable_map_measure hf_aesm hsynth_aemeas]
  refine (MeasureTheory.Integrable.fintype_prod (𝕜 := ℝ)
    (ι := Fin (Module.finrank ℝ E)) (E := ℝ)
    (f := fun i x => exp ((θ * lam i) * x ^ 2))
    (μ := fun _ => gaussianReal 0 1) ?_).congr ?_
  · intro i
    have hθi : 0 ≤ θ * lam i := mul_nonneg hθ (hnonneg i)
    exact integrable_exp_mul_sq_of_hasSubgaussianMGF_of_le
      (μ := gaussianReal 0 1) hasSubgaussianMGF_id_gaussianReal_zero_one
      (θ := θ * lam i) (C0 := 1) hθi (by norm_num) (by norm_num) (by
        simpa [mul_assoc] using hsmall i)
  · filter_upwards with z
    exact (hpoint z).symm

/-- Integrability of the exponential of a Gaussian bilinear form. -/
lemma integrable_exp_inner_toEuclideanCLM_prod_stdGaussian {n : ℕ}
    (A : Matrix (Fin n) (Fin n) ℝ) {l : ℝ}
    (hsmall : (l ^ 2 / 2) * operatorNorm A ^ 2 * exp 1 < 1) :
    Integrable
      (fun p : EuclideanSpace ℝ (Fin n) × EuclideanSpace ℝ (Fin n) =>
        exp (l * inner ℝ (Matrix.toEuclideanCLM (𝕜 := ℝ) A p.1) p.2))
      ((stdGaussian (EuclideanSpace ℝ (Fin n))).prod
        (stdGaussian (EuclideanSpace ℝ (Fin n)))) := by
  let E := EuclideanSpace ℝ (Fin n)
  let γ : Measure E := stdGaussian E
  let T : E →L[ℝ] E := Matrix.toEuclideanCLM (𝕜 := ℝ) A
  let S : E →L[ℝ] E := ContinuousLinearMap.adjoint T ∘L T
  let hSpos : S.toLinearMap.IsPositive :=
    (ContinuousLinearMap.isPositive_adjoint_comp_self T).toLinearMap
  let f : E × E → ℝ := fun p => exp (l * inner ℝ (T p.1) p.2)
  have hf_aesm : AEStronglyMeasurable f (γ.prod γ) := by
    dsimp [f]
    fun_prop
  have hθ_nonneg : 0 ≤ l ^ 2 / 2 := by positivity
  have hsmall_eig :
      ∀ i : Fin (Module.finrank ℝ E),
        (l ^ 2 / 2) * hSpos.isSymmetric.eigenvalues rfl i * exp 1 < 1 := by
    intro i
    have heig_le : hSpos.isSymmetric.eigenvalues rfl i ≤ ‖T‖ ^ 2 := by
      simpa [S, hSpos] using eigenvalue_adjoint_comp_le_opNorm_sq T i
    have hcoef_nonneg : 0 ≤ l ^ 2 / 2 := by positivity
    have hstep :
        (l ^ 2 / 2) * hSpos.isSymmetric.eigenvalues rfl i * exp 1 ≤
          (l ^ 2 / 2) * ‖T‖ ^ 2 * exp 1 := by
      exact mul_le_mul_of_nonneg_right
        (mul_le_mul_of_nonneg_left heig_le hcoef_nonneg) (exp_nonneg 1)
    exact lt_of_le_of_lt hstep (by simpa [operatorNorm, T] using hsmall)
  have hquad_int :
      Integrable (fun y : E => exp ((l ^ 2 / 2) * inner ℝ (S y) y)) γ :=
    integrable_exp_quadratic_stdGaussian S hSpos hθ_nonneg hsmall_eig
  rw [integrable_prod_iff hf_aesm]
  constructor
  · filter_upwards with y
    simpa [f] using integrable_exp_innerSL_stdGaussian (E := E) (T y) l
  · have houter_eq :
        (fun y : E => ∫ x : E, ‖f (y, x)‖ ∂γ) =ᵐ[γ]
          fun y : E => exp ((l ^ 2 / 2) * inner ℝ (S y) y) := by
      filter_upwards with y
      have hinner :
          ∫ x : E, ‖f (y, x)‖ ∂γ =
            ∫ x : E, exp (l * inner ℝ (T y) x) ∂γ := by
        apply integral_congr_ae
        filter_upwards with x
        dsimp [f]
        rw [abs_of_nonneg (exp_nonneg _)]
      rw [hinner, integral_exp_innerSL_stdGaussian (E := E) (T y) l]
      have hnorm_sq : ‖T y‖ ^ 2 = inner ℝ (S y) y := by
        simpa [S] using ContinuousLinearMap.apply_norm_sq_eq_inner_adjoint_left T y
      rw [hnorm_sq]
      ring_nf
    exact hquad_int.congr houter_eq.symm

/-- A Frobenius-scale bound for the exponential moment of a Gaussian bilinear form. -/
lemma integral_exp_inner_toEuclideanCLM_prod_stdGaussian_le {n : ℕ}
    (A : Matrix (Fin n) (Fin n) ℝ) {l : ℝ}
    (hsmall : (l ^ 2 / 2) * operatorNorm A ^ 2 * exp 1 ≤ 1 / 2) :
    ∫ p : EuclideanSpace ℝ (Fin n) × EuclideanSpace ℝ (Fin n),
        exp (l * inner ℝ (Matrix.toEuclideanCLM (𝕜 := ℝ) A p.1) p.2)
      ∂((stdGaussian (EuclideanSpace ℝ (Fin n))).prod
        (stdGaussian (EuclideanSpace ℝ (Fin n)))) ≤
      exp (exp 1 ^ 2 * l ^ 2 * frobeniusNorm A ^ 2) := by
  let E := EuclideanSpace ℝ (Fin n)
  let γ : Measure E := stdGaussian E
  let T : E →L[ℝ] E := Matrix.toEuclideanCLM (𝕜 := ℝ) A
  let S : E →L[ℝ] E := ContinuousLinearMap.adjoint T ∘L T
  let hSpos : S.toLinearMap.IsPositive :=
    (ContinuousLinearMap.isPositive_adjoint_comp_self T).toLinearMap
  let f : E × E → ℝ := fun p => exp (l * inner ℝ (T p.1) p.2)
  have hf_int : Integrable f (γ.prod γ) :=
    integrable_exp_inner_toEuclideanCLM_prod_stdGaussian A (by
      exact lt_of_le_of_lt hsmall (by norm_num))
  have hθ_nonneg : 0 ≤ l ^ 2 / 2 := by positivity
  have hsmall_eig :
      ∀ i : Fin (Module.finrank ℝ E),
        (l ^ 2 / 2) * hSpos.isSymmetric.eigenvalues rfl i * exp 1 ≤ 1 / 2 := by
    intro i
    have heig_le : hSpos.isSymmetric.eigenvalues rfl i ≤ ‖T‖ ^ 2 := by
      simpa [S, hSpos] using eigenvalue_adjoint_comp_le_opNorm_sq T i
    have hcoef_nonneg : 0 ≤ l ^ 2 / 2 := by positivity
    have hstep :
        (l ^ 2 / 2) * hSpos.isSymmetric.eigenvalues rfl i * exp 1 ≤
          (l ^ 2 / 2) * ‖T‖ ^ 2 * exp 1 := by
      exact mul_le_mul_of_nonneg_right
        (mul_le_mul_of_nonneg_left heig_le hcoef_nonneg) (exp_nonneg 1)
    exact hstep.trans (by simpa [operatorNorm, T] using hsmall)
  have hsum_eq :
      (∑ i : Fin (Module.finrank ℝ E), hSpos.isSymmetric.eigenvalues rfl i) =
        frobeniusNormSq A := by
    have htrace_eigs :
        (LinearMap.trace ℝ E) S.toLinearMap =
          ∑ i : Fin (Module.finrank ℝ E), hSpos.isSymmetric.eigenvalues rfl i := by
      simpa using hSpos.isSymmetric.trace_eq_sum_eigenvalues
        (by rfl : Module.finrank ℝ E = Module.finrank ℝ E)
    have htrace_frob :
        (LinearMap.trace ℝ E) S.toLinearMap = frobeniusNormSq A := by
      simpa [E, S, T] using trace_adjoint_comp_toEuclideanCLM_eq_frobeniusNormSq A
    exact htrace_eigs.symm.trans htrace_frob
  have hinner_eq :
      (fun y : E => ∫ x : E, f (y, x) ∂γ) =ᵐ[γ]
        fun y : E => exp ((l ^ 2 / 2) * inner ℝ (S y) y) := by
    filter_upwards with y
    rw [integral_exp_innerSL_stdGaussian (E := E) (T y) l]
    have hnorm_sq : ‖T y‖ ^ 2 = inner ℝ (S y) y := by
      simpa [S] using ContinuousLinearMap.apply_norm_sq_eq_inner_adjoint_left T y
    rw [hnorm_sq]
    ring_nf
  calc
    ∫ p : E × E, f p ∂(γ.prod γ)
        = ∫ y : E, ∫ x : E, f (y, x) ∂γ ∂γ := by
          rw [integral_prod f hf_int]
    _ = ∫ y : E, exp ((l ^ 2 / 2) * inner ℝ (S y) y) ∂γ := by
          exact integral_congr_ae hinner_eq
    _ ≤ exp (2 * exp 1 ^ 2 * (l ^ 2 / 2) *
          (∑ i : Fin (Module.finrank ℝ E), hSpos.isSymmetric.eigenvalues rfl i)) :=
          integral_exp_quadratic_stdGaussian_le S hSpos hθ_nonneg hsmall_eig
    _ = exp (exp 1 ^ 2 * l ^ 2 * frobeniusNorm A ^ 2) := by
          rw [hsum_eq, frobeniusNorm_sq]
          congr 1
          ring

/-- Local exponential integrability for centered squares of sub-Gaussian variables. -/
lemma integrable_exp_centered_sq_of_hasSubgaussianMGF_of_le {μ : Measure Ω}
    [IsProbabilityMeasure μ] {X : Ω → ℝ} {c : ℝ≥0}
    (h : HasSubgaussianMGF X c μ) {θ C0 : ℝ}
    (hC0 : 0 < C0) (hc_le : (c : ℝ) ≤ C0)
    (hθ_abs : |θ| * C0 * exp 1 < 1) :
    Integrable (fun ω => exp (θ * (X ω ^ 2 - ∫ ω, X ω ^ 2 ∂μ))) μ := by
  set m : ℝ := ∫ ω, X ω ^ 2 ∂μ with hm_def
  rcases le_total 0 θ with hθ_nonneg | hθ_nonpos
  · have hsmall : θ * C0 * exp 1 < 1 := by
      simpa [abs_of_nonneg hθ_nonneg] using hθ_abs
    have hsq_int :=
      integrable_exp_mul_sq_of_hasSubgaussianMGF_of_le h hθ_nonneg hC0 hc_le hsmall
    convert hsq_int.const_mul (exp (-θ * m)) using 1
    ext ω
    rw [← exp_add]
    rw [hm_def]
    ring_nf
  · set a : ℝ := -θ with ha_def
    have ha_nonneg : 0 ≤ a := by rw [ha_def]; linarith
    have hX2_int : Integrable (fun ω => X ω ^ 2) μ := by
      exact integrable_pow_of_integrable_exp_mul
        (X := X) (t := 1) one_ne_zero
        (by simpa using h.integrable_exp_mul 1)
        (by simpa using h.integrable_exp_mul (-1)) 2
    have hneg_int : Integrable (fun ω => exp (-a * X ω ^ 2)) μ := by
      exact Integrable.of_bound
        ((hX2_int.aemeasurable.const_mul (-a)).exp.aestronglyMeasurable) 1 (by
          filter_upwards with ω
          rw [Real.norm_eq_abs, abs_of_nonneg (exp_nonneg _)]
          rw [Real.exp_le_one_iff]
          exact mul_nonpos_of_nonpos_of_nonneg (neg_nonpos.mpr ha_nonneg) (sq_nonneg (X ω)))
    convert hneg_int.const_mul (exp (a * m)) using 1
    ext ω
    rw [← exp_add]
    rw [hm_def, ha_def]
    ring_nf

/-- The MGF of the identity under a push-forward law is the MGF of the original variable. -/
lemma mgf_id_map_eq {μ : Measure Ω} {X : Ω → ℝ}
    (hX : AEMeasurable X μ) (t : ℝ) :
    mgf id (μ.map X) t = mgf X μ t := by
  simpa [mgf, Function.comp_def] using
    (integral_map hX ((measurable_id'.const_mul t).exp.aestronglyMeasurable)
      (f := fun x : ℝ => exp (t * id x)))

lemma integrable_exp_mul_id_map {μ : Measure Ω} {X : Ω → ℝ}
    {t : ℝ} (hX : AEMeasurable X μ) (h : Integrable (fun ω => exp (t * X ω)) μ) :
    Integrable (fun x : ℝ => exp (t * id x)) (μ.map X) := by
  rw [integrable_map_measure]
  · simpa [Function.comp_def] using h
  · fun_prop
  · exact hX

lemma hasSubgaussianMGF_id_map {μ : Measure Ω} {X : Ω → ℝ} {c : ℝ≥0}
    (h : HasSubgaussianMGF X c μ) :
    HasSubgaussianMGF id c (μ.map X) where
  integrable_exp_mul t := integrable_exp_mul_id_map h.aemeasurable (h.integrable_exp_mul t)
  mgf_le t := by
    rw [mgf_id_map_eq h.aemeasurable t]
    exact h.mgf_le t

lemma integrable_exp_mul_snd_fst_prod_of_hasSubgaussianMGF_of_le
    {νX νY : Measure ℝ} [IsProbabilityMeasure νX] [IsProbabilityMeasure νY]
    {cX cY : ℝ≥0} (hX : HasSubgaussianMGF id cX νX)
    (hY : HasSubgaussianMGF id cY νY) {C0 θ : ℝ}
    (hC0 : 0 < C0) (hcX : (cX : ℝ) ≤ C0) (hcY : (cY : ℝ) ≤ C0)
    (hθ_half : (C0 * θ ^ 2 / 2) * C0 * exp 1 ≤ 1 / 2) :
    Integrable (fun p : ℝ × ℝ => exp (θ * p.2 * p.1)) (νY.prod νX) := by
  let f : ℝ × ℝ → ℝ := fun p => exp (θ * p.2 * p.1)
  have hf_aesm : AEStronglyMeasurable f (νY.prod νX) := by
    dsimp [f]
    fun_prop
  rw [integrable_prod_iff hf_aesm]
  constructor
  · filter_upwards with y
    have hinner := hX.integrable_exp_mul (θ * y)
    convert hinner using 1
    ext x
    dsimp [f]
    ring_nf
  · have hθ2_nonneg : 0 ≤ C0 * θ ^ 2 / 2 := by positivity
    have hbound_int :
        Integrable (fun y : ℝ => exp ((C0 * θ ^ 2 / 2) * y ^ 2)) νY :=
      integrable_exp_mul_sq_of_hasSubgaussianMGF_of_le hY hθ2_nonneg hC0 hcY (by
        linarith [hθ_half])
    refine Integrable.mono' hbound_int ?_ ?_
    · exact hf_aesm.norm.integral_prod_right'
    · filter_upwards with y
      have hinner_eq :
          ∫ x, ‖f (y, x)‖ ∂νX = mgf id νX (θ * y) := by
        unfold mgf
        apply integral_congr_ae
        filter_upwards with x
        dsimp [f]
        rw [abs_of_nonneg (exp_nonneg _)]
        ring_nf
      have hmgf := hX.mgf_le (θ * y)
      have hquad_le :
          (cX : ℝ) * (θ * y) ^ 2 / 2 ≤ C0 * θ ^ 2 * y ^ 2 / 2 := by
        have hnonneg : 0 ≤ (θ * y) ^ 2 / 2 := by positivity
        calc
          (cX : ℝ) * (θ * y) ^ 2 / 2 =
              (cX : ℝ) * ((θ * y) ^ 2 / 2) := by ring_nf
          _ ≤ C0 * ((θ * y) ^ 2 / 2) := mul_le_mul_of_nonneg_right hcX hnonneg
          _ = C0 * θ ^ 2 * y ^ 2 / 2 := by ring_nf
      have hle :
          ∫ x, ‖f (y, x)‖ ∂νX ≤ exp ((C0 * θ ^ 2 / 2) * y ^ 2) := by
        rw [hinner_eq]
        calc
          mgf id νX (θ * y) ≤ exp ((cX : ℝ) * (θ * y) ^ 2 / 2) := hmgf
          _ ≤ exp (C0 * θ ^ 2 * y ^ 2 / 2) := exp_le_exp.mpr hquad_le
          _ = exp ((C0 * θ ^ 2 / 2) * y ^ 2) := by ring_nf
      rw [Real.norm_eq_abs, abs_of_nonneg ?_]
      · exact hle
      · exact integral_nonneg fun x => norm_nonneg _

lemma integral_exp_mul_snd_fst_prod_le_of_hasSubgaussianMGF_of_le
    {νX νY : Measure ℝ} [IsProbabilityMeasure νX] [IsProbabilityMeasure νY]
    {cX cY : ℝ≥0} (hX : HasSubgaussianMGF id cX νX)
    (hY : HasSubgaussianMGF id cY νY) {C0 θ : ℝ}
    (hC0 : 0 < C0) (hcX : (cX : ℝ) ≤ C0) (hcY : (cY : ℝ) ≤ C0)
    (hθ_half : (C0 * θ ^ 2 / 2) * C0 * exp 1 ≤ 1 / 2) :
    ∫ p : ℝ × ℝ, exp (θ * p.2 * p.1) ∂(νY.prod νX) ≤
      exp (exp 1 ^ 2 * C0 ^ 2 * θ ^ 2) := by
  let f : ℝ × ℝ → ℝ := fun p => exp (θ * p.2 * p.1)
  have hf_int :
      Integrable f (νY.prod νX) :=
    integrable_exp_mul_snd_fst_prod_of_hasSubgaussianMGF_of_le
      hX hY hC0 hcX hcY hθ_half
  have hθ2_nonneg : 0 ≤ C0 * θ ^ 2 / 2 := by positivity
  have hbound_int :
      Integrable (fun y : ℝ => exp ((C0 * θ ^ 2 / 2) * y ^ 2)) νY :=
    integrable_exp_mul_sq_of_hasSubgaussianMGF_of_le hY hθ2_nonneg hC0 hcY (by
      linarith [hθ_half])
  have houter_bound :
      (fun y : ℝ => ∫ x, f (y, x) ∂νX) ≤ᵐ[νY]
        fun y : ℝ => exp ((C0 * θ ^ 2 / 2) * y ^ 2) := by
    filter_upwards with y
    have hinner_eq :
        ∫ x, f (y, x) ∂νX = mgf id νX (θ * y) := by
      unfold mgf
      apply integral_congr_ae
      filter_upwards with x
      dsimp [f]
      ring_nf
    have hmgf := hX.mgf_le (θ * y)
    have hquad_le :
        (cX : ℝ) * (θ * y) ^ 2 / 2 ≤ C0 * θ ^ 2 * y ^ 2 / 2 := by
      have hnonneg : 0 ≤ (θ * y) ^ 2 / 2 := by positivity
      calc
        (cX : ℝ) * (θ * y) ^ 2 / 2 =
            (cX : ℝ) * ((θ * y) ^ 2 / 2) := by ring_nf
        _ ≤ C0 * ((θ * y) ^ 2 / 2) := mul_le_mul_of_nonneg_right hcX hnonneg
        _ = C0 * θ ^ 2 * y ^ 2 / 2 := by ring_nf
    rw [hinner_eq]
    calc
      mgf id νX (θ * y) ≤ exp ((cX : ℝ) * (θ * y) ^ 2 / 2) := hmgf
      _ ≤ exp (C0 * θ ^ 2 * y ^ 2 / 2) := exp_le_exp.mpr hquad_le
      _ = exp ((C0 * θ ^ 2 / 2) * y ^ 2) := by ring_nf
  calc
    ∫ p : ℝ × ℝ, exp (θ * p.2 * p.1) ∂(νY.prod νX)
        = ∫ y, ∫ x, f (y, x) ∂νX ∂νY := by
          rw [integral_prod f hf_int]
    _ ≤ ∫ y, exp ((C0 * θ ^ 2 / 2) * y ^ 2) ∂νY := by
          exact integral_mono_ae hf_int.integral_prod_left hbound_int houter_bound
    _ ≤ exp (2 * exp 1 * ((C0 * θ ^ 2 / 2) * C0 * exp 1)) :=
          integral_exp_mul_sq_le_exp_linear_of_hasSubgaussianMGF_of_le
            hY hθ2_nonneg hC0 hcY hθ_half
    _ = exp (exp 1 ^ 2 * C0 ^ 2 * θ ^ 2) := by
          congr 1
          ring_nf

lemma integrable_exp_mul_prod_of_indepFun_hasSubgaussianMGF_of_le
    {μ : Measure Ω} [IsProbabilityMeasure μ] {X Y : Ω → ℝ}
    (h_indep : X ⟂ᵢ[μ] Y)
    {cX cY : ℝ≥0} (hX : HasSubgaussianMGF X cX μ)
    (hY : HasSubgaussianMGF Y cY μ) {C0 θ : ℝ}
    (hC0 : 0 < C0) (hcX : (cX : ℝ) ≤ C0) (hcY : (cY : ℝ) ≤ C0)
    (hθ_half : (C0 * θ ^ 2 / 2) * C0 * exp 1 ≤ 1 / 2) :
    Integrable (fun ω => exp (θ * X ω * Y ω)) μ := by
  let φ : Ω → ℝ × ℝ := fun ω => (Y ω, X ω)
  let g : ℝ × ℝ → ℝ := fun p => exp (θ * p.2 * p.1)
  have hφ : AEMeasurable φ μ :=
    (hY.aestronglyMeasurable.prodMk hX.aestronglyMeasurable).aemeasurable
  have hg : AEStronglyMeasurable g (μ.map φ) := by
    dsimp [g]
    fun_prop
  haveI : IsProbabilityMeasure (μ.map X) :=
    MeasureTheory.Measure.isProbabilityMeasure_map hX.aemeasurable
  haveI : IsProbabilityMeasure (μ.map Y) :=
    MeasureTheory.Measure.isProbabilityMeasure_map hY.aemeasurable
  have hprod_int :
      Integrable g ((μ.map Y).prod (μ.map X)) :=
    integrable_exp_mul_snd_fst_prod_of_hasSubgaussianMGF_of_le
      (νX := μ.map X) (νY := μ.map Y)
      (hasSubgaussianMGF_id_map hX) (hasSubgaussianMGF_id_map hY)
      hC0 hcX hcY hθ_half
  have hmap_eq :
      μ.map φ = (μ.map Y).prod (μ.map X) := by
    simpa [φ] using (h_indep.symm.map_prod_eq_prod_map_map hY.aemeasurable hX.aemeasurable)
  have hmap_int : Integrable g (μ.map φ) := by
    rwa [hmap_eq]
  have hcomp := (integrable_map_measure hg hφ).1 hmap_int
  convert hcomp using 1
  ext ω
  dsimp [g, φ]

lemma integral_exp_mul_prod_le_of_indepFun_hasSubgaussianMGF_of_le
    {μ : Measure Ω} [IsProbabilityMeasure μ] {X Y : Ω → ℝ}
    (h_indep : X ⟂ᵢ[μ] Y)
    {cX cY : ℝ≥0} (hX : HasSubgaussianMGF X cX μ)
    (hY : HasSubgaussianMGF Y cY μ) {C0 θ : ℝ}
    (hC0 : 0 < C0) (hcX : (cX : ℝ) ≤ C0) (hcY : (cY : ℝ) ≤ C0)
    (hθ_half : (C0 * θ ^ 2 / 2) * C0 * exp 1 ≤ 1 / 2) :
    ∫ ω, exp (θ * X ω * Y ω) ∂μ ≤ exp (exp 1 ^ 2 * C0 ^ 2 * θ ^ 2) := by
  let φ : Ω → ℝ × ℝ := fun ω => (Y ω, X ω)
  let g : ℝ × ℝ → ℝ := fun p => exp (θ * p.2 * p.1)
  have hφ : AEMeasurable φ μ :=
    (hY.aestronglyMeasurable.prodMk hX.aestronglyMeasurable).aemeasurable
  have hg : AEStronglyMeasurable g (μ.map φ) := by
    dsimp [g]
    fun_prop
  haveI : IsProbabilityMeasure (μ.map X) :=
    MeasureTheory.Measure.isProbabilityMeasure_map hX.aemeasurable
  haveI : IsProbabilityMeasure (μ.map Y) :=
    MeasureTheory.Measure.isProbabilityMeasure_map hY.aemeasurable
  have hmap_eq :
      μ.map φ = (μ.map Y).prod (μ.map X) := by
    simpa [φ] using (h_indep.symm.map_prod_eq_prod_map_map hY.aemeasurable hX.aemeasurable)
  have hprod_bound :
      ∫ p : ℝ × ℝ, g p ∂((μ.map Y).prod (μ.map X)) ≤
        exp (exp 1 ^ 2 * C0 ^ 2 * θ ^ 2) :=
    integral_exp_mul_snd_fst_prod_le_of_hasSubgaussianMGF_of_le
      (νX := μ.map X) (νY := μ.map Y)
      (hasSubgaussianMGF_id_map hX) (hasSubgaussianMGF_id_map hY)
      hC0 hcX hcY hθ_half
  calc
    ∫ ω, exp (θ * X ω * Y ω) ∂μ
        = ∫ p : ℝ × ℝ, g p ∂(μ.map φ) := by
          rw [integral_map hφ hg]
    _ = ∫ p : ℝ × ℝ, g p ∂((μ.map Y).prod (μ.map X)) := by rw [hmap_eq]
    _ ≤ exp (exp 1 ^ 2 * C0 ^ 2 * θ ^ 2) := hprod_bound

/-- CGF bound for a centered square of a sub-Gaussian variable. -/
lemma centered_sq_cgf_le_of_hasSubgaussianMGF_of_le {μ : Measure Ω}
    [IsProbabilityMeasure μ] {X : Ω → ℝ} {c : ℝ≥0}
    (h : HasSubgaussianMGF X c μ) {θ C0 : ℝ}
    (hC0 : 0 < C0) (hc_le : (c : ℝ) ≤ C0)
    (hθ_abs : |θ| * C0 * exp 1 ≤ 1 / 2) :
    cgf (fun ω => X ω ^ 2 - ∫ ω, X ω ^ 2 ∂μ) μ θ ≤
      2 * exp 1 * (θ * C0 * exp 1) ^ 2 := by
  have hθ_abs_lt : |θ| * C0 * exp 1 < 1 := by
    linarith
  have hint := integrable_exp_centered_sq_of_hasSubgaussianMGF_of_le h hC0 hc_le hθ_abs_lt
  have hmgf_pos :
      0 < mgf (fun ω => X ω ^ 2 - ∫ ω, X ω ^ 2 ∂μ) μ θ :=
    mgf_pos hint
  have hmgf_le :
      mgf (fun ω => X ω ^ 2 - ∫ ω, X ω ^ 2 ∂μ) μ θ ≤
        exp (2 * exp 1 * (θ * C0 * exp 1) ^ 2) := by
    rcases le_total 0 θ with hθ_nonneg | hθ_nonpos
    · have hhalf : θ * C0 * exp 1 ≤ 1 / 2 := by
        simpa [abs_of_nonneg hθ_nonneg] using hθ_abs
      simpa [mgf] using
        integral_exp_centered_sq_le_exp_quadratic_nonneg h hθ_nonneg hC0 hc_le hhalf
    · have hbound :=
        integral_exp_centered_sq_le_exp_quadratic_nonpos h hθ_nonpos hC0 hc_le
      calc
        mgf (fun ω => X ω ^ 2 - ∫ ω, X ω ^ 2 ∂μ) μ θ
            ≤ exp (2 * exp 1 * ((-θ) * C0 * exp 1) ^ 2) := by
              simpa [mgf] using hbound
        _ = exp (2 * exp 1 * (θ * C0 * exp 1) ^ 2) := by ring_nf
  calc
    cgf (fun ω => X ω ^ 2 - ∫ ω, X ω ^ 2 ∂μ) μ θ
        = log (mgf (fun ω => X ω ^ 2 - ∫ ω, X ω ^ 2 ∂μ) μ θ) := rfl
    _ ≤ log (exp (2 * exp 1 * (θ * C0 * exp 1) ^ 2)) :=
        log_le_log hmgf_pos hmgf_le
    _ = 2 * exp 1 * (θ * C0 * exp 1) ^ 2 := by rw [log_exp]

/-- The centered diagonal part of a quadratic form. -/
def diagonalCenteredQuadraticForm {n : ℕ} (μ : Measure Ω)
    (A : Matrix (Fin n) (Fin n) ℝ) (X : Fin n → Ω → ℝ) : Ω → ℝ :=
  fun ω => ∑ i, A i i * (X i ω ^ 2 - ∫ ω, X i ω ^ 2 ∂μ)

lemma cgf_const_mul {μ : Measure Ω} {Y : Ω → ℝ} (a θ : ℝ) :
    cgf (fun ω => a * Y ω) μ θ = cgf Y μ (a * θ) := by
  rw [cgf, cgf, mgf_const_mul]

lemma iIndepFun_centered_sq {μ : Measure Ω} {n : ℕ} {X : Fin n → Ω → ℝ}
    (h_indep : iIndepFun X μ) :
    iIndepFun (fun i => fun ω => X i ω ^ 2 - ∫ ω, X i ω ^ 2 ∂μ) μ := by
  simpa [Function.comp_def] using
    h_indep.comp
      (fun i x => x ^ 2 - ∫ ω, X i ω ^ 2 ∂μ)
      (fun _ => by fun_prop)

lemma iIndepFun.integrable_exp_mul_sum₀ {ι : Type*} {μ : Measure Ω} [IsFiniteMeasure μ]
    {X : ι → Ω → ℝ}
    (h_indep : iIndepFun X μ) (h_meas : ∀ i, AEMeasurable (X i) μ)
    {s : Finset ι} {t : ℝ}
    (h_int : ∀ i ∈ s, Integrable (fun ω => exp (t * X i ω)) μ) :
    Integrable (fun ω => exp (t * (∑ i ∈ s, X i) ω)) μ := by
  classical
  induction s using Finset.induction_on with
  | empty =>
      simp
  | insert i s hi_notin_s h_rec =>
      have h_int_s : ∀ j ∈ s, Integrable (fun ω : Ω => exp (t * X j ω)) μ :=
        fun j hj => h_int j (Finset.mem_insert_of_mem hj)
      specialize h_rec h_int_s
      rw [Finset.sum_insert hi_notin_s]
      refine IndepFun.integrable_exp_mul_add ?_ (h_int i (Finset.mem_insert_self _ _)) h_rec
      exact (h_indep.indepFun_finsetSum_of_notMem₀ h_meas hi_notin_s).symm

lemma diagonalCenteredQuadraticForm_cgf_le {μ : Measure Ω} [IsProbabilityMeasure μ]
    {n : ℕ} {A : Matrix (Fin n) (Fin n) ℝ} {X : Fin n → Ω → ℝ}
    {K C θ : ℝ} (hK : 0 < K) (hC : 0 < C)
    (hC_domain : exp 1 ≤ C) (hC_quad : 2 * exp 1 ^ 3 ≤ C)
    (hOp : 0 < operatorNorm A)
    (h_indep : iIndepFun X μ)
    (hX_subG : ∀ i, HasSubgaussianMGF (X i) ⟨K ^ 2, sq_nonneg K⟩ μ)
    (hθ : |θ| ≤ (2 * C * K ^ 2 * operatorNorm A)⁻¹) :
    cgf (diagonalCenteredQuadraticForm μ A X) μ θ ≤
      C * θ ^ 2 * K ^ 4 * frobeniusNorm A ^ 2 := by
  let Y : Fin n → Ω → ℝ :=
    fun i ω => X i ω ^ 2 - ∫ ω, X i ω ^ 2 ∂μ
  let Z : Fin n → Ω → ℝ := fun i ω => A i i * Y i ω
  have hKsq_pos : 0 < K ^ 2 := sq_pos_of_pos hK
  have hKsq_nonneg : 0 ≤ K ^ 2 := hKsq_pos.le
  have hC0_le : ((⟨K ^ 2, sq_nonneg K⟩ : ℝ≥0) : ℝ) ≤ K ^ 2 := by simp
  have hscale :
      |θ| * K ^ 2 * operatorNorm A ≤ 1 / (2 * C) := by
    have hfactor_nonneg : 0 ≤ K ^ 2 * operatorNorm A :=
      mul_nonneg hKsq_nonneg (operatorNorm_nonneg A)
    calc
      |θ| * K ^ 2 * operatorNorm A
          = |θ| * (K ^ 2 * operatorNorm A) := by ring_nf
      _ ≤ (2 * C * K ^ 2 * operatorNorm A)⁻¹ * (K ^ 2 * operatorNorm A) :=
          mul_le_mul_of_nonneg_right hθ hfactor_nonneg
      _ = 1 / (2 * C) := by
          field_simp [hC.ne', hKsq_pos.ne', hOp.ne']
  have hhalf_i : ∀ i, |A i i * θ| * K ^ 2 * exp 1 ≤ 1 / 2 := by
    intro i
    have hdiag := abs_diag_le_operatorNorm A i
    have hθ_abs_nonneg : 0 ≤ |θ| := abs_nonneg θ
    have hle_op :
        |θ| * |A i i| * K ^ 2 ≤ |θ| * operatorNorm A * K ^ 2 := by
      exact mul_le_mul_of_nonneg_right
        (mul_le_mul_of_nonneg_left hdiag hθ_abs_nonneg) hKsq_nonneg
    calc
      |A i i * θ| * K ^ 2 * exp 1
          = (|θ| * |A i i| * K ^ 2) * exp 1 := by
              rw [abs_mul]
              ring_nf
      _ ≤ (|θ| * operatorNorm A * K ^ 2) * exp 1 :=
          mul_le_mul_of_nonneg_right hle_op (exp_nonneg 1)
      _ = (|θ| * K ^ 2 * operatorNorm A) * exp 1 := by ring_nf
      _ ≤ (1 / (2 * C)) * exp 1 :=
          mul_le_mul_of_nonneg_right hscale (exp_nonneg 1)
      _ ≤ 1 / 2 := by
          have hdiv : exp 1 / C ≤ 1 := (div_le_one hC).mpr hC_domain
          calc
            (1 / (2 * C)) * exp 1 = (1 / 2) * (exp 1 / C) := by
              field_simp [hC.ne']
            _ ≤ (1 / 2) * 1 := mul_le_mul_of_nonneg_left hdiv (by norm_num)
            _ = 1 / 2 := by norm_num
  have hZ_indep : iIndepFun Z μ := by
    have hY_indep : iIndepFun Y μ := by
      simpa [Y] using iIndepFun_centered_sq (μ := μ) (X := X) h_indep
    simpa [Z, Function.comp_def] using
      hY_indep.comp (fun i x => A i i * x) (fun _ => by fun_prop)
  have hZ_meas : ∀ i, AEMeasurable (Z i) μ := by
    intro i
    dsimp [Z, Y]
    exact (((hX_subG i).aemeasurable.pow_const 2).sub_const _).const_mul _
  have hZ_int :
      ∀ i ∈ (Finset.univ : Finset (Fin n)),
        Integrable (fun ω => exp (θ * Z i ω)) μ := by
    intro i _
    have hsmall_lt : |A i i * θ| * K ^ 2 * exp 1 < 1 := by
      linarith [hhalf_i i]
    have hint :=
      integrable_exp_centered_sq_of_hasSubgaussianMGF_of_le
        (hX_subG i) hKsq_pos hC0_le hsmall_lt
    convert hint using 1
    ext ω
    dsimp [Z, Y]
    ring_nf
  have hcgf_sum :
      cgf (fun ω => ∑ i, Z i ω) μ θ = ∑ i, cgf (Z i) μ θ := by
    have hfun : (fun ω => ∑ i, Z i ω) = (∑ i, Z i) := by
      ext ω
      simp
    rw [hfun]
    exact hZ_indep.cgf_sum₀ hZ_meas (s := Finset.univ) hZ_int
  have hcgf_i :
      ∀ i, cgf (Z i) μ θ ≤
        2 * exp 1 * (A i i * θ * K ^ 2 * exp 1) ^ 2 := by
    intro i
    have hbase :=
      centered_sq_cgf_le_of_hasSubgaussianMGF_of_le
        (hX_subG i) hKsq_pos hC0_le (hhalf_i i)
    calc
      cgf (Z i) μ θ
          = cgf (Y i) μ (A i i * θ) := by
              dsimp [Z]
              exact cgf_const_mul (μ := μ) (Y := Y i) (A i i) θ
      _ ≤ 2 * exp 1 * (A i i * θ * K ^ 2 * exp 1) ^ 2 := hbase
  have hsum_cgf_le :
      (∑ i, cgf (Z i) μ θ) ≤
        ∑ i, 2 * exp 1 * (A i i * θ * K ^ 2 * exp 1) ^ 2 :=
    Finset.sum_le_sum fun i _ => hcgf_i i
  have hsum_eq :
      (∑ i, 2 * exp 1 * (A i i * θ * K ^ 2 * exp 1) ^ 2) =
        (2 * exp 1 ^ 3) * θ ^ 2 * K ^ 4 * ∑ i, A i i ^ 2 := by
    rw [Finset.mul_sum]
    apply Finset.sum_congr rfl
    intro i _
    ring_nf
  have hconst_le :
      (2 * exp 1 ^ 3) * θ ^ 2 * K ^ 4 * ∑ i, A i i ^ 2 ≤
        C * θ ^ 2 * K ^ 4 * frobeniusNorm A ^ 2 := by
    have hnonneg : 0 ≤ θ ^ 2 * K ^ 4 := by positivity
    have hdiag_le := sum_diag_sq_le_frobeniusNorm_sq A
    calc
      (2 * exp 1 ^ 3) * θ ^ 2 * K ^ 4 * ∑ i, A i i ^ 2
          = (2 * exp 1 ^ 3) * (θ ^ 2 * K ^ 4) * ∑ i, A i i ^ 2 := by ring_nf
      _ ≤ C * (θ ^ 2 * K ^ 4) * ∑ i, A i i ^ 2 := by
          exact mul_le_mul_of_nonneg_right
            (mul_le_mul_of_nonneg_right hC_quad hnonneg)
            (Finset.sum_nonneg fun i _ => sq_nonneg (A i i))
      _ ≤ C * (θ ^ 2 * K ^ 4) * frobeniusNorm A ^ 2 := by
          exact mul_le_mul_of_nonneg_left hdiag_le
            (mul_nonneg hC.le hnonneg)
      _ = C * θ ^ 2 * K ^ 4 * frobeniusNorm A ^ 2 := by ring_nf
  calc
    cgf (diagonalCenteredQuadraticForm μ A X) μ θ
        = cgf (fun ω => ∑ i, Z i ω) μ θ := by rfl
    _ = ∑ i, cgf (Z i) μ θ := hcgf_sum
    _ ≤ ∑ i, 2 * exp 1 * (A i i * θ * K ^ 2 * exp 1) ^ 2 := hsum_cgf_le
    _ = (2 * exp 1 ^ 3) * θ ^ 2 * K ^ 4 * ∑ i, A i i ^ 2 := hsum_eq
    _ ≤ C * θ ^ 2 * K ^ 4 * frobeniusNorm A ^ 2 := hconst_le

lemma diagonalCenteredQuadraticForm_integrable_exp {μ : Measure Ω} [IsProbabilityMeasure μ]
    {n : ℕ} {A : Matrix (Fin n) (Fin n) ℝ} {X : Fin n → Ω → ℝ}
    {K C θ : ℝ} (hK : 0 < K) (hC : 0 < C)
    (hC_domain : exp 1 ≤ C) (hOp : 0 < operatorNorm A)
    (h_indep : iIndepFun X μ)
    (hX_subG : ∀ i, HasSubgaussianMGF (X i) ⟨K ^ 2, sq_nonneg K⟩ μ)
    (hθ : |θ| ≤ (2 * C * K ^ 2 * operatorNorm A)⁻¹) :
    Integrable (fun ω => exp (θ * diagonalCenteredQuadraticForm μ A X ω)) μ := by
  let Y : Fin n → Ω → ℝ :=
    fun i ω => X i ω ^ 2 - ∫ ω, X i ω ^ 2 ∂μ
  let Z : Fin n → Ω → ℝ := fun i ω => A i i * Y i ω
  have hKsq_pos : 0 < K ^ 2 := sq_pos_of_pos hK
  have hKsq_nonneg : 0 ≤ K ^ 2 := hKsq_pos.le
  have hC0_le : ((⟨K ^ 2, sq_nonneg K⟩ : ℝ≥0) : ℝ) ≤ K ^ 2 := by simp
  have hscale :
      |θ| * K ^ 2 * operatorNorm A ≤ 1 / (2 * C) := by
    have hfactor_nonneg : 0 ≤ K ^ 2 * operatorNorm A :=
      mul_nonneg hKsq_nonneg (operatorNorm_nonneg A)
    calc
      |θ| * K ^ 2 * operatorNorm A
          = |θ| * (K ^ 2 * operatorNorm A) := by ring_nf
      _ ≤ (2 * C * K ^ 2 * operatorNorm A)⁻¹ * (K ^ 2 * operatorNorm A) :=
          mul_le_mul_of_nonneg_right hθ hfactor_nonneg
      _ = 1 / (2 * C) := by
          field_simp [hC.ne', hKsq_pos.ne', hOp.ne']
  have hhalf_i : ∀ i, |A i i * θ| * K ^ 2 * exp 1 ≤ 1 / 2 := by
    intro i
    have hdiag := abs_diag_le_operatorNorm A i
    have hθ_abs_nonneg : 0 ≤ |θ| := abs_nonneg θ
    have hle_op :
        |θ| * |A i i| * K ^ 2 ≤ |θ| * operatorNorm A * K ^ 2 := by
      exact mul_le_mul_of_nonneg_right
        (mul_le_mul_of_nonneg_left hdiag hθ_abs_nonneg) hKsq_nonneg
    calc
      |A i i * θ| * K ^ 2 * exp 1
          = (|θ| * |A i i| * K ^ 2) * exp 1 := by
              rw [abs_mul]
              ring_nf
      _ ≤ (|θ| * operatorNorm A * K ^ 2) * exp 1 :=
          mul_le_mul_of_nonneg_right hle_op (exp_nonneg 1)
      _ = (|θ| * K ^ 2 * operatorNorm A) * exp 1 := by ring_nf
      _ ≤ (1 / (2 * C)) * exp 1 :=
          mul_le_mul_of_nonneg_right hscale (exp_nonneg 1)
      _ ≤ 1 / 2 := by
          have hdiv : exp 1 / C ≤ 1 := (div_le_one hC).mpr hC_domain
          calc
            (1 / (2 * C)) * exp 1 = (1 / 2) * (exp 1 / C) := by
              field_simp [hC.ne']
            _ ≤ (1 / 2) * 1 := mul_le_mul_of_nonneg_left hdiv (by norm_num)
            _ = 1 / 2 := by norm_num
  have hZ_indep : iIndepFun Z μ := by
    have hY_indep : iIndepFun Y μ := by
      simpa [Y] using iIndepFun_centered_sq (μ := μ) (X := X) h_indep
    simpa [Z, Function.comp_def] using
      hY_indep.comp (fun i x => A i i * x) (fun _ => by fun_prop)
  have hZ_meas : ∀ i, AEMeasurable (Z i) μ := by
    intro i
    dsimp [Z, Y]
    exact (((hX_subG i).aemeasurable.pow_const 2).sub_const _).const_mul _
  have hZ_int :
      ∀ i ∈ (Finset.univ : Finset (Fin n)),
        Integrable (fun ω => exp (θ * Z i ω)) μ := by
    intro i _
    have hsmall_lt : |A i i * θ| * K ^ 2 * exp 1 < 1 := by
      linarith [hhalf_i i]
    have hint :=
      integrable_exp_centered_sq_of_hasSubgaussianMGF_of_le
        (hX_subG i) hKsq_pos hC0_le hsmall_lt
    convert hint using 1
    ext ω
    dsimp [Z, Y]
    ring_nf
  have hsum_int :
      Integrable (fun ω => exp (θ * (∑ i, Z i) ω)) μ :=
    iIndepFun.integrable_exp_mul_sum₀ hZ_indep hZ_meas (s := Finset.univ) hZ_int
  convert hsum_int using 1
  ext ω
  simp [diagonalCenteredQuadraticForm, Z, Y]

lemma quadraticForm_eq_sum_diag_of_offdiag_eq_zero {n : ℕ}
    {A : Matrix (Fin n) (Fin n) ℝ} {x : Fin n → ℝ}
    (hA_diag : ∀ i j, i ≠ j → A i j = 0) :
    quadraticForm A x = ∑ i, A i i * x i ^ 2 := by
  unfold quadraticForm
  apply Finset.sum_congr rfl
  intro i _
  calc
    ∑ j, A i j * x i * x j = A i i * x i * x i := by
      exact Finset.sum_eq_single i
        (fun j _ hji => by
          rw [hA_diag i j hji.symm]
          ring)
        (fun hi => (hi (Finset.mem_univ i)).elim)
    _ = A i i * x i ^ 2 := by ring_nf

lemma centeredQuadraticForm_eq_diagonalCentered_of_offdiag_eq_zero
    {μ : Measure Ω} [IsProbabilityMeasure μ]
    {n : ℕ} {A : Matrix (Fin n) (Fin n) ℝ} {X : Fin n → Ω → ℝ}
    {K : ℝ}
    (hA_diag : ∀ i j, i ≠ j → A i j = 0)
    (hX_subG : ∀ i, HasSubgaussianMGF (X i) ⟨K ^ 2, sq_nonneg K⟩ μ) :
    centeredQuadraticForm μ A X = diagonalCenteredQuadraticForm μ A X := by
  have hterm_int : ∀ i, Integrable (fun ω => A i i * X i ω ^ 2) μ := by
    intro i
    exact ((integrable_pow_of_integrable_exp_mul
      (X := X i) (t := 1) one_ne_zero
      (by simpa using (hX_subG i).integrable_exp_mul 1)
      (by simpa using (hX_subG i).integrable_exp_mul (-1)) 2).const_mul (A i i))
  ext ω
  unfold centeredQuadraticForm randomQuadraticForm diagonalCenteredQuadraticForm
  have hqfun :
      (fun ω => quadraticForm A fun i => X i ω) =
        (fun ω => ∑ i, A i i * X i ω ^ 2) := by
    ext ω
    exact quadraticForm_eq_sum_diag_of_offdiag_eq_zero hA_diag
  rw [quadraticForm_eq_sum_diag_of_offdiag_eq_zero hA_diag]
  rw [show (∫ ω, quadraticForm A (fun i => X i ω) ∂μ) =
      ∫ ω, (∑ i, A i i * X i ω ^ 2) ∂μ by rw [hqfun]]
  have hintegral :
      ∫ ω, (∑ i, A i i * X i ω ^ 2) ∂μ =
        ∑ i, ∫ ω, A i i * X i ω ^ 2 ∂μ := by
    rw [integral_finsetSum]
    intro i _
    exact hterm_int i
  rw [hintegral]
  calc
    (∑ i, A i i * X i ω ^ 2) - ∑ i, ∫ (ω : Ω), A i i * X i ω ^ 2 ∂μ
        = ∑ i, (A i i * X i ω ^ 2 - ∫ (ω : Ω), A i i * X i ω ^ 2 ∂μ) := by
            rw [← Finset.sum_sub_distrib]
    _ = ∑ i, A i i * (X i ω ^ 2 - ∫ (ω : Ω), X i ω ^ 2 ∂μ) := by
        apply Finset.sum_congr rfl
        intro i _
        rw [integral_const_mul]
        ring_nf

/-- Hanson-Wright MGF certificate for diagonal quadratic forms with sub-Gaussian coordinates. -/
theorem hasHansonWrightMGF_diagonal_of_subgaussian {μ : Measure Ω} [IsProbabilityMeasure μ]
    {n : ℕ} {A : Matrix (Fin n) (Fin n) ℝ} {X : Fin n → Ω → ℝ}
    {K C : ℝ} (hK : 0 < K) (hC : 0 < C)
    (hC_domain : exp 1 ≤ C) (hC_quad : 2 * exp 1 ^ 3 ≤ C)
    (hOp : 0 < operatorNorm A)
    (hA_diag : ∀ i j, i ≠ j → A i j = 0)
    (h_indep : iIndepFun X μ)
    (hX_subG : ∀ i, HasSubgaussianMGF (X i) ⟨K ^ 2, sq_nonneg K⟩ μ) :
    HasHansonWrightMGF μ A X K C := by
  have hcenter_eq :
      centeredQuadraticForm μ A X = diagonalCenteredQuadraticForm μ A X :=
    centeredQuadraticForm_eq_diagonalCentered_of_offdiag_eq_zero hA_diag hX_subG
  refine ⟨?_, ?_⟩
  · intro l hl
    rw [hcenter_eq]
    exact diagonalCenteredQuadraticForm_cgf_le hK hC hC_domain hC_quad hOp h_indep hX_subG hl
  · intro l hl
    rw [hcenter_eq]
    exact diagonalCenteredQuadraticForm_integrable_exp hK hC hC_domain hOp h_indep hX_subG hl

private lemma hasSubgaussianMGF_integral_le_zero {μ : Measure Ω} [IsProbabilityMeasure μ]
    {X : Ω → ℝ} {c : ℝ≥0} (h : HasSubgaussianMGF X c μ) :
    ∫ ω, X ω ∂μ ≤ 0 := by
  refine le_of_forall_pos_le_add fun ε hε => ?_
  set t : ℝ := ε / ((c : ℝ) + 1) with ht_def
  have hden_pos : 0 < (c : ℝ) + 1 := by positivity
  have ht_pos : 0 < t := by
    rw [ht_def]
    positivity
  have hmean := mean_le_log_mgf h.integrable ht_pos (h.integrable_exp_mul t)
  have hmgf_pos : 0 < mgf X μ t := mgf_pos (h.integrable_exp_mul t)
  have hlog_le : log (mgf X μ t) ≤ (c : ℝ) * t ^ 2 / 2 := by
    calc
      log (mgf X μ t) ≤ log (exp ((c : ℝ) * t ^ 2 / 2)) :=
        log_le_log hmgf_pos (h.mgf_le t)
      _ = (c : ℝ) * t ^ 2 / 2 := log_exp _
  have hmean_le : ∫ ω, X ω ∂μ ≤ (1 / t) * ((c : ℝ) * t ^ 2 / 2) :=
    hmean.trans (mul_le_mul_of_nonneg_left hlog_le (by positivity))
  calc
    ∫ ω, X ω ∂μ ≤ (1 / t) * ((c : ℝ) * t ^ 2 / 2) := hmean_le
    _ = (c : ℝ) * t / 2 := by field_simp [ht_pos.ne']
    _ ≤ ε := by
      rw [ht_def]
      have hratio : (c : ℝ) / ((c : ℝ) + 1) ≤ 1 :=
        (div_le_one hden_pos).mpr (by linarith)
      calc
        (c : ℝ) * (ε / ((c : ℝ) + 1)) / 2 =
            (ε / 2) * ((c : ℝ) / ((c : ℝ) + 1)) := by ring
        _ ≤ (ε / 2) * 1 := mul_le_mul_of_nonneg_left hratio (by positivity)
        _ ≤ ε := by linarith
    _ = 0 + ε := by ring

/-- A random variable satisfying the global sub-Gaussian MGF bound is centered. -/
lemma hasSubgaussianMGF_integral_eq_zero {μ : Measure Ω} [IsProbabilityMeasure μ]
    {X : Ω → ℝ} {c : ℝ≥0} (h : HasSubgaussianMGF X c μ) :
    ∫ ω, X ω ∂μ = 0 := by
  have hle : ∫ ω, X ω ∂μ ≤ 0 := hasSubgaussianMGF_integral_le_zero h
  have hle_neg : ∫ ω, -X ω ∂μ ≤ 0 := hasSubgaussianMGF_integral_le_zero h.neg
  have hge : 0 ≤ ∫ ω, X ω ∂μ := by
    rw [integral_neg] at hle_neg
    linarith
  exact le_antisymm hle hge

lemma quadraticForm_eq_diag_add_offDiagonal {n : ℕ}
    (A : Matrix (Fin n) (Fin n) ℝ) (x : Fin n → ℝ) :
    quadraticForm A x =
      (∑ i, A i i * x i ^ 2) + quadraticForm (offDiagonalMatrix A) x := by
  classical
  unfold quadraticForm offDiagonalMatrix
  calc
    ∑ i, ∑ j, A i j * x i * x j
        = ∑ i, (A i i * x i ^ 2 +
            ∑ j, (if i = j then 0 else A i j) * x i * x j) := by
          apply Finset.sum_congr rfl
          intro i _
          have hdiag_sum :
              (∑ j, (if i = j then A i i * x i * x i else 0)) =
                A i i * x i * x i := by
            rw [Finset.sum_eq_single i]
            · simp
            · intro j _ hji
              simp [hji.symm]
            · intro hi
              exact (hi (Finset.mem_univ i)).elim
          calc
            ∑ j, A i j * x i * x j
                = (∑ j, (if i = j then A i i * x i * x i else 0)) +
                    ∑ j, (if i = j then 0 else A i j) * x i * x j := by
                  rw [← Finset.sum_add_distrib]
                  apply Finset.sum_congr rfl
                  intro j _
                  by_cases hij : i = j
                  · subst j
                    simp
                  · rw [if_neg hij, if_neg hij]
                    ring
            _ = A i i * x i ^ 2 +
                    ∑ j, (if i = j then 0 else A i j) * x i * x j := by
                  rw [hdiag_sum]
                  ring
    _ = (∑ i, A i i * x i ^ 2) +
          ∑ i, ∑ j, (if i = j then 0 else A i j) * x i * x j := by
          rw [Finset.sum_add_distrib]

lemma integrable_randomQuadraticForm_offDiagonal_of_subgaussian {μ : Measure Ω}
    [IsProbabilityMeasure μ] {n : ℕ} (A : Matrix (Fin n) (Fin n) ℝ)
    {X : Fin n → Ω → ℝ} {K : ℝ}
    (h_indep : iIndepFun X μ)
    (hX_subG : ∀ i, HasSubgaussianMGF (X i) ⟨K ^ 2, sq_nonneg K⟩ μ) :
    Integrable (randomQuadraticForm (offDiagonalMatrix A) X) μ := by
  classical
  have hterm : ∀ i j,
      Integrable (fun ω => offDiagonalMatrix A i j * X i ω * X j ω) μ := by
    intro i j
    by_cases hij : i = j
    · subst j
      simp [offDiagonalMatrix]
    · have hmul : Integrable ((X i) * (X j)) μ :=
        (h_indep.indepFun hij).integrable_mul (hX_subG i).integrable
          (hX_subG j).integrable
      convert hmul.const_mul (A i j) using 1
      ext ω
      simp [offDiagonalMatrix, hij]
      ring
  unfold randomQuadraticForm quadraticForm
  exact integrable_finsetSum _ fun i _ =>
    integrable_finsetSum _ fun j _ => hterm i j

lemma integral_randomQuadraticForm_offDiagonal_eq_zero {μ : Measure Ω}
    [IsProbabilityMeasure μ] {n : ℕ} (A : Matrix (Fin n) (Fin n) ℝ)
    {X : Fin n → Ω → ℝ} {K : ℝ}
    (h_indep : iIndepFun X μ)
    (hX_subG : ∀ i, HasSubgaussianMGF (X i) ⟨K ^ 2, sq_nonneg K⟩ μ) :
    ∫ ω, randomQuadraticForm (offDiagonalMatrix A) X ω ∂μ = 0 := by
  classical
  have hterm_int : ∀ i j,
      Integrable (fun ω => offDiagonalMatrix A i j * X i ω * X j ω) μ := by
    intro i j
    by_cases hij : i = j
    · subst j
      simp [offDiagonalMatrix]
    · have hmul : Integrable ((X i) * (X j)) μ :=
        (h_indep.indepFun hij).integrable_mul (hX_subG i).integrable
          (hX_subG j).integrable
      convert hmul.const_mul (A i j) using 1
      ext ω
      simp [offDiagonalMatrix, hij]
      ring
  have hrow_int : ∀ i,
      Integrable (fun ω => ∑ j, offDiagonalMatrix A i j * X i ω * X j ω) μ := by
    intro i
    exact integrable_finsetSum _ fun j _ => hterm_int i j
  calc
    ∫ ω, randomQuadraticForm (offDiagonalMatrix A) X ω ∂μ
        = ∑ i, ∑ j,
            ∫ ω, offDiagonalMatrix A i j * X i ω * X j ω ∂μ := by
          unfold randomQuadraticForm quadraticForm
          rw [integral_finsetSum]
          · apply Finset.sum_congr rfl
            intro i _
            rw [integral_finsetSum]
            intro j _
            exact hterm_int i j
          · intro i _
            exact hrow_int i
    _ = 0 := by
          apply Finset.sum_eq_zero
          intro i _
          apply Finset.sum_eq_zero
          intro j _
          by_cases hij : i = j
          · subst j
            simp [offDiagonalMatrix]
          · have hpair := h_indep.indepFun hij
            have hmeas_i : AEStronglyMeasurable (X i) μ :=
              (hX_subG i).integrable.aestronglyMeasurable
            have hmeas_j : AEStronglyMeasurable (X j) μ :=
              (hX_subG j).integrable.aestronglyMeasurable
            have hmul_eq :
                ∫ ω, X i ω * X j ω ∂μ =
                  (∫ ω, X i ω ∂μ) * (∫ ω, X j ω ∂μ) :=
              hpair.integral_fun_mul_eq_mul_integral hmeas_i hmeas_j
            calc
              ∫ ω, offDiagonalMatrix A i j * X i ω * X j ω ∂μ
                  = ∫ ω, A i j * (X i ω * X j ω) ∂μ := by
                    apply integral_congr_ae
                    filter_upwards with ω
                    simp [offDiagonalMatrix, hij]
                    ring
              _ = A i j * ∫ ω, X i ω * X j ω ∂μ := by
                    rw [integral_const_mul]
              _ = A i j * ((∫ ω, X i ω ∂μ) * (∫ ω, X j ω ∂μ)) := by
                    rw [hmul_eq]
              _ = 0 := by
                    rw [hasSubgaussianMGF_integral_eq_zero (hX_subG i)]
                    ring

lemma centeredQuadraticForm_eq_diagonal_add_offDiagonal {μ : Measure Ω}
    [IsProbabilityMeasure μ] {n : ℕ} (A : Matrix (Fin n) (Fin n) ℝ)
    {X : Fin n → Ω → ℝ} {K : ℝ}
    (h_indep : iIndepFun X μ)
    (hX_subG : ∀ i, HasSubgaussianMGF (X i) ⟨K ^ 2, sq_nonneg K⟩ μ) :
    centeredQuadraticForm μ A X =
      fun ω => diagonalCenteredQuadraticForm μ A X ω +
        quadraticForm (offDiagonalMatrix A) (fun i => X i ω) := by
  classical
  have hdiag_term_int : ∀ i, Integrable (fun ω => A i i * X i ω ^ 2) μ := by
    intro i
    exact ((integrable_pow_of_integrable_exp_mul
      (X := X i) (t := 1) one_ne_zero
      (by simpa using (hX_subG i).integrable_exp_mul 1)
      (by simpa using (hX_subG i).integrable_exp_mul (-1)) 2).const_mul (A i i))
  have hdiag_int :
      Integrable (fun ω => ∑ i, A i i * X i ω ^ 2) μ :=
    integrable_finsetSum _ fun i _ => hdiag_term_int i
  have hoff_int :
      Integrable (fun ω => quadraticForm (offDiagonalMatrix A) (fun i => X i ω)) μ :=
    integrable_randomQuadraticForm_offDiagonal_of_subgaussian A h_indep hX_subG
  have hqfun :
      (fun ω => randomQuadraticForm A X ω) =
        fun ω => (∑ i, A i i * X i ω ^ 2) +
          quadraticForm (offDiagonalMatrix A) (fun i => X i ω) := by
    ext ω
    exact quadraticForm_eq_diag_add_offDiagonal A (fun i => X i ω)
  ext ω
  unfold centeredQuadraticForm diagonalCenteredQuadraticForm
  rw [show randomQuadraticForm A X ω =
      (∑ i, A i i * X i ω ^ 2) +
        quadraticForm (offDiagonalMatrix A) (fun i => X i ω) by
        exact congrFun hqfun ω]
  rw [show (∫ ω, randomQuadraticForm A X ω ∂μ) =
      ∫ ω, (∑ i, A i i * X i ω ^ 2) +
        quadraticForm (offDiagonalMatrix A) (fun i => X i ω) ∂μ by rw [hqfun]]
  rw [integral_add hdiag_int hoff_int]
  have hoff_zero :
      ∫ a, quadraticForm (offDiagonalMatrix A) (fun i => X i a) ∂μ = 0 := by
    simpa [randomQuadraticForm] using
      integral_randomQuadraticForm_offDiagonal_eq_zero A h_indep hX_subG
  rw [hoff_zero]
  rw [integral_finsetSum]
  · calc
      (∑ i, A i i * X i ω ^ 2) + quadraticForm (offDiagonalMatrix A)
          (fun i => X i ω) - (∑ i, ∫ (ω : Ω), A i i * X i ω ^ 2 ∂μ + 0)
          = (∑ i, (A i i * X i ω ^ 2 -
              ∫ (ω : Ω), A i i * X i ω ^ 2 ∂μ)) +
              quadraticForm (offDiagonalMatrix A) (fun i => X i ω) := by
              let O : ℝ := quadraticForm (offDiagonalMatrix A) (fun i => X i ω)
              let f : Fin n → ℝ := fun i => A i i * X i ω ^ 2
              let g : Fin n → ℝ := fun i => ∫ (ω : Ω), A i i * X i ω ^ 2 ∂μ
              have hsum_sub : (∑ i, f i) - ∑ i, g i = ∑ i, (f i - g i) := by
                rw [← Finset.sum_sub_distrib]
              change (∑ i, f i) + O - (∑ i, g i + 0) =
                (∑ i, (f i - g i)) + O
              rw [← hsum_sub]
              ring
      _ = (∑ i, A i i * (X i ω ^ 2 - ∫ (ω : Ω), X i ω ^ 2 ∂μ)) +
            quadraticForm (offDiagonalMatrix A) (fun i => X i ω) := by
          congr 1
          apply Finset.sum_congr rfl
          intro i _
          rw [integral_const_mul]
          ring
  · intro i _
    exact hdiag_term_int i

lemma hasSubgaussianMGF_mono_param {μ : Measure Ω} {X : Ω → ℝ} {c d : ℝ≥0}
    (h : HasSubgaussianMGF X c μ) (hcd : (c : ℝ) ≤ d) :
    HasSubgaussianMGF X d μ where
  integrable_exp_mul t := h.integrable_exp_mul t
  mgf_le t := by
    have hmul : (c : ℝ) * t ^ 2 ≤ (d : ℝ) * t ^ 2 :=
      mul_le_mul_of_nonneg_right hcd (sq_nonneg t)
    calc
      mgf X μ t ≤ exp ((c : ℝ) * t ^ 2 / 2) := h.mgf_le t
      _ ≤ exp ((d : ℝ) * t ^ 2 / 2) := by
          exact exp_le_exp.mpr (by linarith)

/-- A finite independent linear combination of sub-Gaussian variables is sub-Gaussian. -/
lemma hasSubgaussianMGF_finset_sum_const_mul_of_iIndepFun {ι : Type*} {μ : Measure Ω}
    {X : ι → Ω → ℝ} (h_indep : iIndepFun X μ) {c : ι → ℝ≥0}
    {s : Finset ι} (h_subG : ∀ i ∈ s, HasSubgaussianMGF (X i) (c i) μ)
    (a : ι → ℝ) :
    HasSubgaussianMGF (fun ω => ∑ i ∈ s, a i * X i ω)
      (∑ i ∈ s, Real.toNNReal (a i ^ 2) * c i) μ := by
  have h_indep_mul :
      iIndepFun (fun i => fun ω => a i * X i ω) μ := by
    simpa [Function.comp_def] using
      h_indep.comp (fun i x => a i * x) (fun i => by fun_prop)
  have h_subG_mul :
      ∀ i ∈ s, HasSubgaussianMGF (fun ω => a i * X i ω)
      (Real.toNNReal (a i ^ 2) * c i) μ := by
    intro i hi
    convert (h_subG i hi).const_mul (a i) using 1
    rw [Real.toNNReal_of_nonneg (sq_nonneg (a i))]
    congr 1
  exact HasSubgaussianMGF.sum_of_iIndepFun h_indep_mul h_subG_mul

/-- Fixed linear forms of an independent sub-Gaussian coordinate vector are sub-Gaussian. -/
lemma inner_randomVector_hasSubgaussianMGF_of_iIndepFun {μ : Measure Ω}
    {n : ℕ} {X : Fin n → Ω → ℝ} {K : ℝ}
    (h_indep : iIndepFun X μ)
    (hX_subG : ∀ i, HasSubgaussianMGF (X i) ⟨K ^ 2, sq_nonneg K⟩ μ)
    (v : EuclideanSpace ℝ (Fin n)) :
    HasSubgaussianMGF (fun ω => inner ℝ v (randomVector X ω))
      ⟨K ^ 2 * ‖v‖ ^ 2, mul_nonneg (sq_nonneg K) (sq_nonneg ‖v‖)⟩ μ := by
  let cK : ℝ≥0 := Real.toNNReal (K ^ 2)
  have hcK : cK = (⟨K ^ 2, sq_nonneg K⟩ : ℝ≥0) := by
    ext
    exact Real.coe_toNNReal (K ^ 2) (sq_nonneg K)
  have hX_subG' :
      ∀ i ∈ (Finset.univ : Finset (Fin n)), HasSubgaussianMGF (X i) cK μ := by
    intro i _
    rw [hcK]
    exact hX_subG i
  have hsum :=
    hasSubgaussianMGF_finset_sum_const_mul_of_iIndepFun
      (μ := μ) h_indep (s := Finset.univ) hX_subG' (fun i => v i)
  have hsum_inner :
      HasSubgaussianMGF (fun ω => inner ℝ v (randomVector X ω))
        (∑ i ∈ (Finset.univ : Finset (Fin n)), Real.toNNReal (v i ^ 2) * cK) μ := by
    refine hsum.congr (ae_of_all _ fun ω => ?_)
    change (∑ i : Fin n, v i * X i ω) = inner ℝ v (randomVector X ω)
    rw [PiLp.inner_apply]
    simp only [randomVector_apply, RCLike.inner_apply, conj_trivial]
    apply Finset.sum_congr rfl
    intro i _
    ring
  have hparam :
      (((∑ i ∈ (Finset.univ : Finset (Fin n)), Real.toNNReal (v i ^ 2) * cK) : ℝ≥0) :
          ℝ) ≤ K ^ 2 * ‖v‖ ^ 2 := by
    rw [EuclideanSpace.real_norm_sq_eq]
    simp only [cK, NNReal.coe_sum, NNReal.coe_mul]
    rw [Finset.mul_sum]
    apply le_of_eq
    apply Finset.sum_congr rfl
    intro i _
    rw [Real.coe_toNNReal (v i ^ 2) (sq_nonneg (v i)),
      Real.coe_toNNReal (K ^ 2) (sq_nonneg K)]
    ring
  exact hasSubgaussianMGF_mono_param hsum_inner hparam

/-- Gaussian-comparison square-exponential bound for a linear image of an independent
sub-Gaussian vector. -/
lemma integral_exp_norm_toEuclideanCLM_randomVector_sq_le {μ : Measure Ω}
    [IsProbabilityMeasure μ] {n : ℕ} (A : Matrix (Fin n) (Fin n) ℝ)
    {X : Fin n → Ω → ℝ} {K θ : ℝ} (hθ : 0 ≤ θ)
    (h_indep : iIndepFun X μ)
    (hX_subG : ∀ i, HasSubgaussianMGF (X i) ⟨K ^ 2, sq_nonneg K⟩ μ)
    (hsmall : θ * K ^ 2 * operatorNorm A ^ 2 * exp 1 ≤ 1 / 2) :
    ∫ ω, exp (θ *
        ‖Matrix.toEuclideanCLM (𝕜 := ℝ) A (randomVector X ω)‖ ^ 2) ∂μ ≤
      exp (2 * exp 1 ^ 2 * θ * K ^ 2 * frobeniusNorm A ^ 2) := by
  let E := EuclideanSpace ℝ (Fin n)
  let γ : Measure E := stdGaussian E
  let T : E →L[ℝ] E := Matrix.toEuclideanCLM (𝕜 := ℝ) A
  let U : E →L[ℝ] E := ContinuousLinearMap.adjoint T
  let S : E →L[ℝ] E := ContinuousLinearMap.adjoint U ∘L U
  let hSpos : S.toLinearMap.IsPositive :=
    (ContinuousLinearMap.isPositive_adjoint_comp_self U).toLinearMap
  let s : ℝ := sqrt (2 * θ)
  let f : E × Ω → ℝ := fun p => exp (s * inner ℝ (U p.1) (randomVector X p.2))
  have hs_nonneg : 0 ≤ s := by
    dsimp [s]
    exact sqrt_nonneg _
  have hs_sq : s ^ 2 = 2 * θ := by
    dsimp [s]
    rw [sq_sqrt]
    nlinarith
  have hKθ_nonneg : 0 ≤ K ^ 2 * θ := mul_nonneg (sq_nonneg K) hθ
  have hUnorm : ‖U‖ = operatorNorm A := by
    dsimp [U, T, operatorNorm]
    exact ContinuousLinearMap.adjoint.norm_map _
  have hsmall_eig :
      ∀ i : Fin (Module.finrank ℝ E),
        (K ^ 2 * θ) * hSpos.isSymmetric.eigenvalues rfl i * exp 1 ≤ 1 / 2 := by
    intro i
    have heig_le : hSpos.isSymmetric.eigenvalues rfl i ≤ ‖U‖ ^ 2 := by
      simpa [S, hSpos] using eigenvalue_adjoint_comp_le_opNorm_sq U i
    have hstep :
        (K ^ 2 * θ) * hSpos.isSymmetric.eigenvalues rfl i * exp 1 ≤
          (K ^ 2 * θ) * ‖U‖ ^ 2 * exp 1 := by
      exact mul_le_mul_of_nonneg_right
        (mul_le_mul_of_nonneg_left heig_le hKθ_nonneg) (exp_nonneg 1)
    exact hstep.trans (by
      rw [hUnorm]
      nlinarith [hsmall])
  have hsmall_eig_lt :
      ∀ i : Fin (Module.finrank ℝ E),
        (K ^ 2 * θ) * hSpos.isSymmetric.eigenvalues rfl i * exp 1 < 1 := by
    intro i
    linarith [hsmall_eig i]
  have hquad_int :
      Integrable (fun g : E => exp ((K ^ 2 * θ) * inner ℝ (S g) g)) γ :=
    integrable_exp_quadratic_stdGaussian S hSpos hKθ_nonneg hsmall_eig_lt
  have hf_aesm : AEStronglyMeasurable f (γ.prod μ) := by
    have hX_aemeas : AEMeasurable (randomVector X) μ :=
      randomVector_aemeasurable fun i => (hX_subG i).aemeasurable
    have hU_aemeas :
        AEMeasurable (fun p : E × Ω => U p.1) (γ.prod μ) :=
      AEMeasurable.comp_fst (μ := γ) (ν := μ) U.continuous.aemeasurable
    have hX_prod :
        AEMeasurable (fun p : E × Ω => randomVector X p.2) (γ.prod μ) :=
      AEMeasurable.comp_snd (μ := γ) (ν := μ) hX_aemeas
    have hinner :
        AEMeasurable
          (fun p : E × Ω => inner ℝ (U p.1) (randomVector X p.2)) (γ.prod μ) := by
      exact AEMeasurable.inner hU_aemeas hX_prod
    exact ((hinner.const_mul s).exp).aestronglyMeasurable
  have hinner_bound : (fun g : E => ∫ ω, f (g, ω) ∂μ) ≤ᵐ[γ]
      fun g : E => exp ((K ^ 2 * θ) * inner ℝ (S g) g) := by
    filter_upwards with g
    have hlin := inner_randomVector_hasSubgaussianMGF_of_iIndepFun h_indep hX_subG (U g)
    have hmgf := hlin.mgf_le s
    have hnorm_sq : ‖U g‖ ^ 2 = inner ℝ (S g) g := by
      simpa [S] using ContinuousLinearMap.apply_norm_sq_eq_inner_adjoint_left U g
    calc
      ∫ ω, f (g, ω) ∂μ
          = mgf (fun ω => inner ℝ (U g) (randomVector X ω)) μ s := by
            rfl
      _ ≤ exp ((K ^ 2 * ‖U g‖ ^ 2) * s ^ 2 / 2) := hmgf
      _ = exp ((K ^ 2 * θ) * inner ℝ (S g) g) := by
            rw [hs_sq, hnorm_sq]
            congr 1
            ring
  have hnorm_inner_bound : (fun g : E => ∫ ω, ‖f (g, ω)‖ ∂μ) ≤ᵐ[γ]
      fun g : E => exp ((K ^ 2 * θ) * inner ℝ (S g) g) := by
    filter_upwards [hinner_bound] with g hg
    have hnorm_eq : ∫ ω, ‖f (g, ω)‖ ∂μ = ∫ ω, f (g, ω) ∂μ := by
      apply integral_congr_ae
      filter_upwards with ω
      dsimp [f]
      rw [abs_of_nonneg (exp_nonneg _)]
    rwa [hnorm_eq]
  have hf_int : Integrable f (γ.prod μ) := by
    rw [integrable_prod_iff hf_aesm]
    constructor
    · filter_upwards with g
      exact
        (inner_randomVector_hasSubgaussianMGF_of_iIndepFun h_indep hX_subG (U g)).integrable_exp_mul s
    · have hnorm_outer_bound :
          (fun g : E => ‖∫ ω, ‖f (g, ω)‖ ∂μ‖) ≤ᵐ[γ]
            fun g : E => exp ((K ^ 2 * θ) * inner ℝ (S g) g) := by
        filter_upwards [hnorm_inner_bound] with g hg
        rw [Real.norm_eq_abs, abs_of_nonneg]
        · exact hg
        · exact integral_nonneg fun ω => norm_nonneg _
      exact Integrable.mono' hquad_int hf_aesm.norm.integral_prod_right' hnorm_outer_bound
  have hsum_eq :
      (∑ i : Fin (Module.finrank ℝ E), hSpos.isSymmetric.eigenvalues rfl i) =
        frobeniusNormSq A := by
    have htrace_eigs :
        (LinearMap.trace ℝ E) S.toLinearMap =
          ∑ i : Fin (Module.finrank ℝ E), hSpos.isSymmetric.eigenvalues rfl i := by
      simpa using hSpos.isSymmetric.trace_eq_sum_eigenvalues
        (by rfl : Module.finrank ℝ E = Module.finrank ℝ E)
    have htrace_frob :
        (LinearMap.trace ℝ E) S.toLinearMap = frobeniusNormSq A := by
      have hUeq : U = Matrix.toEuclideanCLM (𝕜 := ℝ) A.conjTranspose := by
        dsimp [U, T]
        exact toEuclideanCLM_adjoint A
      calc
        (LinearMap.trace ℝ E) S.toLinearMap
            = (LinearMap.trace ℝ E)
                ((ContinuousLinearMap.adjoint U ∘L U).toLinearMap) := by
                rfl
        _ = (LinearMap.trace ℝ E)
                ((ContinuousLinearMap.adjoint
                    (Matrix.toEuclideanCLM (𝕜 := ℝ) A.conjTranspose) ∘L
                  Matrix.toEuclideanCLM (𝕜 := ℝ) A.conjTranspose).toLinearMap) := by
                rw [hUeq]
        _ = frobeniusNormSq A.conjTranspose :=
                trace_adjoint_comp_toEuclideanCLM_eq_frobeniusNormSq A.conjTranspose
        _ = frobeniusNormSq A := frobeniusNormSq_conjTranspose A
    exact htrace_eigs.symm.trans htrace_frob
  have hprod_bound :
      ∫ p : E × Ω, f p ∂(γ.prod μ) ≤
        exp (2 * exp 1 ^ 2 * θ * K ^ 2 * frobeniusNorm A ^ 2) := by
    calc
      ∫ p : E × Ω, f p ∂(γ.prod μ)
          = ∫ g : E, ∫ ω, f (g, ω) ∂μ ∂γ := by
              rw [integral_prod f hf_int]
      _ ≤ ∫ g : E, exp ((K ^ 2 * θ) * inner ℝ (S g) g) ∂γ := by
              exact integral_mono_ae hf_int.integral_prod_left hquad_int hinner_bound
      _ ≤ exp (2 * exp 1 ^ 2 * (K ^ 2 * θ) *
            (∑ i : Fin (Module.finrank ℝ E), hSpos.isSymmetric.eigenvalues rfl i)) :=
              integral_exp_quadratic_stdGaussian_le S hSpos hKθ_nonneg hsmall_eig
      _ = exp (2 * exp 1 ^ 2 * θ * K ^ 2 * frobeniusNorm A ^ 2) := by
              rw [hsum_eq, frobeniusNorm_sq]
              congr 1
              ring
  have hleft_eq :
      ∫ ω, exp (θ * ‖T (randomVector X ω)‖ ^ 2) ∂μ =
        ∫ ω, ∫ g : E, f (g, ω) ∂γ ∂μ := by
    apply integral_congr_ae
    filter_upwards with ω
    have hfg : (fun g : E => f (g, ω)) =
        fun g : E => exp (s * inner ℝ (T (randomVector X ω)) g) := by
      ext g
      dsimp [f, U]
      rw [ContinuousLinearMap.adjoint_inner_left T (randomVector X ω) g]
      rw [real_inner_comm]
    rw [hfg, integral_exp_innerSL_stdGaussian (E := E) (T (randomVector X ω)) s]
    rw [hs_sq]
    congr 1
    ring
  calc
      ∫ ω, exp (θ * ‖Matrix.toEuclideanCLM (𝕜 := ℝ) A (randomVector X ω)‖ ^ 2) ∂μ
          = ∫ ω, exp (θ * ‖T (randomVector X ω)‖ ^ 2) ∂μ := by rfl
      _ = ∫ ω, ∫ g : E, f (g, ω) ∂γ ∂μ := hleft_eq
      _ = ∫ q : Ω × E, f q.swap ∂(μ.prod γ) := by
          exact (integral_prod (fun q : Ω × E => f q.swap) hf_int.swap).symm
      _ ≤ exp (2 * exp 1 ^ 2 * θ * K ^ 2 * frobeniusNorm A ^ 2) := by
          rw [integral_prod_swap f]
          exact hprod_bound

lemma integrable_exp_norm_toEuclideanCLM_randomVector_sq {μ : Measure Ω}
    [IsProbabilityMeasure μ] {n : ℕ} (A : Matrix (Fin n) (Fin n) ℝ)
    {X : Fin n → Ω → ℝ} {K θ : ℝ} (hθ : 0 ≤ θ)
    (h_indep : iIndepFun X μ)
    (hX_subG : ∀ i, HasSubgaussianMGF (X i) ⟨K ^ 2, sq_nonneg K⟩ μ)
    (hsmall : θ * K ^ 2 * operatorNorm A ^ 2 * exp 1 < 1) :
    Integrable (fun ω =>
      exp (θ * ‖Matrix.toEuclideanCLM (𝕜 := ℝ) A (randomVector X ω)‖ ^ 2)) μ := by
  let E := EuclideanSpace ℝ (Fin n)
  let γ : Measure E := stdGaussian E
  let T : E →L[ℝ] E := Matrix.toEuclideanCLM (𝕜 := ℝ) A
  let U : E →L[ℝ] E := ContinuousLinearMap.adjoint T
  let S : E →L[ℝ] E := ContinuousLinearMap.adjoint U ∘L U
  let hSpos : S.toLinearMap.IsPositive :=
    (ContinuousLinearMap.isPositive_adjoint_comp_self U).toLinearMap
  let s : ℝ := sqrt (2 * θ)
  let f : E × Ω → ℝ := fun p => exp (s * inner ℝ (U p.1) (randomVector X p.2))
  have hs_sq : s ^ 2 = 2 * θ := by
    dsimp [s]
    rw [sq_sqrt]
    nlinarith
  have hKθ_nonneg : 0 ≤ K ^ 2 * θ := mul_nonneg (sq_nonneg K) hθ
  have hUnorm : ‖U‖ = operatorNorm A := by
    dsimp [U, T, operatorNorm]
    exact ContinuousLinearMap.adjoint.norm_map _
  have hsmall_eig_lt :
      ∀ i : Fin (Module.finrank ℝ E),
        (K ^ 2 * θ) * hSpos.isSymmetric.eigenvalues rfl i * exp 1 < 1 := by
    intro i
    have heig_le : hSpos.isSymmetric.eigenvalues rfl i ≤ ‖U‖ ^ 2 := by
      simpa [S, hSpos] using eigenvalue_adjoint_comp_le_opNorm_sq U i
    have hstep :
        (K ^ 2 * θ) * hSpos.isSymmetric.eigenvalues rfl i * exp 1 ≤
          (K ^ 2 * θ) * ‖U‖ ^ 2 * exp 1 := by
      exact mul_le_mul_of_nonneg_right
        (mul_le_mul_of_nonneg_left heig_le hKθ_nonneg) (exp_nonneg 1)
    exact lt_of_le_of_lt hstep (by
      rw [hUnorm]
      nlinarith [hsmall])
  have hquad_int :
      Integrable (fun g : E => exp ((K ^ 2 * θ) * inner ℝ (S g) g)) γ :=
    integrable_exp_quadratic_stdGaussian S hSpos hKθ_nonneg hsmall_eig_lt
  have hf_aesm : AEStronglyMeasurable f (γ.prod μ) := by
    have hX_aemeas : AEMeasurable (randomVector X) μ :=
      randomVector_aemeasurable fun i => (hX_subG i).aemeasurable
    have hU_aemeas :
        AEMeasurable (fun p : E × Ω => U p.1) (γ.prod μ) :=
      AEMeasurable.comp_fst (μ := γ) (ν := μ) U.continuous.aemeasurable
    have hX_prod :
        AEMeasurable (fun p : E × Ω => randomVector X p.2) (γ.prod μ) :=
      AEMeasurable.comp_snd (μ := γ) (ν := μ) hX_aemeas
    have hinner :
        AEMeasurable
          (fun p : E × Ω => inner ℝ (U p.1) (randomVector X p.2)) (γ.prod μ) := by
      exact AEMeasurable.inner hU_aemeas hX_prod
    exact ((hinner.const_mul s).exp).aestronglyMeasurable
  have hinner_bound : (fun g : E => ∫ ω, f (g, ω) ∂μ) ≤ᵐ[γ]
      fun g : E => exp ((K ^ 2 * θ) * inner ℝ (S g) g) := by
    filter_upwards with g
    have hlin := inner_randomVector_hasSubgaussianMGF_of_iIndepFun h_indep hX_subG (U g)
    have hmgf := hlin.mgf_le s
    have hnorm_sq : ‖U g‖ ^ 2 = inner ℝ (S g) g := by
      simpa [S] using ContinuousLinearMap.apply_norm_sq_eq_inner_adjoint_left U g
    calc
      ∫ ω, f (g, ω) ∂μ
          = mgf (fun ω => inner ℝ (U g) (randomVector X ω)) μ s := by
            rfl
      _ ≤ exp ((K ^ 2 * ‖U g‖ ^ 2) * s ^ 2 / 2) := hmgf
      _ = exp ((K ^ 2 * θ) * inner ℝ (S g) g) := by
            rw [hs_sq, hnorm_sq]
            congr 1
            ring
  have hnorm_inner_bound : (fun g : E => ∫ ω, ‖f (g, ω)‖ ∂μ) ≤ᵐ[γ]
      fun g : E => exp ((K ^ 2 * θ) * inner ℝ (S g) g) := by
    filter_upwards [hinner_bound] with g hg
    have hnorm_eq : ∫ ω, ‖f (g, ω)‖ ∂μ = ∫ ω, f (g, ω) ∂μ := by
      apply integral_congr_ae
      filter_upwards with ω
      dsimp [f]
      rw [abs_of_nonneg (exp_nonneg _)]
    rwa [hnorm_eq]
  have hf_int : Integrable f (γ.prod μ) := by
    rw [integrable_prod_iff hf_aesm]
    constructor
    · filter_upwards with g
      exact
        (inner_randomVector_hasSubgaussianMGF_of_iIndepFun h_indep hX_subG (U g)).integrable_exp_mul s
    · have hnorm_outer_bound :
          (fun g : E => ‖∫ ω, ‖f (g, ω)‖ ∂μ‖) ≤ᵐ[γ]
            fun g : E => exp ((K ^ 2 * θ) * inner ℝ (S g) g) := by
        filter_upwards [hnorm_inner_bound] with g hg
        rw [Real.norm_eq_abs, abs_of_nonneg]
        · exact hg
        · exact integral_nonneg fun ω => norm_nonneg _
      exact Integrable.mono' hquad_int hf_aesm.norm.integral_prod_right' hnorm_outer_bound
  have hleft_eq :
      (fun ω => exp (θ * ‖T (randomVector X ω)‖ ^ 2)) =
        fun ω => ∫ g : E, f (g, ω) ∂γ := by
    ext ω
    have hfg : (fun g : E => f (g, ω)) =
        fun g : E => exp (s * inner ℝ (T (randomVector X ω)) g) := by
      ext g
      dsimp [f, U]
      rw [ContinuousLinearMap.adjoint_inner_left T (randomVector X ω) g]
      rw [real_inner_comm]
    rw [hfg, integral_exp_innerSL_stdGaussian (E := E) (T (randomVector X ω)) s]
    rw [hs_sq]
    congr 1
    ring
  have hprod_inner_int :
      Integrable (fun ω => ∫ g : E, f (g, ω) ∂γ) μ :=
    (hf_int.swap).integral_prod_left
  simpa [T, hleft_eq] using hprod_inner_int

lemma integrable_exp_inner_toEuclideanCLM_randomVector_prod {μ : Measure Ω}
    [IsProbabilityMeasure μ] {n : ℕ} (A : Matrix (Fin n) (Fin n) ℝ)
    {X : Fin n → Ω → ℝ} {K l : ℝ}
    (h_indep : iIndepFun X μ)
    (hX_subG : ∀ i, HasSubgaussianMGF (X i) ⟨K ^ 2, sq_nonneg K⟩ μ)
    (hsmall : (K ^ 2 * l ^ 2 / 2) * K ^ 2 * operatorNorm A ^ 2 * exp 1 < 1) :
    Integrable
      (fun p : Ω × Ω =>
        exp (l * inner ℝ
          (Matrix.toEuclideanCLM (𝕜 := ℝ) A (randomVector X p.1))
          (randomVector X p.2))) (μ.prod μ) := by
  let E := EuclideanSpace ℝ (Fin n)
  let T : E →L[ℝ] E := Matrix.toEuclideanCLM (𝕜 := ℝ) A
  let θ : ℝ := K ^ 2 * l ^ 2 / 2
  let f : Ω × Ω → ℝ :=
    fun p => exp (l * inner ℝ (T (randomVector X p.1)) (randomVector X p.2))
  have hX_aemeas : AEMeasurable (randomVector X) μ :=
    randomVector_aemeasurable fun i => (hX_subG i).aemeasurable
  have hf_aesm : AEStronglyMeasurable f (μ.prod μ) := by
    have hleft :
        AEMeasurable (fun p : Ω × Ω => T (randomVector X p.1)) (μ.prod μ) :=
      T.continuous.aemeasurable.comp_aemeasurable
        (AEMeasurable.comp_fst (μ := μ) (ν := μ) hX_aemeas)
    have hright :
        AEMeasurable (fun p : Ω × Ω => randomVector X p.2) (μ.prod μ) :=
      AEMeasurable.comp_snd (μ := μ) (ν := μ) hX_aemeas
    have hinner :
        AEMeasurable
          (fun p : Ω × Ω => inner ℝ (T (randomVector X p.1)) (randomVector X p.2))
          (μ.prod μ) :=
      AEMeasurable.inner hleft hright
    exact ((hinner.const_mul l).exp).aestronglyMeasurable
  have hθ_nonneg : 0 ≤ θ := by
    dsimp [θ]
    positivity
  have hsquare_int :
      Integrable (fun ω =>
        exp (θ * ‖Matrix.toEuclideanCLM (𝕜 := ℝ) A (randomVector X ω)‖ ^ 2)) μ :=
    integrable_exp_norm_toEuclideanCLM_randomVector_sq A hθ_nonneg
      h_indep hX_subG (by simpa [θ, T] using hsmall)
  rw [integrable_prod_iff hf_aesm]
  constructor
  · filter_upwards with ω₁
    exact
      (inner_randomVector_hasSubgaussianMGF_of_iIndepFun h_indep hX_subG
        (T (randomVector X ω₁))).integrable_exp_mul l
  · have hinner_norm_bound :
        (fun ω₁ => ∫ ω₂, ‖f (ω₁, ω₂)‖ ∂μ) ≤ᵐ[μ]
          fun ω₁ => exp (θ * ‖T (randomVector X ω₁)‖ ^ 2) := by
      filter_upwards with ω₁
      have hnorm_eq :
          ∫ ω₂, ‖f (ω₁, ω₂)‖ ∂μ = ∫ ω₂, f (ω₁, ω₂) ∂μ := by
        apply integral_congr_ae
        filter_upwards with ω₂
        dsimp [f]
        rw [abs_of_nonneg (exp_nonneg _)]
      have hlin :=
        inner_randomVector_hasSubgaussianMGF_of_iIndepFun h_indep hX_subG
          (T (randomVector X ω₁))
      have hmgf := hlin.mgf_le l
      rw [hnorm_eq]
      calc
        ∫ ω₂, f (ω₁, ω₂) ∂μ
            = mgf (fun ω₂ => inner ℝ (T (randomVector X ω₁)) (randomVector X ω₂)) μ l := by
              rfl
        _ ≤ exp ((K ^ 2 * ‖T (randomVector X ω₁)‖ ^ 2) * l ^ 2 / 2) := hmgf
        _ = exp (θ * ‖T (randomVector X ω₁)‖ ^ 2) := by
              congr 1
              dsimp [θ]
              ring
    have houter_bound :
        (fun ω₁ => ‖∫ ω₂, ‖f (ω₁, ω₂)‖ ∂μ‖) ≤ᵐ[μ]
          fun ω₁ => exp (θ * ‖T (randomVector X ω₁)‖ ^ 2) := by
      filter_upwards [hinner_norm_bound] with ω₁ hω₁
      rw [Real.norm_eq_abs, abs_of_nonneg]
      · exact hω₁
      · exact integral_nonneg fun ω₂ => norm_nonneg _
    exact Integrable.mono' (by simpa [θ, T] using hsquare_int)
      hf_aesm.norm.integral_prod_right' houter_bound

lemma integral_exp_inner_toEuclideanCLM_randomVector_prod_le {μ : Measure Ω}
    [IsProbabilityMeasure μ] {n : ℕ} (A : Matrix (Fin n) (Fin n) ℝ)
    {X : Fin n → Ω → ℝ} {K l : ℝ}
    (h_indep : iIndepFun X μ)
    (hX_subG : ∀ i, HasSubgaussianMGF (X i) ⟨K ^ 2, sq_nonneg K⟩ μ)
    (hsmall : (K ^ 2 * l ^ 2 / 2) * K ^ 2 * operatorNorm A ^ 2 * exp 1 ≤ 1 / 2) :
    ∫ p : Ω × Ω,
        exp (l * inner ℝ
          (Matrix.toEuclideanCLM (𝕜 := ℝ) A (randomVector X p.1))
          (randomVector X p.2)) ∂(μ.prod μ) ≤
      exp (exp 1 ^ 2 * l ^ 2 * K ^ 4 * frobeniusNorm A ^ 2) := by
  let E := EuclideanSpace ℝ (Fin n)
  let T : E →L[ℝ] E := Matrix.toEuclideanCLM (𝕜 := ℝ) A
  let θ : ℝ := K ^ 2 * l ^ 2 / 2
  let f : Ω × Ω → ℝ :=
    fun p => exp (l * inner ℝ (T (randomVector X p.1)) (randomVector X p.2))
  have hf_int : Integrable f (μ.prod μ) :=
    integrable_exp_inner_toEuclideanCLM_randomVector_prod A h_indep hX_subG (by
      exact lt_of_le_of_lt hsmall (by norm_num))
  have hθ_nonneg : 0 ≤ θ := by
    dsimp [θ]
    positivity
  have hsquare_bound :
      ∫ ω,
          exp (θ * ‖Matrix.toEuclideanCLM (𝕜 := ℝ) A (randomVector X ω)‖ ^ 2) ∂μ ≤
        exp (2 * exp 1 ^ 2 * θ * K ^ 2 * frobeniusNorm A ^ 2) :=
    integral_exp_norm_toEuclideanCLM_randomVector_sq_le A hθ_nonneg
      h_indep hX_subG (by simpa [θ, T] using hsmall)
  have hsquare_int :
      Integrable (fun ω =>
        exp (θ * ‖T (randomVector X ω)‖ ^ 2)) μ := by
    simpa [T] using
      integrable_exp_norm_toEuclideanCLM_randomVector_sq
        (μ := μ) (A := A) (X := X) (K := K) (θ := θ)
        hθ_nonneg h_indep hX_subG (lt_of_le_of_lt (by simpa [θ, T] using hsmall) (by norm_num))
  have hinner_bound :
      (fun ω₁ => ∫ ω₂, f (ω₁, ω₂) ∂μ) ≤ᵐ[μ]
        fun ω₁ => exp (θ * ‖T (randomVector X ω₁)‖ ^ 2) := by
    filter_upwards with ω₁
    have hlin :=
      inner_randomVector_hasSubgaussianMGF_of_iIndepFun h_indep hX_subG
        (T (randomVector X ω₁))
    have hmgf := hlin.mgf_le l
    calc
      ∫ ω₂, f (ω₁, ω₂) ∂μ
          = mgf (fun ω₂ => inner ℝ (T (randomVector X ω₁)) (randomVector X ω₂)) μ l := by
            rfl
      _ ≤ exp ((K ^ 2 * ‖T (randomVector X ω₁)‖ ^ 2) * l ^ 2 / 2) := hmgf
      _ = exp (θ * ‖T (randomVector X ω₁)‖ ^ 2) := by
            congr 1
            dsimp [θ]
            ring
  calc
    ∫ p : Ω × Ω, f p ∂(μ.prod μ)
        = ∫ ω₁, ∫ ω₂, f (ω₁, ω₂) ∂μ ∂μ := by
            rw [integral_prod f hf_int]
    _ ≤ ∫ ω₁, exp (θ * ‖T (randomVector X ω₁)‖ ^ 2) ∂μ := by
            exact integral_mono_ae hf_int.integral_prod_left
              hsquare_int
              hinner_bound
    _ ≤ exp (2 * exp 1 ^ 2 * θ * K ^ 2 * frobeniusNorm A ^ 2) := by
            simpa [T] using hsquare_bound
    _ = exp (exp 1 ^ 2 * l ^ 2 * K ^ 4 * frobeniusNorm A ^ 2) := by
            congr 1
            dsimp [θ]
            ring

lemma integral_exp_quadraticForm_cutMatrix_eq_prod {μ : Measure Ω}
    [IsProbabilityMeasure μ] {n : ℕ} (A : Matrix (Fin n) (Fin n) ℝ)
    (s : Finset (Fin n)) {X : Fin n → Ω → ℝ} {l : ℝ}
    (h_indep : iIndepFun X μ) (hX_meas : ∀ i, AEMeasurable (X i) μ)
    (hprod_int :
      Integrable
        (fun p : Ω × Ω =>
          exp (l * inner ℝ
            (Matrix.toEuclideanCLM (𝕜 := ℝ) (cutMatrix A s) (randomVector X p.1))
            (randomVector X p.2))) (μ.prod μ)) :
    ∫ ω, exp (l * quadraticForm (cutMatrix A s) (fun i => X i ω)) ∂μ =
      ∫ p : Ω × Ω,
        exp (l * inner ℝ
          (Matrix.toEuclideanCLM (𝕜 := ℝ) (cutMatrix A s) (randomVector X p.1))
          (randomVector X p.2)) ∂(μ.prod μ) := by
  let E := EuclideanSpace ℝ (Fin n)
  let U : Ω → E :=
    fun ω => coordinateMask ((Finset.univ : Finset (Fin n)) \ s) (randomVector X ω)
  let V : Ω → E := fun ω => coordinateMask s (randomVector X ω)
  let φ : Ω × Ω → E × E := fun p => (U p.1, V p.2)
  let ψ : Ω → E × E := fun ω => (U ω, V ω)
  let g : E × E → ℝ :=
    fun p => exp (l * inner ℝ (Matrix.toEuclideanCLM (𝕜 := ℝ) A p.1) p.2)
  let F : Ω × Ω → ℝ :=
    fun p => exp (l * inner ℝ
      (Matrix.toEuclideanCLM (𝕜 := ℝ) (cutMatrix A s) (randomVector X p.1))
      (randomVector X p.2))
  let H : Ω → ℝ :=
    fun ω => exp (l * quadraticForm (cutMatrix A s) (fun i => X i ω))
  have hU_meas : AEMeasurable U μ := by
    dsimp [U]
    exact coordinateMask_aemeasurable _ hX_meas
  have hV_meas : AEMeasurable V μ := by
    dsimp [V]
    exact coordinateMask_aemeasurable _ hX_meas
  have hψ_meas : AEMeasurable ψ μ := hU_meas.prodMk hV_meas
  have hφ_meas : AEMeasurable φ (μ.prod μ) :=
    (AEMeasurable.comp_fst (μ := μ) (ν := μ) hU_meas).prodMk
      (AEMeasurable.comp_snd (μ := μ) (ν := μ) hV_meas)
  have hUV_indep : U ⟂ᵢ[μ] V := by
    dsimp [U, V]
    exact coordinateMask_indepFun_compl h_indep hX_meas s
  have hmap_same : μ.map ψ = (μ.map U).prod (μ.map V) := by
    simpa [ψ] using hUV_indep.map_prod_eq_prod_map_map hU_meas hV_meas
  have hmap_prod : (μ.map U).prod (μ.map V) = (μ.prod μ).map φ := by
    simpa [φ] using measure_map_prod_map_of_aemeasurable (μ := μ) (ν := μ) hU_meas hV_meas
  have hg_aesm : AEStronglyMeasurable g ((μ.map U).prod (μ.map V)) := by
    dsimp [g]
    fun_prop
  have hg_aesm_prod : AEStronglyMeasurable g ((μ.prod μ).map φ) := by
    rwa [← hmap_prod]
  have hF_eq : F = g ∘ φ := by
    ext p
    dsimp [F, g, φ, U, V]
    rw [inner_toEuclideanCLM_cutMatrix_eq_inner_masks]
  have hcomp_int : Integrable (g ∘ φ) (μ.prod μ) := by
    rw [← hF_eq]
    exact hprod_int
  have hg_int_prod : Integrable g ((μ.prod μ).map φ) :=
    (integrable_map_measure hg_aesm_prod hφ_meas).2 hcomp_int
  have hg_int_same : Integrable g (μ.map ψ) := by
    rw [hmap_same, hmap_prod]
    exact hg_int_prod
  have hg_aesm_same : AEStronglyMeasurable g (μ.map ψ) := by
    rw [hmap_same]
    exact hg_aesm
  have hH_eq : H = g ∘ ψ := by
    ext ω
    dsimp [H, g, ψ, U, V]
    have hq :=
      quadraticForm_eq_inner_toEuclideanCLM (cutMatrix A s) (fun i => X i ω)
    rw [hq]
    rw [inner_toEuclideanCLM_cutMatrix_eq_inner_masks]
    rfl
  calc
    ∫ ω, exp (l * quadraticForm (cutMatrix A s) (fun i => X i ω)) ∂μ
        = ∫ ω, H ω ∂μ := by rfl
    _ = ∫ ω, (g ∘ ψ) ω ∂μ := by rw [hH_eq]
    _ = ∫ ω, g (ψ ω) ∂μ := by rfl
    _ = ∫ q, g q ∂(μ.map ψ) := by
        rw [integral_map hψ_meas hg_aesm_same]
    _ = ∫ q, g q ∂((μ.prod μ).map φ) := by
        rw [hmap_same, hmap_prod]
    _ = ∫ p : Ω × Ω, g (φ p) ∂(μ.prod μ) := by
        rw [integral_map hφ_meas hg_aesm_prod]
    _ = ∫ p : Ω × Ω, (g ∘ φ) p ∂(μ.prod μ) := by rfl
    _ = ∫ p : Ω × Ω, F p ∂(μ.prod μ) := by rw [hF_eq]

lemma integrable_exp_quadraticForm_cutMatrix {μ : Measure Ω}
    [IsProbabilityMeasure μ] {n : ℕ} (A : Matrix (Fin n) (Fin n) ℝ)
    (s : Finset (Fin n)) {X : Fin n → Ω → ℝ} {K l : ℝ}
    (h_indep : iIndepFun X μ)
    (hX_subG : ∀ i, HasSubgaussianMGF (X i) ⟨K ^ 2, sq_nonneg K⟩ μ)
    (hsmall :
      (K ^ 2 * l ^ 2 / 2) * K ^ 2 * operatorNorm (cutMatrix A s) ^ 2 * exp 1 < 1) :
    Integrable
      (fun ω => exp (l * quadraticForm (cutMatrix A s) (fun i => X i ω))) μ := by
  let E := EuclideanSpace ℝ (Fin n)
  let U : Ω → E :=
    fun ω => coordinateMask ((Finset.univ : Finset (Fin n)) \ s) (randomVector X ω)
  let V : Ω → E := fun ω => coordinateMask s (randomVector X ω)
  let ψ : Ω → E × E := fun ω => (U ω, V ω)
  let g : E × E → ℝ :=
    fun p => exp (l * inner ℝ (Matrix.toEuclideanCLM (𝕜 := ℝ) A p.1) p.2)
  let F : Ω × Ω → ℝ :=
    fun p => exp (l * inner ℝ
      (Matrix.toEuclideanCLM (𝕜 := ℝ) (cutMatrix A s) (randomVector X p.1))
      (randomVector X p.2))
  let H : Ω → ℝ :=
    fun ω => exp (l * quadraticForm (cutMatrix A s) (fun i => X i ω))
  have hX_meas : ∀ i, AEMeasurable (X i) μ := fun i => (hX_subG i).aemeasurable
  have hprod_int : Integrable F (μ.prod μ) := by
    dsimp [F]
    exact integrable_exp_inner_toEuclideanCLM_randomVector_prod (cutMatrix A s)
      h_indep hX_subG hsmall
  have hU_meas : AEMeasurable U μ := by
    dsimp [U]
    exact coordinateMask_aemeasurable _ hX_meas
  have hV_meas : AEMeasurable V μ := by
    dsimp [V]
    exact coordinateMask_aemeasurable _ hX_meas
  have hψ_meas : AEMeasurable ψ μ := hU_meas.prodMk hV_meas
  let φ : Ω × Ω → E × E := fun p => (U p.1, V p.2)
  have hφ_meas : AEMeasurable φ (μ.prod μ) :=
    (AEMeasurable.comp_fst (μ := μ) (ν := μ) hU_meas).prodMk
      (AEMeasurable.comp_snd (μ := μ) (ν := μ) hV_meas)
  have hUV_indep : U ⟂ᵢ[μ] V := by
    dsimp [U, V]
    exact coordinateMask_indepFun_compl h_indep hX_meas s
  have hmap_same : μ.map ψ = (μ.map U).prod (μ.map V) := by
    simpa [ψ] using hUV_indep.map_prod_eq_prod_map_map hU_meas hV_meas
  have hmap_prod : (μ.map U).prod (μ.map V) = (μ.prod μ).map φ := by
    simpa [φ] using measure_map_prod_map_of_aemeasurable (μ := μ) (ν := μ) hU_meas hV_meas
  have hg_aesm : AEStronglyMeasurable g ((μ.map U).prod (μ.map V)) := by
    dsimp [g]
    fun_prop
  have hg_aesm_prod : AEStronglyMeasurable g ((μ.prod μ).map φ) := by
    rwa [← hmap_prod]
  have hF_eq : F = g ∘ φ := by
    ext p
    dsimp [F, g, φ, U, V]
    rw [inner_toEuclideanCLM_cutMatrix_eq_inner_masks]
  have hcomp_int : Integrable (g ∘ φ) (μ.prod μ) := by
    rw [← hF_eq]
    exact hprod_int
  have hg_int_prod : Integrable g ((μ.prod μ).map φ) :=
    (integrable_map_measure hg_aesm_prod hφ_meas).2 hcomp_int
  have hg_int_same : Integrable g (μ.map ψ) := by
    rw [hmap_same, hmap_prod]
    exact hg_int_prod
  have hH_eq : H = g ∘ ψ := by
    ext ω
    dsimp [H, g, ψ, U, V]
    have hq :=
      quadraticForm_eq_inner_toEuclideanCLM (cutMatrix A s) (fun i => X i ω)
    rw [hq]
    rw [inner_toEuclideanCLM_cutMatrix_eq_inner_masks]
    rfl
  have hcomp_same : Integrable (g ∘ ψ) μ :=
    (integrable_map_measure (by
      rw [hmap_same]
      exact hg_aesm) hψ_meas).1 hg_int_same
  change Integrable H μ
  rw [hH_eq]
  exact hcomp_same

lemma integral_exp_quadraticForm_cutMatrix_le {μ : Measure Ω}
    [IsProbabilityMeasure μ] {n : ℕ} (A : Matrix (Fin n) (Fin n) ℝ)
    (s : Finset (Fin n)) {X : Fin n → Ω → ℝ} {K l : ℝ}
    (h_indep : iIndepFun X μ)
    (hX_subG : ∀ i, HasSubgaussianMGF (X i) ⟨K ^ 2, sq_nonneg K⟩ μ)
    (hsmall :
      (K ^ 2 * l ^ 2 / 2) * K ^ 2 * operatorNorm (cutMatrix A s) ^ 2 * exp 1 ≤ 1 / 2) :
    ∫ ω, exp (l * quadraticForm (cutMatrix A s) (fun i => X i ω)) ∂μ ≤
      exp (exp 1 ^ 2 * l ^ 2 * K ^ 4 * frobeniusNorm (cutMatrix A s) ^ 2) := by
  have hX_meas : ∀ i, AEMeasurable (X i) μ := fun i => (hX_subG i).aemeasurable
  have hprod_int :
      Integrable
        (fun p : Ω × Ω =>
          exp (l * inner ℝ
            (Matrix.toEuclideanCLM (𝕜 := ℝ) (cutMatrix A s) (randomVector X p.1))
            (randomVector X p.2))) (μ.prod μ) :=
    integrable_exp_inner_toEuclideanCLM_randomVector_prod (cutMatrix A s)
      h_indep hX_subG (lt_of_le_of_lt hsmall (by norm_num))
  have heq :=
    integral_exp_quadraticForm_cutMatrix_eq_prod A s h_indep hX_meas hprod_int
  rw [heq]
  exact integral_exp_inner_toEuclideanCLM_randomVector_prod_le (cutMatrix A s)
    h_indep hX_subG hsmall

lemma integrable_exp_quadraticForm_offDiagonal {μ : Measure Ω}
    [IsProbabilityMeasure μ] {n : ℕ} (A : Matrix (Fin n) (Fin n) ℝ)
    {X : Fin n → Ω → ℝ} {K l : ℝ}
    (h_indep : iIndepFun X μ)
    (hX_subG : ∀ i, HasSubgaussianMGF (X i) ⟨K ^ 2, sq_nonneg K⟩ μ)
    (hsmall :
      (K ^ 2 * (4 * l) ^ 2 / 2) * K ^ 2 * operatorNorm A ^ 2 * exp 1 < 1) :
    Integrable
      (fun ω => exp (l * quadraticForm (offDiagonalMatrix A) (fun i => X i ω))) μ := by
  classical
  let P : Finset (Finset (Fin n)) := (Finset.univ : Finset (Fin n)).powerset
  let R : Ω → ℝ := fun ω =>
    (P.card : ℝ)⁻¹ *
      ∑ s ∈ P, exp (4 * l * quadraticForm (cutMatrix A s) (fun i => X i ω))
  have hX_meas : ∀ i, AEMeasurable (X i) μ := fun i => (hX_subG i).aemeasurable
  have hsmall_cut : ∀ s ∈ P,
      (K ^ 2 * (4 * l) ^ 2 / 2) * K ^ 2 *
          operatorNorm (cutMatrix A s) ^ 2 * exp 1 < 1 := by
    intro s hs
    have hop_sq : operatorNorm (cutMatrix A s) ^ 2 ≤ operatorNorm A ^ 2 := by
      nlinarith [operatorNorm_cutMatrix_le A s, operatorNorm_nonneg (cutMatrix A s),
        operatorNorm_nonneg A]
    have hcoef_nonneg :
        0 ≤ (K ^ 2 * (4 * l) ^ 2 / 2) * K ^ 2 * exp 1 := by positivity
    calc
      (K ^ 2 * (4 * l) ^ 2 / 2) * K ^ 2 *
          operatorNorm (cutMatrix A s) ^ 2 * exp 1
          = ((K ^ 2 * (4 * l) ^ 2 / 2) * K ^ 2 * exp 1) *
              operatorNorm (cutMatrix A s) ^ 2 := by ring
      _ ≤ ((K ^ 2 * (4 * l) ^ 2 / 2) * K ^ 2 * exp 1) *
            operatorNorm A ^ 2 :=
          mul_le_mul_of_nonneg_left hop_sq hcoef_nonneg
      _ = (K ^ 2 * (4 * l) ^ 2 / 2) * K ^ 2 * operatorNorm A ^ 2 * exp 1 := by
          ring
      _ < 1 := hsmall
  have hR_int : Integrable R μ := by
    dsimp [R]
    refine (integrable_finsetSum P fun s hs => ?_).const_mul _
    exact integrable_exp_quadraticForm_cutMatrix A s h_indep hX_subG (hsmall_cut s hs)
  have hleft_aesm :
      AEStronglyMeasurable
        (fun ω => exp (l * quadraticForm (offDiagonalMatrix A) (fun i => X i ω))) μ := by
    have hleft_ae :
        AEMeasurable
          (fun ω => exp (l * quadraticForm (offDiagonalMatrix A) (fun i => X i ω))) μ :=
      ((randomQuadraticForm_aemeasurable (offDiagonalMatrix A) hX_meas).const_mul l).exp
    exact hleft_ae.aestronglyMeasurable
  have hpoint :
      (fun ω => exp (l * quadraticForm (offDiagonalMatrix A) (fun i => X i ω)))
        ≤ᵐ[μ] R := by
    filter_upwards with ω
    simpa [R, P] using
      exp_mul_quadraticForm_offDiagonal_le_average_cut A (fun i => X i ω) l
  refine Integrable.mono' hR_int hleft_aesm ?_
  filter_upwards [hpoint] with ω hω
  rw [Real.norm_eq_abs, abs_of_nonneg (exp_nonneg _)]
  exact hω

lemma integral_exp_quadraticForm_offDiagonal_le {μ : Measure Ω}
    [IsProbabilityMeasure μ] {n : ℕ} (A : Matrix (Fin n) (Fin n) ℝ)
    {X : Fin n → Ω → ℝ} {K l : ℝ}
    (h_indep : iIndepFun X μ)
    (hX_subG : ∀ i, HasSubgaussianMGF (X i) ⟨K ^ 2, sq_nonneg K⟩ μ)
    (hsmall :
      (K ^ 2 * (4 * l) ^ 2 / 2) * K ^ 2 * operatorNorm A ^ 2 * exp 1 ≤ 1 / 2) :
    ∫ ω, exp (l * quadraticForm (offDiagonalMatrix A) (fun i => X i ω)) ∂μ ≤
      exp (exp 1 ^ 2 * (4 * l) ^ 2 * K ^ 4 * frobeniusNorm A ^ 2) := by
  classical
  let P : Finset (Finset (Fin n)) := (Finset.univ : Finset (Fin n)).powerset
  let R : Ω → ℝ := fun ω =>
    (P.card : ℝ)⁻¹ *
      ∑ s ∈ P, exp (4 * l * quadraticForm (cutMatrix A s) (fun i => X i ω))
  let B : ℝ := exp (exp 1 ^ 2 * (4 * l) ^ 2 * K ^ 4 * frobeniusNorm A ^ 2)
  have hX_meas : ∀ i, AEMeasurable (X i) μ := fun i => (hX_subG i).aemeasurable
  have hP_pos : 0 < (P.card : ℝ) := by
    exact_mod_cast Finset.card_pos.mpr ⟨∅, Finset.empty_mem_powerset _⟩
  have hsmall_cut : ∀ s ∈ P,
      (K ^ 2 * (4 * l) ^ 2 / 2) * K ^ 2 *
          operatorNorm (cutMatrix A s) ^ 2 * exp 1 ≤ 1 / 2 := by
    intro s hs
    have hop_sq : operatorNorm (cutMatrix A s) ^ 2 ≤ operatorNorm A ^ 2 := by
      nlinarith [operatorNorm_cutMatrix_le A s, operatorNorm_nonneg (cutMatrix A s),
        operatorNorm_nonneg A]
    have hcoef_nonneg :
        0 ≤ (K ^ 2 * (4 * l) ^ 2 / 2) * K ^ 2 * exp 1 := by positivity
    calc
      (K ^ 2 * (4 * l) ^ 2 / 2) * K ^ 2 *
          operatorNorm (cutMatrix A s) ^ 2 * exp 1
          = ((K ^ 2 * (4 * l) ^ 2 / 2) * K ^ 2 * exp 1) *
              operatorNorm (cutMatrix A s) ^ 2 := by ring
      _ ≤ ((K ^ 2 * (4 * l) ^ 2 / 2) * K ^ 2 * exp 1) *
            operatorNorm A ^ 2 :=
          mul_le_mul_of_nonneg_left hop_sq hcoef_nonneg
      _ = (K ^ 2 * (4 * l) ^ 2 / 2) * K ^ 2 * operatorNorm A ^ 2 * exp 1 := by
          ring
      _ ≤ 1 / 2 := hsmall
  have hcut_int : ∀ s ∈ P,
      Integrable (fun ω =>
        exp (4 * l * quadraticForm (cutMatrix A s) (fun i => X i ω))) μ := by
    intro s hs
    exact integrable_exp_quadraticForm_cutMatrix A s h_indep hX_subG
      (lt_of_le_of_lt (hsmall_cut s hs) (by norm_num))
  have hR_int : Integrable R μ := by
    dsimp [R]
    exact (integrable_finsetSum P hcut_int).const_mul _
  have hleft_int :
      Integrable
        (fun ω => exp (l * quadraticForm (offDiagonalMatrix A) (fun i => X i ω))) μ :=
    integrable_exp_quadraticForm_offDiagonal A h_indep hX_subG
      (lt_of_le_of_lt hsmall (by norm_num))
  have hpoint :
      (fun ω => exp (l * quadraticForm (offDiagonalMatrix A) (fun i => X i ω)))
        ≤ᵐ[μ] R := by
    filter_upwards with ω
    simpa [R, P] using
      exp_mul_quadraticForm_offDiagonal_le_average_cut A (fun i => X i ω) l
  have hcut_bound : ∀ s ∈ P,
      ∫ ω, exp (4 * l * quadraticForm (cutMatrix A s) (fun i => X i ω)) ∂μ ≤ B := by
    intro s hs
    have hbase :=
      integral_exp_quadraticForm_cutMatrix_le A s h_indep hX_subG (hsmall_cut s hs)
    have hfrob := frobeniusNorm_cutMatrix_sq_le A s
    have hcoef_nonneg : 0 ≤ exp 1 ^ 2 * (4 * l) ^ 2 * K ^ 4 := by positivity
    have hexp_le :
        exp (exp 1 ^ 2 * (4 * l) ^ 2 * K ^ 4 *
            frobeniusNorm (cutMatrix A s) ^ 2) ≤ B := by
      dsimp [B]
      apply exp_le_exp.mpr
      calc
        exp 1 ^ 2 * (4 * l) ^ 2 * K ^ 4 * frobeniusNorm (cutMatrix A s) ^ 2
            = (exp 1 ^ 2 * (4 * l) ^ 2 * K ^ 4) *
                frobeniusNorm (cutMatrix A s) ^ 2 := by ring
        _ ≤ (exp 1 ^ 2 * (4 * l) ^ 2 * K ^ 4) * frobeniusNorm A ^ 2 :=
            mul_le_mul_of_nonneg_left hfrob hcoef_nonneg
        _ = exp 1 ^ 2 * (4 * l) ^ 2 * K ^ 4 * frobeniusNorm A ^ 2 := by ring
    exact hbase.trans hexp_le
  calc
    ∫ ω, exp (l * quadraticForm (offDiagonalMatrix A) (fun i => X i ω)) ∂μ
        ≤ ∫ ω, R ω ∂μ := integral_mono_ae hleft_int hR_int hpoint
    _ = (P.card : ℝ)⁻¹ *
          ∑ s ∈ P, ∫ ω,
            exp (4 * l * quadraticForm (cutMatrix A s) (fun i => X i ω)) ∂μ := by
          dsimp [R]
          rw [integral_const_mul]
          congr 1
          rw [integral_finsetSum]
          intro s hs
          exact hcut_int s hs
    _ ≤ (P.card : ℝ)⁻¹ * ∑ s ∈ P, B := by
          exact mul_le_mul_of_nonneg_left
            (Finset.sum_le_sum fun s hs => hcut_bound s hs) (inv_nonneg.mpr hP_pos.le)
    _ = B := by
          simp
          field_simp [ne_of_gt hP_pos]

lemma exp_add_le_average_exp_two (a b : ℝ) :
    exp (a + b) ≤ (exp (2 * a) + exp (2 * b)) / 2 := by
  have hconv := convexOn_exp.2 (Set.mem_univ (2 * a)) (Set.mem_univ (2 * b))
    (by norm_num : (0 : ℝ) ≤ 1 / 2) (by norm_num : (0 : ℝ) ≤ 1 / 2)
    (by norm_num : (1 / 2 : ℝ) + 1 / 2 = 1)
  calc
    exp (a + b) = exp ((1 / 2 : ℝ) • (2 * a) + (1 / 2 : ℝ) • (2 * b)) := by
      congr 1
      norm_num
      ring
    _ ≤ (1 / 2 : ℝ) * exp (2 * a) + (1 / 2 : ℝ) * exp (2 * b) := hconv
    _ = (exp (2 * a) + exp (2 * b)) / 2 := by ring

lemma abs_two_mul_le_inv_quarter_of_abs_le {a C l : ℝ}
    (ha : 0 < a) (hC : 0 < C)
    (hl : |l| ≤ (2 * C * a)⁻¹) :
    |2 * l| ≤ (2 * (C / 4) * a)⁻¹ := by
  have hbase : |2 * l| ≤ 2 * (2 * C * a)⁻¹ := by
    calc
      |2 * l| = 2 * |l| := by rw [abs_mul, abs_of_pos two_pos]
      _ ≤ 2 * (2 * C * a)⁻¹ := mul_le_mul_of_nonneg_left hl (by norm_num)
  have hleft : 2 * (2 * C * a)⁻¹ = (C * a)⁻¹ := by
    field_simp [hC.ne', ha.ne']
  have hright : (2 * (C / 4) * a)⁻¹ = 2 * (C * a)⁻¹ := by
    field_simp [hC.ne', ha.ne']
    ring
  calc
    |2 * l| ≤ 2 * (2 * C * a)⁻¹ := hbase
    _ = (C * a)⁻¹ := hleft
    _ ≤ 2 * (C * a)⁻¹ := by
      exact le_mul_of_one_le_left (inv_nonneg.mpr (mul_nonneg hC.le ha.le)) (by norm_num)
    _ = (2 * (C / 4) * a)⁻¹ := hright.symm

lemma abs_mul_le_inv_two_of_abs_le {a C l : ℝ}
    (ha : 0 < a) (hC : 0 < C)
    (hl : |l| ≤ (2 * C * a)⁻¹) :
    |l| * a ≤ 1 / (2 * C) := by
  calc
    |l| * a ≤ (2 * C * a)⁻¹ * a :=
      mul_le_mul_of_nonneg_right hl ha.le
    _ = 1 / (2 * C) := by
      field_simp [hC.ne', ha.ne']

lemma offDiagonal_small_two_mul_of_abs_le {K C op l : ℝ}
    (hK : 0 < K) (hC : 0 < C) (hop : 0 < op)
    (hC_sq : 16 * exp 1 ≤ C ^ 2)
    (hl : |l| ≤ (2 * C * K ^ 2 * op)⁻¹) :
    (K ^ 2 * (4 * (2 * l)) ^ 2 / 2) * K ^ 2 * op ^ 2 * exp 1 ≤ 1 / 2 := by
  let a : ℝ := K ^ 2 * op
  have ha : 0 < a := by
    dsimp [a]
    positivity
  have hscale : |l| * a ≤ 1 / (2 * C) := by
    have hden_eq : 2 * C * a = 2 * C * K ^ 2 * op := by
      dsimp [a]
      ring
    have hl_a : |l| ≤ (2 * C * a)⁻¹ := by
      rw [hden_eq]
      exact hl
    exact abs_mul_le_inv_two_of_abs_le (a := a) (C := C) (l := l) ha hC hl_a
  have hscale_sq :
      (|l| * a) ^ 2 ≤ (1 / (2 * C)) ^ 2 :=
    (sq_le_sq₀ (by positivity) (by positivity)).2 hscale
  have hsq_eq : (|l| * a) ^ 2 = l ^ 2 * K ^ 4 * op ^ 2 := by
    dsimp [a]
    rw [mul_pow, sq_abs]
    ring
  have hcore : l ^ 2 * K ^ 4 * op ^ 2 ≤ (1 / (2 * C)) ^ 2 := by
    rw [← hsq_eq]
    exact hscale_sq
  calc
    (K ^ 2 * (4 * (2 * l)) ^ 2 / 2) * K ^ 2 * op ^ 2 * exp 1
        = 32 * exp 1 * (l ^ 2 * K ^ 4 * op ^ 2) := by ring
    _ ≤ 32 * exp 1 * ((1 / (2 * C)) ^ 2) := by
      exact mul_le_mul_of_nonneg_left hcore (by positivity)
    _ = 8 * exp 1 / C ^ 2 := by
      field_simp [hC.ne']
      ring
    _ ≤ 1 / 2 := by
      have hC2_pos : 0 < C ^ 2 := sq_pos_of_pos hC
      rw [div_le_iff₀ hC2_pos]
      nlinarith [hC_sq]

lemma offDiagonal_exponent_two_mul_le {K C F l : ℝ}
    (hC_offdiag_quad : 64 * exp 1 ^ 2 ≤ C) :
    exp 1 ^ 2 * (4 * (2 * l)) ^ 2 * K ^ 4 * F ^ 2 ≤
      C * l ^ 2 * K ^ 4 * F ^ 2 := by
  have hnonneg : 0 ≤ l ^ 2 * K ^ 4 * F ^ 2 := by positivity
  calc
    exp 1 ^ 2 * (4 * (2 * l)) ^ 2 * K ^ 4 * F ^ 2
        = (64 * exp 1 ^ 2) * (l ^ 2 * K ^ 4 * F ^ 2) := by ring
    _ ≤ C * (l ^ 2 * K ^ 4 * F ^ 2) :=
      mul_le_mul_of_nonneg_right hC_offdiag_quad hnonneg
    _ = C * l ^ 2 * K ^ 4 * F ^ 2 := by ring

/-- Hanson-Wright MGF certificate from independent sub-Gaussian coordinates. -/
theorem hasHansonWrightMGF_of_subgaussian {μ : Measure Ω} [IsProbabilityMeasure μ]
    {n : ℕ} {A : Matrix (Fin n) (Fin n) ℝ} {X : Fin n → Ω → ℝ}
    {K C : ℝ} (hK : 0 < K) (hC : 0 < C)
    (hC_domain : 4 * exp 1 ≤ C)
    (hC_diag_quad : 8 * exp 1 ^ 3 ≤ C)
    (hC_offdiag_domain : 16 * exp 1 ≤ C ^ 2)
    (hC_offdiag_quad : 64 * exp 1 ^ 2 ≤ C)
    (hOp : 0 < operatorNorm A)
    (h_indep : iIndepFun X μ)
    (hX_subG : ∀ i, HasSubgaussianMGF (X i) ⟨K ^ 2, sq_nonneg K⟩ μ) :
    HasHansonWrightMGF μ A X K C := by
  let D : Ω → ℝ := diagonalCenteredQuadraticForm μ A X
  let O : Ω → ℝ := fun ω => quadraticForm (offDiagonalMatrix A) (fun i => X i ω)
  let B : ℝ → ℝ := fun l => C * l ^ 2 * K ^ 4 * frobeniusNorm A ^ 2
  have hX_meas : ∀ i, AEMeasurable (X i) μ := fun i => (hX_subG i).aemeasurable
  have hKsq_op_pos : 0 < K ^ 2 * operatorNorm A := by positivity
  have hCq_pos : 0 < C / 4 := by positivity
  have hCq_domain : exp 1 ≤ C / 4 := by linarith
  have hCq_quad : 2 * exp 1 ^ 3 ≤ C / 4 := by linarith
  have hcenter_eq :
      centeredQuadraticForm μ A X = fun ω => D ω + O ω := by
    dsimp [D, O]
    exact centeredQuadraticForm_eq_diagonal_add_offDiagonal A h_indep hX_subG
  have hD_ae : AEMeasurable D μ := by
    dsimp [D, diagonalCenteredQuadraticForm]
    exact Finset.aemeasurable_fun_sum _ fun i _ =>
      ((((hX_meas i).pow_const 2).sub_const _).const_mul _)
  have hO_ae : AEMeasurable O μ := by
    dsimp [O]
    exact randomQuadraticForm_aemeasurable (offDiagonalMatrix A) hX_meas
  have hmain : ∀ l : ℝ, |l| ≤ (2 * C * K ^ 2 * operatorNorm A)⁻¹ →
      Integrable (fun ω => exp (l * centeredQuadraticForm μ A X ω)) μ ∧
        ∫ ω, exp (l * centeredQuadraticForm μ A X ω) ∂μ ≤ exp (B l) := by
    intro l hl
    have hden_eq : 2 * C * (K ^ 2 * operatorNorm A) =
        2 * C * K ^ 2 * operatorNorm A := by ring
    have hl_a : |l| ≤ (2 * C * (K ^ 2 * operatorNorm A))⁻¹ := by
      rw [hden_eq]
      exact hl
    have hdiag_domain_a :
        |2 * l| ≤ (2 * (C / 4) * (K ^ 2 * operatorNorm A))⁻¹ :=
      abs_two_mul_le_inv_quarter_of_abs_le
        (a := K ^ 2 * operatorNorm A) (C := C) (l := l) hKsq_op_pos hC hl_a
    have hdiag_domain :
        |2 * l| ≤ (2 * (C / 4) * K ^ 2 * operatorNorm A)⁻¹ := by
      simpa [mul_assoc] using hdiag_domain_a
    have hD_int : Integrable (fun ω => exp ((2 * l) * D ω)) μ := by
      dsimp [D]
      exact diagonalCenteredQuadraticForm_integrable_exp
        (A := A) (X := X) (K := K) (C := C / 4) (θ := 2 * l)
        hK hCq_pos hCq_domain hOp h_indep hX_subG hdiag_domain
    have hD_cgf :
        cgf D μ (2 * l) ≤ B l := by
      have hbase :=
        diagonalCenteredQuadraticForm_cgf_le
          (A := A) (X := X) (K := K) (C := C / 4) (θ := 2 * l)
          hK hCq_pos hCq_domain hCq_quad hOp h_indep hX_subG hdiag_domain
      dsimp [D, B] at hbase ⊢
      calc
        cgf (diagonalCenteredQuadraticForm μ A X) μ (2 * l)
            ≤ (C / 4) * (2 * l) ^ 2 * K ^ 4 * frobeniusNorm A ^ 2 := hbase
        _ = C * l ^ 2 * K ^ 4 * frobeniusNorm A ^ 2 := by ring
    have hD_mgf_le : ∫ ω, exp ((2 * l) * D ω) ∂μ ≤ exp (B l) := by
      change mgf D μ (2 * l) ≤ exp (B l)
      rw [← exp_cgf hD_int]
      exact exp_le_exp.mpr hD_cgf
    have hsmall_offdiag :
        (K ^ 2 * (4 * (2 * l)) ^ 2 / 2) * K ^ 2 *
            operatorNorm A ^ 2 * exp 1 ≤ 1 / 2 :=
      offDiagonal_small_two_mul_of_abs_le hK hC hOp hC_offdiag_domain hl
    have hO_int : Integrable (fun ω => exp ((2 * l) * O ω)) μ := by
      dsimp [O]
      exact integrable_exp_quadraticForm_offDiagonal
        (A := A) (X := X) (K := K) (l := 2 * l)
        h_indep hX_subG (lt_of_le_of_lt hsmall_offdiag (by norm_num))
    have hO_mgf_le : ∫ ω, exp ((2 * l) * O ω) ∂μ ≤ exp (B l) := by
      have hbase :=
        integral_exp_quadraticForm_offDiagonal_le
          (A := A) (X := X) (K := K) (l := 2 * l)
          h_indep hX_subG hsmall_offdiag
      have hexp_le :
          exp (exp 1 ^ 2 * (4 * (2 * l)) ^ 2 * K ^ 4 *
              frobeniusNorm A ^ 2) ≤ exp (B l) := by
        apply exp_le_exp.mpr
        dsimp [B]
        exact offDiagonal_exponent_two_mul_le
          (K := K) (C := C) (F := frobeniusNorm A) (l := l) hC_offdiag_quad
      dsimp [O]
      exact hbase.trans hexp_le
    let R : Ω → ℝ :=
      fun ω => (exp ((2 * l) * D ω) + exp ((2 * l) * O ω)) / 2
    have hR_int : Integrable R μ := by
      dsimp [R]
      exact (hD_int.add hO_int).div_const 2
    have hleft_aesm :
        AEStronglyMeasurable (fun ω => exp (l * (D ω + O ω))) μ :=
      (((hD_ae.add hO_ae).const_mul l).exp).aestronglyMeasurable
    have hpoint :
        (fun ω => exp (l * (D ω + O ω))) ≤ᵐ[μ] R := by
      filter_upwards with ω
      dsimp [R]
      have hconv := exp_add_le_average_exp_two (l * D ω) (l * O ω)
      convert hconv using 1 <;> ring_nf
    have hDO_int : Integrable (fun ω => exp (l * (D ω + O ω))) μ := by
      refine Integrable.mono' hR_int hleft_aesm ?_
      filter_upwards [hpoint] with ω hω
      rw [Real.norm_eq_abs, abs_of_nonneg (exp_nonneg _)]
      exact hω
    have hcenter_int : Integrable (fun ω => exp (l * centeredQuadraticForm μ A X ω)) μ :=
      hDO_int.congr (ae_of_all _ fun ω => by
        change exp (l * (D ω + O ω)) = exp (l * centeredQuadraticForm μ A X ω)
        rw [← congrFun hcenter_eq ω])
    have hDO_integral_le :
        ∫ ω, exp (l * (D ω + O ω)) ∂μ ≤ exp (B l) := by
      have hmono : ∫ ω, exp (l * (D ω + O ω)) ∂μ ≤ ∫ ω, R ω ∂μ :=
        integral_mono_ae hDO_int hR_int hpoint
      have hR_integral :
          ∫ ω, R ω ∂μ =
            (∫ ω, exp ((2 * l) * D ω) ∂μ +
              ∫ ω, exp ((2 * l) * O ω) ∂μ) / 2 := by
        dsimp [R]
        rw [integral_div]
        rw [integral_add hD_int hO_int]
      have hsum_le :
          ∫ ω, exp ((2 * l) * D ω) ∂μ +
              ∫ ω, exp ((2 * l) * O ω) ∂μ ≤
            exp (B l) + exp (B l) :=
        add_le_add hD_mgf_le hO_mgf_le
      calc
        ∫ ω, exp (l * (D ω + O ω)) ∂μ ≤ ∫ ω, R ω ∂μ := hmono
        _ = (∫ ω, exp ((2 * l) * D ω) ∂μ +
              ∫ ω, exp ((2 * l) * O ω) ∂μ) / 2 := hR_integral
        _ ≤ (exp (B l) + exp (B l)) / 2 :=
          div_le_div_of_nonneg_right hsum_le (by norm_num)
        _ = exp (B l) := by ring
    have hcenter_integral_le :
        ∫ ω, exp (l * centeredQuadraticForm μ A X ω) ∂μ ≤ exp (B l) := by
      calc
        ∫ ω, exp (l * centeredQuadraticForm μ A X ω) ∂μ
            = ∫ ω, exp (l * (D ω + O ω)) ∂μ := by
              apply integral_congr_ae
              filter_upwards with ω
              rw [congrFun hcenter_eq ω]
        _ ≤ exp (B l) := hDO_integral_le
    exact ⟨hcenter_int, hcenter_integral_le⟩
  refine ⟨?_, ?_⟩
  · intro l hl
    have hres := hmain l hl
    have hmgf_pos : 0 < mgf (centeredQuadraticForm μ A X) μ l := mgf_pos hres.1
    have hmgf_le : mgf (centeredQuadraticForm μ A X) μ l ≤ exp (B l) := by
      simpa [mgf] using hres.2
    calc
      cgf (centeredQuadraticForm μ A X) μ l
          = log (mgf (centeredQuadraticForm μ A X) μ l) := rfl
      _ ≤ log (exp (B l)) := log_le_log hmgf_pos hmgf_le
      _ = C * l ^ 2 * K ^ 4 * frobeniusNorm A ^ 2 := by
        dsimp [B]
        rw [log_exp]
  · intro l hl
    exact (hmain l hl).1

/-- A proved Hanson-Wright MGF certificate for bounded coordinates.

If each coordinate satisfies `|Xᵢ| ≤ K` almost surely, then the quadratic form lies
in the interval `[-K²‖A‖₁, K²‖A‖₁]`.  The centered quadratic form is therefore
bounded by `2K²‖A‖₁`; the local bounded MGF lemma above gives a quadratic CGF
estimate.  The explicit side condition compares this bounded-coordinate constant
with the Frobenius-scale constant used by the Hanson-Wright tail statement. -/
theorem hasHansonWrightMGF_of_bounded {μ : Measure Ω} [IsProbabilityMeasure μ]
    {n : ℕ} {A : Matrix (Fin n) (Fin n) ℝ} {X : Fin n → Ω → ℝ}
    {K C : ℝ} (hK : 0 ≤ K) (hX_meas : ∀ i, AEMeasurable (X i) μ)
    (hX_bound : ∀ i, ∀ᵐ ω ∂μ, |X i ω| ≤ K)
    (hC_bound : 2 * entrywiseL1Norm A ^ 2 ≤ C * frobeniusNorm A ^ 2) :
    HasHansonWrightMGF μ A X K C := by
  set B : ℝ := K ^ 2 * entrywiseL1Norm A with hB_def
  have hB_nonneg : 0 ≤ B := by
    rw [hB_def]
    exact mul_nonneg (sq_nonneg K) (entrywiseL1Norm_nonneg A)
  have hZ_meas : AEMeasurable (randomQuadraticForm A X) μ :=
    randomQuadraticForm_aemeasurable A hX_meas
  have hZ_range :
      ∀ᵐ ω ∂μ, randomQuadraticForm A X ω ∈ Set.Icc (-B) B := by
    simpa [B, hB_def] using
      randomQuadraticForm_mem_Icc_of_ae_coord_bound (μ := μ) A hK hX_bound
  have hZ_int : Integrable (randomQuadraticForm A X) μ :=
    Integrable.of_mem_Icc (-B) B hZ_meas hZ_range
  have hmean_abs : |∫ ω, randomQuadraticForm A X ω ∂μ| ≤ B := by
    have hlower :
        -B ≤ ∫ ω, randomQuadraticForm A X ω ∂μ := by
      calc -B = ∫ _ω, (-B : ℝ) ∂μ := by simp
        _ ≤ ∫ ω, randomQuadraticForm A X ω ∂μ := by
            exact integral_mono_ae (integrable_const _) hZ_int
              (hZ_range.mono fun _ hω => hω.1)
    have hupper :
        ∫ ω, randomQuadraticForm A X ω ∂μ ≤ B := by
      calc ∫ ω, randomQuadraticForm A X ω ∂μ
          ≤ ∫ _ω, (B : ℝ) ∂μ := by
            exact integral_mono_ae hZ_int (integrable_const _)
              (hZ_range.mono fun _ hω => hω.2)
        _ = B := by simp
    exact abs_le.mpr ⟨hlower, hupper⟩
  have hY_meas : AEMeasurable (centeredQuadraticForm μ A X) μ :=
    hZ_meas.sub_const _
  have hY_bound :
      ∀ᵐ ω ∂μ, |centeredQuadraticForm μ A X ω| ≤ 2 * B := by
    filter_upwards [hZ_range] with ω hω
    have hZ_abs : |randomQuadraticForm A X ω| ≤ B := abs_le.mpr hω
    unfold centeredQuadraticForm
    calc |randomQuadraticForm A X ω - ∫ x, randomQuadraticForm A X x ∂μ|
        ≤ |randomQuadraticForm A X ω| + |∫ x, randomQuadraticForm A X x ∂μ| :=
          abs_sub _ _
      _ ≤ B + B := add_le_add hZ_abs hmean_abs
      _ = 2 * B := by ring
  have hY_center : ∫ ω, centeredQuadraticForm μ A X ω ∂μ = 0 := by
    unfold centeredQuadraticForm
    rw [integral_sub hZ_int (integrable_const _)]
    simp
  have h_sg :
      HasSubgaussianMGF (centeredQuadraticForm μ A X)
        ⟨(2 * B) ^ 2, sq_nonneg (2 * B)⟩ μ :=
    hasSubgaussianMGF_of_abs_le_of_integral_eq_zero hY_meas (by positivity)
      hY_bound hY_center
  refine ⟨?_, fun l _ => h_sg.integrable_exp_mul l⟩
  intro l _
  have h_cgf := h_sg.cgf_le l
  have h_const :
      ((⟨(2 * B) ^ 2, sq_nonneg (2 * B)⟩ : ℝ≥0) : ℝ) * l ^ 2 / 2 =
        2 * B ^ 2 * l ^ 2 := by
    simp
    ring
  calc cgf (centeredQuadraticForm μ A X) μ l
      ≤ ((⟨(2 * B) ^ 2, sq_nonneg (2 * B)⟩ : ℝ≥0) : ℝ) * l ^ 2 / 2 := h_cgf
    _ = 2 * B ^ 2 * l ^ 2 := h_const
    _ ≤ C * l ^ 2 * K ^ 4 * frobeniusNorm A ^ 2 := by
        rw [hB_def]
        exact bounded_hansonWright_cgf_constant_le A hC_bound l

private lemma one_sided_tail_of_cgf_bound {μ : Measure Ω} [IsProbabilityMeasure μ]
    {Y : Ω → ℝ} {v b C t : ℝ} (hC : 0 < C) (hv : 0 < v) (hb : 0 < b)
    (hcgf : ∀ l : ℝ, |l| ≤ (2 * C * b)⁻¹ → cgf Y μ l ≤ C * l ^ 2 * v)
    (hint : ∀ l : ℝ, |l| ≤ (2 * C * b)⁻¹ →
      Integrable (fun ω => exp (l * Y ω)) μ) (ht : 0 ≤ t) :
    (μ {ω | t ≤ Y ω}).toReal ≤
      exp (-(1 / (4 * C)) * min (t ^ 2 / v) (t / b)) := by
  rcases eq_or_lt_of_le ht with rfl | ht_pos
  · calc (μ {ω | 0 ≤ Y ω}).toReal
      _ ≤ (1 : ℝ) := ENNReal.toReal_mono ENNReal.one_ne_top prob_le_one
      _ = exp (-(1 / (4 * C)) * min (0 ^ 2 / v) (0 / b)) := by simp
  · set l : ℝ := min (t / (2 * C * v)) ((2 * C * b)⁻¹) with hl_def
    have hden_v : 0 < 2 * C * v := by positivity
    have hden_b : 0 < 2 * C * b := by positivity
    have hl_nonneg : 0 ≤ l := by
      rw [hl_def]
      exact le_min (div_nonneg ht (le_of_lt hden_v)) (inv_nonneg.mpr (le_of_lt hden_b))
    have hl_domain : |l| ≤ (2 * C * b)⁻¹ := by
      rw [abs_of_nonneg hl_nonneg, hl_def]
      exact min_le_right _ _
    have hl_le_v : l ≤ t / (2 * C * v) := by
      rw [hl_def]
      exact min_le_left _ _
    have hl_mul_le_t : l * (2 * C * v) ≤ t := by
      rwa [le_div_iff₀ hden_v] at hl_le_v
    have hquad_le : C * l ^ 2 * v ≤ l * t / 2 := by
      nlinarith [hl_mul_le_t, hl_nonneg, hC.le, hv.le]
    have h_exp_to_half : -l * t + C * l ^ 2 * v ≤ -(l * t / 2) := by
      linarith
    have h_rate :
        l * t / 2 = (1 / (4 * C)) * min (t ^ 2 / v) (t / b) := by
      by_cases hcase : t / (2 * C * v) ≤ (2 * C * b)⁻¹
      · have htb_le_v : t * b ≤ v := by
          rw [inv_eq_one_div] at hcase
          rw [div_le_div_iff₀ hden_v hden_b] at hcase
          nlinarith [hcase, hC, hb]
        have hmin : min (t ^ 2 / v) (t / b) = t ^ 2 / v := by
          rw [min_eq_left]
          rw [div_le_div_iff₀ hv hb]
          nlinarith [htb_le_v, ht_pos]
        rw [hl_def, min_eq_left hcase, hmin]
        field_simp [hC.ne', hv.ne']
        ring
      · have hcase' : (2 * C * b)⁻¹ ≤ t / (2 * C * v) := le_of_not_ge hcase
        have hv_le_tb : v ≤ t * b := by
          rw [inv_eq_one_div] at hcase'
          rw [div_le_div_iff₀ hden_b hden_v] at hcase'
          nlinarith [hcase', hC, hv]
        have hmin : min (t ^ 2 / v) (t / b) = t / b := by
          rw [min_eq_right]
          rw [div_le_div_iff₀ hb hv]
          nlinarith [hv_le_tb, ht_pos]
        rw [hl_def, min_eq_right hcase', hmin]
        field_simp [hC.ne', hb.ne']
        ring
    have h_chernoff := chernoff_bound_cgf hl_nonneg (hint l hl_domain)
      (μ := μ) (X := Y) (ε := t)
    calc (μ {ω | t ≤ Y ω}).toReal
      _ ≤ exp (-l * t + cgf Y μ l) := h_chernoff
      _ ≤ exp (-l * t + C * l ^ 2 * v) := by
        exact exp_le_exp.mpr (by linarith [hcgf l hl_domain])
      _ ≤ exp (-(l * t / 2)) := exp_le_exp.mpr h_exp_to_half
      _ = exp (-(1 / (4 * C)) * min (t ^ 2 / v) (t / b)) := by
        rw [h_rate]
        ring_nf

/-- A two-sided sub-exponential tail bound from a local quadratic CGF estimate. -/
theorem two_sided_tail_of_cgf_bound {μ : Measure Ω} [IsProbabilityMeasure μ]
    {Y : Ω → ℝ} {v b C t : ℝ} (hC : 0 < C) (hv : 0 < v) (hb : 0 < b)
    (hcgf : ∀ l : ℝ, |l| ≤ (2 * C * b)⁻¹ → cgf Y μ l ≤ C * l ^ 2 * v)
    (hint : ∀ l : ℝ, |l| ≤ (2 * C * b)⁻¹ →
      Integrable (fun ω => exp (l * Y ω)) μ) (ht : 0 ≤ t) :
    (μ {ω | t ≤ |Y ω|}).toReal ≤
      2 * exp (-(1 / (4 * C)) * min (t ^ 2 / v) (t / b)) := by
  have hpos := one_sided_tail_of_cgf_bound hC hv hb hcgf hint ht
  have hcgf_neg :
      ∀ l : ℝ, |l| ≤ (2 * C * b)⁻¹ →
        cgf (fun ω => -Y ω) μ l ≤ C * l ^ 2 * v := by
    intro l hl
    have h_eq : cgf (fun ω => -Y ω) μ l = cgf Y μ (-l) := by
      unfold cgf mgf
      congr 1
      apply integral_congr_ae
      filter_upwards with ω
      congr 1
      ring
    rw [h_eq]
    have hl' : |-l| ≤ (2 * C * b)⁻¹ := by simpa [abs_neg] using hl
    simpa using hcgf (-l) hl'
  have hint_neg : ∀ l : ℝ, |l| ≤ (2 * C * b)⁻¹ →
      Integrable (fun ω => exp (l * (-Y ω))) μ := by
    intro l hl
    convert hint (-l) (by simpa [abs_neg] using hl) using 1
    ext ω
    ring_nf
  have hneg := one_sided_tail_of_cgf_bound hC hv hb hcgf_neg hint_neg ht
  have h_subset :
      {ω | t ≤ |Y ω|} ⊆ {ω | t ≤ Y ω} ∪ {ω | t ≤ -Y ω} := by
    intro ω hω
    simp only [Set.mem_setOf_eq, Set.mem_union] at hω ⊢
    rcases le_or_gt 0 (Y ω) with hY | hY
    · left
      simpa [abs_of_nonneg hY] using hω
    · right
      simpa [abs_of_neg hY] using hω
  calc (μ {ω | t ≤ |Y ω|}).toReal
      ≤ (μ ({ω | t ≤ Y ω} ∪ {ω | t ≤ -Y ω})).toReal := by
        exact ENNReal.toReal_mono (measure_ne_top μ _) (measure_mono h_subset)
    _ ≤ (μ {ω | t ≤ Y ω}).toReal + (μ {ω | t ≤ -Y ω}).toReal := by
        rw [← ENNReal.toReal_add (measure_ne_top μ _) (measure_ne_top μ _)]
        exact ENNReal.toReal_mono
          (ENNReal.add_ne_top.mpr ⟨measure_ne_top μ _, measure_ne_top μ _⟩)
          (measure_union_le _ _)
    _ ≤ exp (-(1 / (4 * C)) * min (t ^ 2 / v) (t / b)) +
        exp (-(1 / (4 * C)) * min (t ^ 2 / v) (t / b)) := by
        exact add_le_add hpos hneg
    _ = 2 * exp (-(1 / (4 * C)) * min (t ^ 2 / v) (t / b)) := by ring

/-- Hanson-Wright tail bound with the MGF certificate proved from sub-Gaussian coordinates.

This theorem does not take `HasHansonWrightMGF` as a hypothesis.  Instead it proves
that certificate from independence and coordinate sub-Gaussian MGF bounds, then
optimizes the resulting Chernoff bound. -/
theorem hanson_wright_inequality {μ : Measure Ω} [IsProbabilityMeasure μ]
    {n : ℕ} {A : Matrix (Fin n) (Fin n) ℝ} {X : Fin n → Ω → ℝ}
    {K C t : ℝ} (hK : 0 < K) (hC : 0 < C)
    (hC_domain : 4 * exp 1 ≤ C)
    (hC_diag_quad : 8 * exp 1 ^ 3 ≤ C)
    (hC_offdiag_domain : 16 * exp 1 ≤ C ^ 2)
    (hC_offdiag_quad : 64 * exp 1 ^ 2 ≤ C)
    (hF : 0 < frobeniusNorm A) (hOp : 0 < operatorNorm A)
    (h_indep : iIndepFun X μ)
    (hX_subG : ∀ i, HasSubgaussianMGF (X i) ⟨K ^ 2, sq_nonneg K⟩ μ)
    (ht : 0 ≤ t) :
    (μ {ω | t ≤ |centeredQuadraticForm μ A X ω|}).toReal ≤
      2 * exp (-(1 / (4 * C)) *
        min (t ^ 2 / (K ^ 4 * frobeniusNorm A ^ 2))
          (t / (K ^ 2 * operatorNorm A))) := by
  have hHW : HasHansonWrightMGF μ A X K C :=
    hasHansonWrightMGF_of_subgaussian hK hC hC_domain hC_diag_quad
      hC_offdiag_domain hC_offdiag_quad hOp h_indep hX_subG
  have hv : 0 < K ^ 4 * frobeniusNorm A ^ 2 := by positivity
  have hb : 0 < K ^ 2 * operatorNorm A := by positivity
  exact two_sided_tail_of_cgf_bound hC hv hb
    (Y := centeredQuadraticForm μ A X)
    (fun l hl => by
      have hl' : |l| ≤ (2 * C * K ^ 2 * operatorNorm A)⁻¹ := by
        convert hl using 2
        ring
      simpa [mul_assoc, mul_left_comm, mul_comm] using hHW.1 l hl')
    (fun l hl => by
      have hl' : |l| ≤ (2 * C * K ^ 2 * operatorNorm A)⁻¹ := by
        convert hl using 2
        ring
      exact hHW.2 l hl') ht

private lemma hanson_wright_scaled_rhs_le {K C F O t : ℝ} (hK : 0 < K)
    (hC : 0 < C) (hF : 0 < F) (hO : 0 < O) (ht : 0 ≤ t) :
    2 * exp (-(1 / (4 * (C / 16))) *
        min (t ^ 2 / ((2 * K) ^ 4 * F ^ 2))
          (t / ((2 * K) ^ 2 * O))) ≤
      2 * exp (-(1 / (4 * C)) *
        min (t ^ 2 / (K ^ 4 * F ^ 2))
          (t / (K ^ 2 * O))) := by
  set q : ℝ := t ^ 2 / (K ^ 4 * F ^ 2) with hq_def
  set r : ℝ := t / (K ^ 2 * O) with hr_def
  have hq_nonneg : 0 ≤ q := by
    rw [hq_def]
    positivity
  have hr_nonneg : 0 ≤ r := by
    rw [hr_def]
    positivity
  have hq_scaled :
      t ^ 2 / ((2 * K) ^ 4 * F ^ 2) = q / 16 := by
    rw [hq_def]
    field_simp [hK.ne', hF.ne']
    ring
  have hr_scaled :
      t / ((2 * K) ^ 2 * O) = r / 4 := by
    rw [hr_def]
    field_simp [hK.ne', hO.ne']
    ring
  have hcoef : 1 / (4 * (C / 16)) = 4 / C := by
    field_simp [hC.ne']
    ring
  have hmin_scaled : min q r / 16 ≤ min (q / 16) (r / 4) := by
    apply le_min
    · exact div_le_div_of_nonneg_right (min_le_left q r) (by norm_num)
    · calc
        min q r / 16 ≤ r / 16 :=
          div_le_div_of_nonneg_right (min_le_right q r) (by norm_num)
        _ ≤ r / 4 := by nlinarith [hr_nonneg]
  have harg :
      -(1 / (4 * (C / 16))) *
        min (t ^ 2 / ((2 * K) ^ 4 * F ^ 2)) (t / ((2 * K) ^ 2 * O)) ≤
      -(1 / (4 * C)) *
        min (t ^ 2 / (K ^ 4 * F ^ 2)) (t / (K ^ 2 * O)) := by
    rw [hq_scaled, hr_scaled, hcoef]
    change -(4 / C) * min (q / 16) (r / 4) ≤ -(1 / (4 * C)) * min q r
    have hneg : -(4 / C) ≤ 0 := by
      have hpos : 0 ≤ 4 / C := by positivity
      linarith
    calc
      -(4 / C) * min (q / 16) (r / 4)
          ≤ -(4 / C) * (min q r / 16) :=
            mul_le_mul_of_nonpos_left hmin_scaled hneg
      _ = -(1 / (4 * C)) * min q r := by
          field_simp [hC.ne']
          ring
  exact mul_le_mul_of_nonneg_left (exp_le_exp.mpr harg) (by norm_num)

/-- Hanson-Wright inequality in the HDP normalization.

The scale `K` is the maximum coordinate ψ₂ scale.  The proof expands this
definition into the exact MGF bounds required by `hanson_wright_inequality`. -/
theorem hanson_wright_inequality_hdp_explicit {μ : Measure Ω} [IsProbabilityMeasure μ]
    {n : ℕ} {A : Matrix (Fin n) (Fin n) ℝ} {X : Fin n → Ω → ℝ}
    {K C t : ℝ} (hK_def : K = maxSubGaussianPsi2Norm X μ) (hK : 0 < K)
    (hC : 0 < C) (hC_domain : 4 * exp 1 ≤ C)
    (hC_diag_quad : 8 * exp 1 ^ 3 ≤ C)
    (hC_offdiag_domain : 16 * exp 1 ≤ C ^ 2)
    (hC_offdiag_quad : 64 * exp 1 ^ 2 ≤ C)
    (hF : 0 < frobeniusNorm A) (hOp : 0 < operatorNorm A)
    (h_indep : iIndepFun X μ)
    (hX_center : ∀ i, ∫ ω, X i ω ∂μ = 0)
    (hX_finite : ∀ i, HasFiniteSubGaussianPsi2Norm (X i) μ)
    (ht : 0 ≤ t) :
    (μ {ω | t ≤ |centeredQuadraticForm μ A X ω|}).toReal ≤
      2 * exp (-(1 / (4 * C)) *
        min (t ^ 2 / (K ^ 4 * frobeniusNorm A ^ 2))
          (t / (K ^ 2 * operatorNorm A))) := by
  have hX_center_used : ∀ i, ∫ ω, X i ω ∂μ = 0 := hX_center
  have hX_subG_K :
      ∀ i, HasSubgaussianMGF (X i) ⟨K ^ 2, sq_nonneg K⟩ μ := by
    intro i
    have hnorm_le : subGaussianPsi2Norm (X i) μ ≤ K := by
      rw [hK_def]
      exact subGaussianPsi2Norm_le_maxSubGaussianPsi2Norm (X := X) (μ := μ) i
    exact hasSubgaussianMGF_of_subGaussianPsi2Norm_le (hX_finite i) hnorm_le
  exact
    hanson_wright_inequality (μ := μ) (A := A) (X := X) (K := K) (C := C)
      (t := t) hK hC hC_domain hC_diag_quad hC_offdiag_domain
      hC_offdiag_quad hF hOp h_indep hX_subG_K ht

/-- Hanson-Wright inequality in the form of Theorem 6.2.2 of HDP.

For independent mean-zero coordinates and `K = max_i ‖X_i‖_{ψ₂}`, there is an
absolute positive constant `c` such that the usual Hanson-Wright tail bound
holds for every `t ≥ 0`. -/
theorem hanson_wright_inequality_hdp {μ : Measure Ω} [IsProbabilityMeasure μ]
    {n : ℕ} {A : Matrix (Fin n) (Fin n) ℝ} {X : Fin n → Ω → ℝ}
    {K : ℝ} (hK_def : K = maxSubGaussianPsi2Norm X μ) (hK : 0 < K)
    (h_indep : iIndepFun X μ)
    (hX_center : ∀ i, ∫ ω, X i ω ∂μ = 0)
    (hX_finite : ∀ i, HasFiniteSubGaussianPsi2Norm (X i) μ) :
    ∃ c : ℝ, 0 < c ∧ ∀ t : ℝ, 0 ≤ t →
      (μ {ω | t ≤ |centeredQuadraticForm μ A X ω|}).toReal ≤
        2 * exp (-c *
          min (t ^ 2 / (K ^ 4 * frobeniusNorm A ^ 2))
            (t / (K ^ 2 * operatorNorm A))) := by
  let C0 : ℝ := 64 * exp 1 ^ 2
  have hexp_pos : 0 < exp 1 := exp_pos 1
  have hexp_ge_one : 1 ≤ exp 1 := one_le_exp (by norm_num : (0 : ℝ) ≤ 1)
  have hexp_le_eight : exp 1 ≤ 8 := by
    nlinarith [exp_one_lt_three.le]
  have hC0_pos : 0 < C0 := by
    dsimp [C0]
    positivity
  have hC0_domain : 4 * exp 1 ≤ C0 := by
    dsimp [C0]
    have hmul := mul_le_mul_of_nonneg_left hexp_ge_one
      (by positivity : 0 ≤ 4 * exp 1)
    nlinarith
  have hC0_diag_quad : 8 * exp 1 ^ 3 ≤ C0 := by
    dsimp [C0]
    have hmul := mul_le_mul_of_nonneg_left hexp_le_eight
      (by positivity : 0 ≤ 8 * exp 1 ^ 2)
    nlinarith
  have hC0_ge_offdiag : 16 * exp 1 ≤ C0 := by
    dsimp [C0]
    have hmul := mul_le_mul_of_nonneg_left hexp_ge_one
      (by positivity : 0 ≤ 16 * exp 1)
    nlinarith
  have hC0_ge_one : 1 ≤ C0 := by
    dsimp [C0]
    nlinarith [hexp_ge_one, sq_nonneg (exp 1)]
  have hC0_sq_ge : C0 ≤ C0 ^ 2 := by
    have hC0_nonneg : 0 ≤ C0 := hC0_pos.le
    have hmul := mul_le_mul hC0_ge_one (le_refl C0) hC0_nonneg hC0_nonneg
    simpa [pow_two] using hmul
  have hC0_offdiag_domain : 16 * exp 1 ≤ C0 ^ 2 :=
    hC0_ge_offdiag.trans hC0_sq_ge
  have hC0_offdiag_quad : 64 * exp 1 ^ 2 ≤ C0 := by
    dsimp [C0]
    rfl
  refine ⟨1 / (4 * C0), by positivity, ?_⟩
  intro t ht
  by_cases hF : 0 < frobeniusNorm A
  · by_cases hOp : 0 < operatorNorm A
    · simpa [one_div] using
        hanson_wright_inequality_hdp_explicit (μ := μ) (A := A) (X := X)
          (K := K) (C := C0) (t := t) hK_def hK hC0_pos hC0_domain
          hC0_diag_quad hC0_offdiag_domain hC0_offdiag_quad hF hOp
          h_indep hX_center hX_finite ht
    · have hOp_nonneg : 0 ≤ operatorNorm A := by
        unfold operatorNorm
        exact norm_nonneg _
      have hOp_zero : operatorNorm A = 0 := le_antisymm (le_of_not_gt hOp) hOp_nonneg
      have hfirst_nonneg :
          0 ≤ t ^ 2 / (K ^ 4 * frobeniusNorm A ^ 2) := by positivity
      have hsecond_zero : t / (K ^ 2 * operatorNorm A) = 0 := by
        rw [hOp_zero, mul_zero, div_zero]
      have hmin :
          min (t ^ 2 / (K ^ 4 * frobeniusNorm A ^ 2))
              (t / (K ^ 2 * operatorNorm A)) = 0 := by
        rw [hsecond_zero, min_eq_right hfirst_nonneg]
      calc
        (μ {ω | t ≤ |centeredQuadraticForm μ A X ω|}).toReal ≤ (1 : ℝ) :=
          ENNReal.toReal_mono ENNReal.one_ne_top prob_le_one
        _ ≤ 2 * exp (-(1 / (4 * C0)) *
            min (t ^ 2 / (K ^ 4 * frobeniusNorm A ^ 2))
              (t / (K ^ 2 * operatorNorm A))) := by
            rw [hmin]
            norm_num
  · have hF_nonneg : 0 ≤ frobeniusNorm A := by
      unfold frobeniusNorm
      exact sqrt_nonneg _
    have hF_zero : frobeniusNorm A = 0 := le_antisymm (le_of_not_gt hF) hF_nonneg
    have hOp_nonneg : 0 ≤ operatorNorm A := by
      unfold operatorNorm
      exact norm_nonneg _
    have hfirst_zero :
        t ^ 2 / (K ^ 4 * frobeniusNorm A ^ 2) = 0 := by
      rw [hF_zero]
      simp
    have hsecond_nonneg : 0 ≤ t / (K ^ 2 * operatorNorm A) :=
      div_nonneg ht (mul_nonneg (sq_nonneg K) hOp_nonneg)
    have hmin :
        min (t ^ 2 / (K ^ 4 * frobeniusNorm A ^ 2))
            (t / (K ^ 2 * operatorNorm A)) = 0 := by
      rw [hfirst_zero, min_eq_left hsecond_nonneg]
    calc
      (μ {ω | t ≤ |centeredQuadraticForm μ A X ω|}).toReal ≤ (1 : ℝ) :=
        ENNReal.toReal_mono ENNReal.one_ne_top prob_le_one
      _ ≤ 2 * exp (-(1 / (4 * C0)) *
          min (t ^ 2 / (K ^ 4 * frobeniusNorm A ^ 2))
            (t / (K ^ 2 * operatorNorm A))) := by
          rw [hmin]
          norm_num

end HansonWright
