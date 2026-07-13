/-
Copyright (c) 2026 Yuanhe Zhang. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Yuanhe Zhang, Jason D. Lee, Fanghui Liu
-/
import SLT.MatrixInfra.CourantFischer
import Mathlib.Analysis.CStarAlgebra.CStarMatrix
import Mathlib.Analysis.CStarAlgebra.ContinuousFunctionalCalculus.Instances
import Mathlib.Analysis.CStarAlgebra.Matrix
import Mathlib.Analysis.Matrix.Order
import Mathlib.Analysis.Normed.Algebra.MatrixExponential
import Mathlib.Analysis.SpecialFunctions.ContinuousFunctionalCalculus.ExpLog.Basic
import Mathlib.Analysis.SpecialFunctions.ContinuousFunctionalCalculus.ExpLog.Order
import Mathlib.Analysis.SpecialFunctions.ImproperIntegrals
import Mathlib.Analysis.SpecialFunctions.Integrals.Basic
import Mathlib.LinearAlgebra.Lagrange
import Mathlib.LinearAlgebra.Matrix.SchurComplement
import Mathlib.LinearAlgebra.Matrix.Vec
import Mathlib.MeasureTheory.Integral.Bochner.Basic
import Mathlib.MeasureTheory.Integral.Prod
import Mathlib.Topology.Algebra.Module.ContinuousLinearMap.PiProd

/-!
# Matrix Calculus Infrastructure

This file contains the deterministic matrix-calculus infrastructure needed for HDP
Theorem 5.4.8, Lieb's concavity theorem, in the notation used by the local matrix
infrastructure.

Mathlib already provides the Loewner order on square matrices in the scoped namespace
`MatrixOrder`, with `A ≤ B` meaning `(B - A).PosSemidef`. The retained proof path here is the
relative-entropy route:

1. prove the density-form left/right integral representation of matrix relative entropy;
2. derive joint convexity and Klein's inequality for `traceMatrixRelativeEntropy`;
3. use the Gibbs variational argument to prove deterministic Lieb concavity.

Alternative formalizations of Lieb concavity through variational, Golden-Thompson, or duplicate
relative-entropy routes are intentionally not kept in this file.

## Main definitions

* `Matrix.positiveDefiniteSymmetricMatrices`: positive-definite real matrices as a set.
* `Matrix.leftRightDenominatorMatrix`: the Kronecker matrix for the left/right multiplication
  denominator in the relative-entropy integral formula.
* `Matrix.leftRightRelativeEntropyIntegrand`: the operator-level left/right integrand.
* `Matrix.traceMatrixRelativeEntropy`: `tr(A log A - A log B)`.
* `Matrix.liebTraceFunction`: `X ↦ tr exp(H + log X)`.
* `Matrix.lieb_inequality_hdp`: the proposition form of HDP Theorem 5.4.8.

## Main results

* `Matrix.loewner_eigenvalue_monotonicity_hdp`
* `Matrix.loewner_trace_functionalCalculus_monotonicity_hdp`
* `Matrix.loewner_functionalCalculus_order_hdp`
* `Matrix.leftRightRelativeEntropyIntegrand_jointConvex`
* `Matrix.traceMatrixRelativeEntropy_leftRight_integral_representation`
* `Matrix.traceMatrixRelativeEntropy_jointConvex`
* `Matrix.traceMatrixRelativeEntropy_klein`
* `Matrix.lieb_inequality_hdp_of_leftRight_density_integral_representation`
* `Matrix.lieb_inequality_hdp_5_4_8`
-/

open scoped BigOperators ComplexOrder Kronecker MatrixOrder Matrix.Norms.L2Operator Topology
open MeasureTheory
open NormedSpace
open Module
open Filter

noncomputable section

namespace LinearMap

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- Rayleigh quotients are monotone for the Loewner order on self-adjoint real linear maps. -/
theorem rayleighQuotient_mono_of_le {A B : E →ₗ[ℝ] E} (hAB : A ≤ B) {x : E}
    (hx : x ≠ 0) :
    rayleighQuotient A x ≤ rayleighQuotient B x := by
  have hpos : (B - A).IsPositive := (LinearMap.le_def A B).mp hAB
  have hinner : 0 ≤ inner ℝ ((B - A) x) x := hpos.inner_nonneg_left x
  have hden : 0 < ‖x‖ ^ 2 := sq_pos_of_pos (norm_pos_iff.mpr hx)
  unfold rayleighQuotient
  refine div_le_div_of_nonneg_right ?_ hden.le
  have hadd : inner ℝ (B x) x = inner ℝ (A x) x + inner ℝ ((B - A) x) x := by
    rw [show B x = A x + (B - A) x by simp]
    rw [inner_add_left]
  rw [hadd]
  exact le_add_of_nonneg_right hinner

end LinearMap

namespace Matrix

variable {n : Type*} [Fintype n] [DecidableEq n]

/-! ## CStar bridge for real matrices -/

/-- The entrywise inclusion from real matrices to complex matrices as a star algebra homomorphism. -/
noncomputable def realMatrixToComplexMatrixStarAlgHom :
    Matrix n n ℝ →⋆ₐ[ℝ] Matrix n n ℂ :=
  { (RCLike.ofRealStarAlgHom ℂ).toAlgHom.mapMatrix with
    map_star' := by
      intro A
      ext i j
      simp }

/--
The entrywise inclusion from real matrices to complex `CStarMatrix` as a star algebra
homomorphism.

The target type carries the CStar-algebra structure needed by Mathlib's operator-monotone
functional-calculus theorem for logarithm.
-/
noncomputable def realMatrixToCStarMatrixStarAlgHom :
    Matrix n n ℝ →⋆ₐ[ℝ] CStarMatrix n n ℂ :=
  (((CStarMatrix.ofMatrixStarAlgEquiv (n := n) (A := ℂ) :
      Matrix n n ℂ →⋆ₐ[ℂ] CStarMatrix n n ℂ).restrictScalars ℝ).comp
    (realMatrixToComplexMatrixStarAlgHom (n := n)))

/-- Positive semidefiniteness reflects from the complexification back to the real matrix. -/
theorem realMatrix_posSemidef_of_complexification_posSemidef {A : Matrix n n ℝ}
    (hA : (realMatrixToComplexMatrixStarAlgHom (n := n) A).PosSemidef) :
    A.PosSemidef := by
  rw [posSemidef_iff_dotProduct_mulVec] at hA ⊢
  constructor
  · refine IsHermitian.ext fun i j => ?_
    have hij := hA.1.apply i j
    exact Complex.ofReal_injective (by simpa [realMatrixToComplexMatrixStarAlgHom] using hij)
  · intro x
    let z : n → ℂ := fun i => (x i : ℂ)
    have hz := hA.2 z
    have hquad :
        star z ⬝ᵥ ((realMatrixToComplexMatrixStarAlgHom (n := n) A) *ᵥ z) =
          ((star x ⬝ᵥ (A *ᵥ x) : ℝ) : ℂ) := by
      simp [z, realMatrixToComplexMatrixStarAlgHom, dotProduct, Matrix.mulVec,
        Finset.mul_sum]
    rw [hquad] at hz
    exact Complex.zero_le_real.mp hz

omit [DecidableEq n] in
/-- The `CStarMatrix` spectral order agrees with the ordinary complex matrix order. -/
theorem cstarMatrix_nonneg_iff_matrix_nonneg (M : Matrix n n ℂ) :
    (0 : CStarMatrix n n ℂ) ≤ CStarMatrix.ofMatrix M ↔ (0 : Matrix n n ℂ) ≤ M := by
  rw [StarOrderedRing.nonneg_iff, StarOrderedRing.nonneg_iff]
  rfl

omit [DecidableEq n] in
/-- The `CStarMatrix` spectral order is the usual matrix positive-semidefinite cone. -/
theorem cstarMatrix_nonneg_iff_matrix_posSemidef (M : Matrix n n ℂ) :
    (0 : CStarMatrix n n ℂ) ≤ CStarMatrix.ofMatrix M ↔ M.PosSemidef := by
  rw [cstarMatrix_nonneg_iff_matrix_nonneg M]
  exact Matrix.nonneg_iff_posSemidef

/-- The real-to-`CStarMatrix` inclusion reflects and preserves nonnegativity. -/
theorem realToCStarMatrixStarAlgHom_nonneg_iff {A : Matrix n n ℝ} :
    0 ≤ realMatrixToCStarMatrixStarAlgHom (n := n) A ↔ 0 ≤ A := by
  constructor
  · intro hA
    rw [Matrix.nonneg_iff_posSemidef]
    apply realMatrix_posSemidef_of_complexification_posSemidef (n := n)
    rw [← cstarMatrix_nonneg_iff_matrix_posSemidef]
    rw [CStarMatrix.ofMatrix_eq_ofMatrixStarAlgEquiv]
    simpa [realMatrixToCStarMatrixStarAlgHom] using hA
  · intro hA
    exact map_nonneg (realMatrixToCStarMatrixStarAlgHom (n := n)) hA

/-- The real-to-`CStarMatrix` inclusion reflects and preserves the Loewner order. -/
theorem realToCStarMatrixStarAlgHom_le_iff {A B : Matrix n n ℝ} :
    realMatrixToCStarMatrixStarAlgHom (n := n) A ≤
      realMatrixToCStarMatrixStarAlgHom (n := n) B ↔ A ≤ B := by
  constructor
  · intro hAB
    have hsub :
        0 ≤ realMatrixToCStarMatrixStarAlgHom (n := n) (B - A) := by
      simpa [map_sub] using sub_nonneg.mpr hAB
    exact sub_nonneg.mp ((realToCStarMatrixStarAlgHom_nonneg_iff (n := n)).mp hsub)
  · intro hAB
    have hsub : 0 ≤ realMatrixToCStarMatrixStarAlgHom (n := n) (B - A) :=
      (realToCStarMatrixStarAlgHom_nonneg_iff (n := n)).mpr (sub_nonneg.mpr hAB)
    exact sub_nonneg.mp (by simpa [map_sub] using hsub)

/-- The real-to-`CStarMatrix` inclusion commutes with logarithm on positive-definite matrices. -/
theorem realToCStarMatrixStarAlgHom_map_log_of_posDef {X : Matrix n n ℝ} (hX : X.PosDef) :
    realMatrixToCStarMatrixStarAlgHom (n := n) (CFC.log X) =
      CFC.log (realMatrixToCStarMatrixStarAlgHom (n := n) X) := by
  letI : IsometricContinuousFunctionalCalculus ℝ (CStarMatrix n n ℂ) IsSelfAdjoint :=
    IsSelfAdjoint.instIsometricContinuousFunctionalCalculus (A := CStarMatrix n n ℂ)
  let φ := realMatrixToCStarMatrixStarAlgHom (n := n)
  have hXsa : IsSelfAdjoint X := hX.isHermitian
  have hφcont : Continuous φ := φ.toAlgHom.toLinearMap.continuous_of_finiteDimensional
  have hφXsa : IsSelfAdjoint (φ X) := hXsa.map φ
  have hspos : ∀ x ∈ spectrum ℝ X, 0 < x := by
    exact (StarOrderedRing.isStrictlyPositive_iff_spectrum_pos (R := ℝ) X).mp
      hX.isStrictlyPositive
  have hcont : ContinuousOn Real.log (spectrum ℝ X) := by
    exact Real.continuousOn_log.mono fun x hx => ne_of_gt (hspos x hx)
  change φ (cfc (p := IsSelfAdjoint) Real.log X) =
    cfc (p := IsSelfAdjoint) Real.log (φ X)
  exact StarAlgHom.map_cfc (p := IsSelfAdjoint) (q := IsSelfAdjoint)
    φ Real.log X (hf := hcont) (hφ := hφcont) (ha := hXsa) (hφa := hφXsa)

/- A positive-definite real matrix maps to a strictly positive element of the CStar algebra. -/
theorem realToCStarMatrixStarAlgHom_isStrictlyPositive_of_posDef
    {X : Matrix n n ℝ} (hX : X.PosDef) :
    IsStrictlyPositive (realMatrixToCStarMatrixStarAlgHom (n := n) X) := by
  have hnonneg :
      0 ≤ realMatrixToCStarMatrixStarAlgHom (n := n) X := by
    exact map_nonneg (realMatrixToCStarMatrixStarAlgHom (n := n)) hX.posSemidef.nonneg
  have hunit : IsUnit (realMatrixToCStarMatrixStarAlgHom (n := n) X) := by
    exact hX.isUnit.map (realMatrixToCStarMatrixStarAlgHom (n := n)).toRingHom
  exact hunit.isStrictlyPositive hnonneg

/- Operator monotonicity of logarithm for positive-definite real matrices. -/
theorem cfc_log_le_log_of_le_posDef {X Y : Matrix n n ℝ}
    (hX : X.PosDef) (hXY : X ≤ Y) :
    CFC.log X ≤ CFC.log Y := by
  letI : IsometricContinuousFunctionalCalculus ℝ (CStarMatrix n n ℂ) IsSelfAdjoint :=
    IsSelfAdjoint.instIsometricContinuousFunctionalCalculus (A := CStarMatrix n n ℂ)
  let φ := realMatrixToCStarMatrixStarAlgHom (n := n)
  have hY : Y.PosDef := by
    have hdiff : (Y - X).PosSemidef := Matrix.le_iff.mp hXY
    simpa [sub_eq_add_neg, add_assoc, add_left_comm, add_comm] using
      hX.add_posSemidef hdiff
  have hφXY : φ X ≤ φ Y := by
    exact (realToCStarMatrixStarAlgHom_le_iff (n := n)).mpr hXY
  have hφXpos : IsStrictlyPositive (φ X) :=
    realToCStarMatrixStarAlgHom_isStrictlyPositive_of_posDef hX
  have hlogφ : CFC.log (φ X) ≤ CFC.log (φ Y) := by
    exact CFC.log_le_log hφXY (ha := hφXpos)
  rw [← realToCStarMatrixStarAlgHom_le_iff (n := n)]
  simpa [φ, realToCStarMatrixStarAlgHom_map_log_of_posDef hX,
    realToCStarMatrixStarAlgHom_map_log_of_posDef hY] using hlogφ

/-! ## Proposition 5.4.4: simple properties of Loewner order -/

/--
HDP Proposition 5.4.4(a), eigenvalue monotonicity for Loewner order.

The order `X ≤ Y` is Mathlib's scoped matrix Loewner order, i.e. `(Y - X).PosSemidef`. The
eigenvalues are the decreasingly ordered real eigenvalues obtained from the Euclidean linear map.
-/
theorem loewner_eigenvalue_monotonicity_hdp
    {X Y : Matrix n n ℝ} (hX : X.IsHermitian) (hY : Y.IsHermitian) (hXY : X ≤ Y)
    (i : Fin (Fintype.card n)) :
    hX.eigenvaluesToEuclideanLin i ≤ hY.eigenvaluesToEuclideanLin i := by
  let hXlin : X.toEuclideanLin.IsSymmetric := Matrix.isSymmetric_toEuclideanLin_iff.mpr hX
  let hYlin : Y.toEuclideanLin.IsSymmetric := Matrix.isSymmetric_toEuclideanLin_iff.mpr hY
  let L : Submodule ℝ (EuclideanSpace ℝ n) :=
    hXlin.leadingEigenSubspace finrank_euclideanSpace (Nat.succ_le_of_lt i.2)
  let R : Submodule ℝ (EuclideanSpace ℝ n) :=
    hYlin.trailingEigenSubspace finrank_euclideanSpace i
  have hLdim : finrank ℝ L = i.1 + 1 := by
    simpa [L] using hXlin.finrank_leadingEigenSubspace finrank_euclideanSpace
      (Nat.succ_le_of_lt i.2)
  have hRdim : finrank ℝ R = Fintype.card n - i.1 := by
    simpa [R] using hYlin.finrank_trailingEigenSubspace finrank_euclideanSpace i
  have hdimlt : finrank ℝ (EuclideanSpace ℝ n) < finrank ℝ L + finrank ℝ R := by
    rw [finrank_euclideanSpace, hLdim, hRdim]
    omega
  obtain ⟨x, hxL, hxR, hx0⟩ :=
    Submodule.exists_ne_zero_mem_inf_of_finrank_lt_add_finrank L R hdimlt
  have hlin : X.toEuclideanLin ≤ Y.toEuclideanLin := by
    rw [LinearMap.le_def]
    have hpos : ((Y - X).toEuclideanLin).IsPositive := by
      exact Matrix.isPositive_toEuclideanLin_iff.mpr (Matrix.le_iff.mp hXY)
    convert hpos using 1
    ext v
    simp
  exact (hXlin.eigenvalues_le_rayleighQuotient_of_mem_leadingEigenSubspace
      finrank_euclideanSpace i hxL hx0).trans
    ((LinearMap.rayleighQuotient_mono_of_le hlin hx0).trans
      (hYlin.rayleighQuotient_le_eigenvalues_of_mem_trailingEigenSubspace
        finrank_euclideanSpace i hxR hx0))

/--
The spectral trace `tr f(X)` for a symmetric matrix, written as the sum of `f` over the sorted
eigenvalues. This is the scalar trace of the matrix functional calculus in HDP
Proposition 5.4.4(b).
-/
def spectralTrace {X : Matrix n n ℝ} (hX : X.IsHermitian) (f : ℝ → ℝ) : ℝ :=
  ∑ i : Fin (Fintype.card n), f (hX.eigenvaluesToEuclideanLin i)

/-- The trace of the Hermitian functional calculus is the sum over the sorted eigenvalues. -/
theorem trace_cfc_eq_spectralTrace {X : Matrix n n ℝ} (hX : X.IsHermitian)
    (f : ℝ → ℝ) :
    trace (cfc f X) = spectralTrace hX f := by
  rw [hX.cfc_eq f, Matrix.IsHermitian.cfc]
  simp [spectralTrace, trace_mul_cycle]
  unfold Matrix.IsHermitian.eigenvalues Matrix.IsHermitian.eigenvalues₀
    Matrix.IsHermitian.eigenvaluesToEuclideanLin
  rw [Fintype.sum_equiv (Fintype.equivOfCardEq (Fintype.card_fin (Fintype.card n))).symm]
  intro x
  rfl

/--
HDP Proposition 5.4.4(b), trace monotonicity for weakly increasing scalar functions.

The conclusion is stated through `spectralTrace`, i.e. `tr f(X) = Σᵢ f(λᵢ(X))`.
-/
theorem loewner_trace_functionalCalculus_monotonicity_hdp
    {X Y : Matrix n n ℝ} (hX : X.IsHermitian) (hY : Y.IsHermitian) (hXY : X ≤ Y)
    {f : ℝ → ℝ} (hf : Monotone f) :
    spectralTrace hX f ≤ spectralTrace hY f := by
  unfold spectralTrace
  exact Finset.sum_le_sum fun i _ =>
    hf (loewner_eigenvalue_monotonicity_hdp hX hY hXY i)

/-- Trace exponential is monotone under Loewner order on Hermitian matrices. -/
theorem trace_exp_monotone_of_le
    {X Y : Matrix n n ℝ} (hX : X.IsHermitian) (hY : Y.IsHermitian) (hXY : X ≤ Y) :
    trace (exp X) ≤ trace (exp Y) := by
  rw [← CFC.real_exp_eq_normedSpace_exp (a := X) (ha := (show IsSelfAdjoint X from hX)),
    ← CFC.real_exp_eq_normedSpace_exp (a := Y) (ha := (show IsSelfAdjoint Y from hY))]
  rw [trace_cfc_eq_spectralTrace hX Real.exp, trace_cfc_eq_spectralTrace hY Real.exp]
  exact loewner_trace_functionalCalculus_monotonicity_hdp hX hY hXY Real.exp_monotone

/--
HDP Proposition 5.4.4(d), upgrading a scalar inequality to a Loewner-order inequality under a
spectral norm bound.

Mathlib's CFC order theorem is stated spectrally. This wrapper matches the HDP hypothesis
`‖X‖ ≤ a`, which ensures every spectral value of the Hermitian matrix `X` lies in `[-a, a]`.
-/
theorem loewner_functionalCalculus_order_hdp
    {X : Matrix n n ℝ} (hX : X.IsHermitian) {f g : ℝ → ℝ} {a : ℝ}
    (ha : 0 ≤ a) (hfg : ∀ x : ℝ, |x| ≤ a → f x ≤ g x) (hXnorm : ‖X‖ ≤ a) :
    hX.cfc f ≤ hX.cfc g := by
  rw [← hX.cfc_eq f, ← hX.cfc_eq g]
  have hfcont : ContinuousOn f (spectrum ℝ X) := by
    rw [continuousOn_iff_continuous_restrict]
    fun_prop
  have hgcont : ContinuousOn g (spectrum ℝ X) := by
    rw [continuousOn_iff_continuous_restrict]
    fun_prop
  refine (cfc_le_iff f g X (ha := (show IsSelfAdjoint X from hX))
    (hf := hfcont) (hg := hgcont)).mpr ?_
  intro x hx
  have hone : ‖(1 : Matrix n n ℝ)‖ ≤ 1 := by
    rw [← Matrix.diagonal_one, Matrix.l2_opNorm_diagonal]
    change ‖(fun _ : n => (1 : ℝ))‖ ≤ 1
    simpa using (pi_norm_const_le (ι := n) (1 : ℝ))
  have hxnorm : ‖x‖ ≤ a := by
    refine (le_max_left ‖x‖ 0).trans ?_
    rw [max_le_iff]
    constructor
    · calc
        ‖x‖ ≤ ‖X‖ * ‖(1 : Matrix n n ℝ)‖ := spectrum.norm_le_norm_mul_of_mem hx
        _ ≤ ‖X‖ * 1 := mul_le_mul_of_nonneg_left hone (norm_nonneg X)
        _ = ‖X‖ := by rw [mul_one]
        _ ≤ a := hXnorm
    · exact ha
  exact hfg x (by simpa [Real.norm_eq_abs] using hxnorm)

/-! ## Theorem 5.4.8: Lieb inequality -/

/-- The domain of positive definite real symmetric matrices. -/
def positiveDefiniteSymmetricMatrices (n : Type*) : Set (Matrix n n ℝ) :=
  {X | X.PosDef}

omit [Fintype n] [DecidableEq n] in
/-- The positive-definite cone is convex. -/
theorem positiveDefiniteSymmetricMatrices_convex :
    Convex ℝ (positiveDefiniteSymmetricMatrices n) := by
  intro X hX Y hY a b ha hb hab
  rcases eq_or_lt_of_le ha with rfl | ha_pos
  · have hb_one : b = 1 := by linarith
    simpa [positiveDefiniteSymmetricMatrices, hb_one] using hY
  rcases eq_or_lt_of_le hb with rfl | hb_pos
  · have ha_one : a = 1 := by linarith
    simpa [positiveDefiniteSymmetricMatrices, ha_one] using hX
  exact Matrix.PosDef.add (Matrix.PosDef.smul hX ha_pos) (Matrix.PosDef.smul hY hb_pos)

/-- Matrix logarithm is continuous on the positive-definite real symmetric matrices. -/
theorem continuousOn_cfc_log_positiveDefiniteSymmetricMatrices :
    ContinuousOn (fun X : Matrix n n ℝ => CFC.log X)
      (positiveDefiniteSymmetricMatrices n) := by
  letI : IsometricContinuousFunctionalCalculus ℝ (CStarMatrix n n ℂ) IsSelfAdjoint :=
    IsSelfAdjoint.instIsometricContinuousFunctionalCalculus (A := CStarMatrix n n ℂ)
  let φ := realMatrixToCStarMatrixStarAlgHom (n := n)
  have hφ_cont : ContinuousOn (fun X : Matrix n n ℝ => φ X)
      (positiveDefiniteSymmetricMatrices n) :=
    (φ.toAlgHom.toLinearMap.continuous_of_finiteDimensional).continuousOn
  have hlog_cstar : ContinuousOn (fun X : Matrix n n ℝ => CFC.log (φ X))
      (positiveDefiniteSymmetricMatrices n) := by
    exact (CFC.continuousOn_log (A := CStarMatrix n n ℂ)).comp hφ_cont (by
      intro X hX
      exact ⟨(show IsSelfAdjoint X from hX.isHermitian).map φ,
        (realToCStarMatrixStarAlgHom_isStrictlyPositive_of_posDef (n := n) hX).isUnit⟩)
  change ContinuousOn (fun X : Matrix n n ℝ => fun i j => (CFC.log X) i j)
    (positiveDefiniteSymmetricMatrices n)
  rw [continuousOn_pi]
  intro i
  rw [continuousOn_pi]
  intro j
  have hentry : ContinuousOn
      (fun X : Matrix n n ℝ => ((CFC.log (φ X) : CStarMatrix n n ℂ) i j).re)
      (positiveDefiniteSymmetricMatrices n) := by
    have hev : Continuous (fun M : CStarMatrix n n ℂ => (M i j).re) := by fun_prop
    exact hev.comp_continuousOn hlog_cstar
  refine hentry.congr ?_
  intro X hX
  have hmap := realToCStarMatrixStarAlgHom_map_log_of_posDef (n := n) hX
  have hentry_complex : ((CFC.log X i j : ℝ) : ℂ) =
      (CFC.log (φ X) : CStarMatrix n n ℂ) i j := by
    have h := congr_fun (congr_fun hmap i) j
    dsimp [φ, realMatrixToCStarMatrixStarAlgHom, realMatrixToComplexMatrixStarAlgHom,
      CStarMatrix.ofMatrixStarAlgEquiv, CStarMatrix.ofMatrixRingEquiv, CStarMatrix.ofMatrix]
      at h ⊢
    change (((CFC.log X).map Complex.ofReal) i j) =
      (CFC.log ((X.map Complex.ofReal : Matrix n n ℂ)) : Matrix n n ℂ) i j at h
    change ((CFC.log X i j : ℝ) : ℂ) =
      (CFC.log ((X.map Complex.ofReal : Matrix n n ℂ)) : Matrix n n ℂ) i j
    simpa [Matrix.map_apply] using h
  have hreal := congrArg Complex.re hentry_complex
  simpa using hreal

section InversePerspective

variable {m k : Type*} [Fintype m] [Fintype k] [DecidableEq k]

/-- The Schur-complement block matrix witnessing the inverse perspective is PSD. -/
theorem inversePerspectiveBlock_posSemidef
    (B : Matrix m k ℝ) {D : Matrix k k ℝ} (hD : D.PosDef) :
    (fromBlocks (B * D⁻¹ * Bᴴ) B Bᴴ D).PosSemidef := by
  cases hD.isUnit.nonempty_invertible
  rw [Matrix.PosDef.fromBlocks₂₂ (A := B * D⁻¹ * Bᴴ) B hD]
  simpa using Matrix.PosSemidef.zero

/--
Joint convexity of the matrix inverse perspective `(B, D) ↦ B D⁻¹ Bᴴ`.

This is the Schur-complement form of the standard perspective theorem for the operator-convex
inverse, and is one of the finite-dimensional inputs for matrix relative entropy.
-/
theorem inversePerspective_jointConvex
    {B₁ B₂ : Matrix m k ℝ} {D₁ D₂ : Matrix k k ℝ}
    (hD₁ : D₁.PosDef) (hD₂ : D₂.PosDef)
    {a b : ℝ} (ha : 0 ≤ a) (hb : 0 ≤ b) (hab : a + b = 1) :
    let B := a • B₁ + b • B₂
    let D := a • D₁ + b • D₂
    B * D⁻¹ * Bᴴ ≤
      a • (B₁ * D₁⁻¹ * B₁ᴴ) + b • (B₂ * D₂⁻¹ * B₂ᴴ) := by
  intro B D
  let A₁ := B₁ * D₁⁻¹ * B₁ᴴ
  let A₂ := B₂ * D₂⁻¹ * B₂ᴴ
  let A := a • A₁ + b • A₂
  have hD : D.PosDef := by
    exact positiveDefiniteSymmetricMatrices_convex hD₁ hD₂ ha hb hab
  have hblock₁ : (fromBlocks A₁ B₁ B₁ᴴ D₁).PosSemidef :=
    inversePerspectiveBlock_posSemidef B₁ hD₁
  have hblock₂ : (fromBlocks A₂ B₂ B₂ᴴ D₂).PosSemidef :=
    inversePerspectiveBlock_posSemidef B₂ hD₂
  have hblock_combo :
      (a • fromBlocks A₁ B₁ B₁ᴴ D₁ +
        b • fromBlocks A₂ B₂ B₂ᴴ D₂).PosSemidef :=
    (hblock₁.smul ha).add (hblock₂.smul hb)
  have hblock : (fromBlocks A B Bᴴ D).PosSemidef := by
    convert hblock_combo using 1
    simp [A, B, D, Matrix.fromBlocks_add, Matrix.fromBlocks_smul]
  cases hD.isUnit.nonempty_invertible
  have hschur : (A - B * D⁻¹ * Bᴴ).PosSemidef :=
    (Matrix.PosDef.fromBlocks₂₂ (A := A) B hD).mp hblock
  rw [Matrix.le_iff]
  simpa [A, A₁, A₂] using hschur

end InversePerspective

/-- The Lieb trace function `X ↦ tr exp(H + log X)` from HDP Theorem 5.4.8. -/
def liebTraceFunction (H : Matrix n n ℝ) (X : Matrix n n ℝ) : ℝ :=
  trace (exp (H + CFC.log X))

/-- The Lieb trace function is continuous on positive-definite real symmetric matrices. -/
theorem continuousOn_liebTraceFunction (H : Matrix n n ℝ) :
    ContinuousOn (liebTraceFunction H) (positiveDefiniteSymmetricMatrices n) := by
  have hlog : ContinuousOn (fun X : Matrix n n ℝ => CFC.log X)
      (positiveDefiniteSymmetricMatrices n) :=
    continuousOn_cfc_log_positiveDefiniteSymmetricMatrices (n := n)
  unfold liebTraceFunction
  have hinside : ContinuousOn (fun X : Matrix n n ℝ => H + CFC.log X)
      (positiveDefiniteSymmetricMatrices n) := continuousOn_const.add hlog
  letI : NormedAlgebra ℚ (Matrix n n ℝ) := NormedAlgebra.restrictScalars ℚ ℝ (Matrix n n ℝ)
  have hexp : Continuous (fun A : Matrix n n ℝ => exp A) := exp_continuous
  have htrace : Continuous (fun A : Matrix n n ℝ => trace A) := by fun_prop
  exact htrace.comp_continuousOn (hexp.comp_continuousOn hinside)

/-- Pointwise form of continuity of the Lieb trace function on the positive-definite domain. -/
theorem continuousWithinAt_liebTraceFunction
    (H : Matrix n n ℝ) {X : Matrix n n ℝ}
    (hX : X ∈ positiveDefiniteSymmetricMatrices n) :
    ContinuousWithinAt (liebTraceFunction H) (positiveDefiniteSymmetricMatrices n) X :=
  (continuousOn_liebTraceFunction H).continuousWithinAt hX

/-- The linear trace functional `X ↦ tr(XY)`, for fixed right factor `Y`. -/
def traceRightMulLinearMap (Y : Matrix n n ℝ) : Matrix n n ℝ →ₗ[ℝ] ℝ where
  toFun X := trace (X * Y)
  map_add' X Z := by simp [add_mul, trace_add]
  map_smul' a X := by simp [trace_smul]

@[simp]
theorem traceRightMulLinearMap_apply (Y X : Matrix n n ℝ) :
    traceRightMulLinearMap Y X = trace (X * Y) := rfl

/-- The trace of a product of two positive semidefinite matrices is nonnegative. -/
theorem trace_mul_nonneg_of_posSemidef {X Y : Matrix n n ℝ}
    (hX : X.PosSemidef) (hY : Y.PosSemidef) :
    0 ≤ trace (X * Y) := by
  obtain ⟨B, hB⟩ := CStarAlgebra.nonneg_iff_eq_star_mul_self.mp hY.nonneg
  rw [hB, star_eq_conjTranspose, ← Matrix.mul_assoc, trace_mul_cycle]
  exact (hX.mul_mul_conjTranspose_same B).trace_nonneg

/-- For positive-semidefinite `X`, the functional `Y ↦ tr(XY)` is Loewner-monotone in `Y`. -/
theorem traceRightMulLinearMap_mono_of_posSemidef {X Y Z : Matrix n n ℝ}
    (hX : X.PosSemidef) (hYZ : Y ≤ Z) :
    traceRightMulLinearMap Y X ≤ traceRightMulLinearMap Z X := by
  rw [traceRightMulLinearMap_apply, traceRightMulLinearMap_apply]
  have hdiff : (Z - Y).PosSemidef := Matrix.le_iff.mp hYZ
  have hnonneg : 0 ≤ trace (X * (Z - Y)) :=
    trace_mul_nonneg_of_posSemidef hX hdiff
  have htrace_sub : trace (X * (Z - Y)) = trace (X * Z) - trace (X * Y) := by
    simp [mul_sub, trace_sub]
  linarith

section IntegralPosSemidef

variable {Ω : Type*} [MeasurableSpace Ω]

def matrixEntryContinuousLinearMap {m n : Type*} (i : m) (j : n) :
    Matrix m n ℝ →L[ℝ] ℝ :=
  (ContinuousLinearMap.proj (R := ℝ) (φ := fun _ : n => ℝ) j).comp
    (ContinuousLinearMap.proj (R := ℝ) (φ := fun _ : m => n → ℝ) i)

omit [Fintype n] [DecidableEq n] in
@[simp] theorem matrixEntryContinuousLinearMap_apply {m n : Type*} (i : m) (j : n)
    (M : Matrix m n ℝ) :
    matrixEntryContinuousLinearMap i j M = M i j := rfl

/-- The continuous-linear trace functional `X ↦ tr(XY)`, for fixed right factor `Y`. -/
def traceRightMulContinuousLinearMap (Y : Matrix n n ℝ) :
    Matrix n n ℝ →L[ℝ] ℝ :=
  ∑ i, ∑ j, (Y j i) • matrixEntryContinuousLinearMap i j

omit [DecidableEq n] in
theorem traceRightMulContinuousLinearMap_apply
    (Y X : Matrix n n ℝ) :
    traceRightMulContinuousLinearMap Y X = trace (X * Y) := by
  simp [traceRightMulContinuousLinearMap, Matrix.trace, Matrix.mul_apply]
  exact Finset.sum_congr rfl fun x _ =>
    Finset.sum_congr rfl fun y _ => mul_comm (Y y x) (X x y)

def matrixQuadraticFormContinuousLinearMap (x : n → ℝ) :
    Matrix n n ℝ →L[ℝ] ℝ :=
  ∑ i, ∑ j, (x i * x j) • matrixEntryContinuousLinearMap i j

omit [DecidableEq n] in
theorem matrixQuadraticFormContinuousLinearMap_apply
    (x : n → ℝ) (M : Matrix n n ℝ) :
    matrixQuadraticFormContinuousLinearMap x M =
      dotProduct (star x) (Matrix.mulVec M x) := by
  simp [matrixQuadraticFormContinuousLinearMap, dotProduct, Matrix.mulVec]
  apply Finset.sum_congr rfl
  intro i _
  rw [Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro j _
  change x i * x j * M i j = x i * (M i j * x j)
  ring

/-- Bochner integration preserves positive semidefiniteness of real matrices. -/
theorem integral_posSemidef_of_ae
    {μ : Measure Ω} {M : Ω → Matrix n n ℝ}
    (hM_int : Integrable M μ) (hM_psd : ∀ᵐ ω ∂μ, (M ω).PosSemidef) :
    (∫ ω, M ω ∂μ).PosSemidef := by
  refine Matrix.PosSemidef.of_dotProduct_mulVec_nonneg ?hHerm ?hquad
  · refine Matrix.IsHermitian.ext fun i j => ?_
    have hentry : ∀ᵐ ω ∂μ,
        matrixEntryContinuousLinearMap j i (M ω) =
          matrixEntryContinuousLinearMap i j (M ω) := by
      filter_upwards [hM_psd] with ω hω
      exact hω.1.apply i j
    calc
      star ((∫ ω, M ω ∂μ) j i) =
          matrixEntryContinuousLinearMap j i (∫ ω, M ω ∂μ) := by simp
      _ = ∫ ω, matrixEntryContinuousLinearMap j i (M ω) ∂μ := by
        exact (ContinuousLinearMap.integral_comp_comm
          (matrixEntryContinuousLinearMap j i) hM_int).symm
      _ = ∫ ω, matrixEntryContinuousLinearMap i j (M ω) ∂μ := by
        exact integral_congr_ae hentry
      _ = matrixEntryContinuousLinearMap i j (∫ ω, M ω ∂μ) := by
        exact ContinuousLinearMap.integral_comp_comm
          (matrixEntryContinuousLinearMap i j) hM_int
      _ = (∫ ω, M ω ∂μ) i j := by simp
  · intro x
    have hq_int :
        Integrable
          (fun ω => dotProduct (star x) (Matrix.mulVec (M ω) x)) μ := by
      simpa [matrixQuadraticFormContinuousLinearMap_apply] using
        (matrixQuadraticFormContinuousLinearMap x).integrable_comp hM_int
    have hq_nonneg :
        0 ≤ᵐ[μ] fun ω => dotProduct (star x) (Matrix.mulVec (M ω) x) := by
      filter_upwards [hM_psd] with ω hω
      exact hω.dotProduct_mulVec_nonneg x
    calc
      0 ≤ ∫ ω, dotProduct (star x) (Matrix.mulVec (M ω) x) ∂μ :=
        integral_nonneg_of_ae hq_nonneg
      _ = matrixQuadraticFormContinuousLinearMap x (∫ ω, M ω ∂μ) := by
        calc
          ∫ ω, dotProduct (star x) (Matrix.mulVec (M ω) x) ∂μ =
              ∫ ω, matrixQuadraticFormContinuousLinearMap x (M ω) ∂μ := by
            exact integral_congr_ae (Filter.Eventually.of_forall fun ω =>
              (matrixQuadraticFormContinuousLinearMap_apply x (M ω)).symm)
          _ = matrixQuadraticFormContinuousLinearMap x (∫ ω, M ω ∂μ) := by
            exact ContinuousLinearMap.integral_comp_comm
              (matrixQuadraticFormContinuousLinearMap x) hM_int
      _ = dotProduct (star x) (Matrix.mulVec (∫ ω, M ω ∂μ) x) := by
        rw [matrixQuadraticFormContinuousLinearMap_apply]

end IntegralPosSemidef

section TraceInversePerspective

variable {m k : Type*} [Fintype m] [DecidableEq m] [Fintype k] [DecidableEq k]

/-- Trace-paired scalar form of joint convexity of the matrix inverse perspective. -/
theorem trace_inversePerspective_jointConvex
    {X : Matrix m m ℝ} (hX : X.PosSemidef)
    {B₁ B₂ : Matrix m k ℝ} {D₁ D₂ : Matrix k k ℝ}
    (hD₁ : D₁.PosDef) (hD₂ : D₂.PosDef)
    {a b : ℝ} (ha : 0 ≤ a) (hb : 0 ≤ b) (hab : a + b = 1) :
    let B := a • B₁ + b • B₂
    let D := a • D₁ + b • D₂
    trace (X * (B * D⁻¹ * Bᴴ)) ≤
      a * trace (X * (B₁ * D₁⁻¹ * B₁ᴴ)) +
        b * trace (X * (B₂ * D₂⁻¹ * B₂ᴴ)) := by
  intro B D
  have hpersp :=
    inversePerspective_jointConvex (m := m) (k := k)
      (B₁ := B₁) (B₂ := B₂) (D₁ := D₁) (D₂ := D₂) hD₁ hD₂ ha hb hab
  change B * D⁻¹ * Bᴴ ≤
      a • (B₁ * D₁⁻¹ * B₁ᴴ) + b • (B₂ * D₂⁻¹ * B₂ᴴ) at hpersp
  have htrace :=
    traceRightMulLinearMap_mono_of_posSemidef (X := X) hX hpersp
  simpa [traceRightMulLinearMap_apply, mul_add, mul_smul, trace_add, trace_smul] using htrace

end TraceInversePerspective

/-- Hermitian matrix exponentials are positive definite. -/
theorem exp_posDef_of_isHermitian {Z : Matrix n n ℝ} (hZ : Z.IsHermitian) :
    (exp Z).PosDef := by
  have hsa : IsSelfAdjoint Z := hZ
  have hnonneg : (exp Z).PosSemidef := by
    exact Matrix.nonneg_iff_posSemidef.mp hsa.exp_nonneg
  exact hnonneg.posDef_iff_isUnit.mpr (Matrix.isUnit_exp Z)

section LeftRightDenominator

variable {n : Type*} [Fintype n] [DecidableEq n]

/--
The Kronecker matrix for the left/right multiplication operator
`X ↦ A * X + t • (X * B)` under `Matrix.vec`.

Mathlib vectorizes matrices columnwise, so right multiplication by `B` is represented by
`Bᵀ ⊗ₖ I`, while left multiplication by `A` is represented by `I ⊗ₖ A`.
-/
def leftRightDenominatorMatrix
    (A B : Matrix n n ℝ) (t : ℝ) : Matrix (n × n) (n × n) ℝ :=
  (1 : Matrix n n ℝ) ⊗ₖ A + t • (Bᵀ ⊗ₖ (1 : Matrix n n ℝ))

omit [Fintype n] in
/-- For diagonal inputs, the left/right denominator is diagonal on vectorized indices. -/
theorem leftRightDenominatorMatrix_diagonal
    (d e : n → ℝ) (t : ℝ) :
    leftRightDenominatorMatrix (diagonal d) (diagonal e) t =
      diagonal (fun ij : n × n => d ij.2 + t * e ij.1) := by
  ext ij kl
  rcases ij with ⟨i, j⟩
  rcases kl with ⟨k, l⟩
  simp [leftRightDenominatorMatrix, Matrix.kroneckerMap_apply]
  by_cases h₁ : i = k
  · subst k
    by_cases h₂ : j = l
    · subst l
      simp
    · simp [h₂]
  · simp [h₁]

/--
The inverse of the diagonal left/right denominator, in the `Ring.inverse` form used by
`Matrix.inv_diagonal`. Under pointwise nonzero hypotheses this specializes to scalar reciprocals.
-/
theorem leftRightDenominatorMatrix_diagonal_inv
    (d e : n → ℝ) (t : ℝ) :
    (leftRightDenominatorMatrix (diagonal d) (diagonal e) t)⁻¹ =
      diagonal (Ring.inverse fun ij : n × n => d ij.2 + t * e ij.1) := by
  rw [leftRightDenominatorMatrix_diagonal, Matrix.inv_diagonal]

/--
The left/right denominator is positive definite when `A` and `B` are positive definite and
`t ≥ 0`.
-/
theorem leftRightDenominatorMatrix_posDef
    {A B : Matrix n n ℝ} (hA : A.PosDef) (hB : B.PosDef) {t : ℝ} (ht : 0 ≤ t) :
    (leftRightDenominatorMatrix A B t).PosDef := by
  have hleft : ((1 : Matrix n n ℝ) ⊗ₖ A).PosDef :=
    Matrix.PosDef.one.kronecker hA
  have hright : (Bᵀ ⊗ₖ (1 : Matrix n n ℝ)).PosSemidef :=
    hB.transpose.posSemidef.kronecker Matrix.PosDef.one.posSemidef
  exact hleft.add_posSemidef (hright.smul ht)

theorem leftRightDenominatorMatrix_mulVec_vec
    (A B X : Matrix n n ℝ) (t : ℝ) :
    leftRightDenominatorMatrix A B t *ᵥ vec X =
      vec (A * X + t • (X * B)) := by
  unfold leftRightDenominatorMatrix
  rw [add_mulVec, smul_mulVec]
  rw [← Matrix.vec_mul_eq_mulVec A X, Matrix.kronecker_mulVec_vec]
  simp [transpose_transpose]

theorem leftRightDenominatorMatrix_conjStarAlgAut_mulVec_vec
    (U : unitaryGroup n ℝ) (A B X : Matrix n n ℝ) (t : ℝ) :
    leftRightDenominatorMatrix
        (Unitary.conjStarAlgAut ℝ _ U A) (Unitary.conjStarAlgAut ℝ _ U B) t *ᵥ
      vec (Unitary.conjStarAlgAut ℝ _ U X) =
        vec (Unitary.conjStarAlgAut ℝ _ U (A * X + t • (X * B))) := by
  rw [leftRightDenominatorMatrix_mulVec_vec]
  apply congr_arg Matrix.vec
  calc
    Unitary.conjStarAlgAut ℝ _ U A * Unitary.conjStarAlgAut ℝ _ U X +
          t • (Unitary.conjStarAlgAut ℝ _ U X * Unitary.conjStarAlgAut ℝ _ U B)
        = Unitary.conjStarAlgAut ℝ _ U (A * X) +
          t • Unitary.conjStarAlgAut ℝ _ U (X * B) := by
      have hcancel : ∀ Y : Matrix n n ℝ,
          star (U : Matrix n n ℝ) * ((U : Matrix n n ℝ) * Y) = Y := by
        intro Y
        rw [← Matrix.mul_assoc, Unitary.coe_star_mul_self, Matrix.one_mul]
      simp [Unitary.conjStarAlgAut_apply, Matrix.mul_assoc, hcancel]
    _ = Unitary.conjStarAlgAut ℝ _ U (A * X + t • (X * B)) := by
      rw [map_add, map_smul]

theorem leftRightDenominatorMatrix_conjStarAlgAut_diagonal_mulVec_vec
    (U : unitaryGroup n ℝ) (d e : n → ℝ) (X : Matrix n n ℝ) (t : ℝ) :
    leftRightDenominatorMatrix
        (Unitary.conjStarAlgAut ℝ _ U (diagonal d))
        (Unitary.conjStarAlgAut ℝ _ U (diagonal e)) t *ᵥ
      vec (Unitary.conjStarAlgAut ℝ _ U X) =
        vec (Unitary.conjStarAlgAut ℝ _ U
          (diagonal d * X + t • (X * diagonal e))) := by
  exact leftRightDenominatorMatrix_conjStarAlgAut_mulVec_vec U (diagonal d) (diagonal e) X t

theorem leftRightDenominatorMatrix_twoSidedUnitary_mulVec_vec
    (U V : unitaryGroup n ℝ) (A B X : Matrix n n ℝ) (t : ℝ) :
    leftRightDenominatorMatrix
        (Unitary.conjStarAlgAut ℝ _ U A) (Unitary.conjStarAlgAut ℝ _ V B) t *ᵥ
      vec ((U : Matrix n n ℝ) * X * star (V : Matrix n n ℝ)) =
        vec ((U : Matrix n n ℝ) * (A * X + t • (X * B)) * star (V : Matrix n n ℝ)) := by
  rw [leftRightDenominatorMatrix_mulVec_vec]
  apply congr_arg Matrix.vec
  calc
    Unitary.conjStarAlgAut ℝ _ U A *
          ((U : Matrix n n ℝ) * X * star (V : Matrix n n ℝ)) +
        t • (((U : Matrix n n ℝ) * X * star (V : Matrix n n ℝ)) *
          Unitary.conjStarAlgAut ℝ _ V B)
        = (U : Matrix n n ℝ) * (A * X) * star (V : Matrix n n ℝ) +
          t • ((U : Matrix n n ℝ) * (X * B) * star (V : Matrix n n ℝ)) := by
      have hcancelU : ∀ Y : Matrix n n ℝ,
          star (U : Matrix n n ℝ) * ((U : Matrix n n ℝ) * Y) = Y := by
        intro Y
        rw [← Matrix.mul_assoc, Unitary.coe_star_mul_self, Matrix.one_mul]
      have hcancelV : ∀ Y : Matrix n n ℝ,
          star (V : Matrix n n ℝ) * ((V : Matrix n n ℝ) * Y) = Y := by
        intro Y
        rw [← Matrix.mul_assoc, Unitary.coe_star_mul_self, Matrix.one_mul]
      simp [Unitary.conjStarAlgAut_apply, Matrix.mul_assoc, hcancelU, hcancelV]
    _ = (U : Matrix n n ℝ) * (A * X + t • (X * B)) * star (V : Matrix n n ℝ) := by
      rw [Matrix.mul_add, Matrix.add_mul]
      simp [Matrix.mul_assoc]

theorem leftRightDenominatorMatrix_twoSidedUnitary_diagonal_solve_mulVec_vec
    (U V : unitaryGroup n ℝ) {d e : n → ℝ}
    (hd : ∀ i, 0 < d i) (he : ∀ i, 0 < e i)
    {t : ℝ} (ht : 0 ≤ t) (Y : Matrix n n ℝ) :
    let W : Matrix n n ℝ := fun i j => Y i j / (d i + t * e j)
    leftRightDenominatorMatrix
        (Unitary.conjStarAlgAut ℝ _ U (diagonal d))
        (Unitary.conjStarAlgAut ℝ _ V (diagonal e)) t *ᵥ
      vec ((U : Matrix n n ℝ) * W * star (V : Matrix n n ℝ)) =
        vec ((U : Matrix n n ℝ) * Y * star (V : Matrix n n ℝ)) := by
  intro W
  rw [leftRightDenominatorMatrix_twoSidedUnitary_mulVec_vec]
  apply congr_arg Matrix.vec
  congr 2
  ext i j
  have hden : d i + t * e j ≠ 0 := by
    exact ne_of_gt (add_pos_of_pos_of_nonneg (hd i) (mul_nonneg ht (le_of_lt (he j))))
  simp [W, Matrix.diagonal_mul, Matrix.mul_diagonal, div_eq_mul_inv]
  calc
    d i * (Y i j * (d i + t * e j)⁻¹) +
        t * (Y i j * (d i + t * e j)⁻¹ * e j) =
      (d i + t * e j) * (Y i j * (d i + t * e j)⁻¹) := by
        ring
    _ = Y i j := by
      field_simp [hden]

theorem leftRightDenominatorMatrix_posDef_spectral_solve_mulVec_vec
    {A B : Matrix n n ℝ} (hA : A.PosDef) (hB : B.PosDef)
    {t : ℝ} (ht : 0 ≤ t) :
    let U := hA.isHermitian.eigenvectorUnitary
    let V := hB.isHermitian.eigenvectorUnitary
    let d := hA.isHermitian.eigenvalues
    let e := hB.isHermitian.eigenvalues
    let Y : Matrix n n ℝ := star (U : Matrix n n ℝ) * (A - B) * (V : Matrix n n ℝ)
    let W : Matrix n n ℝ := fun i j => Y i j / (d i + t * e j)
    leftRightDenominatorMatrix A B t *ᵥ
      vec ((U : Matrix n n ℝ) * W * star (V : Matrix n n ℝ)) =
        vec (A - B) := by
  intro U V d e Y W
  have hspecA : A = Unitary.conjStarAlgAut ℝ _ U (diagonal d) := by
    simpa [U, d] using hA.isHermitian.spectral_theorem
  have hspecB : B = Unitary.conjStarAlgAut ℝ _ V (diagonal e) := by
    simpa [V, e] using hB.isHermitian.spectral_theorem
  calc
    leftRightDenominatorMatrix A B t *ᵥ
        vec ((U : Matrix n n ℝ) * W * star (V : Matrix n n ℝ)) =
      leftRightDenominatorMatrix
          (Unitary.conjStarAlgAut ℝ (Matrix n n ℝ) U (diagonal d))
          (Unitary.conjStarAlgAut ℝ (Matrix n n ℝ) V (diagonal e)) t *ᵥ
        vec ((U : Matrix n n ℝ) * W * star (V : Matrix n n ℝ)) := by
          rw [hspecA, hspecB]
    _ =
          vec ((U : Matrix n n ℝ) * Y * star (V : Matrix n n ℝ)) := by
      exact leftRightDenominatorMatrix_twoSidedUnitary_diagonal_solve_mulVec_vec
        U V (fun i => by simpa [d, U] using hA.eigenvalues_pos i)
          (fun i => by simpa [e, V] using hB.eigenvalues_pos i) ht Y
    _ = vec (A - B) := by
      apply congr_arg Matrix.vec
      rw [show Y = star (U : Matrix n n ℝ) * (A - B) * (V : Matrix n n ℝ) from rfl]
      rw [Matrix.mul_assoc (star (U : Matrix n n ℝ)) (A - B) (V : Matrix n n ℝ)]
      rw [← Matrix.mul_assoc (U : Matrix n n ℝ) (star (U : Matrix n n ℝ))
        ((A - B) * (V : Matrix n n ℝ))]
      rw [show (U : Matrix n n ℝ) * star (U : Matrix n n ℝ) = 1 from
        Unitary.coe_mul_star_self U]
      simp [Matrix.mul_assoc]

theorem leftRightDenominatorMatrix_posDef_inv_mulVec_spectral_solve
    {A B : Matrix n n ℝ} (hA : A.PosDef) (hB : B.PosDef)
    {t : ℝ} (ht : 0 ≤ t) :
    let U := hA.isHermitian.eigenvectorUnitary
    let V := hB.isHermitian.eigenvectorUnitary
    let d := hA.isHermitian.eigenvalues
    let e := hB.isHermitian.eigenvalues
    let Y : Matrix n n ℝ := star (U : Matrix n n ℝ) * (A - B) * (V : Matrix n n ℝ)
    let W : Matrix n n ℝ := fun i j => Y i j / (d i + t * e j)
    (leftRightDenominatorMatrix A B t)⁻¹ *ᵥ vec (A - B) =
      vec ((U : Matrix n n ℝ) * W * star (V : Matrix n n ℝ)) := by
  intro U V d e Y W
  let D : Matrix (n × n) (n × n) ℝ := leftRightDenominatorMatrix A B t
  let z : n × n → ℝ := vec ((U : Matrix n n ℝ) * W * star (V : Matrix n n ℝ))
  have hsolve : D *ᵥ z = vec (A - B) := by
    simpa [D, z, U, V, d, e, Y, W] using
      leftRightDenominatorMatrix_posDef_spectral_solve_mulVec_vec hA hB ht
  have hD_pos : D.PosDef := by
    simpa [D] using leftRightDenominatorMatrix_posDef hA hB ht
  have hdet : IsUnit D.det := by
    exact (ne_of_gt hD_pos.det_pos).isUnit
  calc
    D⁻¹ *ᵥ vec (A - B) = D⁻¹ *ᵥ (D *ᵥ z) := by
      rw [hsolve]
    _ = (D⁻¹ * D) *ᵥ z := by
      rw [Matrix.mulVec_mulVec]
    _ = z := by
      rw [Matrix.nonsing_inv_mul D hdet, Matrix.one_mulVec]

omit [Fintype n] in
theorem leftRightDenominatorMatrix_affine
    (A₁ A₂ B₁ B₂ : Matrix n n ℝ) (t a b : ℝ) :
    leftRightDenominatorMatrix (a • A₁ + b • A₂) (a • B₁ + b • B₂) t =
      a • leftRightDenominatorMatrix A₁ B₁ t +
        b • leftRightDenominatorMatrix A₂ B₂ t := by
  ext ij kl
  simp [leftRightDenominatorMatrix, Matrix.kroneckerMap_apply]
  ring

/-- A row matrix with one row and entries from a vector. -/
def rowMatrixOfVec {ι : Type*} (v : ι → ℝ) : Matrix Unit ι ℝ :=
  fun _ j => v j

theorem trace_rowMatrixOfVec_mul_mul_conjTranspose
    {ι : Type*} [Fintype ι] (v : ι → ℝ) (M : Matrix ι ι ℝ) :
    trace (rowMatrixOfVec v * M * (rowMatrixOfVec v)ᴴ) =
      v ⬝ᵥ (M *ᵥ v) := by
  simp [rowMatrixOfVec, trace, Matrix.diag, Matrix.mul_apply, dotProduct,
    Matrix.mulVec, Finset.univ_unique, Finset.mul_sum, Finset.sum_mul, mul_assoc]
  rw [Finset.sum_comm]

theorem vec_dotProduct_twoSidedUnitary
    (U V : unitaryGroup n ℝ) (X W : Matrix n n ℝ) :
    vec X ⬝ᵥ vec ((U : Matrix n n ℝ) * W * star (V : Matrix n n ℝ)) =
      vec (star (U : Matrix n n ℝ) * X * (V : Matrix n n ℝ)) ⬝ᵥ vec W := by
  rw [Matrix.vec_dotProduct_vec, Matrix.vec_dotProduct_vec]
  simp only [star_eq_conjTranspose, Matrix.conjTranspose_eq_transpose_of_trivial,
    transpose_transpose, transpose_mul]
  rw [show (Xᵀ * ((U : Matrix n n ℝ) * W * (V : Matrix n n ℝ)ᵀ)).trace =
      ((V : Matrix n n ℝ)ᵀ * (Xᵀ * ((U : Matrix n n ℝ) * W))).trace by
    rw [show Xᵀ * ((U : Matrix n n ℝ) * W * (V : Matrix n n ℝ)ᵀ) =
        (Xᵀ * ((U : Matrix n n ℝ) * W)) * (V : Matrix n n ℝ)ᵀ by
      simp [Matrix.mul_assoc]]
    rw [trace_mul_comm]]
  simp [Matrix.mul_assoc]

theorem unitaryGroup_sum_sq_row (U : unitaryGroup n ℝ) (i : n) :
    ∑ j, ((U : Matrix n n ℝ) i j) ^ 2 = 1 := by
  have h := congrArg (fun M : Matrix n n ℝ => M i i) (Unitary.coe_mul_star_self U)
  simpa [Matrix.mul_apply, dotProduct, pow_two, star_eq_conjTranspose,
    Matrix.conjTranspose_eq_transpose_of_trivial] using h

theorem unitaryGroup_sum_sq_col (U : unitaryGroup n ℝ) (j : n) :
    ∑ i, ((U : Matrix n n ℝ) i j) ^ 2 = 1 := by
  have h := congrArg (fun M : Matrix n n ℝ => M j j) (Unitary.coe_star_mul_self U)
  simpa [Matrix.mul_apply, dotProduct, pow_two, star_eq_conjTranspose,
    Matrix.conjTranspose_eq_transpose_of_trivial] using h

theorem conjTranspose_diagonal_mul_self_diag
    (C : Matrix n n ℝ) (d : n → ℝ) (j : n) :
    (star C * diagonal d * C) j j = ∑ i, d i * (C i j) ^ 2 := by
  simp [Matrix.mul_apply, Matrix.diagonal, pow_two, star_eq_conjTranspose,
    Matrix.conjTranspose_eq_transpose_of_trivial]
  refine Finset.sum_congr rfl ?_
  intro i hi
  ring

/--
The operator-level integrand in the noncommutative relative-entropy representation, written as an
inverse perspective after vectorizing the left/right multiplication operator.
-/
def leftRightRelativeEntropyIntegrand
    (t : ℝ) (A B : Matrix n n ℝ) : ℝ :=
  trace (rowMatrixOfVec (vec (A - B)) *
    (leftRightDenominatorMatrix A B t)⁻¹ *
      (rowMatrixOfVec (vec (A - B)))ᴴ)

theorem leftRightRelativeEntropyIntegrand_eq_quadratic
    (t : ℝ) (A B : Matrix n n ℝ) :
    leftRightRelativeEntropyIntegrand t A B =
      vec (A - B) ⬝ᵥ ((leftRightDenominatorMatrix A B t)⁻¹ *ᵥ vec (A - B)) := by
  rw [leftRightRelativeEntropyIntegrand, trace_rowMatrixOfVec_mul_mul_conjTranspose]

theorem leftRightRelativeEntropyIntegrand_posDef_spectral
    {A B : Matrix n n ℝ} (hA : A.PosDef) (hB : B.PosDef)
    {t : ℝ} (ht : 0 ≤ t) :
    let U := hA.isHermitian.eigenvectorUnitary
    let V := hB.isHermitian.eigenvectorUnitary
    let d := hA.isHermitian.eigenvalues
    let e := hB.isHermitian.eigenvalues
    let Y : Matrix n n ℝ := star (U : Matrix n n ℝ) * (A - B) * (V : Matrix n n ℝ)
    leftRightRelativeEntropyIntegrand t A B =
      ∑ i, ∑ j, (Y i j) ^ 2 / (d i + t * e j) := by
  intro U V d e Y
  let W : Matrix n n ℝ := fun i j => Y i j / (d i + t * e j)
  rw [leftRightRelativeEntropyIntegrand_eq_quadratic]
  rw [leftRightDenominatorMatrix_posDef_inv_mulVec_spectral_solve hA hB ht]
  change vec (A - B) ⬝ᵥ vec ((U : Matrix n n ℝ) * W * star (V : Matrix n n ℝ)) =
    ∑ i, ∑ j, (Y i j) ^ 2 / (d i + t * e j)
  rw [vec_dotProduct_twoSidedUnitary U V (A - B) W]
  have hY : star (U : Matrix n n ℝ) * (A - B) * (V : Matrix n n ℝ) = Y := rfl
  rw [hY]
  simp [dotProduct, Matrix.vec, W, pow_two, div_eq_mul_inv, Fintype.sum_prod_type, mul_assoc]
  rw [Finset.sum_comm]

theorem posDef_spectral_difference_twoSided_entries
    {A B : Matrix n n ℝ} (hA : A.PosDef) (hB : B.PosDef) :
    let U := hA.isHermitian.eigenvectorUnitary
    let V := hB.isHermitian.eigenvectorUnitary
    let d := hA.isHermitian.eigenvalues
    let e := hB.isHermitian.eigenvalues
    let C : Matrix n n ℝ := star (U : Matrix n n ℝ) * (V : Matrix n n ℝ)
    ∀ i j,
      (star (U : Matrix n n ℝ) * (A - B) * (V : Matrix n n ℝ)) i j =
        (d i - e j) * C i j := by
  intro U V d e C i j
  have hspecA : A = Unitary.conjStarAlgAut ℝ _ U (diagonal d) := by
    simpa [U, d] using hA.isHermitian.spectral_theorem
  have hspecB : B = Unitary.conjStarAlgAut ℝ _ V (diagonal e) := by
    simpa [V, e] using hB.isHermitian.spectral_theorem
  have hC : C = star (U : Matrix n n ℝ) * (V : Matrix n n ℝ) := rfl
  have hcancelU : ∀ Y : Matrix n n ℝ,
      star (U : Matrix n n ℝ) * ((U : Matrix n n ℝ) * Y) = Y := by
    intro Y
    rw [← Matrix.mul_assoc, Unitary.coe_star_mul_self, Matrix.one_mul]
  have hleft :
      star (U : Matrix n n ℝ) * A * (V : Matrix n n ℝ) = diagonal d * C := by
    rw [hspecA]
    simp [hC, Unitary.conjStarAlgAut_apply, Matrix.mul_assoc, hcancelU]
  have hright :
      star (U : Matrix n n ℝ) * B * (V : Matrix n n ℝ) = C * diagonal e := by
    rw [hspecB]
    simp [hC, Unitary.conjStarAlgAut_apply, Matrix.mul_assoc]
  calc
    (star (U : Matrix n n ℝ) * (A - B) * (V : Matrix n n ℝ)) i j =
        (star (U : Matrix n n ℝ) * A * (V : Matrix n n ℝ) -
          star (U : Matrix n n ℝ) * B * (V : Matrix n n ℝ)) i j := by
      simp [Matrix.mul_sub, Matrix.sub_mul, Matrix.mul_assoc]
    _ = (diagonal d * C - C * diagonal e) i j := by
      rw [hleft, hright]
    _ = (d i - e j) * C i j := by
      simp [Matrix.diagonal_mul, Matrix.mul_diagonal]
      ring

theorem posDef_spectral_conj_diag_entries
    {A B : Matrix n n ℝ} (hA : A.PosDef) (hB : B.PosDef) :
    let U := hA.isHermitian.eigenvectorUnitary
    let V := hB.isHermitian.eigenvectorUnitary
    let d := hA.isHermitian.eigenvalues
    let C : Matrix n n ℝ := star (U : Matrix n n ℝ) * (V : Matrix n n ℝ)
    ∀ j,
      (star (V : Matrix n n ℝ) * A * (V : Matrix n n ℝ)) j j =
        ∑ i, d i * (C i j) ^ 2 := by
  intro U V d C j
  have hspecA : A = Unitary.conjStarAlgAut ℝ _ U (diagonal d) := by
    simpa [U, d] using hA.isHermitian.spectral_theorem
  have hC : C = star (U : Matrix n n ℝ) * (V : Matrix n n ℝ) := rfl
  have hconj :
      star (V : Matrix n n ℝ) * A * (V : Matrix n n ℝ) =
        star C * diagonal d * C := by
    rw [hspecA]
    simp [hC, Unitary.conjStarAlgAut_apply, Matrix.mul_assoc,
      star_eq_conjTranspose, Matrix.conjTranspose_eq_transpose_of_trivial]
  rw [hconj]
  exact conjTranspose_diagonal_mul_self_diag C d j

theorem leftRightRelativeEntropyIntegrand_posDef_spectral_overlap
    {A B : Matrix n n ℝ} (hA : A.PosDef) (hB : B.PosDef)
    {t : ℝ} (ht : 0 ≤ t) :
    let U := hA.isHermitian.eigenvectorUnitary
    let V := hB.isHermitian.eigenvectorUnitary
    let d := hA.isHermitian.eigenvalues
    let e := hB.isHermitian.eigenvalues
    let C : Matrix n n ℝ := star (U : Matrix n n ℝ) * (V : Matrix n n ℝ)
    leftRightRelativeEntropyIntegrand t A B =
      ∑ i, ∑ j, (C i j) ^ 2 * ((d i - e j) ^ 2 / (d i + t * e j)) := by
  intro U V d e C
  rw [leftRightRelativeEntropyIntegrand_posDef_spectral hA hB ht]
  change
    ∑ i, ∑ j,
        ((star (U : Matrix n n ℝ) * (A - B) * (V : Matrix n n ℝ)) i j) ^ 2 /
            (d i + t * e j) =
      ∑ i, ∑ j, (C i j) ^ 2 * ((d i - e j) ^ 2 / (d i + t * e j))
  have hentry :
      ∀ i j,
        (star (U : Matrix n n ℝ) * (A - B) * (V : Matrix n n ℝ)) i j =
          (d i - e j) * C i j := by
    simpa [U, V, d, e, C] using posDef_spectral_difference_twoSided_entries hA hB
  refine Finset.sum_congr rfl ?_
  intro i hi
  refine Finset.sum_congr rfl ?_
  intro j hj
  rw [hentry i j]
  ring

/--
Joint convexity of the operator-level relative-entropy integrand represented by the inverse of
`X ↦ A * X + t • (X * B)`.
-/
theorem leftRightRelativeEntropyIntegrand_jointConvex
    {A₁ A₂ B₁ B₂ : Matrix n n ℝ}
    (hA₁ : A₁.PosDef) (hA₂ : A₂.PosDef)
    (hB₁ : B₁.PosDef) (hB₂ : B₂.PosDef)
    {t : ℝ} (ht : 0 ≤ t)
    {a b : ℝ} (ha : 0 ≤ a) (hb : 0 ≤ b) (hab : a + b = 1) :
    leftRightRelativeEntropyIntegrand t
        (a • A₁ + b • A₂) (a • B₁ + b • B₂) ≤
      a * leftRightRelativeEntropyIntegrand t A₁ B₁ +
        b * leftRightRelativeEntropyIntegrand t A₂ B₂ := by
  let N₁ : Matrix Unit (n × n) ℝ := rowMatrixOfVec (vec (A₁ - B₁))
  let N₂ : Matrix Unit (n × n) ℝ := rowMatrixOfVec (vec (A₂ - B₂))
  let D₁ : Matrix (n × n) (n × n) ℝ := leftRightDenominatorMatrix A₁ B₁ t
  let D₂ : Matrix (n × n) (n × n) ℝ := leftRightDenominatorMatrix A₂ B₂ t
  have hD₁ : D₁.PosDef := by
    simpa [D₁] using leftRightDenominatorMatrix_posDef hA₁ hB₁ ht
  have hD₂ : D₂.PosDef := by
    simpa [D₂] using leftRightDenominatorMatrix_posDef hA₂ hB₂ ht
  have hpersp :=
    trace_inversePerspective_jointConvex (m := Unit) (k := n × n)
      (X := (1 : Matrix Unit Unit ℝ)) Matrix.PosDef.one.posSemidef
      (B₁ := N₁) (B₂ := N₂) (D₁ := D₁) (D₂ := D₂)
      hD₁ hD₂ ha hb hab
  change
      trace ((1 : Matrix Unit Unit ℝ) *
        ((a • N₁ + b • N₂) * (a • D₁ + b • D₂)⁻¹ * (a • N₁ + b • N₂)ᴴ)) ≤
      a * trace ((1 : Matrix Unit Unit ℝ) * (N₁ * D₁⁻¹ * N₁ᴴ)) +
        b * trace ((1 : Matrix Unit Unit ℝ) * (N₂ * D₂⁻¹ * N₂ᴴ)) at hpersp
  have hnum :
      a • N₁ + b • N₂ =
        rowMatrixOfVec
          (vec ((a • A₁ + b • A₂) - (a • B₁ + b • B₂))) := by
    ext u ij
    cases u
    change
      a * ((A₁ - B₁) ij.2 ij.1) + b * ((A₂ - B₂) ij.2 ij.1) =
        ((a • A₁ + b • A₂) - (a • B₁ + b • B₂)) ij.2 ij.1
    simp [Matrix.sub_apply, Matrix.add_apply, Matrix.smul_apply]
    ring
  have hden :
      a • D₁ + b • D₂ =
        leftRightDenominatorMatrix
          (a • A₁ + b • A₂) (a • B₁ + b • B₂) t := by
    simpa [D₁, D₂] using
      (leftRightDenominatorMatrix_affine A₁ A₂ B₁ B₂ t a b).symm
  rw [hnum, hden] at hpersp
  simpa [leftRightRelativeEntropyIntegrand, N₁, N₂, D₁, D₂,
    Matrix.mul_assoc] using hpersp

/-- Nonnegativity of the operator-level left/right relative-entropy integrand. -/
theorem leftRightRelativeEntropyIntegrand_nonneg
    {A B : Matrix n n ℝ} (hA : A.PosDef) (hB : B.PosDef) {t : ℝ} (ht : 0 ≤ t) :
    0 ≤ leftRightRelativeEntropyIntegrand t A B := by
  let N : Matrix Unit (n × n) ℝ := rowMatrixOfVec (vec (A - B))
  let D : Matrix (n × n) (n × n) ℝ := leftRightDenominatorMatrix A B t
  have hD : D.PosDef := by
    simpa [D] using leftRightDenominatorMatrix_posDef hA hB ht
  have hpsd : (N * D⁻¹ * Nᴴ).PosSemidef :=
    hD.inv.posSemidef.mul_mul_conjTranspose_same N
  simpa [leftRightRelativeEntropyIntegrand, N, D] using hpsd.trace_nonneg

end LeftRightDenominator

section ScalarRelativeEntropyRepresentation

/--
Scalar improper-integral representation for the perspective of `x log x`.

This is the one-dimensional identity behind the noncommutative left/right representation with
denominator `A + tB`; the corresponding representation measure has density `t / (1 + t)^2` on
`(0, ∞)`.
-/
theorem real_log_perspective_integral {x : ℝ} (hx : 0 < x) :
    ∫ t in Set.Ioi (0 : ℝ), ((x - 1) ^ 2 * t) / ((x + t) * (1 + t) ^ 2) =
      x * Real.log x - x + 1 := by
  let F : ℝ → ℝ :=
    ((fun t : ℝ => -x * Real.log (x + t)) + fun t => x * Real.log (1 + t)) +
      fun t => (x - 1) * (1 + t)⁻¹
  let f : ℝ → ℝ :=
    fun t => ((x - 1) ^ 2 * t) / ((x + t) * (1 + t) ^ 2)
  have hderiv : ∀ t ∈ Set.Ici (0 : ℝ), HasDerivAt F (f t) t := by
    intro t ht
    have ht0 : 0 ≤ t := ht
    have hxt_pos : 0 < x + t := by linarith
    have h1t_pos : 0 < 1 + t := by linarith
    have hxt : x + t ≠ 0 := hxt_pos.ne'
    have h1t : 1 + t ≠ 0 := h1t_pos.ne'
    have hlogx : HasDerivAt (fun y : ℝ => Real.log (x + y)) (1 / (x + t)) t := by
      have hbase : HasDerivAt (fun y : ℝ => x + y) 1 t :=
        (hasDerivAt_id t).const_add x
      simpa [one_div] using hbase.log hxt
    have hlog1 : HasDerivAt (fun y : ℝ => Real.log (1 + y)) (1 / (1 + t)) t := by
      have hbase : HasDerivAt (fun y : ℝ => 1 + y) 1 t :=
        (hasDerivAt_id t).const_add 1
      simpa [one_div] using hbase.log h1t
    have hinv1 : HasDerivAt (fun y : ℝ => (x - 1) * (1 + y)⁻¹)
        ((x - 1) * (-1 / (1 + t) ^ 2)) t := by
      have hbase : HasDerivAt (fun y : ℝ => 1 + y) 1 t :=
        (hasDerivAt_id t).const_add 1
      simpa using (hbase.inv h1t).const_mul (x - 1)
    have hraw : HasDerivAt F
        ((-x) * (1 / (x + t)) + x * (1 / (1 + t)) +
          ((x - 1) * (-1 / (1 + t) ^ 2))) t := by
      simpa [F] using ((hlogx.const_mul (-x)).add (hlog1.const_mul x)).add hinv1
    convert hraw using 1
    change
      ((x - 1) ^ 2 * t) / ((x + t) * (1 + t) ^ 2) =
        -x * (1 / (x + t)) + x * (1 / (1 + t)) +
          (x - 1) * (-1 / (1 + t) ^ 2)
    field_simp [hxt, h1t]
    ring
  have hnonneg : ∀ t ∈ Set.Ioi (0 : ℝ), 0 ≤ f t := by
    intro t ht
    have ht0 : 0 ≤ t := le_of_lt ht
    have hden_pos : 0 < (x + t) * (1 + t) ^ 2 := by positivity
    exact div_nonneg (mul_nonneg (sq_nonneg _) ht0) hden_pos.le
  have hlim : Tendsto F atTop (𝓝 0) := by
    have hlog_ratio :
        Tendsto (fun t : ℝ => Real.log ((1 + t) / (x + t))) atTop (𝓝 0) := by
      have hratio :
          Tendsto (fun t : ℝ => (1 + t) / (x + t)) atTop (𝓝 1) := by
        have hden_atTop : Tendsto (fun t : ℝ => x + t) atTop atTop :=
          tendsto_atTop_add_const_left atTop x tendsto_id
        have hsmall :
            Tendsto (fun t : ℝ => (1 - x) / (x + t)) atTop (𝓝 0) := by
          have hinv : Tendsto (fun t : ℝ => (x + t)⁻¹) atTop (𝓝 0) :=
            tendsto_inv_atTop_zero.comp hden_atTop
          simpa [div_eq_mul_inv] using (tendsto_const_nhds.mul hinv :
            Tendsto (fun t : ℝ => (1 - x) * (x + t)⁻¹) atTop (𝓝 ((1 - x) * 0)))
        have hsum : Tendsto (fun t : ℝ => 1 + (1 - x) / (x + t)) atTop (𝓝 1) := by
          simpa using tendsto_const_nhds.add hsmall
        refine hsum.congr' ?_
        filter_upwards [Ioi_mem_atTop (-x)] with t ht
        have ht' : -x < t := ht
        have hxt : x + t ≠ 0 := by linarith
        field_simp [hxt]
        ring
      simpa [Function.comp_def, Real.log_one] using
        (Real.continuousAt_log (by norm_num : (1 : ℝ) ≠ 0)).tendsto.comp hratio
    have hmain :
        Tendsto (fun t : ℝ => x * Real.log ((1 + t) / (x + t))) atTop (𝓝 0) := by
      simpa using tendsto_const_nhds.mul hlog_ratio
    have htail : Tendsto (fun t : ℝ => (x - 1) * (1 + t)⁻¹) atTop (𝓝 0) := by
      have hden : Tendsto (fun t : ℝ => (1 + t)) atTop atTop :=
        tendsto_atTop_add_const_left atTop (1 : ℝ) tendsto_id
      have hinv : Tendsto (fun t : ℝ => (1 + t)⁻¹) atTop (𝓝 0) :=
        tendsto_inv_atTop_zero.comp hden
      simpa using (tendsto_const_nhds.mul hinv :
        Tendsto (fun t : ℝ => (x - 1) * (1 + t)⁻¹) atTop (𝓝 ((x - 1) * 0)))
    have hF_eq :
        (fun t => F t) =ᶠ[atTop]
          fun t => x * Real.log ((1 + t) / (x + t)) + (x - 1) * (1 + t)⁻¹ := by
      filter_upwards [Ioi_mem_atTop (max 0 (-x))] with t ht
      have hxt_pos : 0 < x + t := by
        have ht' : -x < t := lt_of_le_of_lt (le_max_right 0 (-x)) ht
        linarith
      have h1t_pos : 0 < 1 + t := by
        have ht0 : 0 < t := lt_of_le_of_lt (le_max_left 0 (-x)) ht
        linarith
      simp [F]
      rw [Real.log_div h1t_pos.ne' hxt_pos.ne']
      ring
    simpa using (hmain.add htail).congr' hF_eq.symm
  have hInt := integral_Ioi_of_hasDerivAt_of_nonneg'
      (a := (0 : ℝ)) (g := F) (g' := f) (l := 0) hderiv hnonneg hlim
  calc
    ∫ t in Set.Ioi (0 : ℝ), ((x - 1) ^ 2 * t) / ((x + t) * (1 + t) ^ 2)
        = ∫ t in Set.Ioi (0 : ℝ), f t := rfl
    _ = 0 - F 0 := hInt
    _ = x * Real.log x - x + 1 := by
      simp [F, Real.log_one]
      ring

/--
Two-parameter scalar relative-entropy representation with denominator `a + t b`.

This is the scalar specialization of the left/right representation target:
`a * (log a - log b)` is the affine term `a - b` plus the perspective integral.
-/
theorem real_relativeEntropy_integral_representation
    {a b : ℝ} (ha : 0 < a) (hb : 0 < b) :
    ∫ t in Set.Ioi (0 : ℝ),
        ((a - b) ^ 2 * t) / ((a + t * b) * (1 + t) ^ 2) =
      a * (Real.log a - Real.log b) - (a - b) := by
  have hx : 0 < a / b := div_pos ha hb
  have hpoint :
      Set.EqOn
        (fun t : ℝ => ((a - b) ^ 2 * t) / ((a + t * b) * (1 + t) ^ 2))
        (fun t : ℝ => b * ((((a / b) - 1) ^ 2 * t) / (((a / b) + t) * (1 + t) ^ 2)))
        (Set.Ioi (0 : ℝ)) := by
    intro t ht
    have ht0 : 0 < t := ht
    have hbne : b ≠ 0 := hb.ne'
    have hden₁ : a + t * b ≠ 0 := by positivity
    have hden₂ : a / b + t ≠ 0 := by
      have : 0 < a / b + t := by positivity
      exact this.ne'
    have hden₃ : 1 + t ≠ 0 := by positivity
    field_simp [hbne, hden₁, hden₂, hden₃]
  calc
    ∫ t in Set.Ioi (0 : ℝ), ((a - b) ^ 2 * t) / ((a + t * b) * (1 + t) ^ 2)
        = ∫ t in Set.Ioi (0 : ℝ),
            b * ((((a / b) - 1) ^ 2 * t) / (((a / b) + t) * (1 + t) ^ 2)) := by
      exact setIntegral_congr_fun measurableSet_Ioi hpoint
    _ = b * (∫ t in Set.Ioi (0 : ℝ),
            (((a / b) - 1) ^ 2 * t) / (((a / b) + t) * (1 + t) ^ 2)) := by
      rw [integral_const_mul]
    _ = b * ((a / b) * Real.log (a / b) - (a / b) + 1) := by
      rw [real_log_perspective_integral hx]
    _ = a * (Real.log a - Real.log b) - (a - b) := by
      rw [Real.log_div ha.ne' hb.ne']
      field_simp [hb.ne']
      ring

/-- Density form of `real_relativeEntropy_integral_representation`. -/
theorem real_relativeEntropy_integral_representation_density
    {a b : ℝ} (ha : 0 < a) (hb : 0 < b) :
    ∫ t in Set.Ioi (0 : ℝ),
        (t / (1 + t) ^ 2) * ((a - b) ^ 2 / (a + t * b)) =
      a * (Real.log a - Real.log b) - (a - b) := by
  calc
    ∫ t in Set.Ioi (0 : ℝ),
        (t / (1 + t) ^ 2) * ((a - b) ^ 2 / (a + t * b))
        = ∫ t in Set.Ioi (0 : ℝ),
            ((a - b) ^ 2 * t) / ((a + t * b) * (1 + t) ^ 2) := by
      refine setIntegral_congr_fun measurableSet_Ioi ?_
      intro t ht
      have hden₁ : a + t * b ≠ 0 := by
        exact (add_pos_of_pos_of_nonneg ha (mul_nonneg (le_of_lt ht) hb.le)).ne'
      have hden₂ : (1 + t) ^ 2 ≠ 0 := by
        have ht0 : 0 < t := ht
        have h1t : 1 + t ≠ 0 := by linarith
        exact pow_ne_zero 2 h1t
      field_simp [hden₁, hden₂]
    _ = a * (Real.log a - Real.log b) - (a - b) :=
      real_relativeEntropy_integral_representation ha hb

/-- Integrability of the scalar relative-entropy representation integrand on `(0, ∞)`. -/
theorem real_relativeEntropy_integrand_integrableOn
    {a b : ℝ} (ha : 0 < a) (hb : 0 < b) :
    IntegrableOn
      (fun t : ℝ => ((a - b) ^ 2 * t) / ((a + t * b) * (1 + t) ^ 2))
      (Set.Ioi (0 : ℝ)) := by
  by_cases hab : a = b
  · subst a
    simp
  · rw [IntegrableOn]
    refine Integrable.of_integral_ne_zero ?_
    have hrepr := real_relativeEntropy_integral_representation (a := a) (b := b) ha hb
    have hx : 0 < a / b := div_pos ha hb
    have hxne : a / b ≠ 1 := by
      intro h
      have hbne : b ≠ 0 := hb.ne'
      have : a = b := by
        field_simp [hbne] at h
        exact h
      exact hab this
    have hinner_pos : 0 < (a / b) * Real.log (a / b) - (a / b) + 1 := by
      have hlt := Real.self_sub_one_lt_mul_log hx.le hxne
      linarith
    have hrhs_pos : 0 < a * (Real.log a - Real.log b) - (a - b) := by
      have hrewrite :
          a * (Real.log a - Real.log b) - (a - b) =
            b * ((a / b) * Real.log (a / b) - (a / b) + 1) := by
        rw [Real.log_div ha.ne' hb.ne']
        have hbne : b ≠ 0 := hb.ne'
        field_simp [hbne]
        ring
      rw [hrewrite]
      exact mul_pos hb hinner_pos
    exact ne_of_gt (by simpa [hrepr] using hrhs_pos)

end ScalarRelativeEntropyRepresentation

section DiagonalLeftRightIntegralRepresentation

variable {n : Type*} [Fintype n] [DecidableEq n]

theorem integrableOn_leftRightRelativeEntropyIntegrand_posDef_density
    {A B : Matrix n n ℝ} (hA : A.PosDef) (hB : B.PosDef) :
    IntegrableOn
      (fun t : ℝ => (t / (1 + t) ^ 2) * leftRightRelativeEntropyIntegrand t A B)
      (Set.Ioi (0 : ℝ)) := by
  let U := hA.isHermitian.eigenvectorUnitary
  let V := hB.isHermitian.eigenvectorUnitary
  let d := hA.isHermitian.eigenvalues
  let e := hB.isHermitian.eigenvalues
  let C : Matrix n n ℝ := star (U : Matrix n n ℝ) * (V : Matrix n n ℝ)
  let f : n → n → ℝ → ℝ :=
    fun i j t => (C i j) ^ 2 *
      (((d i - e j) ^ 2 * t) / ((d i + t * e j) * (1 + t) ^ 2))
  have hd : ∀ i, 0 < d i := by
    intro i
    simpa [d, U] using hA.eigenvalues_pos i
  have he : ∀ j, 0 < e j := by
    intro j
    simpa [e, V] using hB.eigenvalues_pos j
  have hf :
      IntegrableOn (fun t : ℝ => ∑ i, ∑ j, f i j t) (Set.Ioi (0 : ℝ)) := by
    rw [IntegrableOn]
    exact integrable_finsetSum Finset.univ fun i hi =>
      integrable_finsetSum Finset.univ fun j hj =>
        (real_relativeEntropy_integrand_integrableOn
          (a := d i) (b := e j) (hd i) (he j)).const_mul ((C i j) ^ 2)
  refine hf.congr_fun ?_ measurableSet_Ioi
  intro t ht
  change (∑ i, ∑ j, f i j t) =
    (t / (1 + t) ^ 2) * leftRightRelativeEntropyIntegrand t A B
  have hoverlap :
      leftRightRelativeEntropyIntegrand t A B =
        ∑ i, ∑ j, (C i j) ^ 2 * ((d i - e j) ^ 2 / (d i + t * e j)) := by
    simpa [U, V, d, e, C] using
      leftRightRelativeEntropyIntegrand_posDef_spectral_overlap hA hB (le_of_lt ht)
  rw [hoverlap]
  simp only [f]
  rw [Finset.mul_sum]
  refine (Finset.sum_congr rfl ?_).symm
  intro i hi
  rw [Finset.mul_sum]
  refine (Finset.sum_congr rfl ?_).symm
  intro j hj
  have hden₁ : d i + t * e j ≠ 0 := by
    exact (add_pos_of_pos_of_nonneg (hd i) (mul_nonneg (le_of_lt ht) (le_of_lt (he j)))).ne'
  have hden₂ : (1 + t) ^ 2 ≠ 0 := by
    have h1t_pos : 0 < 1 + t := by linarith [show 0 < t from ht]
    have h1t : 1 + t ≠ 0 := h1t_pos.ne'
    exact pow_ne_zero 2 h1t
  field_simp [hden₁, hden₂]

theorem integral_leftRightRelativeEntropyIntegrand_posDef_spectral_overlap_density
    {A B : Matrix n n ℝ} (hA : A.PosDef) (hB : B.PosDef) :
    let U := hA.isHermitian.eigenvectorUnitary
    let V := hB.isHermitian.eigenvectorUnitary
    let d := hA.isHermitian.eigenvalues
    let e := hB.isHermitian.eigenvalues
    let C : Matrix n n ℝ := star (U : Matrix n n ℝ) * (V : Matrix n n ℝ)
    ∫ t in Set.Ioi (0 : ℝ),
        (t / (1 + t) ^ 2) * leftRightRelativeEntropyIntegrand t A B =
      ∑ i, ∑ j,
        (C i j) ^ 2 * (d i * (Real.log (d i) - Real.log (e j)) - (d i - e j)) := by
  intro U V d e C
  let f : n → n → ℝ → ℝ :=
    fun i j t => (C i j) ^ 2 *
      (((d i - e j) ^ 2 * t) / ((d i + t * e j) * (1 + t) ^ 2))
  have hd : ∀ i, 0 < d i := by
    intro i
    simpa [d, U] using hA.eigenvalues_pos i
  have he : ∀ j, 0 < e j := by
    intro j
    simpa [e, V] using hB.eigenvalues_pos j
  calc
    ∫ t in Set.Ioi (0 : ℝ),
        (t / (1 + t) ^ 2) * leftRightRelativeEntropyIntegrand t A B =
      ∫ t in Set.Ioi (0 : ℝ), ∑ i, ∑ j, f i j t := by
        refine setIntegral_congr_fun measurableSet_Ioi ?_
        intro t ht
        change (t / (1 + t) ^ 2) * leftRightRelativeEntropyIntegrand t A B =
          ∑ i, ∑ j, f i j t
        have hoverlap :
            leftRightRelativeEntropyIntegrand t A B =
              ∑ i, ∑ j, (C i j) ^ 2 * ((d i - e j) ^ 2 / (d i + t * e j)) := by
          simpa [U, V, d, e, C] using
            leftRightRelativeEntropyIntegrand_posDef_spectral_overlap hA hB (le_of_lt ht)
        rw [hoverlap]
        simp only [f]
        rw [Finset.mul_sum]
        refine Finset.sum_congr rfl ?_
        intro i hi
        rw [Finset.mul_sum]
        refine Finset.sum_congr rfl ?_
        intro j hj
        have hden₁ : d i + t * e j ≠ 0 := by
          exact (add_pos_of_pos_of_nonneg (hd i) (mul_nonneg (le_of_lt ht) (le_of_lt (he j)))).ne'
        have hden₂ : (1 + t) ^ 2 ≠ 0 := by
          have h1t_pos : 0 < 1 + t := by linarith [show 0 < t from ht]
          have h1t : 1 + t ≠ 0 := h1t_pos.ne'
          exact pow_ne_zero 2 h1t
        field_simp [hden₁, hden₂]
    _ = ∑ i, ∫ t in Set.Ioi (0 : ℝ), ∑ j, f i j t := by
      rw [integral_finsetSum]
      intro i hi
      exact integrable_finsetSum Finset.univ fun j hj =>
        (real_relativeEntropy_integrand_integrableOn
          (a := d i) (b := e j) (hd i) (he j)).const_mul ((C i j) ^ 2)
    _ = ∑ i, ∑ j, ∫ t in Set.Ioi (0 : ℝ), f i j t := by
      refine Finset.sum_congr rfl ?_
      intro i hi
      rw [integral_finsetSum]
      intro j hj
      exact (real_relativeEntropy_integrand_integrableOn
        (a := d i) (b := e j) (hd i) (he j)).const_mul ((C i j) ^ 2)
    _ = ∑ i, ∑ j,
        (C i j) ^ 2 * (d i * (Real.log (d i) - Real.log (e j)) - (d i - e j)) := by
      refine Finset.sum_congr rfl ?_
      intro i hi
      refine Finset.sum_congr rfl ?_
      intro j hj
      simp [f, integral_const_mul,
        real_relativeEntropy_integral_representation (a := d i) (b := e j) (hd i) (he j)]

end DiagonalLeftRightIntegralRepresentation

section RelativeEntropyIntegrand

variable {n : Type*} [Fintype n] [DecidableEq n]

/-- Matrix relative entropy in trace form, `tr(A(log A - log B))`. -/
def traceMatrixRelativeEntropy (A B : Matrix n n ℝ) : ℝ :=
  trace (A * (CFC.log A - CFC.log B))

/-- The CFC logarithm written in the Hermitian spectral decomposition. -/
theorem cfc_log_eq_spectral {A : Matrix n n ℝ} (hA : A.IsHermitian) :
    CFC.log A =
      Unitary.conjStarAlgAut ℝ _ hA.eigenvectorUnitary
        (diagonal (fun i => Real.log (hA.eigenvalues i))) := by
  unfold CFC.log
  rw [hA.cfc_eq Real.log]
  rfl

/-- The self-entropy trace term `tr(A log A)` as a sum over Hermitian eigenvalues. -/
theorem trace_mul_cfc_log_self_eq_sum_eigenvalues_log
    {A : Matrix n n ℝ} (hA : A.IsHermitian) :
    trace (A * CFC.log A) =
      ∑ i, hA.eigenvalues i * Real.log (hA.eigenvalues i) := by
  let U := hA.eigenvectorUnitary
  let e := hA.eigenvalues
  let D : Matrix n n ℝ := diagonal e
  let L : Matrix n n ℝ := diagonal (fun i => Real.log (e i))
  have hspec : A = Unitary.conjStarAlgAut ℝ _ U D := by
    simpa [U, e, D] using hA.spectral_theorem
  have hlog : CFC.log A = Unitary.conjStarAlgAut ℝ _ U L := by
    simpa [U, e, L] using cfc_log_eq_spectral hA
  calc
    trace (A * CFC.log A) =
        trace ((Unitary.conjStarAlgAut ℝ _ U D) *
          (Unitary.conjStarAlgAut ℝ _ U L)) := by
      rw [hlog, hspec]
    _ = trace (Unitary.conjStarAlgAut ℝ _ U (D * L)) := by
      rw [← map_mul]
    _ = trace (D * L) := by
      simp [Unitary.conjStarAlgAut_apply, Matrix.mul_assoc]
      rw [trace_mul_comm (U : Matrix n n ℝ) (D * (L * star (U : Matrix n n ℝ)))]
      simp [Matrix.mul_assoc]
    _ = ∑ i, hA.eigenvalues i * Real.log (hA.eigenvalues i) := by
      simp [D, L, e, Matrix.diagonal_mul_diagonal, Matrix.trace_diagonal]

/-- Trace of right multiplication by a diagonal matrix. -/
theorem trace_mul_diagonal (M : Matrix n n ℝ) (d : n → ℝ) :
    trace (M * diagonal d) = ∑ i, M i i * d i := by
  simp [trace, Matrix.diag]

/--
The cross log trace term `tr(A log B)` written in the eigenbasis of the Hermitian matrix `B`.
-/
theorem trace_mul_cfc_log_eq_sum_diag_conj_spectral
    (A B : Matrix n n ℝ) (hB : B.IsHermitian) :
    trace (A * CFC.log B) =
      ∑ i, (star (hB.eigenvectorUnitary : Matrix n n ℝ) * A *
          (hB.eigenvectorUnitary : Matrix n n ℝ)) i i *
        Real.log (hB.eigenvalues i) := by
  let U := hB.eigenvectorUnitary
  let L : Matrix n n ℝ := diagonal (fun i => Real.log (hB.eigenvalues i))
  have hlog : CFC.log B = Unitary.conjStarAlgAut ℝ _ U L := by
    simpa [U, L] using cfc_log_eq_spectral hB
  calc
    trace (A * CFC.log B) =
        trace (A * Unitary.conjStarAlgAut ℝ _ U L) := by
      rw [hlog]
    _ = trace ((star (U : Matrix n n ℝ) * A * (U : Matrix n n ℝ)) * L) := by
      simp only [Unitary.conjStarAlgAut_apply]
      rw [show A * (↑U * L * star ↑U) = A * ↑U * L * star ↑U by
        simp [Matrix.mul_assoc]]
      rw [trace_mul_comm (A * (U : Matrix n n ℝ) * L) (star (U : Matrix n n ℝ))]
      simp [Matrix.mul_assoc]
    _ = ∑ i, (star (U : Matrix n n ℝ) * A * (U : Matrix n n ℝ)) i i *
        Real.log (hB.eigenvalues i) := by
      rw [trace_mul_diagonal]

/-- Trace matrix relative entropy as spectral sums for Hermitian inputs. -/
theorem traceMatrixRelativeEntropy_eq_spectral_sums
    (A B : Matrix n n ℝ) (hA : A.IsHermitian) (hB : B.IsHermitian) :
    traceMatrixRelativeEntropy A B =
      (∑ i, hA.eigenvalues i * Real.log (hA.eigenvalues i)) -
        ∑ i, (star (hB.eigenvectorUnitary : Matrix n n ℝ) * A *
            (hB.eigenvectorUnitary : Matrix n n ℝ)) i i *
          Real.log (hB.eigenvalues i) := by
  unfold traceMatrixRelativeEntropy
  rw [mul_sub, trace_sub]
  rw [trace_mul_cfc_log_self_eq_sum_eigenvalues_log hA]
  rw [trace_mul_cfc_log_eq_sum_diag_conj_spectral A B hB]

theorem traceMatrixRelativeEntropy_posDef_spectral_overlap
    {A B : Matrix n n ℝ} (hA : A.PosDef) (hB : B.PosDef) :
    let U := hA.isHermitian.eigenvectorUnitary
    let V := hB.isHermitian.eigenvectorUnitary
    let d := hA.isHermitian.eigenvalues
    let e := hB.isHermitian.eigenvalues
    let C : Matrix n n ℝ := star (U : Matrix n n ℝ) * (V : Matrix n n ℝ)
    traceMatrixRelativeEntropy A B =
      ∑ i, ∑ j, (C i j) ^ 2 * d i * (Real.log (d i) - Real.log (e j)) := by
  intro U V d e C
  let CU : unitaryGroup n ℝ := star U * V
  have hCU : (CU : Matrix n n ℝ) = C := by
    simp [CU, C, U, V]
  have hrow : ∀ i, ∑ j, (C i j) ^ 2 = 1 := by
    intro i
    simpa [hCU] using unitaryGroup_sum_sq_row CU i
  have hdiag :
      ∀ j, (star (V : Matrix n n ℝ) * A * (V : Matrix n n ℝ)) j j =
        ∑ i, d i * (C i j) ^ 2 := by
    simpa [U, V, d, C] using posDef_spectral_conj_diag_entries hA hB
  rw [traceMatrixRelativeEntropy_eq_spectral_sums A B hA.isHermitian hB.isHermitian]
  change
    (∑ i, d i * Real.log (d i)) -
      ∑ j, (star (V : Matrix n n ℝ) * A * (V : Matrix n n ℝ)) j j *
        Real.log (e j) =
      ∑ i, ∑ j, (C i j) ^ 2 * d i * (Real.log (d i) - Real.log (e j))
  simp_rw [hdiag]
  have hfirst :
      ∑ i, d i * Real.log (d i) =
        ∑ i, ∑ j, (C i j) ^ 2 * (d i * Real.log (d i)) := by
    calc
      ∑ i, d i * Real.log (d i) =
          ∑ i, (∑ j, (C i j) ^ 2) * (d i * Real.log (d i)) := by
        refine Finset.sum_congr rfl ?_
        intro i hi
        rw [hrow i, one_mul]
      _ = ∑ i, ∑ j, (C i j) ^ 2 * (d i * Real.log (d i)) := by
        simp [Finset.sum_mul]
  have hsecond :
      ∑ j, (∑ i, d i * (C i j) ^ 2) * Real.log (e j) =
        ∑ i, ∑ j, (C i j) ^ 2 * d i * Real.log (e j) := by
    calc
      ∑ j, (∑ i, d i * (C i j) ^ 2) * Real.log (e j) =
          ∑ j, ∑ i, d i * (C i j) ^ 2 * Real.log (e j) := by
        simp [Finset.sum_mul]
      _ = ∑ i, ∑ j, d i * (C i j) ^ 2 * Real.log (e j) := by
        rw [Finset.sum_comm]
      _ = ∑ i, ∑ j, (C i j) ^ 2 * d i * Real.log (e j) := by
        refine Finset.sum_congr rfl ?_
        intro i hi
        refine Finset.sum_congr rfl ?_
        intro j hj
        ring
  rw [hfirst, hsecond]
  rw [← Finset.sum_sub_distrib]
  refine Finset.sum_congr rfl ?_
  intro i hi
  rw [← Finset.sum_sub_distrib]
  refine Finset.sum_congr rfl ?_
  intro j hj
  ring

theorem trace_sub_posDef_spectral_overlap
    {A B : Matrix n n ℝ} (hA : A.PosDef) (hB : B.PosDef) :
    let U := hA.isHermitian.eigenvectorUnitary
    let V := hB.isHermitian.eigenvectorUnitary
    let d := hA.isHermitian.eigenvalues
    let e := hB.isHermitian.eigenvalues
    let C : Matrix n n ℝ := star (U : Matrix n n ℝ) * (V : Matrix n n ℝ)
    trace (A - B) = ∑ i, ∑ j, (C i j) ^ 2 * (d i - e j) := by
  intro U V d e C
  let CU : unitaryGroup n ℝ := star U * V
  have hCU : (CU : Matrix n n ℝ) = C := by
    simp [CU, C, U, V]
  have hrow : ∀ i, ∑ j, (C i j) ^ 2 = 1 := by
    intro i
    simpa [hCU] using unitaryGroup_sum_sq_row CU i
  have hcol : ∀ j, ∑ i, (C i j) ^ 2 = 1 := by
    intro j
    simpa [hCU] using unitaryGroup_sum_sq_col CU j
  have hfirst : ∑ i, d i = ∑ i, ∑ j, (C i j) ^ 2 * d i := by
    calc
      ∑ i, d i = ∑ i, (∑ j, (C i j) ^ 2) * d i := by
        refine Finset.sum_congr rfl ?_
        intro i hi
        rw [hrow i, one_mul]
      _ = ∑ i, ∑ j, (C i j) ^ 2 * d i := by
        simp [Finset.sum_mul]
  have hsecond : ∑ j, e j = ∑ i, ∑ j, (C i j) ^ 2 * e j := by
    calc
      ∑ j, e j = ∑ j, (∑ i, (C i j) ^ 2) * e j := by
        refine Finset.sum_congr rfl ?_
        intro j hj
        rw [hcol j, one_mul]
      _ = ∑ j, ∑ i, (C i j) ^ 2 * e j := by
        simp [Finset.sum_mul]
      _ = ∑ i, ∑ j, (C i j) ^ 2 * e j := by
        rw [Finset.sum_comm]
  calc
    trace (A - B) = trace A - trace B := by
      rw [trace_sub]
    _ = ∑ i, d i - ∑ j, e j := by
      rw [hA.isHermitian.trace_eq_sum_eigenvalues, hB.isHermitian.trace_eq_sum_eigenvalues]
      simp [d, e]
    _ = ∑ i, ∑ j, (C i j) ^ 2 * (d i - e j) := by
      rw [hfirst, hsecond]
      rw [← Finset.sum_sub_distrib]
      refine Finset.sum_congr rfl ?_
      intro i hi
      rw [← Finset.sum_sub_distrib]
      refine Finset.sum_congr rfl ?_
      intro j hj
      ring

theorem traceMatrixRelativeEntropy_leftRight_integral_representation
    {A B : Matrix n n ℝ} (hA : A.PosDef) (hB : B.PosDef) :
    traceMatrixRelativeEntropy A B =
      trace (A - B) +
        ∫ t in Set.Ioi (0 : ℝ),
          (t / (1 + t) ^ 2) * leftRightRelativeEntropyIntegrand t A B := by
  let U := hA.isHermitian.eigenvectorUnitary
  let V := hB.isHermitian.eigenvectorUnitary
  let d := hA.isHermitian.eigenvalues
  let e := hB.isHermitian.eigenvalues
  let C : Matrix n n ℝ := star (U : Matrix n n ℝ) * (V : Matrix n n ℝ)
  have hD :
      traceMatrixRelativeEntropy A B =
        ∑ i, ∑ j, (C i j) ^ 2 * d i * (Real.log (d i) - Real.log (e j)) := by
    simpa [U, V, d, e, C] using traceMatrixRelativeEntropy_posDef_spectral_overlap hA hB
  have htrace :
      trace (A - B) = ∑ i, ∑ j, (C i j) ^ 2 * (d i - e j) := by
    simpa [U, V, d, e, C] using trace_sub_posDef_spectral_overlap hA hB
  have hint :
      ∫ t in Set.Ioi (0 : ℝ),
          (t / (1 + t) ^ 2) * leftRightRelativeEntropyIntegrand t A B =
        ∑ i, ∑ j,
          (C i j) ^ 2 * (d i * (Real.log (d i) - Real.log (e j)) - (d i - e j)) := by
    simpa [U, V, d, e, C] using
      integral_leftRightRelativeEntropyIntegrand_posDef_spectral_overlap_density hA hB
  calc
    traceMatrixRelativeEntropy A B =
        ∑ i, ∑ j, (C i j) ^ 2 * d i * (Real.log (d i) - Real.log (e j)) := hD
    _ = (∑ i, ∑ j, (C i j) ^ 2 * (d i - e j)) +
        ∑ i, ∑ j,
          (C i j) ^ 2 * (d i * (Real.log (d i) - Real.log (e j)) - (d i - e j)) := by
      rw [← Finset.sum_add_distrib]
      refine Finset.sum_congr rfl ?_
      intro i hi
      rw [← Finset.sum_add_distrib]
      refine Finset.sum_congr rfl ?_
      intro j hj
      ring
    _ = trace (A - B) +
        ∫ t in Set.Ioi (0 : ℝ),
          (t / (1 + t) ^ 2) * leftRightRelativeEntropyIntegrand t A B := by
      rw [htrace, hint]

theorem traceMatrixRelativeEntropy_klein
    {A B : Matrix n n ℝ} (hA : A.PosDef) (hB : B.PosDef) :
    trace (A - B) ≤ traceMatrixRelativeEntropy A B := by
  have hrepr := traceMatrixRelativeEntropy_leftRight_integral_representation hA hB
  have hnonneg :
      0 ≤ᵐ[volume.restrict (Set.Ioi (0 : ℝ))]
        fun t : ℝ => (t / (1 + t) ^ 2) * leftRightRelativeEntropyIntegrand t A B := by
    filter_upwards [ae_restrict_mem measurableSet_Ioi] with t ht
    have hdensity : 0 ≤ t / (1 + t) ^ 2 :=
      div_nonneg (le_of_lt ht) (sq_nonneg (1 + t))
    exact mul_nonneg hdensity (leftRightRelativeEntropyIntegrand_nonneg hA hB (le_of_lt ht))
  have hintegral_nonneg :
      0 ≤ ∫ t in Set.Ioi (0 : ℝ),
        (t / (1 + t) ^ 2) * leftRightRelativeEntropyIntegrand t A B :=
    setIntegral_nonneg_of_ae_restrict hnonneg
  rw [hrepr]
  exact le_add_of_nonneg_right hintegral_nonneg

theorem traceMatrixRelativeEntropy_jointConvex
    {A₁ A₂ B₁ B₂ : Matrix n n ℝ}
    (hA₁ : A₁.PosDef) (hA₂ : A₂.PosDef)
    (hB₁ : B₁.PosDef) (hB₂ : B₂.PosDef)
    {a b : ℝ} (ha : 0 ≤ a) (hb : 0 ≤ b) (hab : a + b = 1) :
    let A := a • A₁ + b • A₂
    let B := a • B₁ + b • B₂
    traceMatrixRelativeEntropy A B ≤
      a * traceMatrixRelativeEntropy A₁ B₁ +
        b * traceMatrixRelativeEntropy A₂ B₂ := by
  intro A B
  have hA : A.PosDef :=
    positiveDefiniteSymmetricMatrices_convex hA₁ hA₂ ha hb hab
  have hB : B.PosDef :=
    positiveDefiniteSymmetricMatrices_convex hB₁ hB₂ ha hb hab
  let s : Set ℝ := Set.Ioi (0 : ℝ)
  let g : Matrix n n ℝ → Matrix n n ℝ → ℝ → ℝ :=
    fun A B t => (t / (1 + t) ^ 2) * leftRightRelativeEntropyIntegrand t A B
  have hrepr := traceMatrixRelativeEntropy_leftRight_integral_representation hA hB
  have hrepr₁ := traceMatrixRelativeEntropy_leftRight_integral_representation hA₁ hB₁
  have hrepr₂ := traceMatrixRelativeEntropy_leftRight_integral_representation hA₂ hB₂
  have hg_int : Integrable (fun t => g A B t) (volume.restrict s) := by
    simpa [IntegrableOn, s, g] using
      integrableOn_leftRightRelativeEntropyIntegrand_posDef_density hA hB
  have hg₁_int : Integrable (fun t => g A₁ B₁ t) (volume.restrict s) := by
    simpa [IntegrableOn, s, g] using
      integrableOn_leftRightRelativeEntropyIntegrand_posDef_density hA₁ hB₁
  have hg₂_int : Integrable (fun t => g A₂ B₂ t) (volume.restrict s) := by
    simpa [IntegrableOn, s, g] using
      integrableOn_leftRightRelativeEntropyIntegrand_posDef_density hA₂ hB₂
  have hrhs_int : Integrable
      (fun t => a * g A₁ B₁ t + b * g A₂ B₂ t) (volume.restrict s) :=
    (hg₁_int.const_mul a).add (hg₂_int.const_mul b)
  have hpoint :
      g A B ≤ᵐ[volume.restrict s] fun t => a * g A₁ B₁ t + b * g A₂ B₂ t := by
    filter_upwards [ae_restrict_mem measurableSet_Ioi] with t ht
    have hdensity : 0 ≤ t / (1 + t) ^ 2 :=
      div_nonneg (le_of_lt ht) (sq_nonneg (1 + t))
    have hconv :=
      leftRightRelativeEntropyIntegrand_jointConvex (n := n)
        hA₁ hA₂ hB₁ hB₂ (t := t) (le_of_lt ht) (a := a) (b := b) ha hb hab
    change
      (t / (1 + t) ^ 2) * leftRightRelativeEntropyIntegrand t A B ≤
        a * ((t / (1 + t) ^ 2) * leftRightRelativeEntropyIntegrand t A₁ B₁) +
          b * ((t / (1 + t) ^ 2) * leftRightRelativeEntropyIntegrand t A₂ B₂)
    calc
      (t / (1 + t) ^ 2) * leftRightRelativeEntropyIntegrand t A B ≤
          (t / (1 + t) ^ 2) *
            (a * leftRightRelativeEntropyIntegrand t A₁ B₁ +
              b * leftRightRelativeEntropyIntegrand t A₂ B₂) := by
        exact mul_le_mul_of_nonneg_left (by simpa [A, B] using hconv) hdensity
      _ = a * ((t / (1 + t) ^ 2) * leftRightRelativeEntropyIntegrand t A₁ B₁) +
          b * ((t / (1 + t) ^ 2) * leftRightRelativeEntropyIntegrand t A₂ B₂) := by
        ring
  have hint_le :
      (∫ t in s, g A B t) ≤
        ∫ t in s, a * g A₁ B₁ t + b * g A₂ B₂ t :=
    setIntegral_mono_ae_restrict hg_int hrhs_int hpoint
  have hrhs_integral :
      (∫ t in s, a * g A₁ B₁ t + b * g A₂ B₂ t) =
        a * (∫ t in s, g A₁ B₁ t) + b * (∫ t in s, g A₂ B₂ t) := by
    rw [integral_add (hg₁_int.const_mul a) (hg₂_int.const_mul b)]
    simp [integral_const_mul]
  have hint_le' :
      (∫ t in s, g A B t) ≤
        a * (∫ t in s, g A₁ B₁ t) + b * (∫ t in s, g A₂ B₂ t) := by
    rw [hrhs_integral] at hint_le
    exact hint_le
  have htrace_affine :
      trace (A - B) =
        a * trace (A₁ - B₁) + b * trace (A₂ - B₂) := by
    simp [A, B, sub_eq_add_neg, trace_add, trace_smul, mul_add]
    ring
  calc
    traceMatrixRelativeEntropy A B =
        trace (A - B) + ∫ t in s, g A B t := by
      simpa [s, g] using hrepr
    _ ≤ trace (A - B) +
        (a * (∫ t in s, g A₁ B₁ t) + b * (∫ t in s, g A₂ B₂ t)) := by
      exact add_le_add le_rfl hint_le'
    _ = a * (trace (A₁ - B₁) + ∫ t in s, g A₁ B₁ t) +
        b * (trace (A₂ - B₂) + ∫ t in s, g A₂ B₂ t) := by
      rw [htrace_affine]
      ring
    _ = a * traceMatrixRelativeEntropy A₁ B₁ +
        b * traceMatrixRelativeEntropy A₂ B₂ := by
      rw [show trace (A₁ - B₁) + ∫ t in s, g A₁ B₁ t =
          traceMatrixRelativeEntropy A₁ B₁ by simpa [s, g] using hrepr₁.symm]
      rw [show trace (A₂ - B₂) + ∫ t in s, g A₂ B₂ t =
          traceMatrixRelativeEntropy A₂ B₂ by simpa [s, g] using hrepr₂.symm]

/--
Relative entropy against `exp (H + log X)` expands to the entropy against `X` minus the
linear trace term in `H`.
-/
theorem traceMatrixRelativeEntropy_exp_log_eq
    {H X A : Matrix n n ℝ} (hH : H.IsHermitian) :
    traceMatrixRelativeEntropy A (exp (H + CFC.log X)) =
      traceMatrixRelativeEntropy A X - trace (A * H) := by
  have hlogX : (CFC.log X).IsHermitian :=
    (show IsSelfAdjoint (CFC.log X) from IsSelfAdjoint.log).isHermitian
  have hK : (H + CFC.log X).IsHermitian := hH.add hlogX
  unfold traceMatrixRelativeEntropy
  rw [CFC.log_exp (H + CFC.log X) (ha := (show IsSelfAdjoint (H + CFC.log X) from hK))]
  simp [sub_eq_add_neg, mul_add, trace_add, trace_neg]
  ring

/--
Klein's inequality implies the Gibbs variational upper bound needed in the Lieb proof.

For every positive-definite `A`,
`tr(AH) + tr(A) - D(A || X) ≤ tr exp(H + log X)`.
-/
theorem liebTraceFunction_gibbs_upper_of_klein
    (hklein : ∀ ⦃A B : Matrix n n ℝ⦄, A.PosDef → B.PosDef →
      trace (A - B) ≤ traceMatrixRelativeEntropy A B)
    {H X A : Matrix n n ℝ} (hH : H.IsHermitian) (hA : A.PosDef) :
    trace (A * H) + trace A - traceMatrixRelativeEntropy A X ≤
      liebTraceFunction H X := by
  let B : Matrix n n ℝ := exp (H + CFC.log X)
  have hlogX : (CFC.log X).IsHermitian :=
    (show IsSelfAdjoint (CFC.log X) from IsSelfAdjoint.log).isHermitian
  have hB : B.PosDef := by
    exact exp_posDef_of_isHermitian (hH.add hlogX)
  have hklein_B : trace (A - B) ≤ traceMatrixRelativeEntropy A B :=
    hklein hA hB
  have hD :
      traceMatrixRelativeEntropy A B =
        traceMatrixRelativeEntropy A X - trace (A * H) := by
    simpa [B] using
      traceMatrixRelativeEntropy_exp_log_eq (H := H) (X := X) (A := A) hH
  have htrace_sub : trace (A - B) = trace A - trace B := trace_sub A B
  have htraceB : trace B = liebTraceFunction H X := rfl
  rw [hD, htrace_sub, htraceB] at hklein_B
  linarith

theorem traceMatrixRelativeEntropy_self (A : Matrix n n ℝ) :
    traceMatrixRelativeEntropy A A = 0 := by
  simp [traceMatrixRelativeEntropy]

theorem traceMatrixRelativeEntropy_exp_log_right_eq_trace_mul
    {H X : Matrix n n ℝ} (hH : H.IsHermitian) :
    let A : Matrix n n ℝ := exp (H + CFC.log X)
    traceMatrixRelativeEntropy A X = trace (A * H) := by
  intro A
  have hD := traceMatrixRelativeEntropy_exp_log_eq (H := H) (X := X) (A := A) hH
  have hself : traceMatrixRelativeEntropy A A = 0 := traceMatrixRelativeEntropy_self A
  change traceMatrixRelativeEntropy A A =
      traceMatrixRelativeEntropy A X - trace (A * H) at hD
  rw [hself] at hD
  linarith

end RelativeEntropyIntegrand

/--
For Hermitian `Z`, the Lieb trace function at `exp Z` reduces to the trace exponential expression
appearing before Jensen in HDP Lemma 5.4.9.
-/
theorem liebTraceFunction_exp_eq (H : Matrix n n ℝ) {Z : Matrix n n ℝ}
    (hZ : Z.IsHermitian) :
    liebTraceFunction H (exp Z) = trace (exp (H + Z)) := by
  unfold liebTraceFunction
  rw [CFC.log_exp Z (ha := (show IsSelfAdjoint Z from hZ))]

/--
The exact proposition stated by HDP Theorem 5.4.8, Lieb inequality.

For a fixed real symmetric matrix `H`, the function
`X ↦ tr exp(H + log X)` is concave on the positive definite real symmetric matrices.
-/
def lieb_inequality_hdp (H : Matrix n n ℝ) : Prop :=
  H.IsHermitian →
    ConcaveOn ℝ (positiveDefiniteSymmetricMatrices n) (liebTraceFunction H)

/--
Relative-entropy joint convexity plus Klein's inequality imply deterministic Lieb concavity.

This is the Gibbs-variational route with all analytic inputs made explicit. The remaining shortest
path to the unconditional theorem is therefore to prove these two relative-entropy facts, both of
which follow from the left/right integral representation above.
-/
theorem lieb_inequality_hdp_of_relativeEntropy_jointConvex_and_klein
    (H : Matrix n n ℝ)
    (hjconv : ∀ ⦃A₁ A₂ B₁ B₂ : Matrix n n ℝ⦄,
      A₁.PosDef → A₂.PosDef → B₁.PosDef → B₂.PosDef →
      ∀ ⦃a b : ℝ⦄, 0 ≤ a → 0 ≤ b → a + b = 1 →
        let A := a • A₁ + b • A₂
        let B := a • B₁ + b • B₂
        traceMatrixRelativeEntropy A B ≤
          a * traceMatrixRelativeEntropy A₁ B₁ +
            b * traceMatrixRelativeEntropy A₂ B₂)
    (hklein : ∀ ⦃A B : Matrix n n ℝ⦄, A.PosDef → B.PosDef →
      trace (A - B) ≤ traceMatrixRelativeEntropy A B) :
    lieb_inequality_hdp H := by
  intro hH
  refine ⟨positiveDefiniteSymmetricMatrices_convex, ?_⟩
  intro X₁ hX₁ X₂ hX₂ a b ha hb hab
  let X : Matrix n n ℝ := a • X₁ + b • X₂
  let A₁ : Matrix n n ℝ := exp (H + CFC.log X₁)
  let A₂ : Matrix n n ℝ := exp (H + CFC.log X₂)
  let A : Matrix n n ℝ := a • A₁ + b • A₂
  have hlogX₁ : (CFC.log X₁).IsHermitian :=
    (show IsSelfAdjoint (CFC.log X₁) from IsSelfAdjoint.log).isHermitian
  have hlogX₂ : (CFC.log X₂).IsHermitian :=
    (show IsSelfAdjoint (CFC.log X₂) from IsSelfAdjoint.log).isHermitian
  have hA₁ : A₁.PosDef := by
    simpa [A₁] using exp_posDef_of_isHermitian (hH.add hlogX₁)
  have hA₂ : A₂.PosDef := by
    simpa [A₂] using exp_posDef_of_isHermitian (hH.add hlogX₂)
  have hA : A.PosDef := by
    exact positiveDefiniteSymmetricMatrices_convex hA₁ hA₂ ha hb hab
  have hX : X.PosDef := by
    exact positiveDefiniteSymmetricMatrices_convex hX₁ hX₂ ha hb hab
  have hupper :
      trace (A * H) + trace A - traceMatrixRelativeEntropy A X ≤
        liebTraceFunction H X :=
    liebTraceFunction_gibbs_upper_of_klein hklein hH hA
  have hconv :
      traceMatrixRelativeEntropy A X ≤
        a * traceMatrixRelativeEntropy A₁ X₁ +
          b * traceMatrixRelativeEntropy A₂ X₂ := by
    simpa [A, X] using
      (hjconv hA₁ hA₂ hX₁ hX₂ ha hb hab)
  have hD₁ : traceMatrixRelativeEntropy A₁ X₁ = trace (A₁ * H) := by
    simpa [A₁] using
      traceMatrixRelativeEntropy_exp_log_right_eq_trace_mul (H := H) (X := X₁) hH
  have hD₂ : traceMatrixRelativeEntropy A₂ X₂ = trace (A₂ * H) := by
    simpa [A₂] using
      traceMatrixRelativeEntropy_exp_log_right_eq_trace_mul (H := H) (X := X₂) hH
  have htraceH : trace (A * H) =
      a * trace (A₁ * H) + b * trace (A₂ * H) := by
    simp [A, add_mul, trace_add, trace_smul]
  have htraceA : trace A = a * trace A₁ + b * trace A₂ := by
    simp [A, trace_add, trace_smul]
  have hleft :
      a * liebTraceFunction H X₁ + b * liebTraceFunction H X₂ =
        a * trace A₁ + b * trace A₂ := by
    simp [liebTraceFunction, A₁, A₂]
  have hexpr :
      a * liebTraceFunction H X₁ + b * liebTraceFunction H X₂ ≤
        trace (A * H) + trace A - traceMatrixRelativeEntropy A X := by
    rw [hleft, htraceH, htraceA]
    nlinarith
  exact hexpr.trans hupper

theorem lieb_inequality_hdp_of_leftRight_density_integral_representation
    (H : Matrix n n ℝ) :
    lieb_inequality_hdp H :=
  lieb_inequality_hdp_of_relativeEntropy_jointConvex_and_klein H
    (fun {A₁ A₂ B₁ B₂} hA₁ hA₂ hB₁ hB₂ {a b} ha hb hab =>
      traceMatrixRelativeEntropy_jointConvex
        (A₁ := A₁) (A₂ := A₂) (B₁ := B₁) (B₂ := B₂)
        hA₁ hA₂ hB₁ hB₂ (a := a) (b := b) ha hb hab)
    (fun {A B} hA hB =>
      traceMatrixRelativeEntropy_klein (A := A) (B := B) hA hB)

/--
HDP Theorem 5.4.8, Lieb inequality.

For a fixed real symmetric matrix `H`, the function
`X ↦ tr exp(H + log X)` is concave on the positive-definite real symmetric matrices.
-/
theorem lieb_inequality_hdp_5_4_8
    (H : Matrix n n ℝ) (hH : H.IsHermitian) :
    ConcaveOn ℝ (positiveDefiniteSymmetricMatrices n) (liebTraceFunction H) :=
  (lieb_inequality_hdp_of_leftRight_density_integral_representation H) hH

end Matrix
