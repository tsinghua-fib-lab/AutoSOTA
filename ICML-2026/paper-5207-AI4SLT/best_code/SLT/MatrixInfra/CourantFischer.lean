/-
Copyright (c) 2026 Yuanhe Zhang. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Yuanhe Zhang, Jason D. Lee, Fanghui Liu
-/
import SLT.MatrixInfra.Basic

/-!
# Courant-Fischer Min-Max Infrastructure

This file contains the Courant-Fischer and singular-value min-max infrastructure used in HDP,
Section 4.1.

## Main definitions

* `LinearMap.rayleighQuotient`: Rayleigh quotient `Re <Tx, x> / ‖x‖²`.
* `LinearMap.courantFischerMaxMin` and `LinearMap.courantFischerMinMax`: HDP's two eigenvalue
  min-max quantities.
* `LinearMap.singularQuotient`: singular quotient `‖Tx‖ / ‖x‖`.
* `LinearMap.singularCourantFischerMaxMin` and `LinearMap.singularCourantFischerMinMax`: HDP's
  two singular-value min-max quantities.

## Main results

* `LinearMap.IsSymmetric.eigenvalues_eq_courantFischerMaxMin_succ` and
  `LinearMap.IsSymmetric.eigenvalues_eq_courantFischerMinMax_sub`: indexed Courant-Fischer
  min-max equalities for sorted eigenvalues, matching HDP Theorem 4.1.6 in zero-based form.
* `LinearMap.singularValues_eq_singularCourantFischerMaxMin_succ` and
  `LinearMap.singularValues_eq_singularCourantFischerMinMax_sub`: indexed singular-value min-max
  equalities, matching HDP Corollary 4.1.7 in zero-based form.
* `LinearMap.sq_singularQuotient_eq_rayleighQuotient_adjoint_comp_self`: the squared singular
  quotient is the Rayleigh quotient of the Gram operator `T†T`.
* `Matrix.singularQuotient_rightSingularVectorBasis`: right singular vectors attain the
  corresponding singular value in the singular quotient.
-/

open Module

noncomputable section

namespace Submodule

variable {𝕜 E : Type*} [RCLike 𝕜] [NormedAddCommGroup E] [InnerProductSpace 𝕜 E]
variable [FiniteDimensional 𝕜 E]

/--
If two finite-dimensional subspaces have dimensions whose sum exceeds the ambient dimension, then
their intersection contains a nonzero vector. This is the dimension-counting step in the
Courant-Fischer min-max proof.
-/
theorem exists_ne_zero_mem_inf_of_finrank_lt_add_finrank (U V : Submodule 𝕜 E)
    (h : finrank 𝕜 E < finrank 𝕜 U + finrank 𝕜 V) :
    ∃ x : E, x ∈ U ∧ x ∈ V ∧ x ≠ 0 := by
  have hne : U ⊓ V ≠ ⊥ := by
    intro hbot
    have hle : finrank 𝕜 U + finrank 𝕜 V ≤ finrank 𝕜 E := by
      calc
        finrank 𝕜 U + finrank 𝕜 V =
            finrank 𝕜 (U ⊔ V : Submodule 𝕜 E) + finrank 𝕜 (U ⊓ V : Submodule 𝕜 E) := by
          rw [Submodule.finrank_sup_add_finrank_inf_eq]
        _ = finrank 𝕜 (U ⊔ V : Submodule 𝕜 E) := by
          rw [hbot, finrank_bot, add_zero]
        _ ≤ finrank 𝕜 E :=
          Submodule.finrank_le _
    exact (not_lt_of_ge hle) h
  obtain ⟨x, hx, hx0⟩ := (Submodule.ne_bot_iff (U ⊓ V)).mp hne
  exact ⟨x, hx.1, hx.2, hx0⟩

/-- A one-dimensional subspace is the span of any nonzero vector it contains. -/
theorem exists_span_singleton_eq_of_finrank_eq_one (U : Submodule 𝕜 E)
    (hU : finrank 𝕜 U = 1) :
    ∃ x : E, x ∈ U ∧ x ≠ 0 ∧ 𝕜 ∙ x = U := by
  have hU_ne : U ≠ ⊥ := by
    rw [← Submodule.one_le_finrank_iff, hU]
  obtain ⟨x, hxU, hx0⟩ := (Submodule.ne_bot_iff U).mp hU_ne
  have hspan_le : 𝕜 ∙ x ≤ U := by
    rw [Submodule.span_singleton_le_iff_mem]
    exact hxU
  have hspan_eq : 𝕜 ∙ x = U :=
    Submodule.eq_of_le_of_finrank_eq hspan_le (by rw [finrank_span_singleton hx0, hU])
  exact ⟨x, hxU, hx0, hspan_eq⟩

end Submodule


/-! ## Spectral and Rayleigh endpoint principles for self-adjoint operators -/

namespace LinearMap

/-! ### Courant-Fischer quantities -/

variable {𝕜 E F : Type*} [RCLike 𝕜]
variable [NormedAddCommGroup E] [InnerProductSpace 𝕜 E]
variable [NormedAddCommGroup F] [InnerProductSpace 𝕜 F]

/-- The Rayleigh quotient `Re <Tx, x> / ‖x‖²` of a linear endomorphism. -/
noncomputable def rayleighQuotient (T : E →ₗ[𝕜] E) (x : E) : ℝ :=
  RCLike.re (inner 𝕜 (T x) x) / ‖x‖ ^ 2

/-- The linear-map Rayleigh quotient agrees with Mathlib's continuous-linear-map version. -/
theorem rayleighQuotient_toContinuousLinearMap [FiniteDimensional 𝕜 E]
    (T : E →ₗ[𝕜] E) (x : E) :
    LinearMap.rayleighQuotient T x = T.toContinuousLinearMap.rayleighQuotient x := by
  rfl

/-- The Rayleigh quotient is unchanged under nonzero scalar multiplication. -/
theorem rayleighQuotient_smul [FiniteDimensional 𝕜 E]
    (T : E →ₗ[𝕜] E) (x : E) {c : 𝕜} (hc : c ≠ 0) :
    rayleighQuotient T (c • x) = rayleighQuotient T x := by
  rw [T.rayleighQuotient_toContinuousLinearMap (c • x),
    T.rayleighQuotient_toContinuousLinearMap x]
  exact T.toContinuousLinearMap.rayleigh_smul x hc

/-- The Rayleigh quotients of a finite-dimensional linear endomorphism are bounded above. -/
theorem bddAbove_range_rayleighQuotient [FiniteDimensional 𝕜 E]
    (T : E →ₗ[𝕜] E) :
    BddAbove (Set.range fun x : { x : E // x ≠ 0 } => LinearMap.rayleighQuotient T x) := by
  refine ⟨‖T.toContinuousLinearMap‖, ?_⟩
  rintro _ ⟨x, rfl⟩
  change LinearMap.rayleighQuotient T (x : E) ≤ ‖T.toContinuousLinearMap‖
  rw [T.rayleighQuotient_toContinuousLinearMap (x : E)]
  exact le_trans (le_abs_self _)
    (T.toContinuousLinearMap.rayleighQuotient_le_norm (x : E))

/-- The Rayleigh quotients of a finite-dimensional linear endomorphism are bounded below. -/
theorem bddBelow_range_rayleighQuotient [FiniteDimensional 𝕜 E]
    (T : E →ₗ[𝕜] E) :
    BddBelow (Set.range fun x : { x : E // x ≠ 0 } => LinearMap.rayleighQuotient T x) := by
  refine ⟨-‖T.toContinuousLinearMap‖, ?_⟩
  rintro _ ⟨x, rfl⟩
  change -‖T.toContinuousLinearMap‖ ≤ LinearMap.rayleighQuotient T (x : E)
  rw [T.rayleighQuotient_toContinuousLinearMap (x : E)]
  have h := T.toContinuousLinearMap.rayleighQuotient_le_norm (x : E)
  exact neg_le.mp (le_trans (neg_le_abs _) h)

/-- The minimum Rayleigh quotient over the nonzero vectors of a subspace. -/
noncomputable def minRayleighQuotientOn (T : E →ₗ[𝕜] E) (U : Submodule 𝕜 E) : ℝ :=
  ⨅ x : { x : U // (x : E) ≠ 0 }, rayleighQuotient T x

/-- The maximum Rayleigh quotient over the nonzero vectors of a subspace. -/
noncomputable def maxRayleighQuotientOn (T : E →ₗ[𝕜] E) (U : Submodule 𝕜 E) : ℝ :=
  ⨆ x : { x : U // (x : E) ≠ 0 }, rayleighQuotient T x

/-- Rayleigh quotients restricted to a subspace are bounded above. -/
theorem bddAbove_range_rayleighQuotientOn [FiniteDimensional 𝕜 E]
    (T : E →ₗ[𝕜] E) (U : Submodule 𝕜 E) :
    BddAbove (Set.range fun x : { x : U // (x : E) ≠ 0 } => rayleighQuotient T x) := by
  obtain ⟨C, hC⟩ := bddAbove_range_rayleighQuotient T
  refine ⟨C, ?_⟩
  rintro _ ⟨x, rfl⟩
  exact hC ⟨⟨(x : E), x.property⟩, rfl⟩

/-- Rayleigh quotients restricted to a subspace are bounded below. -/
theorem bddBelow_range_rayleighQuotientOn [FiniteDimensional 𝕜 E]
    (T : E →ₗ[𝕜] E) (U : Submodule 𝕜 E) :
    BddBelow (Set.range fun x : { x : U // (x : E) ≠ 0 } => rayleighQuotient T x) := by
  obtain ⟨C, hC⟩ := bddBelow_range_rayleighQuotient T
  refine ⟨C, ?_⟩
  rintro _ ⟨x, rfl⟩
  exact hC ⟨⟨(x : E), x.property⟩, rfl⟩

/-- The restricted Rayleigh infimum is bounded above by every nonzero vector in the subspace. -/
theorem minRayleighQuotientOn_le_of_mem [FiniteDimensional 𝕜 E]
    (T : E →ₗ[𝕜] E) {U : Submodule 𝕜 E} {x : E} (hxU : x ∈ U) (hx0 : x ≠ 0) :
    minRayleighQuotientOn T U ≤ rayleighQuotient T x := by
  exact ciInf_le (bddBelow_range_rayleighQuotientOn T U) ⟨⟨x, hxU⟩, hx0⟩

/-- The restricted Rayleigh supremum is bounded below by every nonzero vector in the subspace. -/
theorem le_maxRayleighQuotientOn_of_mem [FiniteDimensional 𝕜 E]
    (T : E →ₗ[𝕜] E) {U : Submodule 𝕜 E} {x : E} (hxU : x ∈ U) (hx0 : x ≠ 0) :
    rayleighQuotient T x ≤ maxRayleighQuotientOn T U := by
  exact le_ciSup (bddAbove_range_rayleighQuotientOn T U) ⟨⟨x, hxU⟩, hx0⟩

/-- On a nonzero line, every nonzero vector has the same Rayleigh quotient as the generator. -/
theorem rayleighQuotient_eq_of_mem_span_singleton [FiniteDimensional 𝕜 E]
    (T : E →ₗ[𝕜] E) {x y : E} (hy : y ∈ 𝕜 ∙ x) (hy0 : y ≠ 0) :
    rayleighQuotient T y = rayleighQuotient T x := by
  obtain ⟨c, rfl⟩ := Submodule.mem_span_singleton.mp hy
  have hc : c ≠ 0 := by
    intro hc
    exact hy0 (by simp [hc])
  exact rayleighQuotient_smul T x hc

/-- The minimum Rayleigh quotient over a nonzero line is the generator's Rayleigh quotient. -/
theorem minRayleighQuotientOn_span_singleton [FiniteDimensional 𝕜 E]
    (T : E →ₗ[𝕜] E) {x : E} (hx : x ≠ 0) :
    minRayleighQuotientOn T (𝕜 ∙ x) = rayleighQuotient T x := by
  haveI : Nonempty { y : (𝕜 ∙ x : Submodule 𝕜 E) // (y : E) ≠ 0 } :=
    ⟨⟨⟨x, Submodule.mem_span_singleton_self x⟩, hx⟩⟩
  unfold minRayleighQuotientOn
  apply le_antisymm
  · exact ciInf_le (bddBelow_range_rayleighQuotientOn T (𝕜 ∙ x))
      ⟨⟨x, Submodule.mem_span_singleton_self x⟩, hx⟩
  · refine le_ciInf fun y => ?_
    exact (rayleighQuotient_eq_of_mem_span_singleton T y.val.property y.property).ge

/-- The maximum Rayleigh quotient over a nonzero line is the generator's Rayleigh quotient. -/
theorem maxRayleighQuotientOn_span_singleton [FiniteDimensional 𝕜 E]
    (T : E →ₗ[𝕜] E) {x : E} (hx : x ≠ 0) :
    maxRayleighQuotientOn T (𝕜 ∙ x) = rayleighQuotient T x := by
  haveI : Nonempty { y : (𝕜 ∙ x : Submodule 𝕜 E) // (y : E) ≠ 0 } :=
    ⟨⟨⟨x, Submodule.mem_span_singleton_self x⟩, hx⟩⟩
  unfold maxRayleighQuotientOn
  apply le_antisymm
  · refine ciSup_le fun y => ?_
    exact (rayleighQuotient_eq_of_mem_span_singleton T y.val.property y.property).le
  · exact le_ciSup (bddAbove_range_rayleighQuotientOn T (𝕜 ∙ x))
      ⟨⟨x, Submodule.mem_span_singleton_self x⟩, hx⟩

/-- On the full subspace, the subspace infimum is the global Rayleigh infimum. -/
theorem minRayleighQuotientOn_top [FiniteDimensional 𝕜 E] [Nontrivial E]
    (T : E →ₗ[𝕜] E) :
    minRayleighQuotientOn T (⊤ : Submodule 𝕜 E) =
      ⨅ x : { x : E // x ≠ 0 }, rayleighQuotient T x := by
  haveI : Nonempty { x : E // x ≠ 0 } := by
    obtain ⟨x, hx⟩ := exists_ne (0 : E)
    exact ⟨⟨x, hx⟩⟩
  haveI : Nonempty { x : (⊤ : Submodule 𝕜 E) // (x : E) ≠ 0 } := by
    obtain ⟨x, hx⟩ := exists_ne (0 : E)
    exact ⟨⟨⟨x, trivial⟩, hx⟩⟩
  unfold minRayleighQuotientOn
  apply le_antisymm
  · refine le_ciInf fun x => ?_
    exact ciInf_le (bddBelow_range_rayleighQuotientOn T (⊤ : Submodule 𝕜 E))
      ⟨⟨(x : E), trivial⟩, x.property⟩
  · refine le_ciInf fun x => ?_
    exact ciInf_le (bddBelow_range_rayleighQuotient T) ⟨(x : E), x.property⟩

/-- On the full subspace, the subspace supremum is the global Rayleigh supremum. -/
theorem maxRayleighQuotientOn_top [FiniteDimensional 𝕜 E] [Nontrivial E]
    (T : E →ₗ[𝕜] E) :
    maxRayleighQuotientOn T (⊤ : Submodule 𝕜 E) =
      ⨆ x : { x : E // x ≠ 0 }, rayleighQuotient T x := by
  haveI : Nonempty { x : E // x ≠ 0 } := by
    obtain ⟨x, hx⟩ := exists_ne (0 : E)
    exact ⟨⟨x, hx⟩⟩
  haveI : Nonempty { x : (⊤ : Submodule 𝕜 E) // (x : E) ≠ 0 } := by
    obtain ⟨x, hx⟩ := exists_ne (0 : E)
    exact ⟨⟨⟨x, trivial⟩, hx⟩⟩
  unfold maxRayleighQuotientOn
  apply le_antisymm
  · refine ciSup_le fun x => ?_
    exact le_ciSup (bddAbove_range_rayleighQuotient T) ⟨(x : E), x.property⟩
  · refine ciSup_le fun x => ?_
    exact le_ciSup (bddAbove_range_rayleighQuotientOn T (⊤ : Submodule 𝕜 E))
      ⟨⟨(x : E), trivial⟩, x.property⟩

/--
The `max_dim min` side of HDP Theorem 4.1.6, with the subspace dimension supplied explicitly.
For HDP's `λ_k`, use dimension `k` in the 1-based paper convention.
-/
noncomputable def courantFischerMaxMin (T : E →ₗ[𝕜] E) (k : ℕ) : ℝ :=
  ⨆ U : { U : Submodule 𝕜 E // finrank 𝕜 U = k }, minRayleighQuotientOn T U

/--
The `min_dim max` side of HDP Theorem 4.1.6, with the subspace dimension supplied explicitly.
For HDP's `λ_k` in an `n`-dimensional space, use dimension `n - k + 1` in the 1-based paper
convention.
-/
noncomputable def courantFischerMinMax (T : E →ₗ[𝕜] E) (k : ℕ) : ℝ :=
  ⨅ U : { U : Submodule 𝕜 E // finrank 𝕜 U = k }, maxRayleighQuotientOn T U

/--
Full-dimensional endpoint of the `max_dim min` Courant-Fischer quantity: maximizing the minimum
over all full-dimensional subspaces is the global Rayleigh infimum.
-/
theorem courantFischerMaxMin_finrank [FiniteDimensional 𝕜 E] [Nontrivial E]
    (T : E →ₗ[𝕜] E) :
    courantFischerMaxMin T (finrank 𝕜 E) =
      ⨅ x : { x : E // x ≠ 0 }, rayleighQuotient T x := by
  haveI : Nonempty { U : Submodule 𝕜 E // finrank 𝕜 U = finrank 𝕜 E } :=
    ⟨⟨⊤, by rw [finrank_top]⟩⟩
  unfold courantFischerMaxMin
  apply le_antisymm
  · refine ciSup_le fun U => ?_
    change minRayleighQuotientOn T (U : Submodule 𝕜 E) ≤
      ⨅ x : { x : E // x ≠ 0 }, rayleighQuotient T x
    have hU : (U : Submodule 𝕜 E) = ⊤ :=
      Submodule.eq_top_of_finrank_eq U.property
    rw [hU, minRayleighQuotientOn_top]
  · have hbd :
        BddAbove (Set.range fun U : { U : Submodule 𝕜 E // finrank 𝕜 U = finrank 𝕜 E } =>
          minRayleighQuotientOn T U) := by
      refine ⟨⨅ x : { x : E // x ≠ 0 }, rayleighQuotient T x, ?_⟩
      rintro _ ⟨U, rfl⟩
      change minRayleighQuotientOn T (U : Submodule 𝕜 E) ≤
        ⨅ x : { x : E // x ≠ 0 }, rayleighQuotient T x
      have hU : (U : Submodule 𝕜 E) = ⊤ :=
        Submodule.eq_top_of_finrank_eq U.property
      rw [hU, minRayleighQuotientOn_top]
    simpa [minRayleighQuotientOn_top] using
      (le_ciSup hbd ⟨⊤, by rw [finrank_top]⟩)

/--
Full-dimensional endpoint of the `min_dim max` Courant-Fischer quantity: minimizing the maximum
over all full-dimensional subspaces is the global Rayleigh supremum.
-/
theorem courantFischerMinMax_finrank [FiniteDimensional 𝕜 E] [Nontrivial E]
    (T : E →ₗ[𝕜] E) :
    courantFischerMinMax T (finrank 𝕜 E) =
      ⨆ x : { x : E // x ≠ 0 }, rayleighQuotient T x := by
  haveI : Nonempty { U : Submodule 𝕜 E // finrank 𝕜 U = finrank 𝕜 E } :=
    ⟨⟨⊤, by rw [finrank_top]⟩⟩
  unfold courantFischerMinMax
  apply le_antisymm
  · have hbd :
        BddBelow (Set.range fun U : { U : Submodule 𝕜 E // finrank 𝕜 U = finrank 𝕜 E } =>
          maxRayleighQuotientOn T U) := by
      refine ⟨⨆ x : { x : E // x ≠ 0 }, rayleighQuotient T x, ?_⟩
      rintro _ ⟨U, rfl⟩
      change (⨆ x : { x : E // x ≠ 0 }, rayleighQuotient T x) ≤
        maxRayleighQuotientOn T (U : Submodule 𝕜 E)
      have hU : (U : Submodule 𝕜 E) = ⊤ :=
        Submodule.eq_top_of_finrank_eq U.property
      rw [hU, maxRayleighQuotientOn_top]
    simpa [maxRayleighQuotientOn_top] using
      (ciInf_le hbd ⟨⊤, by rw [finrank_top]⟩)
  · refine le_ciInf fun U => ?_
    change (⨆ x : { x : E // x ≠ 0 }, rayleighQuotient T x) ≤
      maxRayleighQuotientOn T (U : Submodule 𝕜 E)
    have hU : (U : Submodule 𝕜 E) = ⊤ :=
      Submodule.eq_top_of_finrank_eq U.property
    rw [hU, maxRayleighQuotientOn_top]

/--
One-dimensional endpoint of the `max_dim min` Courant-Fischer quantity: maximizing over lines is
the global Rayleigh supremum.
-/
theorem courantFischerMaxMin_one [FiniteDimensional 𝕜 E] [Nontrivial E]
    (T : E →ₗ[𝕜] E) :
    courantFischerMaxMin T 1 =
      ⨆ x : { x : E // x ≠ 0 }, rayleighQuotient T x := by
  haveI : Nonempty { U : Submodule 𝕜 E // finrank 𝕜 U = 1 } := by
    obtain ⟨x, hx⟩ := exists_ne (0 : E)
    exact ⟨⟨𝕜 ∙ x, finrank_span_singleton hx⟩⟩
  haveI : Nonempty { x : E // x ≠ 0 } := by
    obtain ⟨x, hx⟩ := exists_ne (0 : E)
    exact ⟨⟨x, hx⟩⟩
  unfold courantFischerMaxMin
  apply le_antisymm
  · refine ciSup_le fun U => ?_
    obtain ⟨x, hxU, hx0, hspan_eq⟩ :=
      Submodule.exists_span_singleton_eq_of_finrank_eq_one (𝕜 := 𝕜) (E := E)
        (U : Submodule 𝕜 E) U.property
    rw [← hspan_eq, minRayleighQuotientOn_span_singleton T hx0]
    exact le_ciSup (bddAbove_range_rayleighQuotient T) ⟨x, hx0⟩
  · have hbd :
        BddAbove (Set.range fun U : { U : Submodule 𝕜 E // finrank 𝕜 U = 1 } =>
          minRayleighQuotientOn T U) := by
      refine ⟨⨆ x : { x : E // x ≠ 0 }, rayleighQuotient T x, ?_⟩
      rintro _ ⟨U, rfl⟩
      obtain ⟨x, hxU, hx0, hspan_eq⟩ :=
        Submodule.exists_span_singleton_eq_of_finrank_eq_one (𝕜 := 𝕜) (E := E) U U.property
      change minRayleighQuotientOn T (U : Submodule 𝕜 E) ≤
        ⨆ x : { x : E // x ≠ 0 }, rayleighQuotient T x
      rw [← hspan_eq, minRayleighQuotientOn_span_singleton T hx0]
      exact le_ciSup (bddAbove_range_rayleighQuotient T) ⟨x, hx0⟩
    refine ciSup_le fun x => ?_
    calc
      rayleighQuotient T (x : E) =
          minRayleighQuotientOn T (𝕜 ∙ (x : E)) := by
        rw [minRayleighQuotientOn_span_singleton T x.property]
      _ ≤ ⨆ U : { U : Submodule 𝕜 E // finrank 𝕜 U = 1 }, minRayleighQuotientOn T U := by
        exact le_ciSup hbd ⟨𝕜 ∙ (x : E), finrank_span_singleton x.property⟩

/--
One-dimensional endpoint of the `min_dim max` Courant-Fischer quantity: minimizing over lines is
the global Rayleigh infimum.
-/
theorem courantFischerMinMax_one [FiniteDimensional 𝕜 E] [Nontrivial E]
    (T : E →ₗ[𝕜] E) :
    courantFischerMinMax T 1 =
      ⨅ x : { x : E // x ≠ 0 }, rayleighQuotient T x := by
  haveI : Nonempty { U : Submodule 𝕜 E // finrank 𝕜 U = 1 } := by
    obtain ⟨x, hx⟩ := exists_ne (0 : E)
    exact ⟨⟨𝕜 ∙ x, finrank_span_singleton hx⟩⟩
  haveI : Nonempty { x : E // x ≠ 0 } := by
    obtain ⟨x, hx⟩ := exists_ne (0 : E)
    exact ⟨⟨x, hx⟩⟩
  unfold courantFischerMinMax
  have hbd :
      BddBelow (Set.range fun U : { U : Submodule 𝕜 E // finrank 𝕜 U = 1 } =>
        maxRayleighQuotientOn T U) := by
    refine ⟨⨅ x : { x : E // x ≠ 0 }, rayleighQuotient T x, ?_⟩
    rintro _ ⟨U, rfl⟩
    obtain ⟨x, hxU, hx0, hspan_eq⟩ :=
      Submodule.exists_span_singleton_eq_of_finrank_eq_one (𝕜 := 𝕜) (E := E) U U.property
    change (⨅ x : { x : E // x ≠ 0 }, rayleighQuotient T x) ≤
      maxRayleighQuotientOn T (U : Submodule 𝕜 E)
    rw [← hspan_eq, maxRayleighQuotientOn_span_singleton T hx0]
    exact ciInf_le (bddBelow_range_rayleighQuotient T) ⟨x, hx0⟩
  apply le_antisymm
  · refine le_ciInf fun x => ?_
    calc
      (⨅ U : { U : Submodule 𝕜 E // finrank 𝕜 U = 1 }, maxRayleighQuotientOn T U) ≤
          maxRayleighQuotientOn T (𝕜 ∙ (x : E)) := by
        exact ciInf_le hbd ⟨𝕜 ∙ (x : E), finrank_span_singleton x.property⟩
      _ = rayleighQuotient T (x : E) := by
        rw [maxRayleighQuotientOn_span_singleton T x.property]
  · refine le_ciInf fun U => ?_
    obtain ⟨x, hxU, hx0, hspan_eq⟩ :=
      Submodule.exists_span_singleton_eq_of_finrank_eq_one (𝕜 := 𝕜) (E := E)
        (U : Submodule 𝕜 E) U.property
    rw [← hspan_eq, maxRayleighQuotientOn_span_singleton T hx0]
    exact ciInf_le (bddBelow_range_rayleighQuotient T) ⟨x, hx0⟩

/-- The singular-value quotient `‖Tx‖ / ‖x‖`. -/
noncomputable def singularQuotient (T : E →ₗ[𝕜] F) (x : E) : ℝ :=
  ‖T x‖ / ‖x‖

/--
The squared singular quotient is the Rayleigh quotient of the positive Gram operator `T†T`.
This is the algebraic bridge from HDP Corollary 4.1.7 to Theorem 4.1.6.
-/
theorem sq_singularQuotient_eq_rayleighQuotient_adjoint_comp_self
    [FiniteDimensional 𝕜 E] [FiniteDimensional 𝕜 F]
    (T : E →ₗ[𝕜] F) (x : E) :
    singularQuotient T x ^ 2 =
      rayleighQuotient (LinearMap.adjoint T ∘ₗ T) x := by
  unfold singularQuotient rayleighQuotient
  rw [div_pow, LinearMap.comp_apply, LinearMap.adjoint_inner_left, inner_self_eq_norm_sq]

/-- The singular quotient is unchanged under nonzero scalar multiplication. -/
theorem singularQuotient_smul [FiniteDimensional 𝕜 E]
    (T : E →ₗ[𝕜] F) (x : E) {c : 𝕜} (hc : c ≠ 0) :
    singularQuotient T (c • x) = singularQuotient T x := by
  unfold singularQuotient
  rw [map_smul, norm_smul, norm_smul, mul_div_mul_left _ _ (norm_ne_zero_iff.mpr hc)]

/-- The minimum singular quotient over the nonzero vectors of a subspace. -/
noncomputable def minSingularQuotientOn (T : E →ₗ[𝕜] F) (U : Submodule 𝕜 E) : ℝ :=
  ⨅ x : { x : U // (x : E) ≠ 0 }, singularQuotient T x

/-- The maximum singular quotient over the nonzero vectors of a subspace. -/
noncomputable def maxSingularQuotientOn (T : E →ₗ[𝕜] F) (U : Submodule 𝕜 E) : ℝ :=
  ⨆ x : { x : U // (x : E) ≠ 0 }, singularQuotient T x

/-- Singular quotients of a finite-dimensional linear map are bounded above. -/
theorem bddAbove_range_singularQuotient [FiniteDimensional 𝕜 E]
    (T : E →ₗ[𝕜] F) :
    BddAbove (Set.range fun x : { x : E // x ≠ 0 } => singularQuotient T x) := by
  refine ⟨‖T.toContinuousLinearMap‖, ?_⟩
  rintro _ ⟨x, rfl⟩
  unfold singularQuotient
  have hx : 0 < ‖(x : E)‖ := norm_pos_iff.mpr x.property
  exact (div_le_iff₀ hx).mpr (T.toContinuousLinearMap.le_opNorm (x : E))

/-- Singular quotients of a finite-dimensional linear map are bounded below. -/
theorem bddBelow_range_singularQuotient [FiniteDimensional 𝕜 E]
    (T : E →ₗ[𝕜] F) :
    BddBelow (Set.range fun x : { x : E // x ≠ 0 } => singularQuotient T x) := by
  refine ⟨0, ?_⟩
  rintro _ ⟨x, rfl⟩
  unfold singularQuotient
  positivity

/-- Singular quotients restricted to a subspace are bounded above. -/
theorem bddAbove_range_singularQuotientOn [FiniteDimensional 𝕜 E]
    (T : E →ₗ[𝕜] F) (U : Submodule 𝕜 E) :
    BddAbove (Set.range fun x : { x : U // (x : E) ≠ 0 } => singularQuotient T x) := by
  obtain ⟨C, hC⟩ := bddAbove_range_singularQuotient T
  refine ⟨C, ?_⟩
  rintro _ ⟨x, rfl⟩
  exact hC ⟨⟨(x : E), x.property⟩, rfl⟩

/-- Singular quotients restricted to a subspace are bounded below. -/
theorem bddBelow_range_singularQuotientOn [FiniteDimensional 𝕜 E]
    (T : E →ₗ[𝕜] F) (U : Submodule 𝕜 E) :
    BddBelow (Set.range fun x : { x : U // (x : E) ≠ 0 } => singularQuotient T x) := by
  obtain ⟨C, hC⟩ := bddBelow_range_singularQuotient T
  refine ⟨C, ?_⟩
  rintro _ ⟨x, rfl⟩
  exact hC ⟨⟨(x : E), x.property⟩, rfl⟩

/-- The restricted singular-quotient infimum is bounded above by every nonzero vector in the subspace. -/
theorem minSingularQuotientOn_le_of_mem [FiniteDimensional 𝕜 E]
    (T : E →ₗ[𝕜] F) {U : Submodule 𝕜 E} {x : E} (hxU : x ∈ U) (hx0 : x ≠ 0) :
    minSingularQuotientOn T U ≤ singularQuotient T x := by
  exact ciInf_le (bddBelow_range_singularQuotientOn T U) ⟨⟨x, hxU⟩, hx0⟩

/-- The restricted singular-quotient supremum is bounded below by every nonzero vector in the subspace. -/
theorem le_maxSingularQuotientOn_of_mem [FiniteDimensional 𝕜 E]
    (T : E →ₗ[𝕜] F) {U : Submodule 𝕜 E} {x : E} (hxU : x ∈ U) (hx0 : x ≠ 0) :
    singularQuotient T x ≤ maxSingularQuotientOn T U := by
  exact le_ciSup (bddAbove_range_singularQuotientOn T U) ⟨⟨x, hxU⟩, hx0⟩

/-- On a nonzero line, every nonzero vector has the same singular quotient as the generator. -/
theorem singularQuotient_eq_of_mem_span_singleton [FiniteDimensional 𝕜 E]
    (T : E →ₗ[𝕜] F) {x y : E} (hy : y ∈ 𝕜 ∙ x) (hy0 : y ≠ 0) :
    singularQuotient T y = singularQuotient T x := by
  obtain ⟨c, rfl⟩ := Submodule.mem_span_singleton.mp hy
  have hc : c ≠ 0 := by
    intro hc
    exact hy0 (by simp [hc])
  exact singularQuotient_smul T x hc

/-- The minimum singular quotient over a nonzero line is the generator's singular quotient. -/
theorem minSingularQuotientOn_span_singleton [FiniteDimensional 𝕜 E]
    (T : E →ₗ[𝕜] F) {x : E} (hx : x ≠ 0) :
    minSingularQuotientOn T (𝕜 ∙ x) = singularQuotient T x := by
  haveI : Nonempty { y : (𝕜 ∙ x : Submodule 𝕜 E) // (y : E) ≠ 0 } :=
    ⟨⟨⟨x, Submodule.mem_span_singleton_self x⟩, hx⟩⟩
  unfold minSingularQuotientOn
  apply le_antisymm
  · exact ciInf_le (bddBelow_range_singularQuotientOn T (𝕜 ∙ x))
      ⟨⟨x, Submodule.mem_span_singleton_self x⟩, hx⟩
  · refine le_ciInf fun y => ?_
    exact (singularQuotient_eq_of_mem_span_singleton T y.val.property y.property).ge

/-- The maximum singular quotient over a nonzero line is the generator's singular quotient. -/
theorem maxSingularQuotientOn_span_singleton [FiniteDimensional 𝕜 E]
    (T : E →ₗ[𝕜] F) {x : E} (hx : x ≠ 0) :
    maxSingularQuotientOn T (𝕜 ∙ x) = singularQuotient T x := by
  haveI : Nonempty { y : (𝕜 ∙ x : Submodule 𝕜 E) // (y : E) ≠ 0 } :=
    ⟨⟨⟨x, Submodule.mem_span_singleton_self x⟩, hx⟩⟩
  unfold maxSingularQuotientOn
  apply le_antisymm
  · refine ciSup_le fun y => ?_
    exact (singularQuotient_eq_of_mem_span_singleton T y.val.property y.property).le
  · exact le_ciSup (bddAbove_range_singularQuotientOn T (𝕜 ∙ x))
      ⟨⟨x, Submodule.mem_span_singleton_self x⟩, hx⟩

/-- On the full subspace, the subspace infimum is the global singular-quotient infimum. -/
theorem minSingularQuotientOn_top [FiniteDimensional 𝕜 E] [Nontrivial E]
    (T : E →ₗ[𝕜] F) :
    minSingularQuotientOn T (⊤ : Submodule 𝕜 E) =
      ⨅ x : { x : E // x ≠ 0 }, singularQuotient T x := by
  haveI : Nonempty { x : E // x ≠ 0 } := by
    obtain ⟨x, hx⟩ := exists_ne (0 : E)
    exact ⟨⟨x, hx⟩⟩
  haveI : Nonempty { x : (⊤ : Submodule 𝕜 E) // (x : E) ≠ 0 } := by
    obtain ⟨x, hx⟩ := exists_ne (0 : E)
    exact ⟨⟨⟨x, trivial⟩, hx⟩⟩
  unfold minSingularQuotientOn
  apply le_antisymm
  · refine le_ciInf fun x => ?_
    exact ciInf_le (bddBelow_range_singularQuotientOn T (⊤ : Submodule 𝕜 E))
      ⟨⟨(x : E), trivial⟩, x.property⟩
  · refine le_ciInf fun x => ?_
    exact ciInf_le (bddBelow_range_singularQuotient T) ⟨(x : E), x.property⟩

/-- On the full subspace, the subspace supremum is the global singular-quotient supremum. -/
theorem maxSingularQuotientOn_top [FiniteDimensional 𝕜 E] [Nontrivial E]
    (T : E →ₗ[𝕜] F) :
    maxSingularQuotientOn T (⊤ : Submodule 𝕜 E) =
      ⨆ x : { x : E // x ≠ 0 }, singularQuotient T x := by
  haveI : Nonempty { x : E // x ≠ 0 } := by
    obtain ⟨x, hx⟩ := exists_ne (0 : E)
    exact ⟨⟨x, hx⟩⟩
  haveI : Nonempty { x : (⊤ : Submodule 𝕜 E) // (x : E) ≠ 0 } := by
    obtain ⟨x, hx⟩ := exists_ne (0 : E)
    exact ⟨⟨⟨x, trivial⟩, hx⟩⟩
  unfold maxSingularQuotientOn
  apply le_antisymm
  · refine ciSup_le fun x => ?_
    exact le_ciSup (bddAbove_range_singularQuotient T) ⟨(x : E), x.property⟩
  · refine ciSup_le fun x => ?_
    exact le_ciSup (bddAbove_range_singularQuotientOn T (⊤ : Submodule 𝕜 E))
      ⟨⟨(x : E), trivial⟩, x.property⟩

/--
The `max_dim min` side of HDP Corollary 4.1.7 for singular values, with the subspace dimension
supplied explicitly.
-/
noncomputable def singularCourantFischerMaxMin (T : E →ₗ[𝕜] F) (k : ℕ) : ℝ :=
  ⨆ U : { U : Submodule 𝕜 E // finrank 𝕜 U = k }, minSingularQuotientOn T U

/--
The `min_dim max` side of HDP Corollary 4.1.7 for singular values, with the subspace dimension
supplied explicitly.
-/
noncomputable def singularCourantFischerMinMax (T : E →ₗ[𝕜] F) (k : ℕ) : ℝ :=
  ⨅ U : { U : Submodule 𝕜 E // finrank 𝕜 U = k }, maxSingularQuotientOn T U

/--
Full-dimensional endpoint of the singular `max_dim min` quantity: maximizing the minimum over
all full-dimensional subspaces is the global singular-quotient infimum.
-/
theorem singularCourantFischerMaxMin_finrank [FiniteDimensional 𝕜 E] [Nontrivial E]
    (T : E →ₗ[𝕜] F) :
    singularCourantFischerMaxMin T (finrank 𝕜 E) =
      ⨅ x : { x : E // x ≠ 0 }, singularQuotient T x := by
  haveI : Nonempty { U : Submodule 𝕜 E // finrank 𝕜 U = finrank 𝕜 E } :=
    ⟨⟨⊤, by rw [finrank_top]⟩⟩
  unfold singularCourantFischerMaxMin
  apply le_antisymm
  · refine ciSup_le fun U => ?_
    change minSingularQuotientOn T (U : Submodule 𝕜 E) ≤
      ⨅ x : { x : E // x ≠ 0 }, singularQuotient T x
    have hU : (U : Submodule 𝕜 E) = ⊤ :=
      Submodule.eq_top_of_finrank_eq U.property
    rw [hU, minSingularQuotientOn_top]
  · have hbd :
        BddAbove (Set.range fun U : { U : Submodule 𝕜 E // finrank 𝕜 U = finrank 𝕜 E } =>
          minSingularQuotientOn T U) := by
      refine ⟨⨅ x : { x : E // x ≠ 0 }, singularQuotient T x, ?_⟩
      rintro _ ⟨U, rfl⟩
      change minSingularQuotientOn T (U : Submodule 𝕜 E) ≤
        ⨅ x : { x : E // x ≠ 0 }, singularQuotient T x
      have hU : (U : Submodule 𝕜 E) = ⊤ :=
        Submodule.eq_top_of_finrank_eq U.property
      rw [hU, minSingularQuotientOn_top]
    simpa [minSingularQuotientOn_top] using
      (le_ciSup hbd ⟨⊤, by rw [finrank_top]⟩)

/--
Full-dimensional endpoint of the singular `min_dim max` quantity: minimizing the maximum over
all full-dimensional subspaces is the global singular-quotient supremum.
-/
theorem singularCourantFischerMinMax_finrank [FiniteDimensional 𝕜 E] [Nontrivial E]
    (T : E →ₗ[𝕜] F) :
    singularCourantFischerMinMax T (finrank 𝕜 E) =
      ⨆ x : { x : E // x ≠ 0 }, singularQuotient T x := by
  haveI : Nonempty { U : Submodule 𝕜 E // finrank 𝕜 U = finrank 𝕜 E } :=
    ⟨⟨⊤, by rw [finrank_top]⟩⟩
  unfold singularCourantFischerMinMax
  apply le_antisymm
  · have hbd :
        BddBelow (Set.range fun U : { U : Submodule 𝕜 E // finrank 𝕜 U = finrank 𝕜 E } =>
          maxSingularQuotientOn T U) := by
      refine ⟨⨆ x : { x : E // x ≠ 0 }, singularQuotient T x, ?_⟩
      rintro _ ⟨U, rfl⟩
      change (⨆ x : { x : E // x ≠ 0 }, singularQuotient T x) ≤
        maxSingularQuotientOn T (U : Submodule 𝕜 E)
      have hU : (U : Submodule 𝕜 E) = ⊤ :=
        Submodule.eq_top_of_finrank_eq U.property
      rw [hU, maxSingularQuotientOn_top]
    simpa [maxSingularQuotientOn_top] using
      (ciInf_le hbd ⟨⊤, by rw [finrank_top]⟩)
  · refine le_ciInf fun U => ?_
    change (⨆ x : { x : E // x ≠ 0 }, singularQuotient T x) ≤
      maxSingularQuotientOn T (U : Submodule 𝕜 E)
    have hU : (U : Submodule 𝕜 E) = ⊤ :=
      Submodule.eq_top_of_finrank_eq U.property
    rw [hU, maxSingularQuotientOn_top]

/--
One-dimensional endpoint of the singular `max_dim min` quantity: maximizing over lines is the
global singular-quotient supremum.
-/
theorem singularCourantFischerMaxMin_one [FiniteDimensional 𝕜 E] [Nontrivial E]
    (T : E →ₗ[𝕜] F) :
    singularCourantFischerMaxMin T 1 =
      ⨆ x : { x : E // x ≠ 0 }, singularQuotient T x := by
  haveI : Nonempty { U : Submodule 𝕜 E // finrank 𝕜 U = 1 } := by
    obtain ⟨x, hx⟩ := exists_ne (0 : E)
    exact ⟨⟨𝕜 ∙ x, finrank_span_singleton hx⟩⟩
  haveI : Nonempty { x : E // x ≠ 0 } := by
    obtain ⟨x, hx⟩ := exists_ne (0 : E)
    exact ⟨⟨x, hx⟩⟩
  unfold singularCourantFischerMaxMin
  apply le_antisymm
  · refine ciSup_le fun U => ?_
    obtain ⟨x, hxU, hx0, hspan_eq⟩ :=
      Submodule.exists_span_singleton_eq_of_finrank_eq_one (𝕜 := 𝕜) (E := E)
        (U : Submodule 𝕜 E) U.property
    rw [← hspan_eq, minSingularQuotientOn_span_singleton T hx0]
    exact le_ciSup (bddAbove_range_singularQuotient T) ⟨x, hx0⟩
  · have hbd :
        BddAbove (Set.range fun U : { U : Submodule 𝕜 E // finrank 𝕜 U = 1 } =>
          minSingularQuotientOn T U) := by
      refine ⟨⨆ x : { x : E // x ≠ 0 }, singularQuotient T x, ?_⟩
      rintro _ ⟨U, rfl⟩
      obtain ⟨x, hxU, hx0, hspan_eq⟩ :=
        Submodule.exists_span_singleton_eq_of_finrank_eq_one (𝕜 := 𝕜) (E := E) U U.property
      change minSingularQuotientOn T (U : Submodule 𝕜 E) ≤
        ⨆ x : { x : E // x ≠ 0 }, singularQuotient T x
      rw [← hspan_eq, minSingularQuotientOn_span_singleton T hx0]
      exact le_ciSup (bddAbove_range_singularQuotient T) ⟨x, hx0⟩
    refine ciSup_le fun x => ?_
    calc
      singularQuotient T (x : E) =
          minSingularQuotientOn T (𝕜 ∙ (x : E)) := by
        rw [minSingularQuotientOn_span_singleton T x.property]
      _ ≤ ⨆ U : { U : Submodule 𝕜 E // finrank 𝕜 U = 1 }, minSingularQuotientOn T U := by
        exact le_ciSup hbd ⟨𝕜 ∙ (x : E), finrank_span_singleton x.property⟩

/--
One-dimensional endpoint of the singular `min_dim max` quantity: minimizing over lines is the
global singular-quotient infimum.
-/
theorem singularCourantFischerMinMax_one [FiniteDimensional 𝕜 E] [Nontrivial E]
    (T : E →ₗ[𝕜] F) :
    singularCourantFischerMinMax T 1 =
      ⨅ x : { x : E // x ≠ 0 }, singularQuotient T x := by
  haveI : Nonempty { U : Submodule 𝕜 E // finrank 𝕜 U = 1 } := by
    obtain ⟨x, hx⟩ := exists_ne (0 : E)
    exact ⟨⟨𝕜 ∙ x, finrank_span_singleton hx⟩⟩
  haveI : Nonempty { x : E // x ≠ 0 } := by
    obtain ⟨x, hx⟩ := exists_ne (0 : E)
    exact ⟨⟨x, hx⟩⟩
  unfold singularCourantFischerMinMax
  have hbd :
      BddBelow (Set.range fun U : { U : Submodule 𝕜 E // finrank 𝕜 U = 1 } =>
        maxSingularQuotientOn T U) := by
    refine ⟨⨅ x : { x : E // x ≠ 0 }, singularQuotient T x, ?_⟩
    rintro _ ⟨U, rfl⟩
    obtain ⟨x, hxU, hx0, hspan_eq⟩ :=
      Submodule.exists_span_singleton_eq_of_finrank_eq_one (𝕜 := 𝕜) (E := E) U U.property
    change (⨅ x : { x : E // x ≠ 0 }, singularQuotient T x) ≤
      maxSingularQuotientOn T (U : Submodule 𝕜 E)
    rw [← hspan_eq, maxSingularQuotientOn_span_singleton T hx0]
    exact ciInf_le (bddBelow_range_singularQuotient T) ⟨x, hx0⟩
  apply le_antisymm
  · refine le_ciInf fun x => ?_
    calc
      (⨅ U : { U : Submodule 𝕜 E // finrank 𝕜 U = 1 }, maxSingularQuotientOn T U) ≤
          maxSingularQuotientOn T (𝕜 ∙ (x : E)) := by
        exact ciInf_le hbd ⟨𝕜 ∙ (x : E), finrank_span_singleton x.property⟩
      _ = singularQuotient T (x : E) := by
        rw [maxSingularQuotientOn_span_singleton T x.property]
  · refine le_ciInf fun U => ?_
    obtain ⟨x, hxU, hx0, hspan_eq⟩ :=
      Submodule.exists_span_singleton_eq_of_finrank_eq_one (𝕜 := 𝕜) (E := E)
        (U : Submodule 𝕜 E) U.property
    rw [← hspan_eq, maxSingularQuotientOn_span_singleton T hx0]
    exact ciInf_le (bddBelow_range_singularQuotient T) ⟨x, hx0⟩

/-! ### Endpoint spectral facts -/

namespace IsSymmetric

variable {𝕜 E : Type*} [RCLike 𝕜] [NormedAddCommGroup E] [InnerProductSpace 𝕜 E]
variable [FiniteDimensional 𝕜 E] {T : E →ₗ[𝕜] E}

local notation "⟪" x ", " y "⟫" => inner 𝕜 x y

/--
The span of the first `k` vectors in the decreasingly ordered self-adjoint eigenbasis. This is the
`k`-dimensional subspace used for the `max_dim min` side of HDP Theorem 4.1.6.
-/
noncomputable def leadingEigenSubspace {n k : ℕ} (hT : T.IsSymmetric)
    (hn : Module.finrank 𝕜 E = n) (hk : k ≤ n) : Submodule 𝕜 E :=
  Submodule.span 𝕜
    (Set.range fun i : Fin k => hT.eigenvectorBasis hn (Fin.castLE hk i))

/-- The eigenbasis vectors used to define `leadingEigenSubspace` belong to it. -/
theorem eigenvectorBasis_mem_leadingEigenSubspace {n k : ℕ} (hT : T.IsSymmetric)
    (hn : Module.finrank 𝕜 E = n) (hk : k ≤ n) (i : Fin k) :
    hT.eigenvectorBasis hn (Fin.castLE hk i) ∈ hT.leadingEigenSubspace hn hk :=
  Submodule.subset_span ⟨i, rfl⟩

/-- The leading eigenspace has the requested dimension. -/
theorem finrank_leadingEigenSubspace {n k : ℕ} (hT : T.IsSymmetric)
    (hn : Module.finrank 𝕜 E = n) (hk : k ≤ n) :
    finrank 𝕜 (hT.leadingEigenSubspace hn hk) = k := by
  unfold leadingEigenSubspace
  rw [finrank_span_eq_card]
  · simp
  · exact (hT.eigenvectorBasis hn).toBasis.linearIndependent.comp
      (fun i : Fin k => Fin.castLE hk i) (Fin.castLE_injective hk)

/--
The span of the eigenbasis vectors from index `i` through the end. This is the
`n - i`-dimensional subspace used for the `min_dim max` side of HDP Theorem 4.1.6, where
`i` is zero-based.
-/
noncomputable def trailingEigenSubspace {n : ℕ} (hT : T.IsSymmetric)
    (hn : Module.finrank 𝕜 E = n) (i : Fin n) : Submodule 𝕜 E :=
  Submodule.span 𝕜
    (Set.range fun j : Fin (n - i.1) =>
      hT.eigenvectorBasis hn (Fin.natAdd_castLEEmb (Nat.sub_le n i.1) j))

/-- The eigenbasis vectors used to define `trailingEigenSubspace` belong to it. -/
theorem eigenvectorBasis_mem_trailingEigenSubspace {n : ℕ} (hT : T.IsSymmetric)
    (hn : Module.finrank 𝕜 E = n) (i : Fin n) (j : Fin (n - i.1)) :
    hT.eigenvectorBasis hn (Fin.natAdd_castLEEmb (Nat.sub_le n i.1) j) ∈
      hT.trailingEigenSubspace hn i :=
  Submodule.subset_span ⟨j, rfl⟩

/-- The trailing eigenspace has dimension `n - i`. -/
theorem finrank_trailingEigenSubspace {n : ℕ} (hT : T.IsSymmetric)
    (hn : Module.finrank 𝕜 E = n) (i : Fin n) :
    finrank 𝕜 (hT.trailingEigenSubspace hn i) = n - i.1 := by
  unfold trailingEigenSubspace
  rw [finrank_span_eq_card]
  · simp
  · exact (hT.eigenvectorBasis hn).toBasis.linearIndependent.comp
      (fun j : Fin (n - i.1) => Fin.natAdd_castLEEmb (Nat.sub_le n i.1) j)
      (Fin.natAdd_castLEEmb (Nat.sub_le n i.1)).injective

/--
Any `(i + 1)`-dimensional subspace intersects the trailing eigenspace nontrivially. This is the
dimension-counting half of the upper bound for the `max_dim min` Courant-Fischer formula.
-/
theorem exists_ne_zero_mem_inf_trailingEigenSubspace_of_finrank_eq_succ {n : ℕ}
    (hT : T.IsSymmetric) (hn : Module.finrank 𝕜 E = n) (i : Fin n)
    (U : Submodule 𝕜 E) (hU : finrank 𝕜 U = i.1 + 1) :
    ∃ x : E, x ∈ U ∧ x ∈ hT.trailingEigenSubspace hn i ∧ x ≠ 0 := by
  refine Submodule.exists_ne_zero_mem_inf_of_finrank_lt_add_finrank U
    (hT.trailingEigenSubspace hn i) ?_
  rw [hn, hU, hT.finrank_trailingEigenSubspace hn i]
  omega

/--
Any `(n - i)`-dimensional subspace intersects the leading eigenspace through index `i`
nontrivially. This is the dimension-counting half of the lower bound for the `min_dim max`
Courant-Fischer formula.
-/
theorem exists_ne_zero_mem_inf_leadingEigenSubspace_of_finrank_eq_sub {n : ℕ}
    (hT : T.IsSymmetric) (hn : Module.finrank 𝕜 E = n) (i : Fin n)
    (U : Submodule 𝕜 E) (hU : finrank 𝕜 U = n - i.1) :
    ∃ x : E, x ∈ U ∧ x ∈ hT.leadingEigenSubspace hn (Nat.succ_le_of_lt i.2) ∧ x ≠ 0 := by
  refine Submodule.exists_ne_zero_mem_inf_of_finrank_lt_add_finrank U
    (hT.leadingEigenSubspace hn (Nat.succ_le_of_lt i.2)) ?_
  rw [hn, hU, hT.finrank_leadingEigenSubspace hn (Nat.succ_le_of_lt i.2)]
  omega

/-- The eigenvalues attached to the leading eigenspace are all at least the endpoint `i`. -/
theorem eigenvalues_le_leadingEigenSubspace_index {n : ℕ} (hT : T.IsSymmetric)
    (hn : Module.finrank 𝕜 E = n) (i : Fin n) (j : Fin (i.1 + 1)) :
    hT.eigenvalues hn i ≤
      hT.eigenvalues hn (Fin.castLE (Nat.succ_le_of_lt i.2) j) := by
  have hji : Fin.castLE (Nat.succ_le_of_lt i.2) j ≤ i := by
    exact Nat.lt_succ_iff.mp j.2
  exact hT.eigenvalues_antitone hn hji

/-- The eigenvalues attached to the trailing eigenspace are all at most the starting index `i`. -/
theorem eigenvalues_trailingEigenSubspace_index_le {n : ℕ} (hT : T.IsSymmetric)
    (hn : Module.finrank 𝕜 E = n) (i : Fin n) (j : Fin (n - i.1)) :
    hT.eigenvalues hn (Fin.natAdd_castLEEmb (Nat.sub_le n i.1) j) ≤
      hT.eigenvalues hn i := by
  have hij : i ≤ (Fin.natAdd_castLEEmb (Nat.sub_le n i.1) j : Fin n) := by
    change i.1 ≤ (Fin.natAdd_castLEEmb (Nat.sub_le n i.1) j : Fin n).1
    simp [Fin.natAdd_castLEEmb]
    omega
  exact hT.eigenvalues_antitone hn hij

/--
Rayleigh numerator in the self-adjoint eigenbasis: `Re <Tx,x>` is the eigenvalue-weighted
sum of squared coordinate magnitudes. This is the coordinate calculation behind the indexed
Courant-Fischer inequalities.
-/
theorem re_inner_apply_self_eq_sum_eigenvalues_mul_norm_sq_repr {n : ℕ}
    (hT : T.IsSymmetric) (hn : Module.finrank 𝕜 E = n) (x : E) :
    RCLike.re (inner 𝕜 (T x) x) =
      ∑ i : Fin n, hT.eigenvalues hn i * ‖(hT.eigenvectorBasis hn).repr x i‖ ^ 2 := by
  let b := hT.eigenvectorBasis hn
  calc
    RCLike.re (inner 𝕜 (T x) x) =
        RCLike.re (∑ i : Fin n, inner 𝕜 (T x) (b i) * inner 𝕜 (b i) x) := by
      rw [OrthonormalBasis.sum_inner_mul_inner b]
    _ = ∑ i : Fin n, hT.eigenvalues hn i * ‖b.repr x i‖ ^ 2 := by
      rw [map_sum]
      refine Finset.sum_congr rfl fun i _ => ?_
      rw [hT x (b i), hT.apply_eigenvectorBasis hn i, inner_smul_right]
      rw [mul_assoc, RCLike.re_ofReal_mul]
      rw [inner_mul_symm_re_eq_norm]
      rw [norm_mul, OrthonormalBasis.repr_apply_apply]
      rw [← inner_conj_symm x (b i), RCLike.norm_conj]
      ring

/--
Parseval identity in the self-adjoint eigenbasis, stated with the same coordinates used in the
Rayleigh-numerator expansion.
-/
theorem norm_sq_eq_sum_norm_sq_repr {n : ℕ} (hT : T.IsSymmetric)
    (hn : Module.finrank 𝕜 E = n) (x : E) :
    ‖x‖ ^ 2 = ∑ i : Fin n, ‖(hT.eigenvectorBasis hn).repr x i‖ ^ 2 := by
  let b := hT.eigenvectorBasis hn
  calc
    ‖x‖ ^ 2 = ‖b.repr x‖ ^ 2 := by
      rw [b.repr.norm_map]
    _ = ∑ i : Fin n, ‖b.repr x i‖ ^ 2 := by
      rw [EuclideanSpace.norm_sq_eq]

/--
Vectors in the leading eigenspace have zero coordinates outside the selected leading block.
This is the support calculation used in the lower-bound half of the indexed
Courant-Fischer theorem.
-/
theorem eigenvectorBasis_repr_eq_zero_of_mem_leadingEigenSubspace {n k : ℕ}
    (hT : T.IsSymmetric) (hn : Module.finrank 𝕜 E = n) (hk : k ≤ n)
    {j : Fin n} (hj : k ≤ j.1) {x : E}
    (hx : x ∈ hT.leadingEigenSubspace hn hk) :
    (hT.eigenvectorBasis hn).repr x j = 0 := by
  rw [OrthonormalBasis.repr_apply_apply]
  let b := hT.eigenvectorBasis hn
  change inner 𝕜 (b j) x = 0
  exact Submodule.span_induction
    (p := fun y _ => inner 𝕜 (b j) y = 0)
    (s := Set.range fun i : Fin k => hT.eigenvectorBasis hn (Fin.castLE hk i))
    (fun y hy => by
      rcases hy with ⟨i, rfl⟩
      have hne : j ≠ Fin.castLE hk i := by
        intro h
        have hlt : j.1 < k := by
          simp [h, i.2]
        omega
      simpa [b] using b.inner_eq_zero hne)
    (by simp)
    (fun y z hy hz hiy hiz => by simp [inner_add_right, hiy, hiz])
    (fun a y hy hiy => by rw [inner_smul_right, hiy, mul_zero])
    (by simpa [leadingEigenSubspace] using hx)

/--
Vectors in the trailing eigenspace have zero coordinates before the selected trailing block.
This is the support calculation used in the upper-bound half of the indexed
Courant-Fischer theorem.
-/
theorem eigenvectorBasis_repr_eq_zero_of_mem_trailingEigenSubspace {n : ℕ}
    (hT : T.IsSymmetric) (hn : Module.finrank 𝕜 E = n) (i : Fin n)
    {j : Fin n} (hj : j.1 < i.1) {x : E}
    (hx : x ∈ hT.trailingEigenSubspace hn i) :
    (hT.eigenvectorBasis hn).repr x j = 0 := by
  rw [OrthonormalBasis.repr_apply_apply]
  let b := hT.eigenvectorBasis hn
  change inner 𝕜 (b j) x = 0
  exact Submodule.span_induction
    (p := fun y _ => inner 𝕜 (b j) y = 0)
    (s := Set.range fun l : Fin (n - i.1) =>
      hT.eigenvectorBasis hn (Fin.natAdd_castLEEmb (Nat.sub_le n i.1) l))
    (fun y hy => by
      rcases hy with ⟨l, rfl⟩
      have hne : j ≠ (Fin.natAdd_castLEEmb (Nat.sub_le n i.1) l : Fin n) := by
        intro h
        have hval := congrArg Fin.val h
        simp [Fin.natAdd_castLEEmb] at hval
        omega
      simpa [b] using b.inner_eq_zero hne)
    (by simp)
    (fun y z hy hz hiy hiz => by simp [inner_add_right, hiy, hiz])
    (fun a y hy hiy => by rw [inner_smul_right, hiy, mul_zero])
    (by simpa [trailingEigenSubspace] using hx)

/--
On the leading eigenspace through `i`, every nonzero vector has Rayleigh quotient at least the
`i`th eigenvalue. This is the lower inequality used in the `max_dim min` half of the indexed
Courant-Fischer theorem.
-/
theorem eigenvalues_le_rayleighQuotient_of_mem_leadingEigenSubspace {n : ℕ}
    (hT : T.IsSymmetric) (hn : Module.finrank 𝕜 E = n) (i : Fin n)
    {x : E} (hx : x ∈ hT.leadingEigenSubspace hn (Nat.succ_le_of_lt i.2))
    (hx0 : x ≠ 0) :
    hT.eigenvalues hn i ≤ LinearMap.rayleighQuotient T x := by
  let b := hT.eigenvectorBasis hn
  have hnorm_pos : 0 < ‖x‖ ^ 2 := sq_pos_of_pos (norm_pos_iff.mpr hx0)
  have hsum :
      hT.eigenvalues hn i * ‖x‖ ^ 2 ≤ RCLike.re (inner 𝕜 (T x) x) := by
    calc
      hT.eigenvalues hn i * ‖x‖ ^ 2 =
          hT.eigenvalues hn i * ∑ j : Fin n, ‖b.repr x j‖ ^ 2 := by
        rw [hT.norm_sq_eq_sum_norm_sq_repr hn x]
      _ = ∑ j : Fin n, hT.eigenvalues hn i * ‖b.repr x j‖ ^ 2 := by
        rw [Finset.mul_sum]
      _ ≤ ∑ j : Fin n, hT.eigenvalues hn j * ‖b.repr x j‖ ^ 2 := by
        refine Finset.sum_le_sum fun j _ => ?_
        by_cases hji : j ≤ i
        · exact mul_le_mul_of_nonneg_right (hT.eigenvalues_antitone hn hji) (sq_nonneg _)
        · have hj : i.1 + 1 ≤ j.1 := by
            change ¬j.1 ≤ i.1 at hji
            omega
          have hsupp : b.repr x j = 0 :=
            hT.eigenvectorBasis_repr_eq_zero_of_mem_leadingEigenSubspace hn
              (Nat.succ_le_of_lt i.2) hj hx
          simp [hsupp]
      _ = RCLike.re (inner 𝕜 (T x) x) := by
        rw [← hT.re_inner_apply_self_eq_sum_eigenvalues_mul_norm_sq_repr hn x]
  unfold LinearMap.rayleighQuotient
  exact (le_div_iff₀ hnorm_pos).2 hsum

/--
On the trailing eigenspace from `i`, every nonzero vector has Rayleigh quotient at most the
`i`th eigenvalue. This is the upper inequality used in the `min_dim max` half of the indexed
Courant-Fischer theorem.
-/
theorem rayleighQuotient_le_eigenvalues_of_mem_trailingEigenSubspace {n : ℕ}
    (hT : T.IsSymmetric) (hn : Module.finrank 𝕜 E = n) (i : Fin n)
    {x : E} (hx : x ∈ hT.trailingEigenSubspace hn i) (hx0 : x ≠ 0) :
    LinearMap.rayleighQuotient T x ≤ hT.eigenvalues hn i := by
  let b := hT.eigenvectorBasis hn
  have hnorm_pos : 0 < ‖x‖ ^ 2 := sq_pos_of_pos (norm_pos_iff.mpr hx0)
  have hsum :
      RCLike.re (inner 𝕜 (T x) x) ≤ hT.eigenvalues hn i * ‖x‖ ^ 2 := by
    calc
      RCLike.re (inner 𝕜 (T x) x) =
          ∑ j : Fin n, hT.eigenvalues hn j * ‖b.repr x j‖ ^ 2 := by
        rw [hT.re_inner_apply_self_eq_sum_eigenvalues_mul_norm_sq_repr hn x]
      _ ≤ ∑ j : Fin n, hT.eigenvalues hn i * ‖b.repr x j‖ ^ 2 := by
        refine Finset.sum_le_sum fun j _ => ?_
        by_cases hij : i ≤ j
        · exact mul_le_mul_of_nonneg_right (hT.eigenvalues_antitone hn hij) (sq_nonneg _)
        · have hj : j.1 < i.1 := by
            change ¬i.1 ≤ j.1 at hij
            omega
          have hsupp : b.repr x j = 0 :=
            hT.eigenvectorBasis_repr_eq_zero_of_mem_trailingEigenSubspace hn i hj hx
          simp [hsupp]
      _ = hT.eigenvalues hn i * ∑ j : Fin n, ‖b.repr x j‖ ^ 2 := by
        rw [Finset.mul_sum]
      _ = hT.eigenvalues hn i * ‖x‖ ^ 2 := by
        rw [hT.norm_sq_eq_sum_norm_sq_repr hn x]
  unfold LinearMap.rayleighQuotient
  exact (div_le_iff₀ hnorm_pos).2 hsum

/--
Indexed `max_dim min` side of HDP Theorem 4.1.6, using zero-based `Fin n` indices:
the `i`th decreasingly ordered eigenvalue is the largest possible minimum Rayleigh quotient
over `(i + 1)`-dimensional subspaces.
-/
theorem eigenvalues_eq_courantFischerMaxMin_succ {n : ℕ}
    (hT : T.IsSymmetric) (hn : Module.finrank 𝕜 E = n) (i : Fin n) :
    hT.eigenvalues hn i = LinearMap.courantFischerMaxMin T (i.1 + 1) := by
  let L : Submodule 𝕜 E := hT.leadingEigenSubspace hn (Nat.succ_le_of_lt i.2)
  have hLdim : finrank 𝕜 L = i.1 + 1 := by
    simpa [L] using hT.finrank_leadingEigenSubspace hn (Nat.succ_le_of_lt i.2)
  have hLmem : hT.eigenvectorBasis hn i ∈ L := by
    let ii : Fin (i.1 + 1) := ⟨i.1, Nat.lt_succ_self i.1⟩
    have hcast : Fin.castLE (Nat.succ_le_of_lt i.2) ii = i := by
      ext
      simp [ii]
    simpa [L, hcast] using
      hT.eigenvectorBasis_mem_leadingEigenSubspace hn (Nat.succ_le_of_lt i.2) ii
  have hb0 : hT.eigenvectorBasis hn i ≠ 0 := by
    simpa using (hT.eigenvectorBasis hn).toBasis.ne_zero i
  haveI : Nonempty { x : L // (x : E) ≠ 0 } :=
    ⟨⟨⟨hT.eigenvectorBasis hn i, hLmem⟩, hb0⟩⟩
  haveI : Nonempty { U : Submodule 𝕜 E // finrank 𝕜 U = i.1 + 1 } :=
    ⟨⟨L, hLdim⟩⟩
  have hbd :
      BddAbove (Set.range fun U : { U : Submodule 𝕜 E // finrank 𝕜 U = i.1 + 1 } =>
        LinearMap.minRayleighQuotientOn T (U : Submodule 𝕜 E)) := by
    refine ⟨‖T.toContinuousLinearMap‖, ?_⟩
    rintro _ ⟨U, rfl⟩
    have hUpos : 0 < finrank 𝕜 (U : Submodule 𝕜 E) := by
      rw [U.property]
      omega
    haveI : Nontrivial (U : Submodule 𝕜 E) := Module.nontrivial_of_finrank_pos hUpos
    obtain ⟨x, hx0U⟩ := exists_ne (0 : (U : Submodule 𝕜 E))
    have hx0 : (x : E) ≠ 0 := by
      intro hx
      exact hx0U (Subtype.ext hx)
    calc
      LinearMap.minRayleighQuotientOn T (U : Submodule 𝕜 E) ≤
          LinearMap.rayleighQuotient T (x : E) :=
        LinearMap.minRayleighQuotientOn_le_of_mem T x.property hx0
      _ ≤ ‖T.toContinuousLinearMap‖ := by
        rw [T.rayleighQuotient_toContinuousLinearMap (x : E)]
        exact le_trans (le_abs_self _)
          (T.toContinuousLinearMap.rayleighQuotient_le_norm (x : E))
  apply le_antisymm
  · calc
      hT.eigenvalues hn i ≤ LinearMap.minRayleighQuotientOn T L := by
        unfold LinearMap.minRayleighQuotientOn
        exact le_ciInf fun x =>
          hT.eigenvalues_le_rayleighQuotient_of_mem_leadingEigenSubspace hn i
            x.val.property x.property
      _ ≤ LinearMap.courantFischerMaxMin T (i.1 + 1) := by
        unfold LinearMap.courantFischerMaxMin
        exact le_ciSup hbd ⟨L, hLdim⟩
  · unfold LinearMap.courantFischerMaxMin
    refine ciSup_le fun U => ?_
    obtain ⟨x, hxU, hxTrailing, hx0⟩ :=
      hT.exists_ne_zero_mem_inf_trailingEigenSubspace_of_finrank_eq_succ hn i U U.property
    exact (LinearMap.minRayleighQuotientOn_le_of_mem T hxU hx0).trans
      (hT.rayleighQuotient_le_eigenvalues_of_mem_trailingEigenSubspace hn i hxTrailing hx0)

/--
Indexed `min_dim max` side of HDP Theorem 4.1.6, using zero-based `Fin n` indices:
the `i`th decreasingly ordered eigenvalue is the smallest possible maximum Rayleigh quotient
over `(n - i)`-dimensional subspaces.
-/
theorem eigenvalues_eq_courantFischerMinMax_sub {n : ℕ}
    (hT : T.IsSymmetric) (hn : Module.finrank 𝕜 E = n) (i : Fin n) :
    hT.eigenvalues hn i = LinearMap.courantFischerMinMax T (n - i.1) := by
  let R : Submodule 𝕜 E := hT.trailingEigenSubspace hn i
  have hRdim : finrank 𝕜 R = n - i.1 := by
    simpa [R] using hT.finrank_trailingEigenSubspace hn i
  have hpos : 0 < n - i.1 := Nat.sub_pos_of_lt i.2
  have hRmem : hT.eigenvectorBasis hn i ∈ R := by
    let j0 : Fin (n - i.1) := ⟨0, hpos⟩
    have hcast :
        (Fin.natAdd_castLEEmb (Nat.sub_le n i.1) j0 : Fin n) = i := by
      ext
      simp [Fin.natAdd_castLEEmb, j0]
      omega
    simpa [R, hcast] using hT.eigenvectorBasis_mem_trailingEigenSubspace hn i j0
  have hb0 : hT.eigenvectorBasis hn i ≠ 0 := by
    simpa using (hT.eigenvectorBasis hn).toBasis.ne_zero i
  haveI : Nonempty { x : R // (x : E) ≠ 0 } :=
    ⟨⟨⟨hT.eigenvectorBasis hn i, hRmem⟩, hb0⟩⟩
  haveI : Nonempty { U : Submodule 𝕜 E // finrank 𝕜 U = n - i.1 } :=
    ⟨⟨R, hRdim⟩⟩
  have hbd :
      BddBelow (Set.range fun U : { U : Submodule 𝕜 E // finrank 𝕜 U = n - i.1 } =>
        LinearMap.maxRayleighQuotientOn T (U : Submodule 𝕜 E)) := by
    refine ⟨-‖T.toContinuousLinearMap‖, ?_⟩
    rintro _ ⟨U, rfl⟩
    have hUpos : 0 < finrank 𝕜 (U : Submodule 𝕜 E) := by
      rw [U.property]
      omega
    haveI : Nontrivial (U : Submodule 𝕜 E) := Module.nontrivial_of_finrank_pos hUpos
    obtain ⟨x, hx0U⟩ := exists_ne (0 : (U : Submodule 𝕜 E))
    have hx0 : (x : E) ≠ 0 := by
      intro hx
      exact hx0U (Subtype.ext hx)
    calc
      -‖T.toContinuousLinearMap‖ ≤ LinearMap.rayleighQuotient T (x : E) := by
        rw [T.rayleighQuotient_toContinuousLinearMap (x : E)]
        have h := T.toContinuousLinearMap.rayleighQuotient_le_norm (x : E)
        exact neg_le.mp (le_trans (neg_le_abs _) h)
      _ ≤ LinearMap.maxRayleighQuotientOn T (U : Submodule 𝕜 E) :=
        LinearMap.le_maxRayleighQuotientOn_of_mem T x.property hx0
  apply le_antisymm
  · unfold LinearMap.courantFischerMinMax
    refine le_ciInf fun U => ?_
    obtain ⟨x, hxU, hxLeading, hx0⟩ :=
      hT.exists_ne_zero_mem_inf_leadingEigenSubspace_of_finrank_eq_sub hn i U U.property
    exact (hT.eigenvalues_le_rayleighQuotient_of_mem_leadingEigenSubspace hn i hxLeading hx0).trans
      (LinearMap.le_maxRayleighQuotientOn_of_mem T hxU hx0)
  · unfold LinearMap.courantFischerMinMax
    calc
      (⨅ U : { U : Submodule 𝕜 E // finrank 𝕜 U = n - i.1 },
          LinearMap.maxRayleighQuotientOn T (U : Submodule 𝕜 E)) ≤
          LinearMap.maxRayleighQuotientOn T R := by
        exact ciInf_le hbd ⟨R, hRdim⟩
      _ ≤ hT.eigenvalues hn i := by
        unfold LinearMap.maxRayleighQuotientOn
        exact ciSup_le fun x =>
          hT.rayleighQuotient_le_eigenvalues_of_mem_trailingEigenSubspace hn i
            x.val.property x.property

/--
The supremum of the Rayleigh quotient is an eigenvalue. This is the upper endpoint case behind
HDP Theorem 4.1.6.
-/
theorem hasEigenvalue_iSup_rayleigh [Nontrivial E] (hT : T.IsSymmetric) :
    Module.End.HasEigenvalue T
      (⨆ x : { x : E // x ≠ 0 }, RCLike.re ⟪T x, x⟫ / ‖(x : E)‖ ^ 2 : ℝ) :=
  hT.hasEigenvalue_iSup_of_finiteDimensional

/--
The infimum of the Rayleigh quotient is an eigenvalue. This is the lower endpoint case behind
HDP Theorem 4.1.6.
-/
theorem hasEigenvalue_iInf_rayleigh [Nontrivial E] (hT : T.IsSymmetric) :
    Module.End.HasEigenvalue T
      (⨅ x : { x : E // x ≠ 0 }, RCLike.re ⟪T x, x⟫ / ‖(x : E)‖ ^ 2 : ℝ) :=
  hT.hasEigenvalue_iInf_of_finiteDimensional

/-- Eigenvectors in the sorted orthonormal eigenbasis attain their eigenvalues as Rayleigh quotients. -/
theorem rayleighQuotient_eigenvectorBasis {n : ℕ} (hT : T.IsSymmetric)
    (hn : Module.finrank 𝕜 E = n) (i : Fin n) :
    LinearMap.rayleighQuotient T (hT.eigenvectorBasis hn i) =
      hT.eigenvalues hn i := by
  unfold LinearMap.rayleighQuotient
  rw [hT.apply_eigenvectorBasis hn i]
  simp [inner_smul_left, OrthonormalBasis.norm_eq_one]

/--
Endpoint case of HDP Theorem 4.1.6: the largest sorted eigenvalue is the global supremum of the
Rayleigh quotient.
-/
theorem eigenvalues_zero_eq_iSup_rayleighQuotient {n : ℕ} [Nontrivial E]
    (hT : T.IsSymmetric) (hn : Module.finrank 𝕜 E = n + 1) :
    hT.eigenvalues hn 0 =
      ⨆ x : { x : E // x ≠ 0 }, LinearMap.rayleighQuotient T x := by
  apply le_antisymm
  · calc
      hT.eigenvalues hn 0 =
          LinearMap.rayleighQuotient T (hT.eigenvectorBasis hn 0) := by
        rw [hT.rayleighQuotient_eigenvectorBasis hn 0]
      _ ≤ ⨆ x : { x : E // x ≠ 0 }, LinearMap.rayleighQuotient T x := by
        have hx : hT.eigenvectorBasis hn 0 ≠ 0 := by
          simpa using (hT.eigenvectorBasis hn).toBasis.ne_zero 0
        exact le_ciSup (bddAbove_range_rayleighQuotient T)
          ⟨hT.eigenvectorBasis hn 0, hx⟩
  · have hsup :
        Module.End.HasEigenvalue T
          (⨆ x : { x : E // x ≠ 0 }, LinearMap.rayleighQuotient T x : ℝ) := by
      simpa [LinearMap.rayleighQuotient] using hT.hasEigenvalue_iSup_rayleigh
    obtain ⟨i, hi⟩ := hT.exists_eigenvalues_eq hn hsup
    rw [← RCLike.ofReal_injective hi]
    exact hT.eigenvalues_antitone hn (Fin.zero_le i)

/--
Endpoint case of HDP Theorem 4.1.6: the smallest sorted eigenvalue is the global infimum of the
Rayleigh quotient.
-/
theorem eigenvalues_last_eq_iInf_rayleighQuotient {n : ℕ} [Nontrivial E]
    (hT : T.IsSymmetric) (hn : Module.finrank 𝕜 E = n + 1) :
    hT.eigenvalues hn (Fin.last n) =
      ⨅ x : { x : E // x ≠ 0 }, LinearMap.rayleighQuotient T x := by
  have hle_left :
      hT.eigenvalues hn (Fin.last n) ≤
        ⨅ x : { x : E // x ≠ 0 }, LinearMap.rayleighQuotient T x := by
    have hinf :
        Module.End.HasEigenvalue T
          (⨅ x : { x : E // x ≠ 0 }, LinearMap.rayleighQuotient T x : ℝ) := by
      simpa [LinearMap.rayleighQuotient] using hT.hasEigenvalue_iInf_rayleigh
    obtain ⟨i, hi⟩ := hT.exists_eigenvalues_eq hn hinf
    have hiℝ :
        hT.eigenvalues hn i =
          (⨅ x : { x : E // x ≠ 0 }, LinearMap.rayleighQuotient T x) :=
      RCLike.ofReal_injective hi
    exact (hT.eigenvalues_antitone hn (Fin.le_last i)).trans_eq hiℝ
  have hle_right :
      (⨅ x : { x : E // x ≠ 0 }, LinearMap.rayleighQuotient T x) ≤
        hT.eigenvalues hn (Fin.last n) := by
    calc
      (⨅ x : { x : E // x ≠ 0 }, LinearMap.rayleighQuotient T x) ≤
          LinearMap.rayleighQuotient T (hT.eigenvectorBasis hn (Fin.last n)) := by
        have hx : hT.eigenvectorBasis hn (Fin.last n) ≠ 0 := by
          simpa using (hT.eigenvectorBasis hn).toBasis.ne_zero (Fin.last n)
        exact ciInf_le (bddBelow_range_rayleighQuotient T)
          ⟨hT.eigenvectorBasis hn (Fin.last n), hx⟩
      _ = hT.eigenvalues hn (Fin.last n) := by
        rw [hT.rayleighQuotient_eigenvectorBasis hn (Fin.last n)]
  exact le_antisymm hle_left hle_right

/--
Full-dimensional endpoint of the `min_dim max` side of HDP Theorem 4.1.6: the largest sorted
eigenvalue is the Courant-Fischer value obtained by ranging over full-dimensional subspaces.
-/
theorem eigenvalues_zero_eq_courantFischerMinMax_finrank {n : ℕ} [Nontrivial E]
    (hT : T.IsSymmetric) (hn : Module.finrank 𝕜 E = n + 1) :
    hT.eigenvalues hn 0 = LinearMap.courantFischerMinMax T (n + 1) := by
  calc
    hT.eigenvalues hn 0 =
        ⨆ x : { x : E // x ≠ 0 }, LinearMap.rayleighQuotient T x :=
      hT.eigenvalues_zero_eq_iSup_rayleighQuotient hn
    _ = LinearMap.courantFischerMinMax T (finrank 𝕜 E) := by
      rw [LinearMap.courantFischerMinMax_finrank]
    _ = LinearMap.courantFischerMinMax T (n + 1) := by
      rw [hn]

/--
Full-dimensional endpoint of the `max_dim min` side of HDP Theorem 4.1.6: the smallest sorted
eigenvalue is the Courant-Fischer value obtained by ranging over full-dimensional subspaces.
-/
theorem eigenvalues_last_eq_courantFischerMaxMin_finrank {n : ℕ} [Nontrivial E]
    (hT : T.IsSymmetric) (hn : Module.finrank 𝕜 E = n + 1) :
    hT.eigenvalues hn (Fin.last n) = LinearMap.courantFischerMaxMin T (n + 1) := by
  calc
    hT.eigenvalues hn (Fin.last n) =
        ⨅ x : { x : E // x ≠ 0 }, LinearMap.rayleighQuotient T x :=
      hT.eigenvalues_last_eq_iInf_rayleighQuotient hn
    _ = LinearMap.courantFischerMaxMin T (finrank 𝕜 E) := by
      rw [LinearMap.courantFischerMaxMin_finrank]
    _ = LinearMap.courantFischerMaxMin T (n + 1) := by
      rw [hn]

/--
One-dimensional endpoint of the `max_dim min` side of HDP Theorem 4.1.6: the largest sorted
eigenvalue is the Courant-Fischer value obtained by ranging over lines.
-/
theorem eigenvalues_zero_eq_courantFischerMaxMin_one {n : ℕ} [Nontrivial E]
    (hT : T.IsSymmetric) (hn : Module.finrank 𝕜 E = n + 1) :
    hT.eigenvalues hn 0 = LinearMap.courantFischerMaxMin T 1 := by
  calc
    hT.eigenvalues hn 0 =
        ⨆ x : { x : E // x ≠ 0 }, LinearMap.rayleighQuotient T x :=
      hT.eigenvalues_zero_eq_iSup_rayleighQuotient hn
    _ = LinearMap.courantFischerMaxMin T 1 := by
      rw [LinearMap.courantFischerMaxMin_one]

/--
One-dimensional endpoint of the `min_dim max` side of HDP Theorem 4.1.6: the smallest sorted
eigenvalue is the Courant-Fischer value obtained by ranging over lines.
-/
theorem eigenvalues_last_eq_courantFischerMinMax_one {n : ℕ} [Nontrivial E]
    (hT : T.IsSymmetric) (hn : Module.finrank 𝕜 E = n + 1) :
    hT.eigenvalues hn (Fin.last n) = LinearMap.courantFischerMinMax T 1 := by
  calc
    hT.eigenvalues hn (Fin.last n) =
        ⨅ x : { x : E // x ≠ 0 }, LinearMap.rayleighQuotient T x :=
      hT.eigenvalues_last_eq_iInf_rayleighQuotient hn
    _ = LinearMap.courantFischerMinMax T 1 := by
      rw [LinearMap.courantFischerMinMax_one]

/-- Rank-one spectral decomposition for a self-adjoint operator. -/
theorem apply_eq_sum_eigenvalue_rankOne {n : ℕ} (hT : T.IsSymmetric)
    (hn : Module.finrank 𝕜 E = n) (x : E) :
    T x =
      ∑ i : Fin n,
        (((hT.eigenvalues hn i : ℝ) : 𝕜) •
          InnerProductSpace.rankOne 𝕜
            (hT.eigenvectorBasis hn i) (hT.eigenvectorBasis hn i)) x := by
  calc
    T x =
        T (∑ i : Fin n, (hT.eigenvectorBasis hn).repr x i •
          hT.eigenvectorBasis hn i) := by
      rw [(hT.eigenvectorBasis hn).sum_repr x]
    _ = ∑ i : Fin n, (hT.eigenvectorBasis hn).repr x i •
          T (hT.eigenvectorBasis hn i) := by
      simp [map_sum, map_smul]
    _ = ∑ i : Fin n, (hT.eigenvectorBasis hn).repr x i •
          (((hT.eigenvalues hn i : ℝ) : 𝕜) • hT.eigenvectorBasis hn i) := by
      refine Finset.sum_congr rfl fun i _ ↦ ?_
      rw [hT.apply_eigenvectorBasis hn i]
    _ = ∑ i : Fin n,
        (((hT.eigenvalues hn i : ℝ) : 𝕜) •
          InnerProductSpace.rankOne 𝕜
            (hT.eigenvectorBasis hn i) (hT.eigenvectorBasis hn i)) x := by
      simp only [OrthonormalBasis.repr_apply_apply]
      refine Finset.sum_congr rfl fun i _ ↦ ?_
      change
        inner 𝕜 (hT.eigenvectorBasis hn i) x •
            (((hT.eigenvalues hn i : ℝ) : 𝕜) • hT.eigenvectorBasis hn i) =
          ((hT.eigenvalues hn i : ℝ) : 𝕜) •
            (inner 𝕜 (hT.eigenvectorBasis hn i) x • hT.eigenvectorBasis hn i)
      rw [smul_smul, smul_smul]
      exact congrArg (fun c : 𝕜 ↦ c • hT.eigenvectorBasis hn i) (mul_comm _ _)

/--
Matrix-form spectral decomposition for a self-adjoint operator, in the orthonormal eigenbasis
provided by Mathlib. This is the linear-map analogue of HDP Remark 4.1.3's diagonal matrix form.
-/
theorem spectral_decomposition_toMatrix {n : ℕ} (hT : T.IsSymmetric)
    (hn : Module.finrank 𝕜 E = n) :
    letI b := (hT.eigenvectorBasis hn).toBasis
    T.toMatrix b b = Matrix.diagonal (RCLike.ofReal ∘ hT.eigenvalues hn) :=
  hT.toMatrix_eigenvectorBasis hn

end IsSymmetric

/--
On the leading eigenspace of the Gram operator `T†T`, every nonzero vector has singular quotient
at least the indexed singular value. This is the singular-value analogue of the leading-subspace
Rayleigh bound.
-/
theorem singularValues_le_singularQuotient_of_mem_gram_leadingEigenSubspace
    [FiniteDimensional 𝕜 E] [FiniteDimensional 𝕜 F]
    (T : E →ₗ[𝕜] F) {n : ℕ} (hn : finrank 𝕜 E = n) (i : Fin n)
    {x : E}
    (hx : x ∈ T.isSymmetric_adjoint_comp_self.leadingEigenSubspace hn (Nat.succ_le_of_lt i.2))
    (hx0 : x ≠ 0) :
    T.singularValues i ≤ singularQuotient T x := by
  have hsq : T.singularValues i ^ 2 ≤ singularQuotient T x ^ 2 := by
    calc
      T.singularValues i ^ 2 =
          T.isSymmetric_adjoint_comp_self.eigenvalues hn i := by
        rw [T.sq_singularValues_fin hn i]
      _ ≤ rayleighQuotient (LinearMap.adjoint T ∘ₗ T) x :=
        T.isSymmetric_adjoint_comp_self.eigenvalues_le_rayleighQuotient_of_mem_leadingEigenSubspace
          hn i hx hx0
      _ = singularQuotient T x ^ 2 := by
        rw [← sq_singularQuotient_eq_rayleighQuotient_adjoint_comp_self T x]
  exact le_of_sq_le_sq hsq (by unfold singularQuotient; positivity)

/--
On the trailing eigenspace of the Gram operator `T†T`, every nonzero vector has singular quotient
at most the indexed singular value. This is the singular-value analogue of the trailing-subspace
Rayleigh bound.
-/
theorem singularQuotient_le_singularValues_of_mem_gram_trailingEigenSubspace
    [FiniteDimensional 𝕜 E] [FiniteDimensional 𝕜 F]
    (T : E →ₗ[𝕜] F) {n : ℕ} (hn : finrank 𝕜 E = n) (i : Fin n)
    {x : E} (hx : x ∈ T.isSymmetric_adjoint_comp_self.trailingEigenSubspace hn i)
    (hx0 : x ≠ 0) :
    singularQuotient T x ≤ T.singularValues i := by
  have hsq : singularQuotient T x ^ 2 ≤ T.singularValues i ^ 2 := by
    calc
      singularQuotient T x ^ 2 =
          rayleighQuotient (LinearMap.adjoint T ∘ₗ T) x := by
        rw [sq_singularQuotient_eq_rayleighQuotient_adjoint_comp_self T x]
      _ ≤ T.isSymmetric_adjoint_comp_self.eigenvalues hn i :=
        T.isSymmetric_adjoint_comp_self.rayleighQuotient_le_eigenvalues_of_mem_trailingEigenSubspace
          hn i hx hx0
      _ = T.singularValues i ^ 2 := by
        rw [T.sq_singularValues_fin hn i]
  exact le_of_sq_le_sq hsq (T.singularValues_nonneg i)

/--
Indexed `max_dim min` side of HDP Corollary 4.1.7, using zero-based `Fin n` indices:
the `i`th singular value is the largest possible minimum singular quotient over
`(i + 1)`-dimensional subspaces of the domain.
-/
theorem singularValues_eq_singularCourantFischerMaxMin_succ
    [FiniteDimensional 𝕜 E] [FiniteDimensional 𝕜 F]
    (T : E →ₗ[𝕜] F) {n : ℕ} (hn : finrank 𝕜 E = n) (i : Fin n) :
    T.singularValues i = singularCourantFischerMaxMin T (i.1 + 1) := by
  let L : Submodule 𝕜 E :=
    T.isSymmetric_adjoint_comp_self.leadingEigenSubspace hn (Nat.succ_le_of_lt i.2)
  have hLdim : finrank 𝕜 L = i.1 + 1 := by
    simpa [L] using
      T.isSymmetric_adjoint_comp_self.finrank_leadingEigenSubspace hn (Nat.succ_le_of_lt i.2)
  have hLmem : T.isSymmetric_adjoint_comp_self.eigenvectorBasis hn i ∈ L := by
    let ii : Fin (i.1 + 1) := ⟨i.1, Nat.lt_succ_self i.1⟩
    have hcast : Fin.castLE (Nat.succ_le_of_lt i.2) ii = i := by
      ext
      simp [ii]
    simpa [L, hcast] using
      T.isSymmetric_adjoint_comp_self.eigenvectorBasis_mem_leadingEigenSubspace hn
        (Nat.succ_le_of_lt i.2) ii
  have hb0 : T.isSymmetric_adjoint_comp_self.eigenvectorBasis hn i ≠ 0 := by
    simpa using (T.isSymmetric_adjoint_comp_self.eigenvectorBasis hn).toBasis.ne_zero i
  haveI : Nonempty { x : L // (x : E) ≠ 0 } :=
    ⟨⟨⟨T.isSymmetric_adjoint_comp_self.eigenvectorBasis hn i, hLmem⟩, hb0⟩⟩
  haveI : Nonempty { U : Submodule 𝕜 E // finrank 𝕜 U = i.1 + 1 } :=
    ⟨⟨L, hLdim⟩⟩
  have hbd :
      BddAbove (Set.range fun U : { U : Submodule 𝕜 E // finrank 𝕜 U = i.1 + 1 } =>
        minSingularQuotientOn T (U : Submodule 𝕜 E)) := by
    refine ⟨‖T.toContinuousLinearMap‖, ?_⟩
    rintro _ ⟨U, rfl⟩
    have hUpos : 0 < finrank 𝕜 (U : Submodule 𝕜 E) := by
      rw [U.property]
      omega
    haveI : Nontrivial (U : Submodule 𝕜 E) := Module.nontrivial_of_finrank_pos hUpos
    obtain ⟨x, hx0U⟩ := exists_ne (0 : (U : Submodule 𝕜 E))
    have hx0 : (x : E) ≠ 0 := by
      intro hx
      exact hx0U (Subtype.ext hx)
    calc
      minSingularQuotientOn T (U : Submodule 𝕜 E) ≤ singularQuotient T (x : E) :=
        minSingularQuotientOn_le_of_mem T x.property hx0
      _ ≤ ‖T.toContinuousLinearMap‖ := by
        unfold singularQuotient
        exact (div_le_iff₀ (norm_pos_iff.mpr hx0)).mpr
          (T.toContinuousLinearMap.le_opNorm (x : E))
  apply le_antisymm
  · calc
      T.singularValues i ≤ minSingularQuotientOn T L := by
        unfold minSingularQuotientOn
        exact le_ciInf fun x =>
          singularValues_le_singularQuotient_of_mem_gram_leadingEigenSubspace T hn i
            x.val.property x.property
      _ ≤ singularCourantFischerMaxMin T (i.1 + 1) := by
        unfold singularCourantFischerMaxMin
        exact le_ciSup hbd ⟨L, hLdim⟩
  · unfold singularCourantFischerMaxMin
    refine ciSup_le fun U => ?_
    obtain ⟨x, hxU, hxTrailing, hx0⟩ :=
      T.isSymmetric_adjoint_comp_self.exists_ne_zero_mem_inf_trailingEigenSubspace_of_finrank_eq_succ
        hn i U U.property
    exact (minSingularQuotientOn_le_of_mem T hxU hx0).trans
      (singularQuotient_le_singularValues_of_mem_gram_trailingEigenSubspace T hn i hxTrailing hx0)

/--
Indexed `min_dim max` side of HDP Corollary 4.1.7, using zero-based `Fin n` indices:
the `i`th singular value is the smallest possible maximum singular quotient over
`(n - i)`-dimensional subspaces of the domain.
-/
theorem singularValues_eq_singularCourantFischerMinMax_sub
    [FiniteDimensional 𝕜 E] [FiniteDimensional 𝕜 F]
    (T : E →ₗ[𝕜] F) {n : ℕ} (hn : finrank 𝕜 E = n) (i : Fin n) :
    T.singularValues i = singularCourantFischerMinMax T (n - i.1) := by
  let R : Submodule 𝕜 E := T.isSymmetric_adjoint_comp_self.trailingEigenSubspace hn i
  have hRdim : finrank 𝕜 R = n - i.1 := by
    simpa [R] using T.isSymmetric_adjoint_comp_self.finrank_trailingEigenSubspace hn i
  have hpos : 0 < n - i.1 := Nat.sub_pos_of_lt i.2
  have hRmem : T.isSymmetric_adjoint_comp_self.eigenvectorBasis hn i ∈ R := by
    let j0 : Fin (n - i.1) := ⟨0, hpos⟩
    have hcast :
        (Fin.natAdd_castLEEmb (Nat.sub_le n i.1) j0 : Fin n) = i := by
      ext
      simp [Fin.natAdd_castLEEmb, j0]
      omega
    simpa [R, hcast] using
      T.isSymmetric_adjoint_comp_self.eigenvectorBasis_mem_trailingEigenSubspace hn i j0
  have hb0 : T.isSymmetric_adjoint_comp_self.eigenvectorBasis hn i ≠ 0 := by
    simpa using (T.isSymmetric_adjoint_comp_self.eigenvectorBasis hn).toBasis.ne_zero i
  haveI : Nonempty { x : R // (x : E) ≠ 0 } :=
    ⟨⟨⟨T.isSymmetric_adjoint_comp_self.eigenvectorBasis hn i, hRmem⟩, hb0⟩⟩
  haveI : Nonempty { U : Submodule 𝕜 E // finrank 𝕜 U = n - i.1 } :=
    ⟨⟨R, hRdim⟩⟩
  have hbd :
      BddBelow (Set.range fun U : { U : Submodule 𝕜 E // finrank 𝕜 U = n - i.1 } =>
        maxSingularQuotientOn T (U : Submodule 𝕜 E)) := by
    refine ⟨0, ?_⟩
    rintro _ ⟨U, rfl⟩
    have hUpos : 0 < finrank 𝕜 (U : Submodule 𝕜 E) := by
      rw [U.property]
      exact hpos
    haveI : Nontrivial (U : Submodule 𝕜 E) := Module.nontrivial_of_finrank_pos hUpos
    obtain ⟨x, hx0U⟩ := exists_ne (0 : (U : Submodule 𝕜 E))
    have hx0 : (x : E) ≠ 0 := by
      intro hx
      exact hx0U (Subtype.ext hx)
    exact (by unfold singularQuotient; positivity : 0 ≤ singularQuotient T (x : E)).trans
      (le_maxSingularQuotientOn_of_mem T x.property hx0)
  apply le_antisymm
  · unfold singularCourantFischerMinMax
    refine le_ciInf fun U => ?_
    obtain ⟨x, hxU, hxLeading, hx0⟩ :=
      T.isSymmetric_adjoint_comp_self.exists_ne_zero_mem_inf_leadingEigenSubspace_of_finrank_eq_sub
        hn i U U.property
    exact (singularValues_le_singularQuotient_of_mem_gram_leadingEigenSubspace T hn i
      hxLeading hx0).trans (le_maxSingularQuotientOn_of_mem T hxU hx0)
  · unfold singularCourantFischerMinMax
    calc
      (⨅ U : { U : Submodule 𝕜 E // finrank 𝕜 U = n - i.1 },
          maxSingularQuotientOn T (U : Submodule 𝕜 E)) ≤ maxSingularQuotientOn T R := by
        exact ciInf_le hbd ⟨R, hRdim⟩
      _ ≤ T.singularValues i := by
        unfold maxSingularQuotientOn
        exact ciSup_le fun x =>
          singularQuotient_le_singularValues_of_mem_gram_trailingEigenSubspace T hn i
            x.val.property x.property

end LinearMap

namespace Matrix

variable {𝕜 m n : Type*} [RCLike 𝕜] [Fintype m] [Fintype n] [DecidableEq n]

/-- Right singular vectors attain the corresponding singular value in the singular quotient. -/
theorem singularQuotient_rightSingularVectorBasis
    (A : Matrix m n 𝕜) (i : Fin (Fintype.card n)) :
    LinearMap.singularQuotient A.toEuclideanLin (A.rightSingularVectorBasis i) =
      A.singularValues i := by
  unfold LinearMap.singularQuotient
  rw [A.norm_apply_rightSingularVectorBasis i]
  simp

end Matrix
