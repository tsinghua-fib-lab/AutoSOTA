/-
Copyright (c) 2026 Yuanhe Zhang. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Yuanhe Zhang, Jason D. Lee, Fanghui Liu
-/
import SLT.MatrixInfra.EYM
import Mathlib.Geometry.Euclidean.Angle.Unoriented.Basic

/-!
# Perturbation Bounds for Spectral Data

This file formalizes the perturbation tools from HDP, Section 4.1.

## Main definitions

* `LinearMap.IsSymmetric.spectralSubspace`: the span of the eigenvectors whose sorted
  eigenvalues lie in a prescribed set.
* `LinearMap.IsSymmetric.spectralProjection`: the orthogonal projection onto that spectral
  subspace.
* `Set.SeparatedBy`: a real-set separation predicate used for spectral
  projection perturbation statements.

## Main results

* `LinearMap.IsSymmetric.abs_eigenvalues_sub_le_opNorm`: HDP Lemma 4.1.14, Weyl inequality
  for decreasingly ordered eigenvalues, in zero-based linear-map form.
* `LinearMap.abs_singularValues_sub_le_opNorm`: HDP Lemma 4.1.14, Weyl inequality for
  decreasingly ordered singular values, in zero-based linear-map form.
* `LinearMap.IsSymmetric.davisKahan_eigenvector_angle_hdp`: HDP Theorem 4.1.15,
  Davis-Kahan eigenvector angle inequality.
* `LinearMap.IsSymmetric.davisKahan_spectralProjection_hdp`: HDP Lemma 4.1.16,
  Davis-Kahan inequality for spectral projections.
-/

open Module

noncomputable section

namespace Set

/-- Two real sets are `δ`-separated if every cross-distance is at least `δ`. -/
def SeparatedBy (δ : ℝ) (I J : Set ℝ) : Prop :=
  ∀ ⦃x : ℝ⦄, x ∈ I → ∀ ⦃y : ℝ⦄, y ∈ J → δ ≤ |x - y|

end Set

namespace LinearMap

variable {𝕜 E F : Type*} [RCLike 𝕜]
variable [NormedAddCommGroup E] [InnerProductSpace 𝕜 E]
variable [NormedAddCommGroup F] [InnerProductSpace 𝕜 F]

/-- Rayleigh quotients are linear in the operator argument. -/
theorem rayleighQuotient_sub (A B : E →ₗ[𝕜] E) (x : E) :
    rayleighQuotient (A - B) x =
      rayleighQuotient A x - rayleighQuotient B x := by
  unfold rayleighQuotient
  simp [LinearMap.sub_apply, inner_sub_left, sub_div]

variable [FiniteDimensional 𝕜 E]

/-- The operator norm is unchanged when reversing a linear-map subtraction. -/
theorem norm_toContinuousLinearMap_sub_rev (A B : E →ₗ[𝕜] F) :
    ‖(B - A).toContinuousLinearMap‖ = ‖(A - B).toContinuousLinearMap‖ := by
  have hneg : (B - A).toContinuousLinearMap = -(A - B).toContinuousLinearMap := by
    ext x
    simp [sub_eq_add_neg]
  rw [hneg, norm_neg]

/-- Rayleigh quotients change by at most the operator norm of the perturbation. -/
theorem abs_rayleighQuotient_sub_le_opNorm (A B : E →ₗ[𝕜] E) (x : E) :
    |rayleighQuotient A x - rayleighQuotient B x| ≤
      ‖(A - B).toContinuousLinearMap‖ := by
  rw [← rayleighQuotient_sub]
  rw [(A - B).rayleighQuotient_toContinuousLinearMap x]
  exact (A - B).toContinuousLinearMap.rayleighQuotient_le_norm x

variable [FiniteDimensional 𝕜 F]

section RealAngle

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

/--
For unit vectors, the sine of their Euclidean angle is the norm of the orthogonal component of
one vector relative to the span of the other.
-/
theorem sin_angle_eq_norm_orthogonal_starProjection_span_singleton
    {x y : V} (hx : ‖x‖ = 1) (hy : ‖y‖ = 1) :
    Real.sin (InnerProductGeometry.angle x y) =
      ‖((ℝ ∙ y)ᗮ.starProjection x : V)‖ := by
  let K : Submodule ℝ V := ℝ ∙ y
  have hx0 : x ≠ 0 := by
    intro h
    have hxnorm : ‖x‖ = 0 := by simp [h]
    linarith
  have hy0 : y ≠ 0 := by
    intro h
    have hynorm : ‖y‖ = 0 := by simp [h]
    linarith
  have hproj : K.starProjection x = (inner ℝ y x) • y := by
    simpa [K] using
      (Submodule.starProjection_unit_singleton (𝕜 := ℝ) (v := y) hy x)
  have hproj_sq : ‖K.starProjection x‖ ^ 2 = (inner ℝ x y) * (inner ℝ x y) := by
    rw [hproj, norm_smul, hy, mul_one, Real.norm_eq_abs, sq_abs]
    rw [real_inner_comm]
    ring
  have hpyth := K.norm_sq_eq_add_norm_sq_starProjection x
  have hcomp_sq :
      ‖Kᗮ.starProjection x‖ ^ 2 = 1 - (inner ℝ x y) * (inner ℝ x y) := by
    rw [hx] at hpyth
    nlinarith
  have hinnerx : inner ℝ x x = 1 := by
    rw [real_inner_self_eq_norm_mul_norm, hx]
    norm_num
  have hinnery : inner ℝ y y = 1 := by
    rw [real_inner_self_eq_norm_mul_norm, hy]
    norm_num
  rw [InnerProductGeometry.sin_angle hx0 hy0]
  rw [hinnerx, hinnery, one_mul, hx, hy, mul_one, div_one]
  rw [← hcomp_sq, Real.sqrt_sq (norm_nonneg _)]

end RealAngle

namespace IsSymmetric

variable {A B : E →ₗ[𝕜] E}

/--
The spectral subspace spanned by the sorted eigenvectors whose eigenvalues lie in `I`.
This is the linear-map version of the subspace underlying the spectral projection
`∑_{i : λ_i ∈ I} u_i u_iᵀ` in HDP Lemma 4.1.16.
-/
noncomputable def spectralSubspace {n : ℕ} (hA : A.IsSymmetric)
    (hn : finrank 𝕜 E = n) (I : Set ℝ) : Submodule 𝕜 E :=
  Submodule.span 𝕜
    (Set.range fun i : { i : Fin n // hA.eigenvalues hn i ∈ I } =>
      hA.eigenvectorBasis hn i)

/--
The spectral projection onto `spectralSubspace`. This is the orthogonal projection
`∑_{i : λ_i ∈ I} u_i u_iᵀ` from HDP Lemma 4.1.16.
-/
noncomputable def spectralProjection {n : ℕ} (hA : A.IsSymmetric)
    (hn : finrank 𝕜 E = n) (I : Set ℝ) : E →ₗ[𝕜] E :=
  ((hA.spectralSubspace hn I).starProjection : E →L[𝕜] E)

/-- The eigenbasis vectors selected by `I` belong to the corresponding spectral subspace. -/
theorem eigenvectorBasis_mem_spectralSubspace {n : ℕ} (hA : A.IsSymmetric)
    (hn : finrank 𝕜 E = n) {I : Set ℝ} {i : Fin n}
    (hi : hA.eigenvalues hn i ∈ I) :
    hA.eigenvectorBasis hn i ∈ hA.spectralSubspace hn I :=
  Submodule.subset_span ⟨⟨i, hi⟩, rfl⟩

/-- A spectral subspace is invariant under the corresponding self-adjoint operator. -/
theorem map_mem_spectralSubspace {n : ℕ} (hA : A.IsSymmetric)
    (hn : finrank 𝕜 E = n) (I : Set ℝ) {x : E}
    (hx : x ∈ hA.spectralSubspace hn I) :
    A x ∈ hA.spectralSubspace hn I := by
  let S : Submodule 𝕜 E := hA.spectralSubspace hn I
  refine Submodule.span_induction
    (p := fun y _ => A y ∈ S)
    (s := Set.range fun i : { i : Fin n // hA.eigenvalues hn i ∈ I } =>
      hA.eigenvectorBasis hn i)
    ?_ ?_ ?_ ?_ (by simpa [spectralSubspace, S] using hx)
  · rintro y ⟨i, rfl⟩
    rw [hA.apply_eigenvectorBasis hn i]
    exact S.smul_mem _ (hA.eigenvectorBasis_mem_spectralSubspace hn i.property)
  · simp [S]
  · intro y z _ _ hy hz
    simpa [map_add] using S.add_mem hy hz
  · intro c y _ hy
    simpa [map_smul] using S.smul_mem c hy

/-- The orthogonal complement of a spectral subspace is also invariant. -/
theorem map_mem_spectralSubspace_orthogonal {n : ℕ} (hA : A.IsSymmetric)
    (hn : finrank 𝕜 E = n) (I : Set ℝ) {x : E}
    (hx : x ∈ (hA.spectralSubspace hn I)ᗮ) :
    A x ∈ (hA.spectralSubspace hn I)ᗮ := by
  intro y hy
  rw [← hA y x]
  exact ((hA.spectralSubspace hn I).mem_orthogonal x).mp hx
    (A y) (hA.map_mem_spectralSubspace hn I hy)

/-- A self-adjoint operator commutes with each of its spectral projections. -/
theorem comp_spectralProjection_eq_spectralProjection_comp {n : ℕ}
    (hA : A.IsSymmetric) (hn : finrank 𝕜 E = n) (I : Set ℝ) :
    A.comp (hA.spectralProjection hn I) =
      (hA.spectralProjection hn I).comp A := by
  ext x
  let S : Submodule 𝕜 E := hA.spectralSubspace hn I
  have hmem : A (hA.spectralProjection hn I x) ∈ S := by
    exact hA.map_mem_spectralSubspace hn I
      (by
        unfold spectralProjection
        exact Submodule.starProjection_apply_mem S x)
  have hxorth : x - hA.spectralProjection hn I x ∈ Sᗮ := by
    unfold spectralProjection
    exact Submodule.sub_starProjection_mem_orthogonal x
  have hortho : A x - A (hA.spectralProjection hn I x) ∈ Sᗮ := by
    simpa [map_sub, S] using hA.map_mem_spectralSubspace_orthogonal hn I hxorth
  change A (hA.spectralProjection hn I x) = hA.spectralProjection hn I (A x)
  symm
  unfold spectralProjection
  exact Submodule.eq_starProjection_of_mem_orthogonal hmem hortho

/-- A centered self-adjoint operator commutes with its spectral projections. -/
theorem shifted_comp_spectralProjection_eq_spectralProjection_comp_shifted {n : ℕ}
    (hA : A.IsSymmetric) (hn : finrank 𝕜 E = n) (I : Set ℝ) (c : ℝ) :
    (A - ((c : 𝕜) • LinearMap.id : E →ₗ[𝕜] E)).comp (hA.spectralProjection hn I) =
      (hA.spectralProjection hn I).comp
        (A - ((c : 𝕜) • LinearMap.id : E →ₗ[𝕜] E)) := by
  ext x
  have hcomm := LinearMap.congr_fun
    (hA.comp_spectralProjection_eq_spectralProjection_comp hn I) x
  change
    A (hA.spectralProjection hn I x) - (c : 𝕜) • hA.spectralProjection hn I x =
      hA.spectralProjection hn I (A x - (c : 𝕜) • x)
  change A (hA.spectralProjection hn I x) = hA.spectralProjection hn I (A x) at hcomm
  rw [hcomm]
  simp [map_sub, map_smul]

/--
Coordinates outside the selected eigenvalue set vanish for vectors in a spectral subspace.
This is the spectral-subspace analogue of the leading/trailing support lemmas used in the
Courant-Fischer proof.
-/
theorem eigenvectorBasis_repr_eq_zero_of_mem_spectralSubspace {n : ℕ}
    (hA : A.IsSymmetric) (hn : finrank 𝕜 E = n) {I : Set ℝ}
    {j : Fin n} (hj : hA.eigenvalues hn j ∉ I) {x : E}
    (hx : x ∈ hA.spectralSubspace hn I) :
    (hA.eigenvectorBasis hn).repr x j = 0 := by
  rw [OrthonormalBasis.repr_apply_apply]
  let b := hA.eigenvectorBasis hn
  change inner 𝕜 (b j) x = 0
  exact Submodule.span_induction
    (p := fun y _ => inner 𝕜 (b j) y = 0)
    (s := Set.range fun i : { i : Fin n // hA.eigenvalues hn i ∈ I } =>
      hA.eigenvectorBasis hn i)
    (fun y hy => by
      rcases hy with ⟨i, rfl⟩
      have hne : j ≠ i := by
        intro hji
        exact hj (by simpa [hji] using i.property)
      simpa [b] using b.inner_eq_zero hne)
    (by simp)
    (fun y z _ _ hiy hiz => by simp [inner_add_right, hiy, hiz])
    (fun c y _ hiy => by rw [inner_smul_right, hiy, mul_zero])
    (by simpa [spectralSubspace] using hx)

/--
If all coordinates outside the selected eigenvalue set vanish, then the vector belongs to the
corresponding spectral subspace.
-/
theorem mem_spectralSubspace_of_eigenvectorBasis_repr_eq_zero {n : ℕ}
    (hA : A.IsSymmetric) (hn : finrank 𝕜 E = n) {I : Set ℝ} {x : E}
    (hzero : ∀ j : Fin n, hA.eigenvalues hn j ∉ I →
      (hA.eigenvectorBasis hn).repr x j = 0) :
    x ∈ hA.spectralSubspace hn I := by
  let b := hA.eigenvectorBasis hn
  have hsupp : ↑(b.toBasis.repr x).support ⊆
      {j : Fin n | hA.eigenvalues hn j ∈ I} := by
    intro j hj
    by_contra hnot
    have hzero' : b.toBasis.repr x j = 0 := by
      rw [b.coe_toBasis_repr_apply]
      exact hzero j hnot
    exact (Finsupp.mem_support_iff.mp hj) hzero'
  have hxspan : x ∈ Submodule.span 𝕜
      (b '' {j : Fin n | hA.eigenvalues hn j ∈ I}) :=
    (b.toBasis.mem_span_image (s := {j : Fin n | hA.eigenvalues hn j ∈ I})).2 hsupp
  unfold spectralSubspace
  refine (Submodule.span_mono ?_) hxspan
  rintro y ⟨j, hj, rfl⟩
  exact ⟨⟨j, hj⟩, rfl⟩

/-- If no eigenvalue is selected by `I`, then the corresponding spectral subspace is trivial. -/
theorem spectralSubspace_eq_bot_of_forall_eigenvalues_notMem {n : ℕ}
    (hA : A.IsSymmetric) (hn : finrank 𝕜 E = n) {I : Set ℝ}
    (hnot : ∀ i : Fin n, hA.eigenvalues hn i ∉ I) :
    hA.spectralSubspace hn I = ⊥ := by
  ext x
  constructor
  · intro hx
    have hrepr : (hA.eigenvectorBasis hn).repr x = 0 := by
      ext i
      exact hA.eigenvectorBasis_repr_eq_zero_of_mem_spectralSubspace hn (hnot i) hx
    have hx0 : x = 0 := (hA.eigenvectorBasis hn).repr.injective (by simpa using hrepr)
    rw [hx0]
    exact Submodule.zero_mem ⊥
  · intro hx
    rw [Submodule.mem_bot] at hx
    rw [hx]
    exact Submodule.zero_mem _

/--
If a selected eigenvalue set contains all basis vectors except `k`, and excludes `k`, then its
spectral subspace is the orthogonal complement of the `k`th eigendirection.
-/
theorem spectralSubspace_eq_orthogonal_span_eigenvectorBasis {n : ℕ}
    (hA : A.IsSymmetric) (hn : finrank 𝕜 E = n) {I : Set ℝ} {k : Fin n}
    (hk_not : hA.eigenvalues hn k ∉ I)
    (hmem : ∀ j : Fin n, j ≠ k → hA.eigenvalues hn j ∈ I) :
    hA.spectralSubspace hn I = (𝕜 ∙ hA.eigenvectorBasis hn k)ᗮ := by
  ext x
  constructor
  · intro hx
    rw [Submodule.mem_orthogonal_singleton_iff_inner_right]
    rw [← OrthonormalBasis.repr_apply_apply]
    exact hA.eigenvectorBasis_repr_eq_zero_of_mem_spectralSubspace hn hk_not hx
  · intro hx
    refine hA.mem_spectralSubspace_of_eigenvectorBasis_repr_eq_zero hn ?_
    intro j hj
    have hjk : j = k := by
      by_contra hne
      exact hj (hmem j hne)
    subst j
    rw [OrthonormalBasis.repr_apply_apply]
    exact (Submodule.mem_orthogonal_singleton_iff_inner_right.mp hx)

/-- Coordinates of a centered self-adjoint operator in its eigenbasis. -/
theorem eigenvectorBasis_repr_shifted_apply {n : ℕ} (hA : A.IsSymmetric)
    (hn : finrank 𝕜 E = n) (c : ℝ) (x : E) (i : Fin n) :
    (hA.eigenvectorBasis hn).repr
        ((A - ((c : 𝕜) • LinearMap.id : E →ₗ[𝕜] E)) x) i =
      (((hA.eigenvalues hn i - c : ℝ) : 𝕜) *
        (hA.eigenvectorBasis hn).repr x i) := by
  simp [hA.eigenvectorBasis_apply_self_apply hn x i, sub_eq_add_neg, add_mul,
    Algebra.smul_def]

/--
If all selected eigenvalues lie in `[c - r, c + r]`, then the centered operator
`A - cI` has norm at most `r` on the corresponding spectral subspace.
-/
theorem norm_shifted_apply_le_of_mem_spectralSubspace {n : ℕ} (hA : A.IsSymmetric)
    (hn : finrank 𝕜 E = n) {I : Set ℝ} {c r : ℝ} (hr : 0 ≤ r)
    (hI : ∀ i : Fin n, hA.eigenvalues hn i ∈ I →
      |hA.eigenvalues hn i - c| ≤ r) {x : E}
    (hx : x ∈ hA.spectralSubspace hn I) :
    ‖(A - ((c : 𝕜) • LinearMap.id : E →ₗ[𝕜] E)) x‖ ≤ r * ‖x‖ := by
  let b := hA.eigenvectorBasis hn
  let T : E →ₗ[𝕜] E := A - ((c : 𝕜) • LinearMap.id : E →ₗ[𝕜] E)
  have hterm : ∀ i : Fin n,
      ‖b.repr (T x) i‖ ^ 2 ≤ r ^ 2 * ‖b.repr x i‖ ^ 2 := by
    intro i
    by_cases hi : hA.eigenvalues hn i ∈ I
    · have hcoeff :
          ‖(((hA.eigenvalues hn i - c : ℝ) : 𝕜))‖ ≤ r := by
        rw [RCLike.norm_ofReal]
        exact hI i hi
      rw [show T x = (A - ((c : 𝕜) • LinearMap.id : E →ₗ[𝕜] E)) x by rfl]
      rw [hA.eigenvectorBasis_repr_shifted_apply hn c x i]
      calc
        ‖(((hA.eigenvalues hn i - c : ℝ) : 𝕜) * b.repr x i)‖ ^ 2 =
            ‖(((hA.eigenvalues hn i - c : ℝ) : 𝕜))‖ ^ 2 * ‖b.repr x i‖ ^ 2 := by
          rw [norm_mul]
          ring
        _ ≤ r ^ 2 * ‖b.repr x i‖ ^ 2 := by
          exact mul_le_mul_of_nonneg_right
            ((sq_le_sq₀ (norm_nonneg _) hr).mpr hcoeff) (sq_nonneg _)
    · have hzero : b.repr x i = 0 :=
        hA.eigenvectorBasis_repr_eq_zero_of_mem_spectralSubspace hn hi hx
      rw [show T x = (A - ((c : 𝕜) • LinearMap.id : E →ₗ[𝕜] E)) x by rfl]
      rw [hA.eigenvectorBasis_repr_shifted_apply hn c x i, hzero, mul_zero]
      simp
  have hsq :
      ‖T x‖ ^ 2 ≤ (r * ‖x‖) ^ 2 := by
    calc
      ‖T x‖ ^ 2 = ∑ i : Fin n, ‖b.repr (T x) i‖ ^ 2 := by
        exact hA.norm_sq_eq_sum_norm_sq_repr hn (T x)
      _ ≤ ∑ i : Fin n, r ^ 2 * ‖b.repr x i‖ ^ 2 :=
        Finset.sum_le_sum fun i _ => hterm i
      _ = r ^ 2 * ∑ i : Fin n, ‖b.repr x i‖ ^ 2 := by
        rw [Finset.mul_sum]
      _ = r ^ 2 * ‖x‖ ^ 2 := by
        rw [← hA.norm_sq_eq_sum_norm_sq_repr hn x]
      _ = (r * ‖x‖) ^ 2 := by
        ring
  exact (sq_le_sq₀ (norm_nonneg _) (mul_nonneg hr (norm_nonneg x))).mp hsq

/--
If all selected eigenvalues are at distance at least `r` from `c`, then the centered operator
`A - cI` expands the corresponding spectral subspace by at least `r`.
-/
theorem mul_norm_le_norm_shifted_apply_of_mem_spectralSubspace {n : ℕ}
    (hA : A.IsSymmetric) (hn : finrank 𝕜 E = n) {J : Set ℝ} {c r : ℝ}
    (hr : 0 ≤ r) (hJ : ∀ i : Fin n, hA.eigenvalues hn i ∈ J →
      r ≤ |hA.eigenvalues hn i - c|) {x : E}
    (hx : x ∈ hA.spectralSubspace hn J) :
    r * ‖x‖ ≤ ‖(A - ((c : 𝕜) • LinearMap.id : E →ₗ[𝕜] E)) x‖ := by
  let b := hA.eigenvectorBasis hn
  let T : E →ₗ[𝕜] E := A - ((c : 𝕜) • LinearMap.id : E →ₗ[𝕜] E)
  have hterm : ∀ i : Fin n,
      r ^ 2 * ‖b.repr x i‖ ^ 2 ≤ ‖b.repr (T x) i‖ ^ 2 := by
    intro i
    by_cases hi : hA.eigenvalues hn i ∈ J
    · have hcoeff :
          r ≤ ‖(((hA.eigenvalues hn i - c : ℝ) : 𝕜))‖ := by
        rw [RCLike.norm_ofReal]
        exact hJ i hi
      rw [show T x = (A - ((c : 𝕜) • LinearMap.id : E →ₗ[𝕜] E)) x by rfl]
      rw [hA.eigenvectorBasis_repr_shifted_apply hn c x i]
      calc
        r ^ 2 * ‖b.repr x i‖ ^ 2 ≤
            ‖(((hA.eigenvalues hn i - c : ℝ) : 𝕜))‖ ^ 2 * ‖b.repr x i‖ ^ 2 := by
          exact mul_le_mul_of_nonneg_right
            ((sq_le_sq₀ hr (norm_nonneg _)).mpr hcoeff) (sq_nonneg _)
        _ = ‖(((hA.eigenvalues hn i - c : ℝ) : 𝕜) * b.repr x i)‖ ^ 2 := by
          rw [norm_mul]
          ring
    · have hzero : b.repr x i = 0 :=
        hA.eigenvectorBasis_repr_eq_zero_of_mem_spectralSubspace hn hi hx
      simp [hzero]
  have hsq :
      (r * ‖x‖) ^ 2 ≤ ‖T x‖ ^ 2 := by
    calc
      (r * ‖x‖) ^ 2 = r ^ 2 * ‖x‖ ^ 2 := by
        ring
      _ = r ^ 2 * ∑ i : Fin n, ‖b.repr x i‖ ^ 2 := by
        rw [hA.norm_sq_eq_sum_norm_sq_repr hn x]
      _ = ∑ i : Fin n, r ^ 2 * ‖b.repr x i‖ ^ 2 := by
        rw [Finset.mul_sum]
      _ ≤ ∑ i : Fin n, ‖b.repr (T x) i‖ ^ 2 :=
        Finset.sum_le_sum fun i _ => hterm i
      _ = ‖T x‖ ^ 2 := by
        rw [hA.norm_sq_eq_sum_norm_sq_repr hn (T x)]
  exact (sq_le_sq₀ (mul_nonneg hr (norm_nonneg x)) (norm_nonneg _)).mp hsq

/--
Operator-norm form of `mul_norm_le_norm_shifted_apply_of_mem_spectralSubspace`: if the range of
`S` lies in a spectral subspace on which `A - cI` expands by at least `r`, then
`r * ‖S‖ ≤ ‖(A - cI)S‖`.
-/
theorem mul_norm_le_norm_shifted_comp_of_forall_mem_spectralSubspace {n : ℕ}
    (hA : A.IsSymmetric) (hn : finrank 𝕜 E = n) {J : Set ℝ} {c r : ℝ}
    (hr : 0 ≤ r) (hJ : ∀ i : Fin n, hA.eigenvalues hn i ∈ J →
      r ≤ |hA.eigenvalues hn i - c|)
    (S : E →L[𝕜] E) (hS : ∀ x : E, S x ∈ hA.spectralSubspace hn J) :
    r * ‖S‖ ≤
      ‖(A - ((c : 𝕜) • LinearMap.id : E →ₗ[𝕜] E)).toContinuousLinearMap.comp S‖ := by
  by_cases hr0 : r = 0
  · simp [hr0]
  have hrpos : 0 < r := lt_of_le_of_ne hr (Ne.symm hr0)
  let T : E →ₗ[𝕜] E := A - ((c : 𝕜) • LinearMap.id : E →ₗ[𝕜] E)
  have hbound :
      ‖S‖ ≤ r⁻¹ * ‖T.toContinuousLinearMap.comp S‖ := by
    refine ContinuousLinearMap.opNorm_le_bound _ (mul_nonneg (inv_nonneg.mpr hr) (norm_nonneg _)) ?_
    intro x
    have hexpand : r * ‖S x‖ ≤ ‖T (S x)‖ := by
      exact hA.mul_norm_le_norm_shifted_apply_of_mem_spectralSubspace hn hr hJ (hS x)
    have hop : ‖T (S x)‖ ≤ ‖T.toContinuousLinearMap.comp S‖ * ‖x‖ := by
      simpa [T] using (T.toContinuousLinearMap.comp S).le_opNorm x
    calc
      ‖S x‖ = r⁻¹ * (r * ‖S x‖) := by
        rw [← mul_assoc, inv_mul_cancel₀ hrpos.ne', one_mul]
      _ ≤ r⁻¹ * ‖T (S x)‖ :=
        mul_le_mul_of_nonneg_left hexpand (inv_nonneg.mpr hr)
      _ ≤ r⁻¹ * (‖T.toContinuousLinearMap.comp S‖ * ‖x‖) :=
        mul_le_mul_of_nonneg_left hop (inv_nonneg.mpr hr)
      _ = (r⁻¹ * ‖T.toContinuousLinearMap.comp S‖) * ‖x‖ := by
        ring
  calc
    r * ‖S‖ ≤ r * (r⁻¹ * ‖T.toContinuousLinearMap.comp S‖) :=
      mul_le_mul_of_nonneg_left hbound hr
    _ = ‖T.toContinuousLinearMap.comp S‖ := by
      rw [← mul_assoc, mul_inv_cancel₀ hrpos.ne', one_mul]

/--
Operator-norm form of `norm_shifted_apply_le_of_mem_spectralSubspace`: if the selected
eigenvalues lie in `[c - r, c + r]`, then `‖(A - cI)P‖ ≤ r`.
-/
theorem norm_shifted_comp_spectralProjection_le {n : ℕ} (hA : A.IsSymmetric)
    (hn : finrank 𝕜 E = n) {I : Set ℝ} {c r : ℝ} (hr : 0 ≤ r)
    (hI : ∀ i : Fin n, hA.eigenvalues hn i ∈ I →
      |hA.eigenvalues hn i - c| ≤ r) :
    ‖(A - ((c : 𝕜) • LinearMap.id : E →ₗ[𝕜] E)).toContinuousLinearMap.comp
        (hA.spectralProjection hn I).toContinuousLinearMap‖ ≤ r := by
  refine ContinuousLinearMap.opNorm_le_bound _ hr ?_
  intro x
  have hxP :
      hA.spectralProjection hn I x ∈ hA.spectralSubspace hn I := by
    unfold spectralProjection
    exact Submodule.starProjection_apply_mem (hA.spectralSubspace hn I) x
  have hmain :
      ‖(A - ((c : 𝕜) • LinearMap.id : E →ₗ[𝕜] E)) (hA.spectralProjection hn I x)‖ ≤
        r * ‖hA.spectralProjection hn I x‖ :=
    hA.norm_shifted_apply_le_of_mem_spectralSubspace hn hr hI hxP
  have hproj : ‖hA.spectralProjection hn I x‖ ≤ ‖x‖ := by
    unfold spectralProjection
    exact Submodule.norm_starProjection_apply_le (hA.spectralSubspace hn I) x
  exact hmain.trans (mul_le_mul_of_nonneg_left hproj hr)

/-- The spectral projection lands in its spectral subspace. -/
theorem spectralProjection_apply_mem {n : ℕ} (hA : A.IsSymmetric)
    (hn : finrank 𝕜 E = n) (I : Set ℝ) (x : E) :
    hA.spectralProjection hn I x ∈ hA.spectralSubspace hn I := by
  unfold spectralProjection
  exact Submodule.starProjection_apply_mem (hA.spectralSubspace hn I) x

/-- The range of a spectral projection is its spectral subspace. -/
theorem range_spectralProjection {n : ℕ} (hA : A.IsSymmetric)
    (hn : finrank 𝕜 E = n) (I : Set ℝ) :
    (hA.spectralProjection hn I).range = hA.spectralSubspace hn I := by
  unfold spectralProjection
  change
    LinearMap.range (((hA.spectralSubspace hn I).starProjection : E →L[𝕜] E) : E →ₗ[𝕜] E) =
      hA.spectralSubspace hn I
  exact Submodule.range_starProjection (hA.spectralSubspace hn I)

/-- A spectral projection fixes exactly the vectors in its spectral subspace. -/
theorem spectralProjection_eq_self_iff {n : ℕ} (hA : A.IsSymmetric)
    (hn : finrank 𝕜 E = n) (I : Set ℝ) {x : E} :
    hA.spectralProjection hn I x = x ↔ x ∈ hA.spectralSubspace hn I := by
  unfold spectralProjection
  exact Submodule.starProjection_eq_self_iff

/-- A spectral projection is contractive. -/
theorem norm_spectralProjection_le {n : ℕ} (hA : A.IsSymmetric)
    (hn : finrank 𝕜 E = n) (I : Set ℝ) :
    ‖(hA.spectralProjection hn I).toContinuousLinearMap‖ ≤ 1 := by
  unfold spectralProjection
  change ‖(hA.spectralSubspace hn I).starProjection‖ ≤ 1
  exact (hA.spectralSubspace hn I).starProjection_norm_le

/-- Pointwise contractivity of a spectral projection. -/
theorem norm_spectralProjection_apply_le {n : ℕ} (hA : A.IsSymmetric)
    (hn : finrank 𝕜 E = n) (I : Set ℝ) (x : E) :
    ‖hA.spectralProjection hn I x‖ ≤ ‖x‖ := by
  unfold spectralProjection
  exact Submodule.norm_starProjection_apply_le (hA.spectralSubspace hn I) x

/-- A centered self-adjoint operator preserves each spectral subspace. -/
theorem shifted_apply_mem_spectralSubspace {n : ℕ} (hA : A.IsSymmetric)
    (hn : finrank 𝕜 E = n) (I : Set ℝ) (c : ℝ) {x : E}
    (hx : x ∈ hA.spectralSubspace hn I) :
    (A - ((c : 𝕜) • LinearMap.id : E →ₗ[𝕜] E)) x ∈
      hA.spectralSubspace hn I := by
  let S : Submodule 𝕜 E := hA.spectralSubspace hn I
  exact S.sub_mem (hA.map_mem_spectralSubspace hn I hx) (S.smul_mem (c : 𝕜) hx)

/--
HDP Lemma 4.1.16 in the centered form used in the proof.

If the selected eigenvalues of `A` lie within distance `r` of `c`, while the selected
eigenvalues of `B` are at least distance `r + δ` from `c`, then the product of the corresponding
spectral projections satisfies `‖QP‖ ≤ ‖A - B‖ / δ`.
-/
theorem davisKahan_spectralProjection_centered {n : ℕ}
    (hA : A.IsSymmetric) (hB : B.IsSymmetric) (hn : finrank 𝕜 E = n)
    {I J : Set ℝ} {c r δ : ℝ} (hr : 0 ≤ r) (hδ : 0 < δ)
    (hI : ∀ i : Fin n, hA.eigenvalues hn i ∈ I →
      |hA.eigenvalues hn i - c| ≤ r)
    (hJ : ∀ j : Fin n, hB.eigenvalues hn j ∈ J →
      r + δ ≤ |hB.eigenvalues hn j - c|) :
    ‖(hB.spectralProjection hn J).toContinuousLinearMap.comp
        (hA.spectralProjection hn I).toContinuousLinearMap‖ ≤
      ‖(A - B).toContinuousLinearMap‖ / δ := by
  let P : E →L[𝕜] E := (hA.spectralProjection hn I).toContinuousLinearMap
  let Q : E →L[𝕜] E := (hB.spectralProjection hn J).toContinuousLinearMap
  let S : E →L[𝕜] E := Q.comp P
  let TA : E →L[𝕜] E :=
    (A - ((c : 𝕜) • LinearMap.id : E →ₗ[𝕜] E)).toContinuousLinearMap
  let TB : E →L[𝕜] E :=
    (B - ((c : 𝕜) • LinearMap.id : E →ₗ[𝕜] E)).toContinuousLinearMap
  let H : E →L[𝕜] E := (B - A).toContinuousLinearMap
  have hgap_nonneg : 0 ≤ r + δ := add_nonneg hr hδ.le
  have hSrange : ∀ x : E, S x ∈ hB.spectralSubspace hn J := by
    intro x
    exact hB.spectralProjection_apply_mem hn J (P x)
  have hlower :
      (r + δ) * ‖S‖ ≤ ‖TB.comp S‖ := by
    simpa [TB] using
      hB.mul_norm_le_norm_shifted_comp_of_forall_mem_spectralSubspace
        hn hgap_nonneg hJ S hSrange
  have hHnorm : ‖H‖ = ‖(A - B).toContinuousLinearMap‖ := by
    simpa [H] using norm_toContinuousLinearMap_sub_rev A B
  have hHterm :
      ‖Q.comp (H.comp P)‖ ≤ ‖(A - B).toContinuousLinearMap‖ := by
    refine ContinuousLinearMap.opNorm_le_bound _ (norm_nonneg _) ?_
    intro x
    have hQle : ‖Q ((H.comp P) x)‖ ≤ ‖(H.comp P) x‖ := by
      simpa [Q] using hB.norm_spectralProjection_apply_le hn J ((H.comp P) x)
    have hHle : ‖(H.comp P) x‖ ≤ ‖H‖ * ‖P x‖ := H.le_opNorm (P x)
    have hPle : ‖P x‖ ≤ ‖x‖ := by
      simpa [P] using hA.norm_spectralProjection_apply_le hn I x
    calc
      ‖(Q.comp (H.comp P)) x‖ = ‖Q ((H.comp P) x)‖ := rfl
      _ ≤ ‖(H.comp P) x‖ := hQle
      _ ≤ ‖H‖ * ‖P x‖ := hHle
      _ ≤ ‖H‖ * ‖x‖ := mul_le_mul_of_nonneg_left hPle (norm_nonneg H)
      _ = ‖(A - B).toContinuousLinearMap‖ * ‖x‖ := by rw [hHnorm]
  have hTAfactor : Q.comp (TA.comp P) = S.comp (TA.comp P) := by
    ext x
    have hxP : P x ∈ hA.spectralSubspace hn I := by
      simpa [P] using hA.spectralProjection_apply_mem hn I x
    have hTAmem :
        (A - ((c : 𝕜) • LinearMap.id : E →ₗ[𝕜] E)) (P x) ∈
          hA.spectralSubspace hn I :=
      hA.shifted_apply_mem_spectralSubspace hn I c hxP
    have hfix :
        hA.spectralProjection hn I
            ((A - ((c : 𝕜) • LinearMap.id : E →ₗ[𝕜] E)) (P x)) =
          (A - ((c : 𝕜) • LinearMap.id : E →ₗ[𝕜] E)) (P x) :=
      (hA.spectralProjection_eq_self_iff hn I).mpr hTAmem
    change
      Q ((A - ((c : 𝕜) • LinearMap.id : E →ₗ[𝕜] E)) (P x)) =
        Q (hA.spectralProjection hn I
          ((A - ((c : 𝕜) • LinearMap.id : E →ₗ[𝕜] E)) (P x)))
    exact congrArg Q hfix.symm
  have hTAop : ‖TA.comp P‖ ≤ r := by
    simpa [TA, P] using hA.norm_shifted_comp_spectralProjection_le hn hr hI
  have hTAterm :
      ‖Q.comp (TA.comp P)‖ ≤ r * ‖S‖ := by
    rw [hTAfactor]
    calc
      ‖S.comp (TA.comp P)‖ ≤ ‖S‖ * ‖TA.comp P‖ :=
        ContinuousLinearMap.opNorm_comp_le S (TA.comp P)
      _ ≤ ‖S‖ * r := mul_le_mul_of_nonneg_left hTAop (norm_nonneg S)
      _ = r * ‖S‖ := by ring
  have hdecomp :
      TB.comp S = Q.comp (H.comp P) + Q.comp (TA.comp P) := by
    ext x
    have hcommB := LinearMap.congr_fun
      (hB.shifted_comp_spectralProjection_eq_spectralProjection_comp_shifted hn J c)
      (P x)
    change
      (B - ((c : 𝕜) • LinearMap.id : E →ₗ[𝕜] E))
          (hB.spectralProjection hn J (P x)) =
        hB.spectralProjection hn J
            ((B - ((c : 𝕜) • LinearMap.id : E →ₗ[𝕜] E)) (P x)) at hcommB
    change
      (B - ((c : 𝕜) • LinearMap.id : E →ₗ[𝕜] E))
          (hB.spectralProjection hn J (P x)) =
        hB.spectralProjection hn J ((B - A) (P x)) +
          hB.spectralProjection hn J
            ((A - ((c : 𝕜) • LinearMap.id : E →ₗ[𝕜] E)) (P x))
    rw [hcommB, ← map_add]
    congr 1
    simp [sub_eq_add_neg, add_assoc]
  have hupper :
      ‖TB.comp S‖ ≤ ‖(A - B).toContinuousLinearMap‖ + r * ‖S‖ := by
    rw [hdecomp]
    exact (norm_add_le _ _).trans (add_le_add hHterm hTAterm)
  have hdeltaS : δ * ‖S‖ ≤ ‖(A - B).toContinuousLinearMap‖ := by
    nlinarith
  have htarget : ‖S‖ * δ ≤ ‖(A - B).toContinuousLinearMap‖ := by
    nlinarith
  have hmain := (le_div_iff₀ hδ).mpr htarget
  simpa [S, P, Q] using hmain

/--
HDP Lemma 4.1.16, Davis-Kahan inequality for spectral projections, in the closed-interval
normalization used in the proof.  The interval `I` is `[c - r, c + r]`, and `J` is any set
`δ`-separated from it.
-/
theorem davisKahan_spectralProjection_closedInterval_hdp {n : ℕ}
    (hA : A.IsSymmetric) (hB : B.IsSymmetric) (hn : finrank 𝕜 E = n)
    {J : Set ℝ} {c r δ : ℝ} (hr : 0 ≤ r) (hδ : 0 < δ)
    (hsep : Set.SeparatedBy δ (Set.Icc (c - r) (c + r)) J) :
    ‖(hB.spectralProjection hn J).toContinuousLinearMap.comp
        (hA.spectralProjection hn (Set.Icc (c - r) (c + r))).toContinuousLinearMap‖ ≤
      ‖(A - B).toContinuousLinearMap‖ / δ := by
  refine davisKahan_spectralProjection_centered (A := A) (B := B) (hA := hA) (hB := hB)
    (hn := hn) (I := Set.Icc (c - r) (c + r)) (J := J) (c := c) (r := r)
    (δ := δ) hr hδ ?_ ?_
  · intro i hi
    have h := (Set.mem_Icc_iff_abs_le (x := c)
      (y := hA.eigenvalues hn i) (z := r)).mpr hi
    simpa [abs_sub_comm] using h
  · intro j hj
    let y : ℝ := hB.eigenvalues hn j
    have hleft_mem : c - r ∈ Set.Icc (c - r) (c + r) := by
      constructor
      · rfl
      · linarith
    have hright_mem : c + r ∈ Set.Icc (c - r) (c + r) := by
      constructor
      · linarith
      · rfl
    have hleft_sep : δ ≤ |(c - r) - y| := hsep hleft_mem hj
    have hright_sep : δ ≤ |(c + r) - y| := hsep hright_mem hj
    by_cases hyleft : y < c - r
    · have hyc : |y - c| = c - y := by
        rw [abs_of_nonpos (sub_nonpos.mpr (hyleft.le.trans (sub_le_self c hr)))]
        ring
      have hleft_abs : |(c - r) - y| = (c - r) - y := by
        rw [abs_of_nonneg]
        exact sub_nonneg.mpr hyleft.le
      rw [hleft_abs] at hleft_sep
      rw [hyc]
      linarith
    · by_cases hyright : c + r < y
      · have hyc : |y - c| = y - c := by
          rw [abs_of_nonneg]
          exact sub_nonneg.mpr ((le_add_of_nonneg_right hr).trans hyright.le)
        have hright_abs : |(c + r) - y| = y - (c + r) := by
          rw [abs_of_nonpos (sub_nonpos.mpr hyright.le)]
          ring
        rw [hright_abs] at hright_sep
        rw [hyc]
        linarith
      · have hyI : y ∈ Set.Icc (c - r) (c + r) :=
          ⟨le_of_not_gt hyleft, le_of_not_gt hyright⟩
        have hzero : δ ≤ |y - y| := hsep hyI hj
        simp at hzero
        nlinarith

/--
HDP Lemma 4.1.16, Davis-Kahan inequality for spectral projections.  If `I` and `J`
are `δ`-separated real sets and `I` is an interval, then the spectral projections onto the
eigenvectors with eigenvalues in `I` and `J` satisfy `‖QP‖ ≤ ‖A - B‖ / δ`.
-/
theorem davisKahan_spectralProjection_hdp {n : ℕ}
    (hA : A.IsSymmetric) (hB : B.IsSymmetric) (hn : finrank 𝕜 E = n)
    {I J : Set ℝ} {δ : ℝ} (hI_interval : Set.OrdConnected I) (hδ : 0 < δ)
    (hsep : Set.SeparatedBy δ I J) :
    ‖(hB.spectralProjection hn J).toContinuousLinearMap.comp
        (hA.spectralProjection hn I).toContinuousLinearMap‖ ≤
      ‖(A - B).toContinuousLinearMap‖ / δ := by
  classical
  let S : Finset ℝ :=
    (Finset.univ.filter fun i : Fin n => hA.eigenvalues hn i ∈ I).image
      (fun i : Fin n => hA.eigenvalues hn i)
  by_cases hS : S.Nonempty
  · let a : ℝ := S.min' hS
    let b : ℝ := S.max' hS
    have haS : a ∈ S := by
      simpa [a] using S.min'_mem hS
    have hbS : b ∈ S := by
      simpa [b] using S.max'_mem hS
    have haI : a ∈ I := by
      rcases Finset.mem_image.mp haS with ⟨i, hi, hia⟩
      rw [← hia]
      exact (Finset.mem_filter.mp hi).2
    have hbI : b ∈ I := by
      rcases Finset.mem_image.mp hbS with ⟨i, hi, hib⟩
      rw [← hib]
      exact (Finset.mem_filter.mp hi).2
    have hab : a ≤ b := by
      simpa [a, b] using S.min'_le_max' hS
    have hIcc_subset : Set.Icc a b ⊆ I :=
      hI_interval.out haI hbI
    have hr : 0 ≤ (b - a) / 2 := by
      nlinarith
    refine davisKahan_spectralProjection_centered (A := A) (B := B) (hA := hA)
      (hB := hB) (hn := hn) (I := I) (J := J) (c := (a + b) / 2)
      (r := (b - a) / 2) (δ := δ) hr hδ ?_ ?_
    · intro i hi
      have hiS : hA.eigenvalues hn i ∈ S := by
        exact Finset.mem_image.mpr ⟨i, by simp [hi], rfl⟩
      have hai : a ≤ hA.eigenvalues hn i := by
        simpa [a] using S.min'_le (hA.eigenvalues hn i) hiS
      have hib : hA.eigenvalues hn i ≤ b := by
        simpa [b] using S.le_max' (hA.eigenvalues hn i) hiS
      rw [abs_le]
      constructor <;> nlinarith
    · intro j hj
      let y : ℝ := hB.eigenvalues hn j
      have hleft_sep : δ ≤ |a - y| := hsep haI hj
      have hright_sep : δ ≤ |b - y| := hsep hbI hj
      by_cases hyleft : y < a
      · have hyc : |y - (a + b) / 2| = (a + b) / 2 - y := by
          calc
            |y - (a + b) / 2| = -(y - (a + b) / 2) :=
              abs_of_nonpos (sub_nonpos.mpr (by nlinarith [hyleft.le, hab]))
            _ = (a + b) / 2 - y := by ring
        have hleft_abs : |a - y| = a - y := by
          exact abs_of_nonneg (sub_nonneg.mpr hyleft.le)
        rw [hleft_abs] at hleft_sep
        rw [hyc]
        nlinarith
      · by_cases hyright : b < y
        · have hyc : |y - (a + b) / 2| = y - (a + b) / 2 := by
            exact abs_of_nonneg (sub_nonneg.mpr (by nlinarith [hyright.le, hab]))
          have hright_abs : |b - y| = y - b := by
            calc
              |b - y| = -(b - y) := abs_of_nonpos (sub_nonpos.mpr hyright.le)
              _ = y - b := by ring
          rw [hright_abs] at hright_sep
          rw [hyc]
          nlinarith
        · have hyIcc : y ∈ Set.Icc a b := ⟨le_of_not_gt hyleft, le_of_not_gt hyright⟩
          have hyI : y ∈ I := hIcc_subset hyIcc
          have hzero : δ ≤ |y - y| := hsep hyI hj
          simp at hzero
          nlinarith
  · have hnot : ∀ i : Fin n, hA.eigenvalues hn i ∉ I := by
      intro i hi
      exact hS ⟨hA.eigenvalues hn i,
        Finset.mem_image.mpr ⟨i, by simp [hi], rfl⟩⟩
    have hsub : hA.spectralSubspace hn I = ⊥ :=
      hA.spectralSubspace_eq_bot_of_forall_eigenvalues_notMem hn hnot
    have hPzero : hA.spectralProjection hn I = 0 := by
      ext x
      simp [spectralProjection, hsub]
    have hcomp_zero :
        (hB.spectralProjection hn J).toContinuousLinearMap.comp
            (hA.spectralProjection hn I).toContinuousLinearMap = 0 := by
      ext x
      simp [hPzero]
    rw [hcomp_zero, norm_zero]
    exact div_nonneg (norm_nonneg _) hδ.le

/--
One side of Weyl's eigenvalue inequality: the `i`th decreasing eigenvalue of `A` is at most
the corresponding eigenvalue of `B`, up to the operator norm of `A - B`.
-/
theorem eigenvalues_sub_le_opNorm {n : ℕ} (hA : A.IsSymmetric) (hB : B.IsSymmetric)
    (hn : finrank 𝕜 E = n) (i : Fin n) :
    hA.eigenvalues hn i - hB.eigenvalues hn i ≤
      ‖(A - B).toContinuousLinearMap‖ := by
  let L : Submodule 𝕜 E := hA.leadingEigenSubspace hn (Nat.succ_le_of_lt i.2)
  have hLdim : finrank 𝕜 L = i.1 + 1 := by
    simpa [L] using hA.finrank_leadingEigenSubspace hn (Nat.succ_le_of_lt i.2)
  have hbd :
      BddAbove (Set.range fun U : { U : Submodule 𝕜 E // finrank 𝕜 U = i.1 + 1 } =>
        minRayleighQuotientOn B (U : Submodule 𝕜 E)) := by
    refine ⟨‖B.toContinuousLinearMap‖, ?_⟩
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
      minRayleighQuotientOn B (U : Submodule 𝕜 E) ≤ rayleighQuotient B (x : E) :=
        minRayleighQuotientOn_le_of_mem B x.property hx0
      _ ≤ ‖B.toContinuousLinearMap‖ := by
        rw [B.rayleighQuotient_toContinuousLinearMap (x : E)]
        exact le_trans (le_abs_self _)
          (B.toContinuousLinearMap.rayleighQuotient_le_norm (x : E))
  have hL_le_cf :
      minRayleighQuotientOn B L ≤ courantFischerMaxMin B (i.1 + 1) := by
    unfold courantFischerMaxMin
    exact le_ciSup hbd ⟨L, hLdim⟩
  have hA_le_min :
      hA.eigenvalues hn i - ‖(A - B).toContinuousLinearMap‖ ≤
        minRayleighQuotientOn B L := by
    have hLpos : 0 < finrank 𝕜 L := by
      rw [hLdim]
      omega
    haveI : Nontrivial L := Module.nontrivial_of_finrank_pos hLpos
    obtain ⟨x0, hx0L⟩ := exists_ne (0 : L)
    have hx0 : (x0 : E) ≠ 0 := by
      intro h
      exact hx0L (Subtype.ext h)
    haveI : Nonempty { x : L // (x : E) ≠ 0 } := ⟨⟨x0, hx0⟩⟩
    unfold minRayleighQuotientOn
    refine le_ciInf fun x => ?_
    have hxLeading :
        (x : E) ∈ hA.leadingEigenSubspace hn (Nat.succ_le_of_lt i.2) := x.val.property
    have hA_le_ray :
        hA.eigenvalues hn i ≤ rayleighQuotient A (x : E) :=
      hA.eigenvalues_le_rayleighQuotient_of_mem_leadingEigenSubspace hn i hxLeading x.property
    have hpert := abs_rayleighQuotient_sub_le_opNorm A B (x : E)
    have hray :
        rayleighQuotient A (x : E) ≤
          rayleighQuotient B (x : E) + ‖(A - B).toContinuousLinearMap‖ := by
      nlinarith [le_of_abs_le hpert]
    nlinarith
  have hcf := hB.eigenvalues_eq_courantFischerMaxMin_succ hn i
  nlinarith

/-- HDP Lemma 4.1.14, Weyl inequality for decreasingly ordered eigenvalues. -/
theorem abs_eigenvalues_sub_le_opNorm {n : ℕ} (hA : A.IsSymmetric) (hB : B.IsSymmetric)
    (hn : finrank 𝕜 E = n) (i : Fin n) :
    |hA.eigenvalues hn i - hB.eigenvalues hn i| ≤
      ‖(A - B).toContinuousLinearMap‖ := by
  rw [abs_sub_le_iff]
  constructor
  · exact hA.eigenvalues_sub_le_opNorm hB hn i
  · have h := hB.eigenvalues_sub_le_opNorm hA hn i
    rw [norm_toContinuousLinearMap_sub_rev A B] at h
    exact h

section RealDavisKahan

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] [FiniteDimensional ℝ V]
variable {A0 B0 : V →ₗ[ℝ] V}

/--
Davis-Kahan eigenvector perturbation in projection form.  If the `k`th eigenvalue of `A` is
separated from all other eigenvalues by at least `δ`, then the component of `u_k(A)` orthogonal
to `v_k(B)` is bounded by `2 ‖A - B‖ / δ`.
-/
theorem davisKahan_eigenvector_projection_of_gap_le {n : ℕ}
    (hA : A0.IsSymmetric) (hB : B0.IsSymmetric) (hn : finrank ℝ V = n)
    (k : Fin n) {δ : ℝ} (hδ : 0 < δ)
    (hgap : ∀ i : Fin n, i ≠ k →
      δ ≤ |hA.eigenvalues hn k - hA.eigenvalues hn i|) :
    ‖(((ℝ ∙ hB.eigenvectorBasis hn k)ᗮ).starProjection
        (hA.eigenvectorBasis hn k) : V)‖ ≤
      2 * ‖(A0 - B0).toContinuousLinearMap‖ / δ := by
  let eps : ℝ := ‖(A0 - B0).toContinuousLinearMap‖
  let u : V := hA.eigenvectorBasis hn k
  let v : V := hB.eigenvectorBasis hn k
  let K : Submodule ℝ V := ℝ ∙ v
  by_cases hlarge : δ ≤ 2 * eps
  · have hu_norm : ‖u‖ = 1 := by
      simp [u]
    have hproj_le_one : ‖Kᗮ.starProjection u‖ ≤ 1 := by
      exact (Submodule.norm_starProjection_apply_le (K := Kᗮ) u).trans_eq hu_norm
    have hone_le : 1 ≤ 2 * eps / δ := by
      exact (le_div_iff₀ hδ).mpr (by simpa using hlarge)
    simpa [eps, u, v, K] using hproj_le_one.trans hone_le
  · have hsmall : 2 * eps < δ := lt_of_not_ge hlarge
    have heps_lt_half : eps < δ / 2 := by nlinarith
    let lam : ℝ := hA.eigenvalues hn k
    let I : Set ℝ := {lam}
    let J : Set ℝ := {mu : ℝ | mu ≠ hB.eigenvalues hn k}
    have hB_distinct : ∀ j : Fin n, j ≠ k →
        hB.eigenvalues hn j ≠ hB.eigenvalues hn k := by
      intro j hjne hmu
      have hgap_j : δ ≤ |hA.eigenvalues hn k - hA.eigenvalues hn j| :=
        hgap j hjne
      have hwj : |hA.eigenvalues hn j - hB.eigenvalues hn j| ≤ eps := by
        simpa [eps] using hA.abs_eigenvalues_sub_le_opNorm hB hn j
      have hwk : |hA.eigenvalues hn k - hB.eigenvalues hn k| ≤ eps := by
        simpa [eps] using hA.abs_eigenvalues_sub_le_opNorm hB hn k
      have htri :
          |hA.eigenvalues hn k - hA.eigenvalues hn j| ≤
            |hA.eigenvalues hn k - hB.eigenvalues hn k| +
              |hA.eigenvalues hn j - hB.eigenvalues hn j| := by
        simpa [hmu, abs_sub_comm] using
          abs_sub_le (hA.eigenvalues hn k) (hB.eigenvalues hn k)
            (hA.eigenvalues hn j)
      nlinarith
    have hJdist : ∀ j : Fin n, hB.eigenvalues hn j ∈ J →
        δ / 2 ≤ |hB.eigenvalues hn j - lam| := by
      intro j hj
      have hjne : j ≠ k := by
        intro h
        exact hj (by simp [h])
      have hgap_j : δ ≤ |hA.eigenvalues hn k - hA.eigenvalues hn j| :=
        hgap j hjne
      have hwj : |hA.eigenvalues hn j - hB.eigenvalues hn j| ≤ eps := by
        simpa [eps] using hA.abs_eigenvalues_sub_le_opNorm hB hn j
      have htri :
          |hA.eigenvalues hn k - hA.eigenvalues hn j| ≤
            |hA.eigenvalues hn j - hB.eigenvalues hn j| +
              |hB.eigenvalues hn j - hA.eigenvalues hn k| := by
        have h := abs_sub_le (hA.eigenvalues hn k) (hB.eigenvalues hn j)
          (hA.eigenvalues hn j)
        simpa [abs_sub_comm, add_comm, add_left_comm, add_assoc] using h
      dsimp [lam]
      nlinarith
    have hIzero : ∀ i : Fin n, hA.eigenvalues hn i ∈ I →
        |hA.eigenvalues hn i - lam| ≤ 0 := by
      intro i hi
      rw [Set.mem_singleton_iff] at hi
      rw [hi, sub_self, abs_zero]
    have hproj_bound :
        ‖(hB.spectralProjection hn J).toContinuousLinearMap.comp
            (hA.spectralProjection hn I).toContinuousLinearMap‖ ≤ eps / (δ / 2) := by
      simpa [eps] using
        (davisKahan_spectralProjection_centered (A := A0) (B := B0)
          (hA := hA) (hB := hB) (hn := hn) (I := I) (J := J)
          (c := lam) (r := 0) (δ := δ / 2) (by norm_num) (half_pos hδ)
          hIzero (by simpa [zero_add] using hJdist))
    have hu_mem : u ∈ hA.spectralSubspace hn I := by
      exact hA.eigenvectorBasis_mem_spectralSubspace hn (by simp [I, lam])
    have hPu : hA.spectralProjection hn I u = u :=
      (hA.spectralProjection_eq_self_iff hn I).mpr hu_mem
    have hsubB : hB.spectralSubspace hn J = Kᗮ := by
      simpa [K, v, J] using
        hB.spectralSubspace_eq_orthogonal_span_eigenvectorBasis hn
          (I := J) (k := k) (by simp [J])
          (by
            intro j hjne
            exact hB_distinct j hjne)
    have hQeq : hB.spectralProjection hn J u = Kᗮ.starProjection u := by
      simp [spectralProjection, hsubB, K]
    have hcomp_apply :
        ((hB.spectralProjection hn J).toContinuousLinearMap.comp
          (hA.spectralProjection hn I).toContinuousLinearMap) u =
            hB.spectralProjection hn J u := by
      change hB.spectralProjection hn J (hA.spectralProjection hn I u) =
        hB.spectralProjection hn J u
      rw [hPu]
    have hu_norm : ‖u‖ = 1 := by
      simp [u]
    have happ_le :
        ‖hB.spectralProjection hn J u‖ ≤
          ‖(hB.spectralProjection hn J).toContinuousLinearMap.comp
            (hA.spectralProjection hn I).toContinuousLinearMap‖ := by
      have hle :=
        ((hB.spectralProjection hn J).toContinuousLinearMap.comp
          (hA.spectralProjection hn I).toContinuousLinearMap).le_opNorm u
      rw [hcomp_apply] at hle
      simpa [hu_norm] using hle
    calc
      ‖Kᗮ.starProjection u‖ = ‖hB.spectralProjection hn J u‖ := by rw [hQeq]
      _ ≤ ‖(hB.spectralProjection hn J).toContinuousLinearMap.comp
            (hA.spectralProjection hn I).toContinuousLinearMap‖ := happ_le
      _ ≤ eps / (δ / 2) := hproj_bound
      _ = 2 * eps / δ := by
        field_simp [hδ.ne']

/-- Davis-Kahan eigenvector angle inequality under a lower-bound gap hypothesis. -/
theorem davisKahan_eigenvector_angle_of_gap_le {n : ℕ}
    (hA : A0.IsSymmetric) (hB : B0.IsSymmetric) (hn : finrank ℝ V = n)
    (k : Fin n) {δ : ℝ} (hδ : 0 < δ)
    (hgap : ∀ i : Fin n, i ≠ k →
      δ ≤ |hA.eigenvalues hn k - hA.eigenvalues hn i|) :
    Real.sin (InnerProductGeometry.angle
        (hA.eigenvectorBasis hn k) (hB.eigenvectorBasis hn k)) ≤
      2 * ‖(A0 - B0).toContinuousLinearMap‖ / δ := by
  rw [sin_angle_eq_norm_orthogonal_starProjection_span_singleton]
  · exact hA.davisKahan_eigenvector_projection_of_gap_le hB hn k hδ hgap
  · simp
  · simp

/--
HDP Theorem 4.1.15, Davis-Kahan eigenvector angle inequality.  The gap hypothesis is stated as
`δ = min_{i : i ≠ k} |λ_k - λ_i|`, encoded by `IsLeast` for the finite set of non-`k` gaps.
-/
theorem davisKahan_eigenvector_angle_hdp {n : ℕ}
    (hA : A0.IsSymmetric) (hB : B0.IsSymmetric) (hn : finrank ℝ V = n)
    (k : Fin n) {δ : ℝ} (hδ : 0 < δ)
    (hgap : IsLeast
      ((fun i : { i : Fin n // i ≠ k } =>
        |hA.eigenvalues hn k - hA.eigenvalues hn i.1|) '' Set.univ) δ) :
    Real.sin (InnerProductGeometry.angle
        (hA.eigenvectorBasis hn k) (hB.eigenvectorBasis hn k)) ≤
      2 * ‖(A0 - B0).toContinuousLinearMap‖ / δ := by
  exact hA.davisKahan_eigenvector_angle_of_gap_le hB hn k hδ fun i hi =>
    hgap.2 ⟨⟨i, hi⟩, Set.mem_univ _, rfl⟩

end RealDavisKahan

end IsSymmetric

/--
One side of Weyl's singular-value inequality: the `i`th decreasing singular value of `A` is at
most the corresponding singular value of `B`, up to the operator norm of `A - B`.
-/
theorem singularValues_sub_le_opNorm
    (A B : E →ₗ[𝕜] F) {n : ℕ} (hn : finrank 𝕜 E = n) (i : Fin n) :
    A.singularValues i - B.singularValues i ≤ ‖(A - B).toContinuousLinearMap‖ := by
  obtain ⟨C, hCrank, hCerr, _⟩ :=
    eckartYoungMirsky_opNorm_rank_le_attained B hn i.2
  have hA_lower :
      A.singularValues i ≤ ‖(A - C).toContinuousLinearMap‖ :=
    eckartYoungMirsky_opNorm_lower_bound A C hn i.2 hCrank
  have htri :
      ‖(A - C).toContinuousLinearMap‖ ≤
        ‖(A - B).toContinuousLinearMap‖ + ‖(B - C).toContinuousLinearMap‖ := by
    have hdecomp :
        (A - C).toContinuousLinearMap =
          (A - B).toContinuousLinearMap + (B - C).toContinuousLinearMap := by
      ext x
      simp [sub_eq_add_neg, add_assoc]
    rw [hdecomp]
    exact norm_add_le _ _
  nlinarith

/-- HDP Lemma 4.1.14, Weyl inequality for decreasingly ordered singular values. -/
theorem abs_singularValues_sub_le_opNorm
    (A B : E →ₗ[𝕜] F) {n : ℕ} (hn : finrank 𝕜 E = n) (i : Fin n) :
    |A.singularValues i - B.singularValues i| ≤ ‖(A - B).toContinuousLinearMap‖ := by
  rw [abs_sub_le_iff]
  constructor
  · exact singularValues_sub_le_opNorm A B hn i
  · have h := singularValues_sub_le_opNorm B A hn i
    rw [norm_toContinuousLinearMap_sub_rev A B] at h
    exact h

end LinearMap
