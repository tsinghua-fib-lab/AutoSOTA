/-
Copyright (c) 2026 Yuanhe Zhang. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Yuanhe Zhang, Jason D. Lee, Fanghui Liu
-/
import SLT.MatrixInfra.CourantFischer

/-!
# Eckart-Young-Mirsky Infrastructure

This file starts the formalization of HDP Theorem 4.1.13, the
Eckart-Young-Mirsky theorem.

## Main definitions

This file uses Mathlib's `LinearMap.singularValues` and the operator norm of
`LinearMap.toContinuousLinearMap`.

* `LinearMap.eckartYoungMirskyApproximant`: the rank-`k` truncation obtained by projecting the
  domain onto the leading Gram eigenspace.

## Main results

* `LinearMap.eckartYoungMirsky_opNorm_lower_bound`: every rank-`k` approximation has operator
  error at least the next singular value.
* `LinearMap.eckartYoungMirskyApproximant_opNorm_error_eq`: the projection truncation attains
  that lower bound.
* `LinearMap.eckartYoungMirsky_opNorm_rank_le_attained`: the operator-norm EYM theorem, stated
  for rank at most `k`.
* `LinearMap.IsSymmetric.leadingEigenSubspace_orthogonal_le_trailingEigenSubspace`: the
  orthogonal complement of the leading eigenspace is contained in the corresponding trailing
  eigenspace.
-/

open Module

noncomputable section

namespace LinearMap

variable {𝕜 E F : Type*} [RCLike 𝕜]
variable [NormedAddCommGroup E] [InnerProductSpace 𝕜 E] [FiniteDimensional 𝕜 E]
variable [NormedAddCommGroup F] [InnerProductSpace 𝕜 F] [FiniteDimensional 𝕜 F]

namespace IsSymmetric

variable {S : E →ₗ[𝕜] E}

/--
If all coordinates before the cutoff vanish in the sorted eigenbasis, then the vector lies in the
trailing eigenspace from that cutoff.
-/
theorem mem_trailingEigenSubspace_of_repr_eq_zero_lt
    (hS : S.IsSymmetric) {n : ℕ} (hn : finrank 𝕜 E = n) (i : Fin n)
    {x : E} (hzero : ∀ j : Fin n, j.1 < i.1 → (hS.eigenvectorBasis hn).repr x j = 0) :
    x ∈ hS.trailingEigenSubspace hn i := by
  let b := hS.eigenvectorBasis hn
  have hsupp : ↑(b.toBasis.repr x).support ⊆ {j : Fin n | i.1 ≤ j.1} := by
    intro j hj
    by_contra hjlt
    have hzero' : b.toBasis.repr x j = 0 := by
      rw [b.coe_toBasis_repr_apply]
      exact hzero j (Nat.lt_of_not_ge hjlt)
    exact (Finsupp.mem_support_iff.mp hj) hzero'
  have hxspan : x ∈ Submodule.span 𝕜 (b '' {j : Fin n | i.1 ≤ j.1}) :=
    (b.toBasis.mem_span_image (s := {j : Fin n | i.1 ≤ j.1})).2 hsupp
  unfold trailingEigenSubspace
  refine (Submodule.span_mono ?_) hxspan
  rintro y ⟨j, hj, rfl⟩
  have hjrange : j ∈ Set.range (Fin.natAdd_castLEEmb (Nat.sub_le n i.1)) := by
    rw [Fin.range_natAdd_castLEEmb]
    change n - (n - i.1) ≤ j.1
    rw [Nat.sub_sub_self (Nat.le_of_lt i.2)]
    exact hj
  rcases hjrange with ⟨l, rfl⟩
  exact ⟨l, rfl⟩

/--
The orthogonal complement of the leading eigenspace through index `i - 1` is contained in the
trailing eigenspace from index `i`.
-/
theorem leadingEigenSubspace_orthogonal_le_trailingEigenSubspace
    (hS : S.IsSymmetric) {n : ℕ} (hn : finrank 𝕜 E = n) (i : Fin n) :
    (hS.leadingEigenSubspace hn (Nat.le_of_lt i.2))ᗮ ≤ hS.trailingEigenSubspace hn i := by
  intro x hx
  refine hS.mem_trailingEigenSubspace_of_repr_eq_zero_lt hn i ?_
  intro j hj
  rw [OrthonormalBasis.repr_apply_apply]
  let L : Submodule 𝕜 E := hS.leadingEigenSubspace hn (Nat.le_of_lt i.2)
  have hmem : hS.eigenvectorBasis hn j ∈ L := by
    let jj : Fin i.1 := ⟨j.1, hj⟩
    have hcast : Fin.castLE (Nat.le_of_lt i.2) jj = j := by
      ext
      simp [jj]
    simpa [L, hcast] using
      hS.eigenvectorBasis_mem_leadingEigenSubspace hn (Nat.le_of_lt i.2) jj
  exact (L.mem_orthogonal x).mp hx (hS.eigenvectorBasis hn j) hmem

end IsSymmetric

/--
The canonical rank-`k` approximation used in the operator-norm Eckart-Young-Mirsky theorem:
project the domain onto the leading `k`-dimensional eigenspace of `T†T`, then apply `T`.
-/
noncomputable def eckartYoungMirskyApproximant
    (T : E →ₗ[𝕜] F) {n k : ℕ} (hn : finrank 𝕜 E = n) (hk : k ≤ n) : E →ₗ[𝕜] F :=
  let L : Submodule 𝕜 E := T.isSymmetric_adjoint_comp_self.leadingEigenSubspace hn hk
  haveI : CompleteSpace L := FiniteDimensional.complete 𝕜 L
  T.comp ((L.starProjection : E →L[𝕜] E) : E →ₗ[𝕜] E)

/-- The Eckart-Young-Mirsky approximant has rank at most `k`. -/
theorem finrank_range_eckartYoungMirskyApproximant_le
    (T : E →ₗ[𝕜] F) {n k : ℕ} (hn : finrank 𝕜 E = n) (hk : k ≤ n) :
    finrank 𝕜 (T.eckartYoungMirskyApproximant hn hk).range ≤ k := by
  let L : Submodule 𝕜 E := T.isSymmetric_adjoint_comp_self.leadingEigenSubspace hn hk
  haveI : CompleteSpace L := FiniteDimensional.complete 𝕜 L
  unfold eckartYoungMirskyApproximant
  change finrank 𝕜 (LinearMap.range (T.comp ((L.starProjection : E →L[𝕜] E) : E →ₗ[𝕜] E))) ≤ k
  rw [LinearMap.range_comp, Submodule.range_starProjection]
  exact (Submodule.finrank_map_le T L).trans_eq
    (T.isSymmetric_adjoint_comp_self.finrank_leadingEigenSubspace hn hk)

/--
If `T` has full column rank, then the Eckart-Young-Mirsky approximant has rank exactly `k`.
-/
theorem finrank_range_eckartYoungMirskyApproximant_eq_of_finrank_range_eq
    (T : E →ₗ[𝕜] F) {n k : ℕ} (hn : finrank 𝕜 E = n)
    (hTfull : finrank 𝕜 T.range = n) (hk : k ≤ n) :
    finrank 𝕜 (T.eckartYoungMirskyApproximant hn hk).range = k := by
  let L : Submodule 𝕜 E := T.isSymmetric_adjoint_comp_self.leadingEigenSubspace hn hk
  haveI : CompleteSpace L := FiniteDimensional.complete 𝕜 L
  have hkerdim : finrank 𝕜 T.ker = 0 := by
    have h := T.finrank_range_add_finrank_ker
    rw [hTfull, hn] at h
    omega
  have hinj : Function.Injective T := by
    rw [← LinearMap.ker_eq_bot]
    exact Submodule.finrank_eq_zero.mp hkerdim
  let TL : L →ₗ[𝕜] F := T.comp L.subtype
  have hTLinj : Function.Injective TL := by
    intro x y hxy
    exact Subtype.ext (hinj hxy)
  have hTLrange : LinearMap.range TL = L.map T := by
    ext y
    constructor
    · rintro ⟨x, rfl⟩
      exact ⟨x, x.property, rfl⟩
    · rintro ⟨x, hx, rfl⟩
      exact ⟨⟨x, hx⟩, rfl⟩
  unfold eckartYoungMirskyApproximant
  change finrank 𝕜 (LinearMap.range (T.comp ((L.starProjection : E →L[𝕜] E) : E →ₗ[𝕜] E))) = k
  rw [LinearMap.range_comp, Submodule.range_starProjection]
  calc
    finrank 𝕜 (L.map T) = finrank 𝕜 (LinearMap.range TL) := by rw [hTLrange]
    _ = finrank 𝕜 L := LinearMap.finrank_range_of_inj hTLinj
    _ = k := T.isSymmetric_adjoint_comp_self.finrank_leadingEigenSubspace hn hk

/--
If `k` is at most the rank of `T`, then the Eckart-Young-Mirsky truncation has rank exactly `k`.
This is the rank-exact version needed for HDP Theorem 4.1.13, where `n` denotes the number of
nonzero singular values rather than the ambient domain dimension.
-/
theorem finrank_range_eckartYoungMirskyApproximant_eq_of_le_finrank_range
    (T : E →ₗ[𝕜] F) {d k : ℕ} (hd : finrank 𝕜 E = d)
    (hkdim : k ≤ d) (hk : k ≤ finrank 𝕜 T.range) :
    finrank 𝕜 (T.eckartYoungMirskyApproximant hd hkdim).range = k := by
  by_cases hk0 : k = 0
  · have hle := finrank_range_eckartYoungMirskyApproximant_le T hd hkdim
    omega
  let L : Submodule 𝕜 E := T.isSymmetric_adjoint_comp_self.leadingEigenSubspace hd hkdim
  haveI : CompleteSpace L := FiniteDimensional.complete 𝕜 L
  let TL : L →ₗ[𝕜] F := T.comp L.subtype
  have hkpos : 0 < k := Nat.pos_of_ne_zero hk0
  have hTLinj : Function.Injective TL := by
    intro x y hxy
    apply Subtype.ext
    let z : E := (x : E) - (y : E)
    have hzL : z ∈ L := by
      dsimp [z]
      exact L.sub_mem x.property y.property
    have hTz : T z = 0 := by
      have hsub : TL (x - y) = 0 := by
        rw [map_sub, hxy, sub_self]
      simpa [TL, z] using hsub
    have hz0 : z = 0 := by
      by_contra hz_ne
      let i : Fin d := ⟨k - 1, by omega⟩
      have hi_succ : i.1 + 1 = k := by
        simp [i, Nat.sub_add_cancel hkpos]
      have hsing_pos : 0 < T.singularValues i := by
        rw [T.singularValues_pos_iff_lt_finrank_range]
        have hi_lt_k : (i : ℕ) < k := by
          simp [i, Nat.sub_lt hkpos Nat.zero_lt_one]
        exact hi_lt_k.trans_le hk
      have hsing_le : T.singularValues i ≤ singularQuotient T z := by
        have hzLeading :
            z ∈
              T.isSymmetric_adjoint_comp_self.leadingEigenSubspace hd
                (Nat.succ_le_of_lt i.2) := by
          simpa [L, hi_succ] using hzL
        exact singularValues_le_singularQuotient_of_mem_gram_leadingEigenSubspace
          T hd i hzLeading hz_ne
      have hquot_zero : singularQuotient T z = 0 := by
        unfold singularQuotient
        rw [hTz, norm_zero, zero_div]
      have : 0 < singularQuotient T z := hsing_pos.trans_le hsing_le
      rw [hquot_zero] at this
      exact (lt_irrefl (0 : ℝ)) this
    simpa [z] using sub_eq_zero.mp hz0
  have hTLrange : LinearMap.range TL = L.map T := by
    ext y
    constructor
    · rintro ⟨x, rfl⟩
      exact ⟨x, x.property, rfl⟩
    · rintro ⟨x, hx, rfl⟩
      exact ⟨⟨x, hx⟩, rfl⟩
  unfold eckartYoungMirskyApproximant
  change finrank 𝕜 (LinearMap.range (T.comp ((L.starProjection : E →L[𝕜] E) : E →ₗ[𝕜] E))) = k
  rw [LinearMap.range_comp, Submodule.range_starProjection]
  calc
    finrank 𝕜 (L.map T) = finrank 𝕜 (LinearMap.range TL) := by rw [hTLrange]
    _ = finrank 𝕜 L := LinearMap.finrank_range_of_inj hTLinj
    _ = k := T.isSymmetric_adjoint_comp_self.finrank_leadingEigenSubspace hd hkdim

/--
The lower-bound half of the operator-norm Eckart-Young-Mirsky theorem.

With zero-based singular values, if `B` has rank at most `k`, then the operator-norm error
`‖T - B‖` is at least `T.singularValues k`, the first singular value not captured by a rank-`k`
truncation. This is the min-max obstruction in HDP Theorem 4.1.13.
-/
theorem eckartYoungMirsky_opNorm_lower_bound
    (T B : E →ₗ[𝕜] F) {n k : ℕ} (hn : finrank 𝕜 E = n) (hk : k < n)
    (hB : finrank 𝕜 B.range ≤ k) :
    T.singularValues k ≤ ‖(T - B).toContinuousLinearMap‖ := by
  let i : Fin n := ⟨k, hk⟩
  let L : Submodule 𝕜 E :=
    T.isSymmetric_adjoint_comp_self.leadingEigenSubspace hn (Nat.succ_le_of_lt i.2)
  have hLdim : finrank 𝕜 L = k + 1 := by
    simpa [L, i] using
      T.isSymmetric_adjoint_comp_self.finrank_leadingEigenSubspace hn
        (Nat.succ_le_of_lt i.2)
  have hkerdim : finrank 𝕜 B.ker = n - finrank 𝕜 B.range := by
    have h := B.finrank_range_add_finrank_ker
    rw [hn] at h
    omega
  have hinter :
      finrank 𝕜 E < finrank 𝕜 L + finrank 𝕜 B.ker := by
    rw [hn, hLdim, hkerdim]
    omega
  obtain ⟨x, hxL, hxker, hx0⟩ :=
    Submodule.exists_ne_zero_mem_inf_of_finrank_lt_add_finrank L B.ker hinter
  have hsing_le : T.singularValues k ≤ singularQuotient T x := by
    simpa [i] using
      singularValues_le_singularQuotient_of_mem_gram_leadingEigenSubspace T hn i
        (by simpa [L] using hxL) hx0
  have hquot_le : singularQuotient T x ≤ ‖(T - B).toContinuousLinearMap‖ := by
    unfold singularQuotient
    have hxnorm : 0 < ‖x‖ := norm_pos_iff.mpr hx0
    have hTx : T x = (T - B) x := by
      simp [LinearMap.sub_apply, LinearMap.mem_ker.mp hxker]
    rw [hTx]
    exact (div_le_iff₀ hxnorm).mpr ((T - B).toContinuousLinearMap.le_opNorm x)
  exact hsing_le.trans hquot_le

/-- Exact-rank approximants satisfy the same Eckart-Young-Mirsky lower bound. -/
theorem eckartYoungMirsky_opNorm_lower_bound_of_finrank_range_eq
    (T B : E →ₗ[𝕜] F) {n k : ℕ} (hn : finrank 𝕜 E = n) (hk : k < n)
    (hB : finrank 𝕜 B.range = k) :
    T.singularValues k ≤ ‖(T - B).toContinuousLinearMap‖ :=
  eckartYoungMirsky_opNorm_lower_bound T B hn hk hB.le

/--
The canonical rank-`k` truncation has operator-norm error at most the next singular value.
-/
theorem eckartYoungMirskyApproximant_opNorm_error_le
    (T : E →ₗ[𝕜] F) {n k : ℕ} (hn : finrank 𝕜 E = n) (hk : k < n) :
    ‖(T - T.eckartYoungMirskyApproximant hn (Nat.le_of_lt hk)).toContinuousLinearMap‖ ≤
      T.singularValues k := by
  let i : Fin n := ⟨k, hk⟩
  let L : Submodule 𝕜 E :=
    T.isSymmetric_adjoint_comp_self.leadingEigenSubspace hn (Nat.le_of_lt hk)
  haveI : CompleteSpace L := FiniteDimensional.complete 𝕜 L
  let B : E →ₗ[𝕜] F := T.eckartYoungMirskyApproximant hn (Nat.le_of_lt hk)
  refine ((T - B).toContinuousLinearMap).opNorm_le_bound (T.singularValues_nonneg k) ?_
  intro x
  change ‖(T - B) x‖ ≤ T.singularValues k * ‖x‖
  let y : E := x - L.starProjection x
  have h_apply : (T - B) x = T y := by
    simp [B, eckartYoungMirskyApproximant, L, y, LinearMap.sub_apply, map_sub]
  rw [h_apply]
  by_cases hy0 : y = 0
  · rw [hy0, map_zero, norm_zero]
    exact mul_nonneg (T.singularValues_nonneg k) (norm_nonneg x)
  have hyOrth : y ∈ Lᗮ := by
    simp [y]
  have hyTrailing :
      y ∈ T.isSymmetric_adjoint_comp_self.trailingEigenSubspace hn i := by
    simpa [L, i] using
      T.isSymmetric_adjoint_comp_self.leadingEigenSubspace_orthogonal_le_trailingEigenSubspace
        hn i hyOrth
  have hquot : singularQuotient T y ≤ T.singularValues k := by
    simpa [i] using
      singularQuotient_le_singularValues_of_mem_gram_trailingEigenSubspace T hn i hyTrailing hy0
  have hTy : ‖T y‖ ≤ T.singularValues k * ‖y‖ := by
    unfold singularQuotient at hquot
    exact (div_le_iff₀ (norm_pos_iff.mpr hy0)).mp hquot
  have hnorm_y : ‖y‖ ≤ ‖x‖ := by
    have hproj : Lᗮ.starProjection x = y := by
      simp [y]
    calc
      ‖y‖ = ‖Lᗮ.starProjection x‖ := by rw [← hproj]
      _ ≤ ‖x‖ := Submodule.norm_starProjection_apply_le (K := Lᗮ) x
  exact hTy.trans (mul_le_mul_of_nonneg_left hnorm_y (T.singularValues_nonneg k))

/--
The canonical rank-`k` truncation attains the operator-norm Eckart-Young-Mirsky value.
-/
theorem eckartYoungMirskyApproximant_opNorm_error_eq
    (T : E →ₗ[𝕜] F) {n k : ℕ} (hn : finrank 𝕜 E = n) (hk : k < n) :
    ‖(T - T.eckartYoungMirskyApproximant hn (Nat.le_of_lt hk)).toContinuousLinearMap‖ =
      T.singularValues k := by
  apply le_antisymm
  · exact eckartYoungMirskyApproximant_opNorm_error_le T hn hk
  · exact eckartYoungMirsky_opNorm_lower_bound T
      (T.eckartYoungMirskyApproximant hn (Nat.le_of_lt hk)) hn hk
      (finrank_range_eckartYoungMirskyApproximant_le T hn (Nat.le_of_lt hk))

/--
Operator-norm Eckart-Young-Mirsky theorem in zero-based singular-value indexing: among all maps
with rank at most `k`, the optimal error is `T.singularValues k`, and it is attained by the
projection truncation onto the leading Gram eigenspace.
-/
theorem eckartYoungMirsky_opNorm_rank_le_attained
    (T : E →ₗ[𝕜] F) {n k : ℕ} (hn : finrank 𝕜 E = n) (hk : k < n) :
    ∃ B : E →ₗ[𝕜] F,
      finrank 𝕜 B.range ≤ k ∧
        ‖(T - B).toContinuousLinearMap‖ = T.singularValues k ∧
          ∀ C : E →ₗ[𝕜] F, finrank 𝕜 C.range ≤ k →
            T.singularValues k ≤ ‖(T - C).toContinuousLinearMap‖ := by
  refine ⟨T.eckartYoungMirskyApproximant hn (Nat.le_of_lt hk), ?_, ?_, ?_⟩
  · exact finrank_range_eckartYoungMirskyApproximant_le T hn (Nat.le_of_lt hk)
  · exact eckartYoungMirskyApproximant_opNorm_error_eq T hn hk
  · intro C hC
    exact eckartYoungMirsky_opNorm_lower_bound T C hn hk hC

/--
HDP Theorem 4.1.13, Eckart-Young-Mirsky theorem, in zero-based singular-value indexing.

The hypothesis `finrank 𝕜 T.range = n` says that `T` has rank `n`, equivalently exactly `n`
nonzero singular values. Thus, for every `k ≤ n`, there is an exact-rank-`k` map attaining
operator-norm error `T.singularValues k`, i.e. HDP's `s_{k+1}` with the endpoint convention
`s_{n+1} = 0`. The competitors are also required to have rank exactly `k`, matching the statement
of Theorem 4.1.13.
-/
theorem eckartYoungMirsky_hdp
    (T : E →ₗ[𝕜] F) {n k : ℕ} (hTrank : finrank 𝕜 T.range = n)
    (hk : k ≤ n) :
    ∃ B : E →ₗ[𝕜] F,
      finrank 𝕜 B.range = k ∧
        ‖(T - B).toContinuousLinearMap‖ = T.singularValues k ∧
          ∀ C : E →ₗ[𝕜] F, finrank 𝕜 C.range = k →
            T.singularValues k ≤ ‖(T - C).toContinuousLinearMap‖ := by
  let d := finrank 𝕜 E
  have hd : finrank 𝕜 E = d := rfl
  have hn_le_d : n ≤ d := by
    have h := T.finrank_range_add_finrank_ker
    rw [hTrank] at h
    omega
  by_cases hkn : k < n
  · have hkdim : k ≤ d := hk.trans hn_le_d
    have hkd : k < d := hkn.trans_le hn_le_d
    refine ⟨T.eckartYoungMirskyApproximant hd hkdim, ?_, ?_, ?_⟩
    · exact finrank_range_eckartYoungMirskyApproximant_eq_of_le_finrank_range
        T hd hkdim (by rwa [hTrank])
    · exact eckartYoungMirskyApproximant_opNorm_error_eq T hd hkd
    · intro C hC
      exact eckartYoungMirsky_opNorm_lower_bound_of_finrank_range_eq T C hd hkd hC
  · have hk_eq : (k : ℕ) = n := by omega
    refine ⟨T, ?_, ?_, ?_⟩
    · simpa [hk_eq] using hTrank
    · have hsing : T.singularValues (k : ℕ) = 0 := by
        rw [hk_eq]
        simpa [hTrank] using T.singularValues_finrank_range_self
      simp [hsing]
    · intro C hC
      have hsing : T.singularValues (k : ℕ) = 0 := by
        rw [hk_eq]
        simpa [hTrank] using T.singularValues_finrank_range_self
      simp [hsing]

end LinearMap

namespace Matrix

variable {𝕜 ι κ : Type*} [RCLike 𝕜] [Fintype ι] [Fintype κ] [DecidableEq κ]

/--
HDP Theorem 4.1.13, Eckart-Young-Mirsky theorem, in matrix form.

If an `ι × κ` matrix `A` has rank `n` and `k ≤ n`, then the best operator-norm rank-`k`
approximation error is the zero-based singular value `A.singularValues k`, i.e. HDP's
`s_{k+1}`. The minimum is attained by a matrix of rank exactly `k`, and the competitors are also
required to have rank exactly `k`.
-/
theorem eckartYoungMirsky_hdp
    (A : Matrix ι κ 𝕜) {n k : ℕ}
    (hArank : finrank 𝕜 A.toEuclideanLin.range = n) (hk : k ≤ n) :
    ∃ B : Matrix ι κ 𝕜,
      finrank 𝕜 B.toEuclideanLin.range = k ∧
        ‖(A.toEuclideanLin - B.toEuclideanLin).toContinuousLinearMap‖ = A.singularValues k ∧
          ∀ C : Matrix ι κ 𝕜, finrank 𝕜 C.toEuclideanLin.range = k →
            A.singularValues k ≤
              ‖(A.toEuclideanLin - C.toEuclideanLin).toContinuousLinearMap‖ := by
  obtain ⟨Blin, hBrank, hBerr, hBmin⟩ :=
    LinearMap.eckartYoungMirsky_hdp A.toEuclideanLin hArank hk
  let B : Matrix ι κ 𝕜 :=
    LinearMap.toMatrix (EuclideanSpace.basisFun κ 𝕜).toBasis
      (EuclideanSpace.basisFun ι 𝕜).toBasis Blin
  have hBlin : B.toEuclideanLin = Blin := by
    rw [Matrix.toEuclideanLin_eq_toLin_orthonormal]
    exact Matrix.toLin_toMatrix (EuclideanSpace.basisFun κ 𝕜).toBasis
      (EuclideanSpace.basisFun ι 𝕜).toBasis Blin
  refine ⟨B, ?_, ?_, ?_⟩
  · rw [hBlin]
    exact hBrank
  · rw [hBlin]
    exact hBerr
  · intro C hCrank
    exact hBmin C.toEuclideanLin hCrank

end Matrix
