/-
Copyright (c) 2026 Yuanhe Zhang. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Yuanhe Zhang, Jason D. Lee, Fanghui Liu
-/
import SLT.SubGaussian
import SLT.Chaining
import SLT.MetricEntropy
import SLT.SeparableSpaceSup

/-!
# Dudley's Entropy Integral Bound

This file proves Dudley's entropy integral bound for sub-Gaussian processes.

## Main results

* `dudley`: The main Dudley theorem - expected supremum bounded by entropy integral

-/

open Set Metric Real MeasureTheory ProbabilityTheory TopologicalSpace
open scoped BigOperators ENNReal

noncomputable section

universe u v

variable {Ω : Type u} [MeasurableSpace Ω]
variable {A : Type v} [PseudoMetricSpace A]

/-!
## Separability-Based Supremum Reduction

Using separability-based argument, we convert the uncountable supremum over s to a
countable supremum over denseSeq.

Key insight: For a set s in a separable space A, the subtype ↥s inherits separability.
We use denseSeq (↥s) to get a countable dense sequence in s.
-/

/-- Dense sequence in the subtype ↥s, derived from totally boundedness.
    This wrapper avoids the need for SeparableSpace instance at definition time. -/
def denseSeqInTB {s : Set A} (hs : TotallyBounded s) (hsne : s.Nonempty) (n : ℕ) : ↥s :=
  letI : Nonempty (↥s) := hsne.to_subtype
  letI : SeparableSpace (↥s) := hs.isSeparable.separableSpace
  denseSeq (↥s) n

/-- For a continuous function on a totally bounded nonempty set, the supremum equals
    the supremum over the dense sequence. -/
lemma sup_subtype_eq_iSup_denseSeq {s : Set A} (hs : TotallyBounded s) (hsne : s.Nonempty)
    {f : A → ℝ} (hcont : Continuous (fun (t : ↥s) => f t.1)) :
    ⨆ (t : ↥s), f t.1 = ⨆ n : ℕ, f (denseSeqInTB hs hsne n).1 := by
  letI : Nonempty (↥s) := hsne.to_subtype
  letI : SeparableSpace (↥s) := hs.isSeparable.separableSpace
  unfold denseSeqInTB
  exact separableSpaceSup_eq_real hcont

omit [MeasurableSpace Ω] in
/-- Pathwise equality: the supremum over s equals supremum over denseSeq for each ω. -/
lemma sup_over_s_eq_iSup_denseSeq_pathwise
    {s : Set A} (hs : TotallyBounded s) (hsne : s.Nonempty)
    {X : A → Ω → ℝ}
    (hcont : ∀ ω, Continuous (fun (t : ↥s) => X t.1 ω)) :
    ∀ ω, (⨆ (t : ↥s), X t.1 ω) = ⨆ n : ℕ, X (denseSeqInTB hs hsne n).1 ω := by
  intro ω
  have h := sup_subtype_eq_iSup_denseSeq hs hsne (f := fun a => X a ω) (hcont ω)
  convert h using 2

omit [PseudoMetricSpace A] in
/-- The biSup over s equals iSup over subtype for ℝ-valued functions with a zero.
    Works for both bounded and unbounded cases (using Real.iSup_of_not_bddAbove). -/
lemma biSup_eq_iSup_subtype_real {s : Set A} {f : A → ℝ}
    (hne : s.Nonempty) (hzero : ∃ t ∈ s, f t = 0) :
    ⨆ t ∈ s, f t = ⨆ (t : ↥s), f t.1 := by
  haveI : Nonempty ↥s := hne.to_subtype
  haveI : Nonempty A := ⟨hne.some⟩
  obtain ⟨t₀, ht₀, hft₀⟩ := hzero

  -- Check if bounded above
  by_cases hbdd : BddAbove (f '' s)
  · -- Bounded case: use the conditional lemmas
    have hbdd_subtype : BddAbove (range (fun (u : ↥s) => f u.1)) := by
      obtain ⟨M, hM⟩ := hbdd
      use M
      rintro _ ⟨⟨u, hu⟩, rfl⟩
      exact hM ⟨u, hu, rfl⟩
    have hbdd_biSup : BddAbove (range (fun t => ⨆ (_ : t ∈ s), f t)) := by
      obtain ⟨M, hM⟩ := hbdd
      use max M 0
      rintro _ ⟨t, rfl⟩
      by_cases ht' : t ∈ s
      · simp only [ciSup_pos ht']
        exact le_max_of_le_left (hM ⟨t, ht', rfl⟩)
      · simp only
        rw [ciSup_neg ht', Real.sSup_empty]
        exact le_max_right M 0
    apply le_antisymm
    · apply ciSup_le
      intro t
      by_cases ht : t ∈ s
      · simp only [ciSup_pos ht]
        exact le_ciSup hbdd_subtype ⟨t, ht⟩
      · rw [ciSup_neg ht, Real.sSup_empty]
        calc (0 : ℝ) = f t₀ := hft₀.symm
          _ ≤ ⨆ (t : ↥s), f t.1 := le_ciSup hbdd_subtype ⟨t₀, ht₀⟩
    · apply ciSup_le
      intro ⟨t, ht⟩
      have h : f t = ⨆ (_ : t ∈ s), f t := by simp [ht]
      rw [h]
      exact le_ciSup hbdd_biSup t

  · -- Unbounded case: both sides are 0 for ℝ
    have hbdd_subtype : ¬ BddAbove (range (fun (u : ↥s) => f u.1)) := by
      intro ⟨M, hM⟩
      apply hbdd
      use M
      rintro _ ⟨t, ht, rfl⟩
      exact hM ⟨⟨t, ht⟩, rfl⟩
    have hbdd_biSup : ¬ BddAbove (range (fun t => ⨆ (_ : t ∈ s), f t)) := by
      intro ⟨M, hM⟩
      apply hbdd
      use M
      rintro _ ⟨t, ht, rfl⟩
      have := hM ⟨t, rfl⟩
      simp only [ciSup_pos ht] at this
      exact this
    rw [Real.iSup_of_not_bddAbove hbdd_biSup]
    rw [Real.iSup_of_not_bddAbove hbdd_subtype]


set_option maxHeartbeats 400000

/-- **Core Dudley Discrete Bound**

Bounds the expected supremum of a sub-Gaussian process over a finite dyadic net at level K:

  𝔼[sup_{u ∈ nets_K} (X u - X t₀)] ≤ (6√2) σ · dyadicRHS(s, D, K+1)

**Proof outline:**
1. Telescope via transitive projection: X u - X t₀ = SBase + ∑ₖ SΔ_trans k
2. Bound base term: 𝔼[SBase] ≤ (2√2) σ · dyadicRHS
3. Bound increments: 𝔼[∑ₖ SΔ_trans k] ≤ (4√2) σ · dyadicRHS

**Key insight:** Transitive projection gives dist ≤ εₖ (not 3εₖ/2) and
|pairs| ≤ |nets_{k+1}| (not |nets_K|), improving the constant from 11√2 to 6√2.

The tail term is handled separately in `dudley_chaining_bound_countable`.
-/
lemma dudley_chaining_bound_core {Ω : Type u} [MeasurableSpace Ω] {A : Type v} [PseudoMetricSpace A]
    {μ : Measure Ω} [IsProbabilityMeasure μ]
    {X : A → Ω → ℝ} {σ : ℝ} (hσ : 0 < σ)
    (hX : IsSubGaussianProcess μ X σ)
    {s : Set A} (hs : TotallyBounded s)
    {D : ℝ} (hD : 0 < D) (hdiam : Metric.diam s ≤ D)
    (K : ℕ) (hK : 1 ≤ K)
    (hX_meas : ∀ t, Measurable (X t))
    (hX_int : ∀ t s : A, Integrable (fun ω => X t ω - X s ω) μ)
    (hX_centered : ∀ t s : A, ∫ ω, (X t ω - X s ω) ∂μ = 0)
    (hX_int_exp : ∀ t s : A, ∀ l : ℝ, Integrable (fun ω => Real.exp (l * (X t ω - X s ω))) μ)
    (dn : DyadicNets s D)
    (hcard_bound : ∀ k, (dn.nets k).card ≤ (coveringNumber (dyadicScale D (k + 1)) s).untop
        (ne_top_of_lt (coveringNumber_lt_top_of_totallyBounded (dyadicScale_pos hD (k + 1)) hs)))
    (hnet_nonempty : ∀ k, (dn.nets k).Nonempty)
    (t₀ : A) :
    ∫ ω, (dn.nets K).sup' (hnet_nonempty K) (fun u => X u ω - X t₀ ω) ∂μ ≤
        (6 * Real.sqrt 2) * σ * dyadicRHS_real s D (K + 1) := by
  classical

  -- ===== DEFINE CHAINING COMPONENTS =====
  -- Define the finite supremum functions
  -- SBase: sup over level-0 net
  let SBase_fin : Ω → ℝ := fun ω =>
    (dn.nets 0).sup' (hnet_nonempty 0) (fun u => X u ω - X t₀ ω)

  -- For increments, define edge sets: pairs (u, v) with u ∈ net_k, v ∈ net_{k+1}
  let edges : ℕ → Finset (A × A) := fun k =>
    (dn.nets k).product (dn.nets (k + 1))

  have h_edges_nonempty : ∀ k, (edges k).Nonempty := fun k =>
    (hnet_nonempty k).product (hnet_nonempty (k + 1))

  -- SΔ_fin k: sup over edge pairs at level k
  let SΔ_fin : ℕ → Ω → ℝ := fun k ω =>
    (edges k).sup' (h_edges_nonempty k) (fun p => X p.2 ω - X p.1 ω)

  -- Transitive projection: gives dist ≤ εₖ and |pairs| ≤ |nets_{k+1}|
  let transProj : ℕ → A → A := fun k u => transitiveProj dn hnet_nonempty K k u

  -- Chaining pairs: uniquely determined by second component
  let chaining_pairs_trans : ℕ → Set (A × A) := fun k =>
    { p | ∃ u ∈ dn.nets K, p.1 = transProj k u ∧ p.2 = transProj (k + 1) u }

  -- Transitive chaining pairs form a subset of edges k (for finiteness)
  have h_chain_trans_subset : ∀ k, k < K → chaining_pairs_trans k ⊆ ↑(edges k) := by
    intro k hk p ⟨u, hu, hp1, hp2⟩
    apply Finset.mem_coe.mpr
    apply Finset.mem_product.mpr
    constructor
    · simp only [transProj] at hp1
      rw [hp1]
      exact transitiveProj_mem_nets dn hnet_nonempty K k hk u
    · simp only [transProj] at hp2
      rw [hp2]
      by_cases hk1 : k + 1 < K
      · exact transitiveProj_mem_nets dn hnet_nonempty K (k + 1) hk1 u
      · push Not at hk1
        have hk1_eq : k + 1 = K := by omega
        rw [hk1_eq, transitiveProj_self]
        exact hu

  -- Transitive chaining pairs is finite (subset of finite edges k for k < K)
  have h_chain_trans_finite : ∀ k, (chaining_pairs_trans k).Finite := by
    intro k
    by_cases hk : k < K
    · exact Set.Finite.subset (Finset.finite_toSet (edges k)) (h_chain_trans_subset k hk)
    · -- When k ≥ K, both transProj k u = u and transProj (k+1) u = u
      -- So chaining_pairs_trans k ⊆ {(u, u) : u ∈ nets K} ⊆ (nets K) × (nets K)
      push Not at hk
      apply Set.Finite.subset ((dn.nets K).finite_toSet.prod (dn.nets K).finite_toSet)
      intro p ⟨u, hu, hp1, hp2⟩
      simp only [Set.mem_prod, Finset.mem_coe, transProj] at hp1 hp2 ⊢
      constructor
      · rw [hp1, transitiveProj_of_le dn hnet_nonempty K k hk]; exact hu
      · rw [hp2, transitiveProj_of_le dn hnet_nonempty K (k + 1) (by omega)]; exact hu

  -- Transitive chaining pairs is nonempty (project any element of nets_K)
  have h_chain_trans_nonempty : ∀ k, (chaining_pairs_trans k).Nonempty := fun k =>
    let ⟨u, hu⟩ := hnet_nonempty K
    ⟨(transProj k u, transProj (k + 1) u), u, hu, rfl, rfl⟩

  -- ===== DISTANCE AND CARDINALITY BOUNDS =====
  -- Distance bound: dist ≤ εₖ (sharp, via dist_transitiveProj_consecutive)
  have h_chain_trans_dist : ∀ k, k < K → ∀ (p : A × A), p ∈ chaining_pairs_trans k →
      dist p.1 p.2 ≤ dyadicScale D k := by
    intro k hk p ⟨u, hu, hp1, hp2⟩
    calc dist p.1 p.2 = dist (transProj k u) (transProj (k + 1) u) := congrArg₂ dist hp1 hp2
      _ ≤ dyadicScale D k := dist_transitiveProj_consecutive dn hnet_nonempty K k hk u hu

  -- SΔ_trans k: sup over transitive chaining pairs (matches pathwise decomposition)
  let SΔ_trans : ℕ → Ω → ℝ := fun k ω =>
    (h_chain_trans_finite k).toFinset.sup' (by
      rw [Set.Finite.toFinset_nonempty]
      exact h_chain_trans_nonempty k) (fun p => X p.2 ω - X p.1 ω)

  -- SΔ_trans k ≤ SΔ_fin k (transitive pairs ⊆ edges k for k < K)
  have h_SΔ_trans_le_fin : ∀ k, k < K → ∀ ω, SΔ_trans k ω ≤ SΔ_fin k ω := by
    intro k hk ω
    apply Finset.sup'_le
    intro p hp
    simp only [Set.Finite.mem_toFinset] at hp
    have hp' := h_chain_trans_subset k hk hp
    simp only [edges, Finset.mem_coe] at hp'
    exact Finset.le_sup' (fun q => X q.2 ω - X q.1 ω) hp'

  -- Cardinality bound: |pairs| ≤ |nets_{k+1}| (gives single sqrtE term)
  have h_chain_trans_card_le : ∀ k, k < K →
      (h_chain_trans_finite k).toFinset.card ≤ (dn.nets (k + 1)).card := by
    intro k hk
    -- Each pair (transProj k u, transProj (k+1) u) is determined by transProj (k+1) u
    -- because transProj k u = projectToNet k (transProj (k+1) u)
    -- So we map pairs to their second component, which lands in nets (k+1)
    let second : A × A → A := fun p => p.2
    have h_second_inj : Set.InjOn second (chaining_pairs_trans k) := by
      intro p1 hp1 p2 hp2 h_eq
      obtain ⟨u1, hu1, hp1_1, hp1_2⟩ := hp1
      obtain ⟨u2, hu2, hp2_1, hp2_2⟩ := hp2
      -- h_eq says p1.2 = p2.2, i.e., transProj (k+1) u1 = transProj (k+1) u2
      simp only [transProj] at hp1_1 hp1_2 hp2_1 hp2_2
      have h_snd_eq : transitiveProj dn hnet_nonempty K (k + 1) u1 =
          transitiveProj dn hnet_nonempty K (k + 1) u2 := by
        rw [← hp1_2, ← hp2_2]; exact h_eq
      -- Then transProj k u1 = projectToNet k (transProj (k+1) u1)
      --                    = projectToNet k (transProj (k+1) u2)
      --                    = transProj k u2
      have h_fst_eq : transitiveProj dn hnet_nonempty K k u1 =
          transitiveProj dn hnet_nonempty K k u2 := by
        rw [transitiveProj_of_lt dn hnet_nonempty K k hk,
            transitiveProj_of_lt dn hnet_nonempty K k hk, h_snd_eq]
      ext
      · rw [hp1_1, hp2_1, h_fst_eq]
      · rw [hp1_2, hp2_2, h_snd_eq]
    -- The image of pairs under `second` lands in nets (k+1)
    have h_image_subset : second '' (chaining_pairs_trans k) ⊆ ↑(dn.nets (k + 1)) := by
      intro v ⟨p, hp, hpv⟩
      obtain ⟨u, hu, _, hp2⟩ := hp
      simp only [transProj, second] at hpv hp2
      rw [← hpv, hp2]
      by_cases hk1 : k + 1 < K
      · exact transitiveProj_mem_nets dn hnet_nonempty K (k + 1) hk1 u
      · push Not at hk1
        have hk1_eq : k + 1 = K := by omega
        rw [hk1_eq, transitiveProj_self]
        exact hu
    -- Use injectivity to bound cardinality
    calc (h_chain_trans_finite k).toFinset.card
      _ = ((h_chain_trans_finite k).toFinset.image second).card := by
          rw [Finset.card_image_of_injOn]
          intro x hx y hy hxy
          exact h_second_inj ((h_chain_trans_finite k).mem_toFinset.mp hx)
            ((h_chain_trans_finite k).mem_toFinset.mp hy) hxy
      _ ≤ (dn.nets (k + 1)).card := by
          apply Finset.card_le_card
          intro v hv
          simp only [Finset.mem_image] at hv
          obtain ⟨p, hp, hpv⟩ := hv
          have hp' := (h_chain_trans_finite k).mem_toFinset.mp hp
          have := h_image_subset ⟨p, hp', hpv⟩
          simp only [Finset.mem_coe] at this
          exact this

  -- ===== INTEGRABILITY =====
  -- Helper: bound for finite sup in terms of sum of abs values
  have h_sup'_abs_bound : ∀ {T : Finset A} (hT : T.Nonempty) (f : A → ℝ),
      |T.sup' hT f| ≤ ∑ u ∈ T, |f u| := by
    intro T hT f
    rw [abs_le]
    have ⟨a₀, ha₀⟩ := hT
    constructor
    · -- Lower bound: -(∑ |f u|) ≤ sup' f
      calc -(∑ u ∈ T, |f u|)
        _ ≤ -|f a₀| := by
            apply neg_le_neg
            exact Finset.single_le_sum (f := fun u => |f u|) (fun _ _ => abs_nonneg _) ha₀
        _ ≤ f a₀ := neg_abs_le _
        _ ≤ T.sup' hT f := Finset.le_sup' f ha₀
    · -- Upper bound: sup' f ≤ ∑ |f u|
      apply Finset.sup'_le hT
      intro u hu
      calc f u ≤ |f u| := le_abs_self _
        _ ≤ ∑ v ∈ T, |f v| := Finset.single_le_sum (f := fun v => |f v|) (fun _ _ => abs_nonneg _) hu

  -- Similar bound for pairs
  have h_sup'_abs_bound_pair : ∀ {T : Finset (A × A)} (hT : T.Nonempty) (f : A × A → ℝ),
      |T.sup' hT f| ≤ ∑ p ∈ T, |f p| := by
    intro T hT f
    rw [abs_le]
    have ⟨a₀, ha₀⟩ := hT
    constructor
    · calc -(∑ p ∈ T, |f p|)
        _ ≤ -|f a₀| := by
            apply neg_le_neg
            exact Finset.single_le_sum (f := fun p => |f p|) (fun _ _ => abs_nonneg _) ha₀
        _ ≤ f a₀ := neg_abs_le _
        _ ≤ T.sup' hT f := Finset.le_sup' f ha₀
    · apply Finset.sup'_le hT
      intro p hp
      calc f p ≤ |f p| := le_abs_self _
        _ ≤ ∑ q ∈ T, |f q| := Finset.single_le_sum (f := fun q => |f q|) (fun _ _ => abs_nonneg _) hp

  have h_SBase_fin_int : Integrable SBase_fin μ := by
    have h_dom_int : Integrable (fun ω => ∑ u ∈ dn.nets 0, |X u ω - X t₀ ω|) μ :=
      integrable_finsetSum _ (fun u _ => (hX_int u t₀).abs)
    -- Measurability: via Finset.measurable_sup' with explicit application
    -- Define the function-valued form that Finset.measurable_sup' expects
    let SBase_fn : Ω → ℝ := (dn.nets 0).sup' (hnet_nonempty 0) (fun u ω => X u ω - X t₀ ω)
    have h_base_meas' : Measurable SBase_fn :=
      Finset.measurable_sup' (hnet_nonempty 0) (fun u _ => (hX_meas u).sub (hX_meas t₀))
    -- Show SBase_fin = SBase_fn pointwise using Finset.sup'_apply
    have h_eq : SBase_fin = SBase_fn := by
      ext ω
      simp only [SBase_fin, SBase_fn, Finset.sup'_apply]
    have h_base_meas : Measurable SBase_fin := h_eq ▸ h_base_meas'
    refine h_dom_int.mono' h_base_meas.aestronglyMeasurable ?_
    filter_upwards with ω
    rw [Real.norm_eq_abs]
    exact h_sup'_abs_bound (hnet_nonempty 0) (fun u => X u ω - X t₀ ω)

  have h_SΔ_fin_int : ∀ k, Integrable (SΔ_fin k) μ := by
    intro k
    have h_dom_int : Integrable (fun ω => ∑ p ∈ edges k, |X p.2 ω - X p.1 ω|) μ :=
      integrable_finsetSum _ (fun p _ => (hX_int p.2 p.1).abs)
    -- Define the function-valued form
    let SΔ_fn : Ω → ℝ := (edges k).sup' (h_edges_nonempty k) (fun p ω => X p.2 ω - X p.1 ω)
    have h_incr_meas' : Measurable SΔ_fn :=
      Finset.measurable_sup' (h_edges_nonempty k) (fun p _ => (hX_meas p.2).sub (hX_meas p.1))
    -- Show SΔ_fin k = SΔ_fn pointwise
    have h_eq : SΔ_fin k = SΔ_fn := by
      ext ω
      simp only [SΔ_fin, SΔ_fn, Finset.sup'_apply]
    have h_incr_meas : Measurable (SΔ_fin k) := h_eq ▸ h_incr_meas'
    refine h_dom_int.mono' h_incr_meas.aestronglyMeasurable ?_
    filter_upwards with ω
    rw [Real.norm_eq_abs]
    exact h_sup'_abs_bound_pair (h_edges_nonempty k) (fun p => X p.2 ω - X p.1 ω)

  -- ===== BASE LEVEL BOUNDS =====
  -- Diameter bound: diam(net_k) ≤ diam(s) ≤ D
  have h_netk_diam_bound : ∀ k, Metric.diam ((dn.nets k) : Set A) ≤ D := by
    intro k
    -- s is bounded (totally bounded implies bounded)
    have hs_bdd : Bornology.IsBounded s := hs.isBounded
    -- Net points are in s, so diam(net) ≤ diam(s) ≤ D
    calc Metric.diam ((dn.nets k) : Set A)
      _ ≤ Metric.diam s := Metric.diam_mono (dn.nets_subset k) hs_bdd
      _ ≤ D := hdiam

  -- Net cardinality ≥ 1 (from nonemptiness)
  have h_net_card_pos : ∀ k, 1 ≤ (dn.nets k).card :=
    fun k => Finset.Nonempty.card_pos (hnet_nonempty k)

  -- Base level integral bound: E[SBase_fin] ≤ C * σ * D * √(log |net_0|)
  have h_base_bound_card_ge_2 : (dn.nets 0).card ≥ 2 →
      ∃ C : ℝ, C > 0 ∧ C ≤ Real.sqrt 2 ∧ ∫ ω, SBase_fin ω ∂μ ≤ C * σ * D * Real.sqrt (Real.log (dn.nets 0).card) := by
    intro hcard
    -- Pick a base point u₀ from net_0 using the nonemptiness witness
    let u₀ := (hnet_nonempty 0).choose
    have hu₀_mem : u₀ ∈ dn.nets 0 := (hnet_nonempty 0).choose_spec
    -- Apply subGaussian_finite_max_bound' to get the bound structure
    -- The lemma internally uses C = √2, so the bound is: ∫ ... ≤ √2 * σ * D * √(log |T|)
    have hbound := subGaussian_finite_max_bound' hσ hX
      (dn.nets 0) hcard D hD (h_netk_diam_bound 0) hX_meas hX_int hX_int_exp
    obtain ⟨C, hC_pos, hC_le_sqrt2, hC_bound_all⟩ := hbound
    -- The existential now provides C ≤ √2 directly
    refine ⟨C, hC_pos, hC_le_sqrt2, ?_⟩
    -- For the sup' to biSup conversion, we work with the direct integral
    -- First establish the shifted sup' form
    let SBase_shifted : Ω → ℝ := fun ω =>
      (dn.nets 0).sup' (hnet_nonempty 0) (fun u => X u ω - X u₀ ω)
    -- Relate SBase_fin to SBase_shifted + constant
    have h_shift : ∀ ω, SBase_fin ω = SBase_shifted ω + (X u₀ ω - X t₀ ω) := by
      intro ω
      simp only [SBase_fin, SBase_shifted]
      -- sup_u (X u - X t₀) = sup_u ((X u - X u₀) + (X u₀ - X t₀))
      have h_rewrite : ∀ u, X u ω - X t₀ ω = (X u ω - X u₀ ω) + (X u₀ ω - X t₀ ω) := fun u => by ring
      have h_eq_fns : (fun u => X u ω - X t₀ ω) = (fun u => (X u ω - X u₀ ω) + (X u₀ ω - X t₀ ω)) :=
        funext h_rewrite
      rw [h_eq_fns]
      -- sup (f + c) = sup f + c for constant c
      -- Prove this manually using le_antisymm, sup'_le_iff, and le_sup'
      apply le_antisymm
      · -- (≤): sup' (f + c) ≤ sup' f + c
        rw [Finset.sup'_le_iff]
        intro b hb
        calc (X b ω - X u₀ ω) + (X u₀ ω - X t₀ ω)
          _ ≤ (dn.nets 0).sup' (hnet_nonempty 0) (fun u => X u ω - X u₀ ω) + (X u₀ ω - X t₀ ω) := by
              gcongr
              exact Finset.le_sup' (fun u => X u ω - X u₀ ω) hb
      · -- (≥): sup' f + c ≤ sup' (f + c)
        -- We show: for any u ∈ T, f u ≤ sup' (f + c) - c, hence sup' f ≤ sup' (f + c) - c
        rw [le_sub_iff_add_le.symm]
        rw [Finset.sup'_le_iff]
        intro b hb
        -- Need: X b ω - X u₀ ω ≤ sup' (f + c) - c
        have h1 : (X b ω - X u₀ ω) + (X u₀ ω - X t₀ ω) ≤
            (dn.nets 0).sup' (hnet_nonempty 0) (fun u => (X u ω - X u₀ ω) + (X u₀ ω - X t₀ ω)) :=
          Finset.le_sup' (fun u => (X u ω - X u₀ ω) + (X u₀ ω - X t₀ ω)) hb
        linarith
    -- Integrability of SBase_shifted
    have h_shifted_int : Integrable SBase_shifted μ := by
      have h_dom_int : Integrable (fun ω => ∑ u ∈ dn.nets 0, |X u ω - X u₀ ω|) μ :=
        integrable_finsetSum _ (fun u _ => (hX_int u u₀).abs)
      let SBase_fn : Ω → ℝ := (dn.nets 0).sup' (hnet_nonempty 0) (fun u ω => X u ω - X u₀ ω)
      have h_meas' : Measurable SBase_fn :=
        Finset.measurable_sup' (hnet_nonempty 0) (fun u _ => (hX_meas u).sub (hX_meas u₀))
      have h_eq : SBase_shifted = SBase_fn := by
        ext ω
        simp only [SBase_shifted, SBase_fn, Finset.sup'_apply]
      rw [h_eq]
      refine h_dom_int.mono' h_meas'.aestronglyMeasurable ?_
      filter_upwards with ω
      rw [Real.norm_eq_abs]
      -- SBase_fn ω = (dn.nets 0).sup' ... (fun u ω' => ...) evaluated at ω
      -- By Finset.sup'_apply, this equals (dn.nets 0).sup' ... (fun u => ... ω)
      simp only [SBase_fn, Finset.sup'_apply]
      exact h_sup'_abs_bound (hnet_nonempty 0) (fun u => X u ω - X u₀ ω)
    -- Integrate both sides of h_shift
    have h_int_eq : ∫ ω, SBase_fin ω ∂μ = ∫ ω, SBase_shifted ω ∂μ + ∫ ω, (X u₀ ω - X t₀ ω) ∂μ := by
      calc ∫ ω, SBase_fin ω ∂μ
        _ = ∫ ω, SBase_shifted ω + (X u₀ ω - X t₀ ω) ∂μ := by congr; exact funext h_shift
        _ = ∫ ω, SBase_shifted ω ∂μ + ∫ ω, (X u₀ ω - X t₀ ω) ∂μ := integral_add h_shifted_int (hX_int u₀ t₀)
    -- The centering kills the second term
    rw [h_int_eq, hX_centered u₀ t₀, add_zero]
    -- Now bound the shifted sup using subGaussian_finite_max_bound'
    -- We need to convert sup' to biSup for hbound
    -- Use biSup_eq_sup'_of_finset: requires ∃ t ∈ T, 0 ≤ f t
    have h_nonneg_witness : ∀ ω, ∃ t ∈ dn.nets 0, 0 ≤ X t ω - X u₀ ω := by
      intro ω
      use u₀, hu₀_mem
      simp
    -- Use iSup_subtype_eq_sup' from SubGaussian.lean to relate forms
    have h_sup'_to_biSup : ∀ ω, SBase_shifted ω = ⨆ u ∈ dn.nets 0, (X u ω - X u₀ ω) := by
      intro ω
      simp only [SBase_shifted]
      rw [← biSup_eq_sup'_of_finset (hnet_nonempty 0) (fun u => X u ω - X u₀ ω) (h_nonneg_witness ω)]
    calc ∫ ω, SBase_shifted ω ∂μ
      _ = ∫ ω, ⨆ u ∈ dn.nets 0, (X u ω - X u₀ ω) ∂μ := by congr; exact funext h_sup'_to_biSup
      _ ≤ C * σ * D * Real.sqrt (Real.log (dn.nets 0).card) := hC_bound_all u₀ hu₀_mem

  -- Base level bound when |net_0| = 1 (trivial case)
  have h_base_bound_card_1 : (dn.nets 0).card = 1 →
      ∫ ω, SBase_fin ω ∂μ ≤ 0 := by
    intro hcard
    -- When |net_0| = 1, the sup is over a singleton
    rw [Finset.card_eq_one] at hcard
    obtain ⟨u₀, hu₀⟩ := hcard
    -- SBase_fin ω = X u₀ ω - X t₀ ω
    have h_eq : ∀ ω, SBase_fin ω = X u₀ ω - X t₀ ω := by
      intro ω
      simp only [SBase_fin, hu₀, Finset.sup'_singleton]
    calc ∫ ω, SBase_fin ω ∂μ
      _ = ∫ ω, X u₀ ω - X t₀ ω ∂μ := by congr; exact funext h_eq
      _ = 0 := hX_centered u₀ t₀
      _ ≤ 0 := le_refl 0

  -- ===== PATHWISE DECOMPOSITION =====
  -- S_net: the finite net supremum we're bounding
  let S_net : Ω → ℝ := fun ω => (dn.nets K).sup' (hnet_nonempty K) (fun u => X u ω - X t₀ ω)

  have h_S_net_int : Integrable S_net μ := by
    -- Dominating function: sum of |X u - X t₀| over net_K
    have h_dom : Integrable (fun ω => ∑ u ∈ dn.nets K, |X u ω - X t₀ ω|) μ :=
      integrable_finsetSum _ (fun u _ => (hX_int u t₀).abs)
    -- Measurability
    have h_meas : Measurable S_net := by
      simp only [S_net]
      have hf : ∀ u ∈ dn.nets K, Measurable (fun ω => X u ω - X t₀ ω) :=
        fun u _ => (hX_meas u).sub (hX_meas t₀)
      convert Finset.measurable_sup' (hnet_nonempty K) hf using 1
      ext ω
      simp only [Finset.sup'_apply]
    refine h_dom.mono' h_meas.aestronglyMeasurable ?_
    filter_upwards with ω
    rw [Real.norm_eq_abs]
    exact h_sup'_abs_bound (hnet_nonempty K) (fun u => X u ω - X t₀ ω)

  -- Pathwise bound: S_net ≤ SBase_fin + ∑ SΔ_trans k (tail is exactly zero)
  have h_S_net_pathwise_trans : ∀ᵐ ω ∂μ, S_net ω ≤ SBase_fin ω + ∑ k ∈ Finset.range K, SΔ_trans k ω := by
    apply Filter.Eventually.of_forall
    intro ω
    apply Finset.sup'_le (hnet_nonempty K)
    intro u hu
    -- Base: X (transProj 0 u) - X t₀ ≤ SBase_fin
    have h_base : X (transProj 0 u) ω - X t₀ ω ≤ SBase_fin ω := by
      have h_mem : transProj 0 u ∈ dn.nets 0 := by
        simp only [transProj]
        exact transitiveProj_mem_nets dn hnet_nonempty K 0 (Nat.zero_lt_of_lt hK) u
      exact Finset.le_sup' (fun v => X v ω - X t₀ ω) h_mem
    -- Increments: X (transProj (k+1) u) - X (transProj k u) ≤ SΔ_trans k
    have h_incr : ∀ k ∈ Finset.range K,
        X (transProj (k + 1) u) ω - X (transProj k u) ω ≤ SΔ_trans k ω := by
      intro k hk
      have h_pair_mem : (transProj k u, transProj (k + 1) u) ∈ (h_chain_trans_finite k).toFinset := by
        simp only [Set.Finite.mem_toFinset]
        exact ⟨u, hu, rfl, rfl⟩
      exact Finset.le_sup' (fun p => X p.2 ω - X p.1 ω) h_pair_mem
    -- Tail is exactly zero: transProj K u = u for u ∈ nets K
    have h_tail_zero : X u ω - X (transProj K u) ω = 0 := by
      simp only [transProj, transitiveProj_self]; ring
    -- Combine using telescoping
    calc X u ω - X t₀ ω
      _ = (X (transProj 0 u) ω - X t₀ ω) + (X (transProj K u) ω - X (transProj 0 u) ω) +
          (X u ω - X (transProj K u) ω) := by ring
      _ = (X (transProj 0 u) ω - X t₀ ω) +
          (∑ k ∈ Finset.range K, (X (transProj (k + 1) u) ω - X (transProj k u) ω)) +
          (X u ω - X (transProj K u) ω) := by
          congr 1
          have h_tele : ∑ k ∈ Finset.range K, (X (transProj (k + 1) u) ω - X (transProj k u) ω) =
              X (transProj K u) ω - X (transProj 0 u) ω := by
            rw [Finset.sum_range_sub (fun k => X (transProj k u) ω)]
          rw [← h_tele]
      _ = (X (transProj 0 u) ω - X t₀ ω) +
          (∑ k ∈ Finset.range K, (X (transProj (k + 1) u) ω - X (transProj k u) ω)) + 0 := by
          rw [h_tail_zero]
      _ ≤ SBase_fin ω + (∑ k ∈ Finset.range K, SΔ_trans k ω) + 0 := by
          apply add_le_add
          apply add_le_add
          · exact h_base
          · exact Finset.sum_le_sum h_incr
          · exact le_refl 0
      _ = SBase_fin ω + ∑ k ∈ Finset.range K, SΔ_trans k ω := by ring

  -- ===== MAIN BOUND ASSEMBLY =====
  have h_goal : ∫ ω, S_net ω ∂μ ≤ (6 * Real.sqrt 2) * σ * dyadicRHS_real s D (K + 1) := by
    -- Base bound with C ≤ √2
    have h_base_int_bound : ∃ C_base : ℝ, 0 ≤ C_base ∧ C_base ≤ Real.sqrt 2 ∧
        ∫ ω, SBase_fin ω ∂μ ≤ C_base * σ * D * Real.sqrt (Real.log (dn.nets 0).card) := by
      by_cases hcard : (dn.nets 0).card ≥ 2
      · -- Case: |net_0| ≥ 2
        obtain ⟨C, hC_pos, hC_le_sqrt2, hC_bound⟩ := h_base_bound_card_ge_2 hcard
        exact ⟨C, le_of_lt hC_pos, hC_le_sqrt2, hC_bound⟩
      · -- Case: |net_0| < 2, so |net_0| = 1 (since nonempty)
        push Not at hcard
        have h1 : (dn.nets 0).card = 1 := by
          have hpos := h_net_card_pos 0
          omega
        have hbound := h_base_bound_card_1 h1
        -- When card = 1, √(log 1) = √0 = 0, so any C works
        use 0
        refine ⟨le_refl 0, ?_, ?_⟩
        · exact Real.sqrt_nonneg 2
        · simp only [zero_mul]
          exact hbound

    -- Convert net cardinalities to sqrtEntropy using hcard_bound
    have h_net_sqrt_le : ∀ k, Real.sqrt (Real.log (dn.nets k).card) ≤
        sqrtEntropy (dyadicScale D (k + 1)) s := by
      intro k
      apply sqrt_log_card_le_sqrtEntropy_of_card_le (dyadicScale_pos hD (k + 1)) hs
      -- hcard_bound k gives |nets k| ≤ coveringNumber(ε_{k+1}, s)
      -- Convert from ℕ ≤ ℕ (after untop) to WithTop ℕ ≤ WithTop ℕ
      have hne : coveringNumber (dyadicScale D (k + 1)) s ≠ ⊤ :=
        ne_top_of_lt (coveringNumber_lt_top_of_totallyBounded (dyadicScale_pos hD (k + 1)) hs)
      rw [← WithTop.coe_untop (coveringNumber (dyadicScale D (k + 1)) s) hne]
      exact WithTop.coe_le_coe.mpr (hcard_bound k)

    -- Extract bounds from existential hypotheses
    obtain ⟨C_base, hC_base_nonneg, hC_base_le_sqrt2, h_base_bound⟩ := h_base_int_bound

    -- Bound in terms of sqrtEntropy
    have h_base_sqrtE : ∫ ω, SBase_fin ω ∂μ ≤ C_base * σ * D * sqrtEntropy (dyadicScale D 1) s := by
      calc ∫ ω, SBase_fin ω ∂μ
        _ ≤ C_base * σ * D * Real.sqrt (Real.log (dn.nets 0).card) := h_base_bound
        _ ≤ C_base * σ * D * sqrtEntropy (dyadicScale D 1) s := by
            apply mul_le_mul_of_nonneg_left (h_net_sqrt_le 0)
            apply mul_nonneg (mul_nonneg hC_base_nonneg (le_of_lt hσ)) (le_of_lt hD)

    -- h1: ∫ SBase_fin ≤ (2√2) σ · dyadicRHS
    have h1 : ∫ (ω : Ω), SBase_fin ω ∂μ ≤ (2 * Real.sqrt 2) * σ * dyadicRHS_real s D (K + 1) := by
      -- D · sqrtE(ε₁) ≤ 2 · dyadicRHS (k=0 term gives D · 2⁻¹ · sqrtE(ε₁))
            have h_sqrtE_le_rhs : D * sqrtEntropy (dyadicScale D 1) s ≤ 2 * dyadicRHS_real s D (K + 1) := by
              unfold dyadicRHS_real dyadicScale
              -- For (K + 1), the sum is over Finset.range (K + 1), which includes k = 0
              have h_sum_has_zero : 0 ∈ Finset.range (K + 1) := Finset.mem_range.mpr (Nat.zero_lt_succ K)
              -- The sum includes at least the k=0 term, which has value (1/2) * sqrtE(D/2, s)
              let f : ℕ → ℝ := fun k => (2 : ℝ)^(-((k : ℤ) + 1)) * sqrtEntropy (D * (2 : ℝ)^(-((k : ℤ) + 1))) s
              have h_single := Finset.single_le_sum
                (f := f) (a := 0)
                (fun k _ => mul_nonneg (by positivity) (sqrtEntropy_nonneg _ _))
                h_sum_has_zero
              -- So: f 0 = (1/2) * sqrtE(D/2, s) ≤ sum
              have h_tail_nonneg : 0 ≤ D * (2 : ℝ)^(-((K + 1) : ℤ)) * sqrtEntropy (D * (2 : ℝ)^(-((K + 1) : ℤ))) s := by
                apply mul_nonneg (mul_nonneg (le_of_lt hD) (by positivity)) (sqrtEntropy_nonneg _ _)
              -- Now the main calc: D * sqrtE ≤ 2 * dyadicRHS_real
              -- Convert h_single to explicit form: f 0 = 2^(-1) * sqrtE(D * 2^(-1), s)
              have hf0 : f 0 = (2 : ℝ)⁻¹ * sqrtEntropy (D * (2 : ℝ)⁻¹) s := by
                simp only [f]; norm_num
              have h_D_single : D * f 0 ≤ D * ∑ k ∈ Finset.range (K + 1), f k :=
                mul_le_mul_of_nonneg_left h_single (le_of_lt hD)
              rw [hf0] at h_D_single
              -- Normalize 2^(-1) = 2⁻¹
              have h_two_inv : (2 : ℝ)^(-(1 : ℤ)) = (2 : ℝ)⁻¹ := by norm_num
              calc D * sqrtEntropy (D * (2 : ℝ)^(-(1 : ℤ))) s
                _ = D * sqrtEntropy (D * (2 : ℝ)⁻¹) s := by rw [h_two_inv]
                _ = 2 * (D * (2 : ℝ)⁻¹ * sqrtEntropy (D * (2 : ℝ)⁻¹) s) := by ring
                _ ≤ 2 * (D * ∑ k ∈ Finset.range (K + 1), f k) := by
                    have h_assoc : D * 2⁻¹ * sqrtEntropy (D * 2⁻¹) s = D * (2⁻¹ * sqrtEntropy (D * 2⁻¹) s) := by ring
                    rw [h_assoc]
                    exact mul_le_mul_of_nonneg_left h_D_single (by norm_num : (0 : ℝ) ≤ 2)
                _ = 2 * (D * ∑ k ∈ Finset.range (K + 1), (2 : ℝ)^(-((k : ℤ) + 1)) *
                    sqrtEntropy (D * (2 : ℝ)^(-((k : ℤ) + 1))) s) := by rfl
                _ ≤ 2 * (D * ∑ k ∈ Finset.range (K + 1), (2 : ℝ)^(-((k : ℤ) + 1)) *
                    sqrtEntropy (D * (2 : ℝ)^(-((k : ℤ) + 1))) s +
                    D * (2 : ℝ)^(-((K + 1) : ℤ)) * sqrtEntropy (D * (2 : ℝ)^(-((K + 1) : ℤ))) s) := by
                    linarith
            calc ∫ (ω : Ω), SBase_fin ω ∂μ
              _ ≤ C_base * σ * D * sqrtEntropy (dyadicScale D 1) s := h_base_sqrtE
              _ = C_base * σ * (D * sqrtEntropy (dyadicScale D 1) s) := by ring
              _ ≤ C_base * σ * (2 * dyadicRHS_real s D (K + 1)) := by
                  apply mul_le_mul_of_nonneg_left h_sqrtE_le_rhs
                  exact mul_nonneg hC_base_nonneg (le_of_lt hσ)
              _ = C_base * 2 * σ * dyadicRHS_real s D (K + 1) := by ring
              _ ≤ (2 * Real.sqrt 2) * σ * dyadicRHS_real s D (K + 1) := by
                  -- We have hC_base_le_sqrt2 : C_base ≤ √2, so C_base * 2 ≤ 2√2 exactly
                  have h_CB_bound : C_base * 2 ≤ 2 * Real.sqrt 2 := by
                    calc C_base * 2 ≤ Real.sqrt 2 * 2 :=
                        mul_le_mul_of_nonneg_right hC_base_le_sqrt2 (by norm_num : (0:ℝ) ≤ 2)
                      _ = 2 * Real.sqrt 2 := by ring
                  have h_rhs_nonneg' : 0 ≤ σ * dyadicRHS_real s D (K + 1) := by
                    apply mul_nonneg (le_of_lt hσ)
                    unfold dyadicRHS_real
                    apply add_nonneg
                    · apply mul_nonneg (le_of_lt hD)
                      apply Finset.sum_nonneg
                      intro k _
                      apply mul_nonneg (zpow_nonneg (by norm_num : (0:ℝ) ≤ 2) _) (sqrtEntropy_nonneg _ _)
                    · apply mul_nonneg (mul_nonneg (le_of_lt hD) (zpow_nonneg (by norm_num : (0:ℝ) ≤ 2) _))
                      exact sqrtEntropy_nonneg _ _
                  calc C_base * 2 * σ * dyadicRHS_real s D (K + 1)
                    _ = C_base * 2 * (σ * dyadicRHS_real s D (K + 1)) := by ring
                    _ ≤ (2 * Real.sqrt 2) * (σ * dyadicRHS_real s D (K + 1)) := mul_le_mul_of_nonneg_right h_CB_bound h_rhs_nonneg'
                    _ = (2 * Real.sqrt 2) * σ * dyadicRHS_real s D (K + 1) := by ring

    -- Integrability of SΔ_trans (via domination by SΔ_fin for k < K, constant 0 for k ≥ K)
    have h_SΔ_trans_int : ∀ k, Integrable (fun ω => SΔ_trans k ω) μ := by
      intro k
      have h_ne : ((h_chain_trans_finite k).toFinset).Nonempty := by
        rw [Set.Finite.toFinset_nonempty]; exact h_chain_trans_nonempty k
      obtain ⟨p₀, hp₀⟩ := h_ne
      have h_lower : ∀ ω, X p₀.2 ω - X p₀.1 ω ≤ SΔ_trans k ω :=
        fun ω => Finset.le_sup' (fun p => X p.2 ω - X p.1 ω) hp₀
      have hp₀_int : Integrable (fun ω => X p₀.2 ω - X p₀.1 ω) μ := hX_int p₀.2 p₀.1
      have h_meas : AEStronglyMeasurable (fun ω => SΔ_trans k ω) μ := by
        apply Measurable.aestronglyMeasurable
        have h_ne' : ((h_chain_trans_finite k).toFinset).Nonempty := by
          rw [Set.Finite.toFinset_nonempty]; exact h_chain_trans_nonempty k
        have h_meas_sup : Measurable ((h_chain_trans_finite k).toFinset.sup' h_ne'
            (fun p => fun ω => X p.2 ω - X p.1 ω)) := by
          apply Finset.measurable_sup' h_ne'
          intro p _
          exact (hX_meas p.2).sub (hX_meas p.1)
        convert h_meas_sup using 1
        ext ω
        simp only [SΔ_trans, Finset.sup'_apply]
      by_cases hk : k < K
      · -- For k < K: bound by SΔ_fin k
        have h_upper : ∀ ω, SΔ_trans k ω ≤ SΔ_fin k ω := fun ω => h_SΔ_trans_le_fin k hk ω
        have h_bound : ∀ ω, |SΔ_trans k ω| ≤ |SΔ_fin k ω| + |X p₀.2 ω - X p₀.1 ω| := by
          intro ω
          apply abs_le.mpr
          constructor
          · calc -(|SΔ_fin k ω| + |X p₀.2 ω - X p₀.1 ω|)
              _ ≤ -|X p₀.2 ω - X p₀.1 ω| := by linarith [abs_nonneg (SΔ_fin k ω)]
              _ ≤ X p₀.2 ω - X p₀.1 ω := neg_abs_le _
              _ ≤ SΔ_trans k ω := h_lower ω
          · calc SΔ_trans k ω
              _ ≤ SΔ_fin k ω := h_upper ω
              _ ≤ |SΔ_fin k ω| := le_abs_self _
              _ ≤ |SΔ_fin k ω| + |X p₀.2 ω - X p₀.1 ω| := by linarith [abs_nonneg (X p₀.2 ω - X p₀.1 ω)]
        apply Integrable.mono ((h_SΔ_fin_int k).abs.add hp₀_int.abs) h_meas
        filter_upwards with ω
        simp only [Pi.add_apply, Real.norm_eq_abs]
        have h1 := h_bound ω
        have h2 : |SΔ_fin k ω| + |X p₀.2 ω - X p₀.1 ω| =
            |(|SΔ_fin k ω| + |X p₀.2 ω - X p₀.1 ω|)| :=
          (abs_of_nonneg (add_nonneg (abs_nonneg _) (abs_nonneg _))).symm
        linarith
      · -- For k ≥ K: SΔ_trans k = 0 (all pairs collapse to (u, u))
        push Not at hk
        -- Show SΔ_trans k ω = 0 for all ω
        have h_zero : ∀ ω, SΔ_trans k ω = 0 := by
          intro ω
          apply le_antisymm
          · apply Finset.sup'_le
            intro p hp
            simp only [Set.Finite.mem_toFinset] at hp
            obtain ⟨u, hu, hp1, hp2⟩ := hp
            simp only [transProj] at hp1 hp2
            have h1 : transitiveProj dn hnet_nonempty K k u = u := transitiveProj_of_le dn hnet_nonempty K k hk u
            have h2 : transitiveProj dn hnet_nonempty K (k + 1) u = u := transitiveProj_of_le dn hnet_nonempty K (k + 1) (by omega) u
            rw [hp1, hp2, h1, h2, sub_self]
          · -- 0 ≤ sup (since all elements equal 0)
            have hp₀' := (h_chain_trans_finite k).mem_toFinset.mp hp₀
            obtain ⟨u₀, hu₀, hp₀_1, hp₀_2⟩ := hp₀'
            simp only [transProj] at hp₀_1 hp₀_2
            have h1 : transitiveProj dn hnet_nonempty K k u₀ = u₀ := transitiveProj_of_le dn hnet_nonempty K k hk u₀
            have h2 : transitiveProj dn hnet_nonempty K (k + 1) u₀ = u₀ := transitiveProj_of_le dn hnet_nonempty K (k + 1) (by omega) u₀
            calc (0 : ℝ) = X u₀ ω - X u₀ ω := by ring
              _ = X p₀.2 ω - X p₀.1 ω := by rw [hp₀_1, hp₀_2, h1, h2]
              _ ≤ SΔ_trans k ω := h_lower ω
        simp_rw [h_zero]
        exact integrable_const 0

    -- Per-level bound: √2 factor (cardinality ≤ |nets_{k+1}|)
    have h2a_trans : ∀ k ∈ Finset.range K, ∫ (ω : Ω), SΔ_trans k ω ∂μ ≤
        Real.sqrt 2 * σ * dyadicScale D k * Real.sqrt (Real.log (dn.nets (k + 1)).card) := by
      intro k hk
      have hk' : k < K := Finset.mem_range.mp hk
      by_cases hcard_trans : (h_chain_trans_finite k).toFinset.card ≥ 2
      case pos =>
        have h_ne : ((h_chain_trans_finite k).toFinset).Nonempty := by
          rw [Set.Finite.toFinset_nonempty]; exact h_chain_trans_nonempty k
        have h_ε_k_pos : 0 < dyadicScale D k := dyadicScale_pos hD k
        have hσ'_pos : 0 < σ * dyadicScale D k := by positivity
        -- CGF bound using dist ≤ εₖ
        have hY_cgf : ∀ p ∈ (h_chain_trans_finite k).toFinset, ∀ t : ℝ,
            ProbabilityTheory.cgf (fun ω => X p.2 ω - X p.1 ω) μ t ≤
            t^2 * (σ * dyadicScale D k)^2 / 2 := by
          intro p hp t
          have hp' : p ∈ chaining_pairs_trans k := (h_chain_trans_finite k).mem_toFinset.mp hp
          have h_dist : dist p.1 p.2 ≤ dyadicScale D k := h_chain_trans_dist k hk' p hp'
          rw [ProbabilityTheory.cgf]
          have h_mgf_bound : ProbabilityTheory.mgf (fun ω => X p.2 ω - X p.1 ω) μ t ≤
              Real.exp (t^2 * (σ * dyadicScale D k)^2 / 2) := by
            simp only [ProbabilityTheory.mgf]
            calc μ[fun ω => Real.exp (t * (X p.2 ω - X p.1 ω))]
              _ ≤ Real.exp (t^2 * σ^2 * (dist p.2 p.1)^2 / 2) := hX p.2 p.1 t
              _ ≤ Real.exp (t^2 * σ^2 * (dyadicScale D k)^2 / 2) := by
                  apply Real.exp_le_exp.mpr
                  apply div_le_div_of_nonneg_right _ (by norm_num : (0:ℝ) ≤ 2)
                  apply mul_le_mul_of_nonneg_left _ (by positivity)
                  rw [dist_comm] at h_dist
                  have h_dist_nonneg : 0 ≤ dist p.2 p.1 := dist_nonneg
                  exact sq_le_sq' (by linarith [dyadicScale_pos hD k]) h_dist
              _ = Real.exp (t^2 * (σ * dyadicScale D k)^2 / 2) := by ring_nf
          have h_mgf_pos : 0 < ProbabilityTheory.mgf (fun ω => X p.2 ω - X p.1 ω) μ t := by
            simp only [ProbabilityTheory.mgf]
            exact integral_exp_pos (hX_int_exp p.2 p.1 t)
          calc Real.log (ProbabilityTheory.mgf (fun ω => X p.2 ω - X p.1 ω) μ t)
            _ ≤ Real.log (Real.exp (t^2 * (σ * dyadicScale D k)^2 / 2)) :=
                Real.log_le_log h_mgf_pos h_mgf_bound
            _ = t^2 * (σ * dyadicScale D k)^2 / 2 := Real.log_exp _
        calc ∫ ω, SΔ_trans k ω ∂μ
          _ ≤ (σ * dyadicScale D k) * Real.sqrt (2 * Real.log (h_chain_trans_finite k).toFinset.card) :=
              expected_max_subGaussian hσ'_pos h_ne hcard_trans
                (fun p _ => (hX_meas p.2).sub (hX_meas p.1))
                (fun p _ => hX_int p.2 p.1)
                hY_cgf (fun p _ t => hX_int_exp p.2 p.1 t)
          _ ≤ (σ * dyadicScale D k) * Real.sqrt (2 * Real.log (dn.nets (k + 1)).card) := by
              apply mul_le_mul_of_nonneg_left _ (le_of_lt hσ'_pos)
              apply Real.sqrt_le_sqrt
              apply mul_le_mul_of_nonneg_left _ (by norm_num : (0:ℝ) ≤ 2)
              apply Real.log_le_log
              · exact Nat.cast_pos.mpr (Finset.Nonempty.card_pos h_ne)
              · exact Nat.cast_le.mpr (h_chain_trans_card_le k hk')
          _ = Real.sqrt 2 * σ * dyadicScale D k * Real.sqrt (Real.log (dn.nets (k + 1)).card) := by
              have h_sqrt_mul : Real.sqrt (2 * Real.log (dn.nets (k + 1)).card) =
                  Real.sqrt 2 * Real.sqrt (Real.log (dn.nets (k + 1)).card) :=
                Real.sqrt_mul (by norm_num : (0 : ℝ) ≤ 2) _
              rw [h_sqrt_mul]; ring
      case neg =>
        push Not at hcard_trans
        have h_ne : ((h_chain_trans_finite k).toFinset).Nonempty := by
          rw [Set.Finite.toFinset_nonempty]; exact h_chain_trans_nonempty k
        have h_card_1 : (h_chain_trans_finite k).toFinset.card = 1 := by
          have h_ge_1 := Finset.Nonempty.card_pos h_ne; omega
        obtain ⟨p₀, hp₀_mem⟩ := h_ne
        have h_singleton : (h_chain_trans_finite k).toFinset = {p₀} := by
          apply Finset.eq_singleton_iff_unique_mem.mpr
          constructor
          · exact hp₀_mem
          · intro p hp
            have h1 : (h_chain_trans_finite k).toFinset.card = 1 := h_card_1
            rw [Finset.card_eq_one] at h1
            obtain ⟨a, ha⟩ := h1
            rw [ha] at hp₀_mem hp
            simp only [Finset.mem_singleton] at hp₀_mem hp
            rw [hp, hp₀_mem]
        calc ∫ ω, SΔ_trans k ω ∂μ
          _ = ∫ ω, X p₀.2 ω - X p₀.1 ω ∂μ := by
              congr 1; ext ω
              simp only [SΔ_trans, h_singleton, Finset.sup'_singleton]
          _ = 0 := hX_centered p₀.2 p₀.1
          _ ≤ Real.sqrt 2 * σ * dyadicScale D k * Real.sqrt (Real.log (dn.nets (k + 1)).card) := by
              apply mul_nonneg; apply mul_nonneg; apply mul_nonneg
              · exact Real.sqrt_nonneg _
              · exact le_of_lt hσ
              · exact le_of_lt (dyadicScale_pos hD k)
              · exact Real.sqrt_nonneg _

    -- h2_trans: sum of increments ≤ (4√2) σ · dyadicRHS
    have h2_trans : ∫ (ω : Ω), ∑ k ∈ Finset.range K, SΔ_trans k ω ∂μ ≤
        (4 * Real.sqrt 2) * σ * dyadicRHS_real s D (K + 1) := by
      calc ∫ (ω : Ω), ∑ k ∈ Finset.range K, SΔ_trans k ω ∂μ
        _ = ∑ k ∈ Finset.range K, ∫ (ω : Ω), SΔ_trans k ω ∂μ := by
            rw [integral_finsetSum]; intro k _; exact h_SΔ_trans_int k
        _ ≤ ∑ k ∈ Finset.range K, Real.sqrt 2 * σ * dyadicScale D k *
            Real.sqrt (Real.log (dn.nets (k + 1)).card) :=
            Finset.sum_le_sum (fun k hk => h2a_trans k hk)
        _ ≤ (4 * Real.sqrt 2) * σ * dyadicRHS_real s D (K + 1) := by
            -- Factor out √2 σ and convert to sqrtEntropy structure
            have h_sum_factored : ∑ k ∈ Finset.range K, Real.sqrt 2 * σ * dyadicScale D k *
                Real.sqrt (Real.log (dn.nets (k + 1)).card) =
                Real.sqrt 2 * σ * ∑ k ∈ Finset.range K, dyadicScale D k *
                Real.sqrt (Real.log (dn.nets (k + 1)).card) := by
              rw [Finset.mul_sum]; congr 1; ext k; ring
            rw [h_sum_factored]
            -- KEY: √(log |nets(k+1)|) ≤ sqrtEntropy(ε_{k+2}) (single term, not sum)
            have h_sqrt_log_le : ∀ k, Real.sqrt (Real.log (dn.nets (k + 1)).card) ≤
                sqrtEntropy (dyadicScale D (k + 2)) s := by
              intro k
              -- Use sqrt_log_card_le_sqrtEntropy_of_card_le (same as h_net_sqrt_le)
              apply sqrt_log_card_le_sqrtEntropy_of_card_le (dyadicScale_pos hD (k + 2)) hs
              -- Convert from ℕ ≤ ℕ (after untop) to WithTop ℕ ≤ WithTop ℕ
              have hne : coveringNumber (dyadicScale D (k + 2)) s ≠ ⊤ :=
                ne_top_of_lt (coveringNumber_lt_top_of_totallyBounded (dyadicScale_pos hD (k + 2)) hs)
              rw [← WithTop.coe_untop (coveringNumber (dyadicScale D (k + 2)) s) hne]
              exact WithTop.coe_le_coe.mpr (hcard_bound (k + 1))
            have h_sum_le : ∑ k ∈ Finset.range K, dyadicScale D k *
                Real.sqrt (Real.log (dn.nets (k + 1)).card) ≤
                ∑ k ∈ Finset.range K, dyadicScale D k * sqrtEntropy (dyadicScale D (k + 2)) s := by
              apply Finset.sum_le_sum; intro k _
              apply mul_le_mul_of_nonneg_left (h_sqrt_log_le k)
              exact le_of_lt (dyadicScale_pos hD k)
            -- Factor: ε_k = 4 × ε_{k+2}
            have h2d_trans : ∑ k ∈ Finset.range K, dyadicScale D k *
                sqrtEntropy (dyadicScale D (k + 2)) s ≤ 4 * dyadicRHS_real s D (K + 1) := by
              have h_scale_factor : ∀ k, dyadicScale D k = 4 * dyadicScale D (k + 2) := by
                intro k; simp only [dyadicScale]; rw [zpow_neg, zpow_neg, zpow_natCast, zpow_natCast]
                have h_pow : (2:ℝ)^(k+2) = (2:ℝ)^k * 4 := by rw [pow_add]; norm_num
                rw [h_pow]; field_simp
              have h_sum_eq : ∑ k ∈ Finset.range K, dyadicScale D k * sqrtEntropy (dyadicScale D (k + 2)) s =
                  4 * ∑ k ∈ Finset.range K, dyadicScale D (k + 2) * sqrtEntropy (dyadicScale D (k + 2)) s := by
                have : ∀ k ∈ Finset.range K, dyadicScale D k * sqrtEntropy (dyadicScale D (k + 2)) s =
                    4 * (dyadicScale D (k + 2) * sqrtEntropy (dyadicScale D (k + 2)) s) := by
                  intro k _; rw [h_scale_factor k]; ring
                rw [Finset.sum_congr rfl this, ← Finset.mul_sum]
              rw [h_sum_eq]
              apply mul_le_mul_of_nonneg_left _ (by norm_num : (0:ℝ) ≤ 4)
              unfold dyadicRHS_real
              have h_unfold_scale_k2 : ∀ k, dyadicScale D (k + 2) = D * (2:ℝ)^(-((k:ℤ) + 2)) := by
                intro k; simp only [dyadicScale, Nat.cast_add, Nat.cast_ofNat]
              have h_sum_rewrite : ∑ k ∈ Finset.range K, dyadicScale D (k + 2) * sqrtEntropy (dyadicScale D (k + 2)) s =
                  D * ∑ k ∈ Finset.range K, (2:ℝ)^(-((k:ℤ) + 2)) * sqrtEntropy (D * (2:ℝ)^(-((k:ℤ) + 2))) s := by
                have : ∀ k ∈ Finset.range K, dyadicScale D (k + 2) * sqrtEntropy (dyadicScale D (k + 2)) s =
                    D * ((2:ℝ)^(-((k:ℤ) + 2)) * sqrtEntropy (D * (2:ℝ)^(-((k:ℤ) + 2))) s) := by
                  intro k _; rw [h_unfold_scale_k2 k]; ring
                rw [Finset.sum_congr rfl this, ← Finset.mul_sum]
              rw [h_sum_rewrite]
              have h_index_shift : ∑ k ∈ Finset.range K, (2:ℝ)^(-((k:ℤ) + 2)) * sqrtEntropy (D * (2:ℝ)^(-((k:ℤ) + 2))) s ≤
                  ∑ k ∈ Finset.range (K + 1), (2:ℝ)^(-((k:ℤ) + 1)) * sqrtEntropy (D * (2:ℝ)^(-((k:ℤ) + 1))) s := by
                let f := fun j : ℕ => (2:ℝ)^(-((j:ℤ) + 1)) * sqrtEntropy (D * (2:ℝ)^(-((j:ℤ) + 1))) s
                have h_lhs_eq : ∑ k ∈ Finset.range K, (2:ℝ)^(-((k:ℤ) + 2)) * sqrtEntropy (D * (2:ℝ)^(-((k:ℤ) + 2))) s =
                    ∑ k ∈ Finset.range K, f (k + 1) := by
                  apply Finset.sum_congr rfl; intro k _; simp only [f]; congr 2
                rw [h_lhs_eq]
                have h_sum_ico : ∑ k ∈ Finset.range K, f (k + 1) = ∑ j ∈ Finset.Ico 1 (K + 1), f j := by
                  symm; rw [Finset.sum_Ico_eq_sum_range]; simp only [Nat.add_sub_cancel]
                  apply Finset.sum_congr rfl; intro x _; ring_nf
                rw [h_sum_ico]
                apply Finset.sum_le_sum_of_subset_of_nonneg
                · intro j hj; simp only [Finset.mem_Ico, Finset.mem_range] at hj ⊢; omega
                · intro j _ _; apply mul_nonneg (zpow_nonneg (by norm_num : (0:ℝ) ≤ 2) _) (sqrtEntropy_nonneg _ _)
              calc D * ∑ k ∈ Finset.range K, (2:ℝ)^(-((k:ℤ) + 2)) * sqrtEntropy (D * (2:ℝ)^(-((k:ℤ) + 2))) s
                _ ≤ D * ∑ k ∈ Finset.range (K + 1), (2:ℝ)^(-((k:ℤ) + 1)) * sqrtEntropy (D * (2:ℝ)^(-((k:ℤ) + 1))) s := by
                    apply mul_le_mul_of_nonneg_left h_index_shift (le_of_lt hD)
                _ ≤ D * ∑ k ∈ Finset.range (K + 1), (2:ℝ)^(-((k:ℤ) + 1)) * sqrtEntropy (D * (2:ℝ)^(-((k:ℤ) + 1))) s +
                    D * (2:ℝ)^(-((K + 1):ℤ)) * sqrtEntropy (D * (2:ℝ)^(-((K + 1):ℤ))) s := by
                    apply le_add_of_nonneg_right
                    apply mul_nonneg (mul_nonneg (le_of_lt hD) _) (sqrtEntropy_nonneg _ _)
                    apply zpow_nonneg (by norm_num : (0:ℝ) ≤ 2)
            -- Final assembly: √2 × 4 = 4√2 (exact)
            calc Real.sqrt 2 * σ * ∑ k ∈ Finset.range K, dyadicScale D k * Real.sqrt (Real.log (dn.nets (k + 1)).card)
              _ ≤ Real.sqrt 2 * σ * ∑ k ∈ Finset.range K, dyadicScale D k * sqrtEntropy (dyadicScale D (k + 2)) s := by
                  apply mul_le_mul_of_nonneg_left h_sum_le
                  apply mul_nonneg (Real.sqrt_nonneg _) (le_of_lt hσ)
              _ ≤ Real.sqrt 2 * σ * (4 * dyadicRHS_real s D (K + 1)) := by
                  apply mul_le_mul_of_nonneg_left h2d_trans
                  apply mul_nonneg (Real.sqrt_nonneg _) (le_of_lt hσ)
              _ = (4 * Real.sqrt 2) * σ * dyadicRHS_real s D (K + 1) := by ring

    -- Integrability for transitive sum
    have h_SΔ_trans_sum_int : Integrable (fun ω => ∑ k ∈ Finset.range K, SΔ_trans k ω) μ := by
      apply integrable_finsetSum
      intro k _
      exact h_SΔ_trans_int k

    -- Sum bound using transitive projection (6√2)
    have h_sum_bound_trans : ∫ (ω : Ω), SBase_fin ω ∂μ + ∫ (ω : Ω), ∑ k ∈ Finset.range K, SΔ_trans k ω ∂μ
        ≤ (6 * Real.sqrt 2) * σ * dyadicRHS_real s D (K + 1) := by
      calc ∫ (ω : Ω), SBase_fin ω ∂μ + ∫ (ω : Ω), ∑ k ∈ Finset.range K, SΔ_trans k ω ∂μ
        _ ≤ (2 * Real.sqrt 2) * σ * dyadicRHS_real s D (K + 1) + (4 * Real.sqrt 2) * σ * dyadicRHS_real s D (K + 1) :=
            add_le_add h1 h2_trans
        _ = (2 * Real.sqrt 2 + 4 * Real.sqrt 2) * σ * dyadicRHS_real s D (K + 1) := by ring
        _ = (6 * Real.sqrt 2) * σ * dyadicRHS_real s D (K + 1) := by ring

    -- Final assembly via pathwise decomposition
    calc ∫ (ω : Ω), S_net ω ∂μ
      _ ≤ ∫ (ω : Ω), SBase_fin ω + ∑ k ∈ Finset.range K, SΔ_trans k ω ∂μ := by
          apply integral_mono_ae h_S_net_int (h_SBase_fin_int.add h_SΔ_trans_sum_int)
          exact h_S_net_pathwise_trans
      _ = ∫ (ω : Ω), SBase_fin ω ∂μ + ∫ (ω : Ω), ∑ k ∈ Finset.range K, SΔ_trans k ω ∂μ := by
          rw [integral_add h_SBase_fin_int h_SΔ_trans_sum_int]
      _ ≤ (6 * Real.sqrt 2) * σ * dyadicRHS_real s D (K + 1) := h_sum_bound_trans

  exact h_goal

/-- Helper lemma: if u → L and u ≤ v pointwise with v bounded below,
and liminf v < ⊤ (in EReal sense), then L ≤ liminf v.
-/
lemma tendsto_le_liminf_of_le' {u v : ℕ → ℝ} {L : ℝ}
    (hu : Filter.Tendsto u Filter.atTop (nhds L))
    (h : ∀ n, u n ≤ v n)
    (hv_bdd_below : Filter.IsBoundedUnder (· ≥ ·) Filter.atTop v)
    (hv_liminf_ereal_ne_top : Filter.liminf (fun n => (v n : EReal)) Filter.atTop ≠ ⊤) :
    L ≤ Filter.liminf v Filter.atTop := by
  -- Use EReal for monotonicity, then convert back using the finite liminf assumption
  have h_liminf_ne_bot : Filter.liminf (fun n => (v n : EReal)) Filter.atTop ≠ ⊥ := by
    obtain ⟨M, hM⟩ := hv_bdd_below
    rw [Filter.eventually_map] at hM
    intro h_eq
    have h_le_M : (M : EReal) ≤ Filter.liminf (fun n => (v n : EReal)) Filter.atTop := by
      apply Filter.le_liminf_of_le
      · simp only [Filter.IsCoboundedUnder, Filter.IsCobounded]; use ⊤; intro a _; exact le_top
      · filter_upwards [hM] with n hn; exact EReal.coe_le_coe_iff.mpr hn
    rw [h_eq] at h_le_M; exact (EReal.bot_lt_coe M).not_ge h_le_M
  -- Now convert EReal liminf to Real liminf
  rw [Filter.liminf_eq]
  have h_bdd_above : BddAbove {a | ∀ᶠ n in Filter.atTop, a ≤ v n} := by
    use (Filter.liminf (fun n => (v n : EReal)) Filter.atTop).toReal + 1
    intro a ha; simp only [Set.mem_setOf_eq] at ha
    have h_a_le : (a : EReal) ≤ Filter.liminf (fun n => (v n : EReal)) Filter.atTop := by
      apply Filter.le_liminf_of_le
      · simp only [Filter.IsCoboundedUnder, Filter.IsCobounded]; use ⊤; intro b _; exact le_top
      · filter_upwards [ha] with n hn; exact EReal.coe_le_coe_iff.mpr hn
    have h_toReal_eq := EReal.coe_toReal hv_liminf_ereal_ne_top h_liminf_ne_bot
    rw [← h_toReal_eq] at h_a_le
    exact le_of_lt (lt_of_le_of_lt (EReal.coe_le_coe_iff.mp h_a_le) (by linarith))
  have h_dense : ∀ ε > 0, L - ε ≤ sSup {a | ∀ᶠ n in Filter.atTop, a ≤ v n} := by
    intro ε hε
    have h_L_eps_in : L - ε ∈ {a | ∀ᶠ n in Filter.atTop, a ≤ v n} := by
      simp only [Set.mem_setOf_eq]
      have h_ev : ∀ᶠ n in Filter.atTop, L - ε < u n :=
        hu.eventually (Ioi_mem_nhds (by linarith : L - ε < L))
      filter_upwards [h_ev] with n hn
      exact le_of_lt (lt_of_lt_of_le hn (h n))
    exact le_csSup h_bdd_above h_L_eps_in
  by_contra h_neg; push Not at h_neg
  specialize h_dense ((L - sSup {a | ∀ᶠ n in Filter.atTop, a ≤ v n}) / 2) (by linarith)
  linarith

/-- **Dudley Chaining Bound for Countable Supremum**

For a σ-sub-Gaussian process over a countable dense sequence from a totally bounded set s,
the expected supremum is bounded by the entropy integral:
  𝔼[sup_n X(tₙ)] ≤ 12√2 · σ · ∫₀^D √(log N(ε,s)) dε

where {tₙ} is a dense sequence in s. The constant 12√2 ≈ 17 arises from transitive projection
in the dyadic chaining argument.
-/
theorem dudley_chaining_bound_countable {Ω : Type u} [MeasurableSpace Ω] {A : Type v} [PseudoMetricSpace A]
    {μ : Measure Ω} [IsProbabilityMeasure μ]
    {X : A → Ω → ℝ} {σ : ℝ} (hσ : 0 < σ)
    (hX : IsSubGaussianProcess μ X σ)
    {s : Set A} (hs : TotallyBounded s)
    {D : ℝ} (hD : 0 < D) (hdiam : Metric.diam s ≤ D)
    -- Centering assumption: process is centered at a base point
    (t₀ : A) (ht₀ : t₀ ∈ s) (hcenter : ∀ ω, X t₀ ω = 0)
    -- Sub-Gaussian hypotheses
    (hX_meas : ∀ t, Measurable (X t))
    (hX_int_exp : ∀ t s : A, ∀ l : ℝ, Integrable (fun ω => Real.exp (l * (X t ω - X s ω))) μ)
    -- Finiteness and continuity for limit argument
    (hfinite : entropyIntegralENNReal s D ≠ ⊤)
    (hcont : ∀ ω, Continuous (fun (t : ↥s) => X t.1 ω)) :
    ∫ ω, ⨆ n : ℕ, X (denseSeqInTB hs ⟨t₀, ht₀⟩ n).1 ω ∂μ ≤ (12 * Real.sqrt 2) * σ * entropyIntegral s D := by
  classical

  -- ══════════════════════════════════════════════════════════════════════════════
  -- DERIVED HYPOTHESES (from minimal signature)
  -- ══════════════════════════════════════════════════════════════════════════════
  have hsne : s.Nonempty := ⟨t₀, ht₀⟩
  have hX_int : ∀ t s : A, Integrable (fun ω => X t ω - X s ω) μ := fun t' s' =>
    integrable_of_integrable_exp_all (hX_int_exp t' s')
  have hX_centered : ∀ t s : A, ∫ ω, (X t ω - X s ω) ∂μ = 0 := fun t' s' =>
    subGaussian_process_centered hX t' s' (hX_int_exp t' s')

  -- ══════════════════════════════════════════════════════════════════════════════
  -- SECTION 1: SETUP
  -- Dense sequence {tₙ}, dyadic nets, and projections proj_K(tₙ) to level-K nets.
  -- ══════════════════════════════════════════════════════════════════════════════

  let t : ℕ → A := fun n => (denseSeqInTB hs hsne n).1
  have ht_mem : ∀ n, t n ∈ s := fun n => (denseSeqInTB hs hsne n).2

  obtain ⟨dn_good⟩ := exists_goodDyadicNets hs hsne hD
  let dn : DyadicNets s D := dn_good.toDyadicNets

  have hnet_nonempty : ∀ k, (dn.nets k).Nonempty := by
    intro k
    have h_cover := dn.isNet k
    have ⟨x, hx⟩ := hsne
    obtain ⟨y, hy_mem, _⟩ := exists_near_in_net h_cover hx
    exact ⟨y, hy_mem⟩

  let proj : ℕ → ℕ → A := fun K n => projectToNet dn K (hnet_nonempty K) (t n)
  have h_proj_mem : ∀ K n, proj K n ∈ dn.nets K :=
    fun K n => projectToNet_mem dn K (hnet_nonempty K) (t n)
  have h_proj_dist : ∀ K n, dist (t n) (proj K n) ≤ dyadicScale D K :=
    fun K n => dist_projectToNet_le dn K (hnet_nonempty K) (ht_mem n)

  -- Y_K(ω) = sup_{u ∈ nets K} X(u)(ω)
  let Y : ℕ → Ω → ℝ := fun K ω =>
    (dn.nets K).sup' (hnet_nonempty K) (fun u => X u ω)

  -- ══════════════════════════════════════════════════════════════════════════════
  -- SECTION 2: CORE BOUND (Lemma 4.2)
  -- 𝔼[Y_K] ≤ 6√2 · σ · dyadicRHS(K+1), via dudley_chaining_bound_core.
  -- ══════════════════════════════════════════════════════════════════════════════

  have h_core_bound : ∀ K, 1 ≤ K →
      ∫ ω, Y K ω ∂μ ≤ (6 * Real.sqrt 2) * σ * dyadicRHS_real s D (K + 1) := by
    intro K hK
    -- Y_K = sup_{u ∈ nets K} (X(u) - X(t₀)) since X(t₀) = 0
    have h_Y_eq : ∀ ω, Y K ω = (dn.nets K).sup' (hnet_nonempty K) (fun u => X u ω - X t₀ ω) := by
      intro ω
      simp only [Y, hcenter ω, sub_zero]
    conv_lhs => arg 2; ext ω; rw [h_Y_eq ω]
    have hcard_bound : ∀ k, (dn.nets k).card ≤ (coveringNumber (dyadicScale D (k + 1)) s).untop
        (ne_top_of_lt (coveringNumber_lt_top_of_totallyBounded (dyadicScale_pos hD (k + 1)) hs)) := by
      intro k
      exact dn_good.card_bound k
    exact dudley_chaining_bound_core hσ hX hs hD hdiam K hK hX_meas hX_int hX_centered
      hX_int_exp dn hcard_bound hnet_nonempty t₀

  -- ══════════════════════════════════════════════════════════════════════════════
  -- SECTION 3: CONVERGENCE AND DOMINATION (Lemmas 4.3-4.5)
  -- proj_K(tₙ) → tₙ, hence X(proj_K(tₙ)) → X(tₙ), and X(proj_K(tₙ)) ≤ Y_K.
  -- ══════════════════════════════════════════════════════════════════════════════

  -- Lemma 4.3: proj_K(tₙ) → tₙ as K → ∞
  have h_proj_tendsto : ∀ n, Filter.Tendsto (fun K => proj K n) Filter.atTop (nhds (t n)) := by
    intro n
    rw [Metric.tendsto_atTop]
    intro ε hε
    have h_scale_tendsto : Filter.Tendsto (fun K : ℕ => dyadicScale D K) Filter.atTop (nhds 0) := by
      simp only [dyadicScale]
      -- D * (1/2)^K → D * 0 = 0 as K → ∞
      have h1 : (2 : ℝ)⁻¹ < 1 := by norm_num
      have h2 : 0 ≤ (2 : ℝ)⁻¹ := by norm_num
      have h3 : Filter.Tendsto (fun K : ℕ => (2 : ℝ)⁻¹^K) Filter.atTop (nhds 0) :=
        tendsto_pow_atTop_nhds_zero_of_lt_one h2 h1
      have h4 : ∀ K : ℕ, (2 : ℝ)^(-(K : ℤ)) = (2 : ℝ)⁻¹^K := by
        intro K
        rw [zpow_neg, zpow_natCast, inv_pow]
      simp_rw [h4]
      have h5 := h3.const_mul D
      simp only [mul_zero] at h5
      exact h5.congr (fun K => by ring)
    rw [Metric.tendsto_atTop] at h_scale_tendsto
    obtain ⟨N, hN⟩ := h_scale_tendsto ε hε
    use N
    intro K hK
    calc dist (proj K n) (t n) = dist (t n) (proj K n) := dist_comm _ _
      _ ≤ dyadicScale D K := h_proj_dist K n
      _ = |dyadicScale D K - 0| := by simp [abs_of_pos (dyadicScale_pos hD K)]
      _ < ε := hN K hK

  -- Lemma 4.4: X(proj_K(tₙ))(ω) → X(tₙ)(ω) by path continuity
  have h_X_tendsto : ∀ n ω, Filter.Tendsto (fun K => X (proj K n) ω) Filter.atTop (nhds (X (t n) ω)) := by
    intro n ω
    have h_cont_at := (hcont ω).continuousAt (x := ⟨t n, ht_mem n⟩)
    have h_proj_in_s : ∀ K, proj K n ∈ s := fun K => dn.nets_subset K (h_proj_mem K n)
    let proj_sub : ℕ → ↥s := fun K => ⟨proj K n, h_proj_in_s K⟩
    have h_proj_sub_tendsto : Filter.Tendsto proj_sub Filter.atTop (nhds ⟨t n, ht_mem n⟩) := by
      rw [tendsto_subtype_rng]
      exact h_proj_tendsto n
    exact h_cont_at.tendsto.comp h_proj_sub_tendsto

  -- Lemma 4.5: X(proj_K(tₙ))(ω) ≤ Y_K(ω)
  have h_le_Y : ∀ K n ω, X (proj K n) ω ≤ Y K ω := by
    intro K n ω
    exact Finset.le_sup' (fun u => X u ω) (h_proj_mem K n)

  -- ══════════════════════════════════════════════════════════════════════════════
  -- SECTION 4: BOUNDEDNESS (Lemmas 4.6-4.7)
  -- Y_K is bounded below pointwise, integrable for K ≥ 1, and liminf Y_K ≠ +∞ a.e.
  -- ══════════════════════════════════════════════════════════════════════════════

  -- Lemma 4.6: Y_K bounded below via convergence of X(proj_K(t₀))
  have h_Y_bdd_below : ∀ ω, Filter.IsBoundedUnder (· ≥ ·) Filter.atTop (fun K => Y K ω) := by
    intro ω
    have h_tendsto := h_X_tendsto 0 ω
    have h_bdd := h_tendsto.isBoundedUnder_ge
    obtain ⟨b, hb⟩ := h_bdd
    refine ⟨b, ?_⟩
    rw [Filter.eventually_map] at hb ⊢
    filter_upwards [hb] with K hK
    exact le_trans hK (h_le_Y K 0 ω)

  -- Y_K integrable: |Y_K| ≤ Σ_{u ∈ nets K} |X u|
  have h_Y_int : ∀ K, 1 ≤ K → Integrable (Y K) μ := by
    intro K hK
    have h_X_int' : ∀ u, u ∈ dn.nets K → Integrable (fun ω => X u ω) μ := by
      intro u hu
      have h_int_diff := hX_int u t₀
      simp only [hcenter, sub_zero] at h_int_diff
      exact h_int_diff
    refine Integrable.mono (integrable_finsetSum (dn.nets K)
      (fun u hu => (h_X_int' u hu).abs)) ?_ ?_
    · apply Measurable.aestronglyMeasurable
      convert Finset.measurable_sup' (hnet_nonempty K) (fun u _ => hX_meas u) using 1
      funext ω
      simp only [Y, Finset.sup'_apply]
    · refine ae_of_all μ (fun ω => ?_)
      simp only [Real.norm_eq_abs]
      have h_sum_nonneg : 0 ≤ ∑ u ∈ dn.nets K, |X u ω| :=
        Finset.sum_nonneg (fun u _ => abs_nonneg _)
      rw [abs_of_nonneg h_sum_nonneg]
      simp only [Y]
      calc |(dn.nets K).sup' (hnet_nonempty K) (fun u => X u ω)|
        _ ≤ ∑ u ∈ dn.nets K, |X u ω| := abs_sup'_le_sum (hnet_nonempty K) (fun u => X u ω)

  -- Lemma 4.7: liminf Y_K(ω) < +∞ a.e. via uniform L¹ bounds
  have h_Y_liminf_ne_top : ∀ᵐ ω ∂μ,
      Filter.liminf (fun K => (Y K ω : EReal)) Filter.atTop ≠ ⊤ := by

    have h_Y_meas : ∀ K, Measurable (Y K) := by
      intro K
      simp only [Y]
      convert Finset.measurable_sup' (hnet_nonempty K) (fun u _ => hX_meas u)
      simp only [Finset.sup'_apply]

    -- Uniform L¹ bound: 𝔼[|Y_K|] = 𝔼[Y_K] + 2𝔼[Y_K⁻] ≤ C₁ + 2√(2π)σD
    have h_bound_exist : ∃ R : NNReal, ∀ K, 1 ≤ K → MeasureTheory.eLpNorm (Y K) 1 μ ≤ (R : ℝ≥0∞) := by
      -- dyadicRHS bounded by 4 * entropyIntegral
      have h_rhs_bounded : ∃ C₁ : ℝ, 0 < C₁ ∧ ∀ K, 1 ≤ K → dyadicRHS_real s D (K + 1) ≤ C₁ := by
        have hδ_pos : ∀ K : ℕ, 0 < D * (2 : ℝ)^(-((K + 1) : ℤ)) := by
          intro K
          apply mul_pos hD
          apply zpow_pos (by norm_num : (0 : ℝ) < 2)

        have h_dyadic_upper : ∀ K, dyadicRHS_real s D (K + 1) ≤
            2 * entropyIntegralTrunc s (D * (2 : ℝ)^(-((K + 1) : ℤ))) D +
            2 * (D * (2 : ℝ)^(-((K + 1) : ℤ))) * sqrtEntropy (D * (2 : ℝ)^(-((K + 1) : ℤ))) s := by
          intro K
          exact dyadicRHS_le_twice_entropyIntegralTrunc_add_tail hs hD (Nat.le_add_left 1 K)

        have h_trunc_le_full : ∀ K : ℕ, entropyIntegralTrunc s (D * (2 : ℝ)^(-((K + 1) : ℤ))) D ≤
            entropyIntegral s D := by
          intro K
          unfold entropyIntegralTrunc entropyIntegral
          apply ENNReal.toReal_mono hfinite
          apply MeasureTheory.lintegral_mono'
          · apply Measure.restrict_mono
            · intro x hx; exact ⟨lt_of_lt_of_le (hδ_pos K) (le_of_lt hx.1), hx.2⟩
            · exact le_rfl
          · exact fun _ => le_rfl

        have h_tail_le_integral : ∀ K : ℕ,
            (D * (2 : ℝ)^(-((K + 1) : ℤ))) * sqrtEntropy (D * (2 : ℝ)^(-((K + 1) : ℤ))) s ≤
            entropyIntegral s D := by
          intro K
          let δ := D * (2 : ℝ)^(-((K + 1) : ℤ))
          have hδ_pos' : 0 < δ := hδ_pos K
          have hδ_le_D : δ ≤ D := by
            simp only [δ]
            have h1 : (2 : ℝ) ^ (-((K + 1) : ℤ)) ≤ 1 := by
              apply zpow_le_one_of_nonpos₀ (by norm_num : (1 : ℝ) ≤ 2)
              simp only [Left.neg_nonpos_iff]
              exact Int.natCast_nonneg (K + 1)
            calc D * (2 : ℝ) ^ (-((K + 1) : ℤ)) ≤ D * 1 := mul_le_mul_of_nonneg_left h1 (le_of_lt hD)
              _ = D := mul_one D
          have h_le_gap : δ * sqrtEntropy δ s ≤ entropyIntegral s D - entropyIntegralTrunc s δ D := by
            have h_gap_eq : entropyIntegral s D - entropyIntegralTrunc s δ D =
                (entropyIntegralENNReal s D - entropyIntegralENNRealTrunc s δ D).toReal := by
              unfold entropyIntegral entropyIntegralTrunc
              have h_trunc_le : entropyIntegralENNRealTrunc s δ D ≤ entropyIntegralENNReal s D := by
                unfold entropyIntegralENNRealTrunc entropyIntegralENNReal
                apply MeasureTheory.lintegral_mono'
                · apply Measure.restrict_mono
                  · intro x hx; exact ⟨lt_of_lt_of_le hδ_pos' (le_of_lt hx.1), hx.2⟩
                  · exact le_rfl
                · exact fun _ => le_rfl
              rw [ENNReal.toReal_sub_of_le h_trunc_le hfinite]
            have h_ennreal_gap_eq : entropyIntegralENNReal s D - entropyIntegralENNRealTrunc s δ D =
                ∫⁻ eps in Set.Ioc 0 δ, dudleyIntegrand eps s := by
              unfold entropyIntegralENNReal entropyIntegralENNRealTrunc
              have h_union : Set.Ioc 0 D = Set.Ioc 0 δ ∪ Set.Ioc δ D :=
                (Set.Ioc_union_Ioc_eq_Ioc (le_of_lt hδ_pos') hδ_le_D).symm
              have h_disjoint : Disjoint (Set.Ioc 0 δ) (Set.Ioc δ D) :=
                Set.Ioc_disjoint_Ioc_of_le (le_refl δ)
              conv_lhs => rw [h_union]
              rw [MeasureTheory.lintegral_union measurableSet_Ioc h_disjoint]
              rw [ENNReal.add_sub_cancel_right]
              have h_trunc_lt_top : ∫⁻ x in Set.Ioc δ D, dudleyIntegrand x s < ⊤ :=
                entropyIntegralENNRealTrunc_lt_top hδ_pos' hs
              exact ne_top_of_lt h_trunc_lt_top
            have h_lower : ENNReal.ofReal δ * dudleyIntegrand δ s ≤
                ∫⁻ eps in Set.Ioc 0 δ, dudleyIntegrand eps s := by
              have h_anti : ∀ x ∈ Set.Ioc 0 δ, dudleyIntegrand δ s ≤ dudleyIntegrand x s := by
                intro x hx
                have hx_pos : 0 < x := hx.1
                have hx_le_δ : x ≤ δ := hx.2
                exact dudleyIntegrand_anti_eps_of_totallyBounded hx_pos hx_le_δ hs
              calc ENNReal.ofReal δ * dudleyIntegrand δ s
                  = ∫⁻ _ in Set.Ioc 0 δ, dudleyIntegrand δ s := by
                    rw [MeasureTheory.lintegral_const,
                        MeasureTheory.Measure.restrict_apply MeasurableSet.univ]
                    simp only [Set.univ_inter, Real.volume_Ioc, sub_zero]
                    rw [mul_comm]
                _ ≤ ∫⁻ x in Set.Ioc 0 δ, dudleyIntegrand x s := by
                    exact MeasureTheory.setLIntegral_mono' measurableSet_Ioc h_anti
            have h_gap_finite : ∫⁻ eps in Set.Ioc 0 δ, dudleyIntegrand eps s ≠ ⊤ := by
              rw [← h_ennreal_gap_eq]
              exact ENNReal.sub_ne_top hfinite
            calc δ * sqrtEntropy δ s
                = (ENNReal.ofReal δ * dudleyIntegrand δ s).toReal := by
                  rw [ENNReal.toReal_mul, ENNReal.toReal_ofReal (le_of_lt hδ_pos')]
                  unfold dudleyIntegrand
                  rw [ENNReal.toReal_ofReal (sqrtEntropy_nonneg δ s)]
              _ ≤ (∫⁻ eps in Set.Ioc 0 δ, dudleyIntegrand eps s).toReal := by
                  apply ENNReal.toReal_mono h_gap_finite
                  exact h_lower
              _ = entropyIntegral s D - entropyIntegralTrunc s δ D := by
                  rw [h_gap_eq, h_ennreal_gap_eq]
          have h_trunc_nonneg : 0 ≤ entropyIntegralTrunc s δ D := ENNReal.toReal_nonneg
          simp only [δ] at h_le_gap h_trunc_nonneg ⊢
          linarith

        have h_dyadic_le_4integral : ∀ K, dyadicRHS_real s D (K + 1) ≤ 4 * entropyIntegral s D := by
          intro K
          calc dyadicRHS_real s D (K + 1)
              ≤ 2 * entropyIntegralTrunc s (D * (2 : ℝ)^(-((K + 1) : ℤ))) D +
                2 * (D * (2 : ℝ)^(-((K + 1) : ℤ))) * sqrtEntropy (D * (2 : ℝ)^(-((K + 1) : ℤ))) s :=
                h_dyadic_upper K
            _ ≤ 2 * entropyIntegral s D + 2 * entropyIntegral s D := by
                have h1 := h_trunc_le_full K
                have h2 := h_tail_le_integral K
                linarith
            _ = 4 * entropyIntegral s D := by ring

        use 4 * entropyIntegral s D + 1
        constructor
        · have h_int_nonneg : 0 ≤ entropyIntegral s D := entropyIntegral_nonneg
          linarith
        · intro K hK
          have := h_dyadic_le_4integral K
          linarith

      obtain ⟨C₁, hC₁_pos, hC₁⟩ := h_rhs_bounded

      -- 𝔼[|X(proj_K(t₀))|] ≤ √(2π)σD via sub-Gaussian first moment
      have h_proj_abs_bound : ∀ K, ∫ ω, |X (proj K 0) ω| ∂μ ≤ Real.sqrt (2 * Real.pi) * σ * D := by
        intro K
        have h_eq : ∀ ω, X (proj K 0) ω = X (proj K 0) ω - X t₀ ω := by
          intro ω; simp [hcenter ω]
        conv_lhs => arg 2; ext ω; rw [h_eq ω]
        have h_dist_bound : dist (proj K 0) t₀ ≤ D := by
          have h1 := dn.nets_subset K (h_proj_mem K 0)
          have h2 : t₀ ∈ s := ht₀
          exact Metric.dist_le_diam_of_mem (TotallyBounded.isBounded hs) h1 h2 |>.trans hdiam
        calc ∫ ω, |X (proj K 0) ω - X t₀ ω| ∂μ
          _ ≤ Real.sqrt (2 * Real.pi) * σ * dist (proj K 0) t₀ :=
              subGaussian_first_moment_bound hσ hX (proj K 0) t₀
                (hX_int (proj K 0) t₀) (hX_int_exp (proj K 0) t₀)
          _ ≤ Real.sqrt (2 * Real.pi) * σ * D := by
              apply mul_le_mul_of_nonneg_left h_dist_bound
              apply mul_nonneg (Real.sqrt_nonneg _) (le_of_lt hσ)

      -- Y_K⁻ ≤ |X(proj_K(t₀))| since Y_K ≥ X(proj_K(t₀))
      have h_neg_part_bound : ∀ K, 1 ≤ K → ∫ ω, (Y K ω)⁻ ∂μ ≤ Real.sqrt (2 * Real.pi) * σ * D := by
        intro K hK
        have h_Y_ge_X : ∀ ω, X (proj K 0) ω ≤ Y K ω := h_le_Y K 0
        have h_neg_le : ∀ ω, (Y K ω)⁻ ≤ |X (proj K 0) ω| := by
          intro ω
          by_cases hY : Y K ω ≥ 0
          · simp only [negPart_def, max_eq_right (by linarith : -Y K ω ≤ 0)]
            exact abs_nonneg _
          · push Not at hY
            simp only [negPart_def, max_eq_left (by linarith : 0 ≤ -Y K ω)]
            have : -Y K ω ≤ -X (proj K 0) ω := by linarith [h_Y_ge_X ω]
            calc -Y K ω ≤ -X (proj K 0) ω := this
              _ ≤ |X (proj K 0) ω| := neg_le_abs _
        have h_abs_eq_diff : ∀ ω, |X (proj K 0) ω| = |X (proj K 0) ω - X t₀ ω| := by
          intro ω; simp [hcenter ω]
        have h_int_abs : Integrable (fun ω => |X (proj K 0) ω|) μ := by
          have h := (hX_int (proj K 0) t₀).abs
          convert h using 2 with ω
          exact h_abs_eq_diff ω
        calc ∫ ω, (Y K ω)⁻ ∂μ
          _ ≤ ∫ ω, |X (proj K 0) ω| ∂μ := by
              apply integral_mono
              · exact (h_Y_int K hK).neg_part
              · exact h_int_abs
              · exact h_neg_le
          _ ≤ Real.sqrt (2 * Real.pi) * σ * D := h_proj_abs_bound K

      -- E[|Y_K|] ≤ C via |Y| = Y + 2Y⁻
      have h_abs_bound : ∀ K, 1 ≤ K → ∫ ω, |Y K ω| ∂μ ≤ (6 * Real.sqrt 2) * σ * C₁ + 2 * Real.sqrt (2 * Real.pi) * σ * D := by
        intro K hK
        have h_abs_eq : ∀ ω, |Y K ω| = Y K ω + 2 * (Y K ω)⁻ := by
          intro ω
          have h1 : (Y K ω)⁺ - (Y K ω)⁻ = Y K ω := posPart_sub_negPart (Y K ω)
          have h2 : (Y K ω)⁺ + (Y K ω)⁻ = |Y K ω| := posPart_add_negPart (Y K ω)
          linarith
        have h_int_split : ∫ ω, Y K ω + 2 * (Y K ω)⁻ ∂μ = ∫ ω, Y K ω ∂μ + 2 * ∫ ω, (Y K ω)⁻ ∂μ := by
          have h1 : ∫ ω, Y K ω + 2 * (Y K ω)⁻ ∂μ = ∫ ω, Y K ω ∂μ + ∫ ω, 2 * (Y K ω)⁻ ∂μ :=
            integral_add (h_Y_int K hK) ((h_Y_int K hK).neg_part.const_mul 2)
          have h2 : ∫ ω, 2 * (Y K ω)⁻ ∂μ = 2 * ∫ ω, (Y K ω)⁻ ∂μ :=
            MeasureTheory.integral_const_mul 2 _
          rw [h1, h2]
        conv_lhs => arg 2; ext ω; rw [h_abs_eq ω]
        rw [h_int_split]
        calc ∫ ω, Y K ω ∂μ + 2 * ∫ ω, (Y K ω)⁻ ∂μ
          _ ≤ (6 * Real.sqrt 2) * σ * dyadicRHS_real s D (K + 1) + 2 * (Real.sqrt (2 * Real.pi) * σ * D) := by
              apply add_le_add (h_core_bound K hK)
              apply mul_le_mul_of_nonneg_left (h_neg_part_bound K hK) (by norm_num)
          _ ≤ (6 * Real.sqrt 2) * σ * C₁ + 2 * (Real.sqrt (2 * Real.pi) * σ * D) := by
              gcongr
              exact hC₁ K hK
          _ = (6 * Real.sqrt 2) * σ * C₁ + 2 * Real.sqrt (2 * Real.pi) * σ * D := by ring

      -- Convert to eLpNorm bound
      let R_real := (6 * Real.sqrt 2) * σ * C₁ + 2 * Real.sqrt (2 * Real.pi) * σ * D
      have hR_real_nonneg : 0 ≤ R_real := by
        simp only [R_real]
        have h1 : 0 ≤ (6 * Real.sqrt 2) * σ * C₁ := by positivity
        have h2 : 0 ≤ 2 * Real.sqrt (2 * Real.pi) * σ * D := by positivity
        linarith
      use ⟨R_real, hR_real_nonneg⟩
      intro K hK
      rw [MeasureTheory.eLpNorm_one_eq_lintegral_enorm]
      simp only [Real.enorm_eq_ofReal_abs]
      have h_int_abs := (h_Y_int K hK).abs
      rw [← MeasureTheory.ofReal_integral_eq_lintegral_ofReal h_int_abs (ae_of_all μ (fun ω => abs_nonneg _))]
      rw [ENNReal.coe_nnreal_eq]
      apply ENNReal.ofReal_le_ofReal
      exact h_abs_bound K hK

    obtain ⟨R, hR⟩ := h_bound_exist

    -- Apply ae_bdd_liminf using shifted sequence Y'(K) = Y(K+1) for uniform bound
    have h_ae_liminf_finite : ∀ᵐ ω ∂μ,
        Filter.liminf (fun K => ‖Y K ω‖ₑ) Filter.atTop < ⊤ := by
      let Y' : ℕ → Ω → ℝ := fun K => Y (K + 1)
      have hY'_meas : ∀ K, Measurable (Y' K) := fun K => h_Y_meas (K + 1)
      have hY'_bound : ∀ K, MeasureTheory.eLpNorm (Y' K) 1 μ ≤ R := by
        intro K
        exact hR (K + 1) (Nat.le_add_left 1 K)
      have h := MeasureTheory.ae_bdd_liminf_atTop_of_eLpNorm_bdd (p := 1)
        (by norm_num : (1 : ℝ≥0∞) ≠ 0) hY'_meas hY'_bound
      filter_upwards [h] with ω hω
      -- liminf_nat_add: shifting doesn't change liminf
      have h_shift : Filter.liminf (fun K => ‖Y K ω‖ₑ) Filter.atTop =
          Filter.liminf (fun K => ‖Y (K + 1) ω‖ₑ) Filter.atTop := by
        exact (Filter.liminf_nat_add (fun K => ‖Y K ω‖ₑ) 1).symm
      rw [h_shift]
      exact hω

    -- Convert to EReal: liminf ‖Y‖ₑ < ⊤ implies liminf (Y : EReal) ≠ ⊤
    filter_upwards [h_ae_liminf_finite] with ω h_finite
    have h_liminf_ne_top : Filter.liminf (fun K => ‖Y K ω‖ₑ) Filter.atTop ≠ ⊤ :=
      ne_top_of_lt h_finite
    obtain ⟨R, hR⟩ := ENNReal.exists_frequently_lt_of_liminf_ne_top h_liminf_ne_top
    have hR_le : ∃ᶠ K in Filter.atTop, Y K ω ≤ R := hR.mono (fun K hK => le_of_lt hK)
    have h_liminf_le_R : Filter.liminf (fun K => (Y K ω : EReal)) Filter.atTop ≤ (R : EReal) := by
      apply Filter.liminf_le_of_frequently_le'
      exact hR_le.mono (fun K hK => EReal.coe_le_coe_iff.mpr hK)
    intro h_eq_top
    rw [h_eq_top] at h_liminf_le_R
    exact (EReal.coe_ne_top R) (le_antisymm h_liminf_le_R le_top).symm

  -- ══════════════════════════════════════════════════════════════════════════════
  -- SECTION 5: POINTWISE BOUND (Lemma 4.8)
  -- For a.e. ω: sup_n X(tₙ)(ω) ≤ liminf_K Y_K(ω) via tendsto_le_liminf_of_le'.
  -- ══════════════════════════════════════════════════════════════════════════════
  have h_pointwise : ∀ᵐ ω ∂μ, ∀ n, X (t n) ω ≤ Filter.liminf (fun K => Y K ω) Filter.atTop := by
    filter_upwards [h_Y_liminf_ne_top] with ω h_liminf_ne_top n
    exact tendsto_le_liminf_of_le' (h_X_tendsto n ω) (fun K => h_le_Y K n ω)
      (h_Y_bdd_below ω) h_liminf_ne_top

  have h_pointwise_sup : ∀ᵐ ω ∂μ, (⨆ n, X (t n) ω) ≤ Filter.liminf (fun K => Y K ω) Filter.atTop := by
    filter_upwards [h_pointwise] with ω h_all_n
    by_cases h_bdd : BddAbove (Set.range (fun n => X (t n) ω))
    · exact ciSup_le (fun n => h_all_n n)
    · -- Unbounded case: sup = 0 by convention, and liminf ≥ 0 since some X(tₙ) > 0
      rw [Real.iSup_of_not_bddAbove h_bdd]
      have h_liminf_ge_0 : 0 ≤ Filter.liminf (fun K => Y K ω) Filter.atTop := by
        have h_exists_pos : ∃ n, X (t n) ω > 0 := by
          by_contra h_all_le_0
          push Not at h_all_le_0
          have h_bdd' : BddAbove (Set.range (fun n => X (t n) ω)) := by
            use 0
            intro y hy
            obtain ⟨n, rfl⟩ := hy
            exact h_all_le_0 n
          exact h_bdd h_bdd'
        obtain ⟨n₀, hn₀⟩ := h_exists_pos
        calc 0 ≤ X (t n₀) ω := le_of_lt hn₀
          _ ≤ Filter.liminf (fun K => Y K ω) Filter.atTop := h_all_n n₀
      exact h_liminf_ge_0

  -- ══════════════════════════════════════════════════════════════════════════════
  -- SECTION 6: FATOU'S LEMMA (Lemma 4.9)
  -- ∫ sup_n X(tₙ) ≤ liminf_K ∫ Y_K using shifted-Fatou: Z_K = Y_K - g ≥ 0 where
  -- g = inf_K X(proj_K(0)). The shift g is integrable via telescoping bound.
  -- ══════════════════════════════════════════════════════════════════════════════
  have h_fatou : ∫ ω, ⨆ n, X (t n) ω ∂μ ≤ Filter.liminf (fun K => ∫ ω, Y K ω ∂μ) Filter.atTop := by

    -- Define shift function g = inf_K X(proj_K(0))
    let g : Ω → ℝ := fun ω => ⨅ K, X (proj K 0) ω

    -- g is bounded below since X(proj_K(0)) → 0
    have h_g_bdd : ∀ ω, BddBelow (Set.range (fun K => X (proj K 0) ω)) := by
      intro ω
      have h_tendsto := h_X_tendsto 0 ω
      exact h_tendsto.isBoundedUnder_ge.bddBelow_range

    -- Y_K ≥ g for all K, making Z_K = Y_K - g non-negative
    have h_Y_ge_g : ∀ K ω, g ω ≤ Y K ω := by
      intro K ω
      calc g ω = ⨅ J, X (proj J 0) ω := rfl
        _ ≤ X (proj K 0) ω := ciInf_le (h_g_bdd ω) K
        _ ≤ Y K ω := h_le_Y K 0 ω

    -- g is integrable via telescoping: |g| ≤ |X(proj_0(0))| + ∑_K |Δ_K| a.e.
    have h_g_int : Integrable g μ := by
      have h_t0_int : Integrable (fun ω => X (t 0) ω) μ := by
        have h := hX_int (t 0) t₀
        simp only [hcenter, sub_zero] at h
        exact h
      have h_proj0_int : Integrable (fun ω => X (proj 0 0) ω) μ := by
        have h := hX_int (proj 0 0) t₀
        simp only [hcenter, sub_zero] at h
        exact h
      have h_g_meas : Measurable g := Measurable.iInf (fun K => hX_meas (proj K 0))

      -- Δ_K = X(proj_{K+1}(0)) - X(proj_K(0)), consecutive differences
      let Δ : ℕ → Ω → ℝ := fun K ω => X (proj (K + 1) 0) ω - X (proj K 0) ω
      have h_Δ_int : ∀ K, Integrable (Δ K) μ := by
        intro K
        have h1 := hX_int (proj (K + 1) 0) t₀
        have h2 := hX_int (proj K 0) t₀
        simp only [hcenter, sub_zero] at h1 h2
        exact h1.sub h2

      -- E[|Δ_K|] ≤ 3√(2π)σD·2^{-K} via sub-Gaussian first moment bound
      have h_Δ_moment : ∀ K, ∫ ω, |Δ K ω| ∂μ ≤ 3 * Real.sqrt (2 * Real.pi) * σ * D * (2 : ℝ)^(-(K : ℤ)) := by
        intro K
        have h_bound := subGaussian_first_moment_bound hσ hX (proj (K + 1) 0) (proj K 0)
            (hX_int (proj (K + 1) 0) (proj K 0)) (hX_int_exp (proj (K + 1) 0) (proj K 0))
        have h_dist : dist (proj (K + 1) 0) (proj K 0) ≤ dyadicScale D (K + 1) + dyadicScale D K := by
          calc dist (proj (K + 1) 0) (proj K 0)
              ≤ dist (proj (K + 1) 0) (t 0) + dist (t 0) (proj K 0) := dist_triangle _ _ _
            _ ≤ dyadicScale D (K + 1) + dyadicScale D K := by
                apply add_le_add
                · rw [dist_comm]; exact h_proj_dist (K + 1) 0
                · exact h_proj_dist K 0
        have h_scale_bound : dyadicScale D (K + 1) + dyadicScale D K ≤ 3 * D * (2 : ℝ)^(-(K : ℤ)) := by
          simp only [dyadicScale]
          have h1 : (2 : ℝ)^(-(↑(K + 1) : ℤ)) = (2 : ℝ)^(-(K : ℤ)) / 2 := by
            have : (↑(K + 1) : ℤ) = (K : ℤ) + 1 := by simp
            rw [this, neg_add, zpow_add₀ (by norm_num : (2 : ℝ) ≠ 0)]
            simp only [zpow_neg_one]; ring
          rw [h1]
          calc D * ((2 : ℝ)^(-(K : ℤ)) / 2) + D * (2 : ℝ)^(-(K : ℤ))
              = D * (2 : ℝ)^(-(K : ℤ)) * (1/2 + 1) := by ring
            _ = D * (2 : ℝ)^(-(K : ℤ)) * (3/2) := by norm_num
            _ ≤ 3 * D * (2 : ℝ)^(-(K : ℤ)) := by nlinarith [hD.le, zpow_nonneg (by norm_num : (0 : ℝ) ≤ 2) (-(K : ℤ))]
        calc ∫ ω, |Δ K ω| ∂μ = ∫ ω, |X (proj (K + 1) 0) ω - X (proj K 0) ω| ∂μ := rfl
          _ ≤ Real.sqrt (2 * Real.pi) * σ * dist (proj (K + 1) 0) (proj K 0) := h_bound
          _ ≤ Real.sqrt (2 * Real.pi) * σ * (dyadicScale D (K + 1) + dyadicScale D K) := by
              apply mul_le_mul_of_nonneg_left h_dist
              exact mul_nonneg (Real.sqrt_nonneg _) (le_of_lt hσ)
          _ ≤ Real.sqrt (2 * Real.pi) * σ * (3 * D * (2 : ℝ)^(-(K : ℤ))) := by
              apply mul_le_mul_of_nonneg_left h_scale_bound
              exact mul_nonneg (Real.sqrt_nonneg _) (le_of_lt hσ)
          _ = 3 * Real.sqrt (2 * Real.pi) * σ * D * (2 : ℝ)^(-(K : ℤ)) := by ring

      -- ∑_K E[|Δ_K|] < ∞ (geometric series)
      have h_geom : Summable (fun K : ℕ => (2 : ℝ)^(-(K : ℤ))) := by
        have h := summable_geometric_two
        convert h using 1
        ext K
        simp only [one_div, zpow_neg, zpow_natCast, inv_pow]
      have h_sum_Δ_summable : Summable (fun K => ∫ ω, |Δ K ω| ∂μ) := by
        apply Summable.of_nonneg_of_le
        · intro K; exact integral_nonneg (fun ω => abs_nonneg _)
        · exact h_Δ_moment
        · exact h_geom.mul_left _

      -- ∑_K |Δ_K| is a.e. finite via Fubini
      have h_Δ_summable_ae : ∀ᵐ ω ∂μ, Summable (fun K => |Δ K ω|) := by
        have h_lintegral_sum : ∫⁻ ω, ∑' K, ENNReal.ofReal |Δ K ω| ∂μ < ⊤ := by
          calc ∫⁻ ω, ∑' K, ENNReal.ofReal |Δ K ω| ∂μ
              = ∑' K, ∫⁻ ω, ENNReal.ofReal |Δ K ω| ∂μ := by
                apply lintegral_tsum
                intro K
                exact (hX_meas (proj (K + 1) 0)).sub (hX_meas (proj K 0)) |>.abs.ennreal_ofReal.aemeasurable
            _ = ∑' K, ENNReal.ofReal (∫ ω, |Δ K ω| ∂μ) := by
                congr 1
                ext K
                rw [eq_comm]
                apply ofReal_integral_eq_lintegral_ofReal
                · have h_int_norm := (h_Δ_int K).norm
                  simp only [Real.norm_eq_abs] at h_int_norm
                  exact h_int_norm
                · exact ae_of_all μ (fun ω => abs_nonneg _)
            _ ≤ ∑' (K : ℕ), ENNReal.ofReal (3 * Real.sqrt (2 * Real.pi) * σ * D * (2 : ℝ)^(-(K : ℤ))) := by
                apply ENNReal.tsum_le_tsum
                intro K
                exact ENNReal.ofReal_le_ofReal (h_Δ_moment K)
            _ < ⊤ := by
                calc ∑' (K : ℕ), ENNReal.ofReal (3 * Real.sqrt (2 * Real.pi) * σ * D * (2 : ℝ)^(-(K : ℤ)))
                    = ENNReal.ofReal (∑' (K : ℕ), 3 * Real.sqrt (2 * Real.pi) * σ * D * (2 : ℝ)^(-(K : ℤ))) := by
                      rw [ENNReal.ofReal_tsum_of_nonneg]
                      · intro K
                        apply mul_nonneg
                        apply mul_nonneg
                        apply mul_nonneg
                        apply mul_nonneg
                        · linarith
                        · exact Real.sqrt_nonneg _
                        · exact le_of_lt hσ
                        · exact le_of_lt hD
                        · exact zpow_nonneg (by norm_num : (0 : ℝ) ≤ 2) _
                      · exact h_geom.mul_left _
                  _ < ⊤ := ENNReal.ofReal_lt_top
        have h_ae_lt_top : ∀ᵐ ω ∂μ, ∑' K, ENNReal.ofReal |Δ K ω| < ⊤ :=
          ae_lt_top (Measurable.tsum (fun K =>
            (hX_meas (proj (K + 1) 0)).sub (hX_meas (proj K 0)) |>.abs.ennreal_ofReal))
            h_lintegral_sum.ne
        filter_upwards [h_ae_lt_top] with ω h_lt_top
        have h_conv : ∀ K, ENNReal.ofReal |Δ K ω| = (Real.toNNReal |Δ K ω| : ℝ≥0∞) := fun K => rfl
        simp only [h_conv] at h_lt_top
        have h_ne_top : ∑' K, (Real.toNNReal |Δ K ω| : ℝ≥0∞) ≠ ⊤ := h_lt_top.ne
        rw [ENNReal.tsum_coe_ne_top_iff_summable] at h_ne_top
        have h_nnreal_summable : Summable fun K => Real.toNNReal |Δ K ω| := h_ne_top
        have h_real_summable : Summable fun K => (Real.toNNReal |Δ K ω| : ℝ) :=
          NNReal.summable_coe.mpr h_nnreal_summable
        convert h_real_summable using 1
        ext K
        exact (Real.coe_toNNReal _ (abs_nonneg _)).symm

      -- Telescoping bound: |g| ≤ |X(t₀)| + |X(proj_0(0)) - X(t₀)| + ∑_K |Δ_K|
      have h_g_bound_ae : ∀ᵐ ω ∂μ, |g ω| ≤ |X (t 0) ω| + |X (proj 0 0) ω - X (t 0) ω| + ∑' K, |Δ K ω| := by
        filter_upwards [h_Δ_summable_ae] with ω h_summable
        have h_conv := h_X_tendsto 0 ω
        have h_telescope : ∀ K, X (proj K 0) ω - X (proj 0 0) ω = ∑ j ∈ Finset.range K, Δ j ω := by
          intro K
          induction K with
          | zero => simp
          | succ K ih =>
            rw [Finset.sum_range_succ, ← ih]
            ring
        have h_sup_bound : ∀ K, |X (proj K 0) ω - X (t 0) ω| ≤
            |X (proj 0 0) ω - X (t 0) ω| + ∑' J, |Δ J ω| := by
          intro K
          calc |X (proj K 0) ω - X (t 0) ω|
              = |(X (proj K 0) ω - X (proj 0 0) ω) + (X (proj 0 0) ω - X (t 0) ω)| := by ring_nf
            _ ≤ |X (proj K 0) ω - X (proj 0 0) ω| + |X (proj 0 0) ω - X (t 0) ω| := abs_add_le _ _
            _ = |∑ j ∈ Finset.range K, Δ j ω| + |X (proj 0 0) ω - X (t 0) ω| := by rw [h_telescope K]
            _ ≤ ∑ j ∈ Finset.range K, |Δ j ω| + |X (proj 0 0) ω - X (t 0) ω| := by
                gcongr
                exact Finset.abs_sum_le_sum_abs _ _
            _ ≤ ∑' j, |Δ j ω| + |X (proj 0 0) ω - X (t 0) ω| := by
                gcongr
                exact h_summable.sum_le_tsum _ (fun j _ => abs_nonneg _)
            _ = |X (proj 0 0) ω - X (t 0) ω| + ∑' j, |Δ j ω| := by ring
        -- |g - X(t₀)| ≤ bound since g = inf_K X(proj_K(0)) ≤ X(t₀)
        have h_g_diff : |g ω - X (t 0) ω| ≤ |X (proj 0 0) ω - X (t 0) ω| + ∑' J, |Δ J ω| := by
          have h_bdd := h_g_bdd ω
          have h_g_le_t0 : g ω ≤ X (t 0) ω := by
            apply ge_of_tendsto h_conv
            filter_upwards with K
            exact ciInf_le h_bdd K
          have h_abs_eq : |g ω - X (t 0) ω| = X (t 0) ω - g ω := by
            rw [abs_of_nonpos (sub_nonpos.mpr h_g_le_t0)]
            ring
          rw [h_abs_eq]
          -- X(proj_K(0)) ≥ X(t₀) - bound for all K, so g ≥ X(t₀) - bound
          let bound := |X (proj 0 0) ω - X (t 0) ω| + ∑' J, |Δ J ω|
          have h_lower : ∀ K, X (proj K 0) ω ≥ X (t 0) ω - bound := by
            intro K
            have h1 : X (t 0) ω - X (proj K 0) ω ≤ |X (proj K 0) ω - X (t 0) ω| := by
              rw [abs_sub_comm]
              exact le_abs_self _
            have h2 : |X (proj K 0) ω - X (t 0) ω| ≤ bound := h_sup_bound K
            linarith
          have h_ciInf_ge : g ω ≥ X (t 0) ω - bound := by
            apply le_ciInf
            intro K
            exact h_lower K
          simp only [bound] at h_ciInf_ge
          linarith
        calc |g ω| ≤ |g ω - X (t 0) ω| + |X (t 0) ω| := by
              have h := abs_sub_le (g ω) (X (t 0) ω) 0; simp at h; exact h
          _ ≤ (|X (proj 0 0) ω - X (t 0) ω| + ∑' J, |Δ J ω|) + |X (t 0) ω| := by gcongr
          _ = |X (t 0) ω| + |X (proj 0 0) ω - X (t 0) ω| + ∑' J, |Δ J ω| := by ring

      -- Dominating function is integrable (∑' K, |Δ K| via interchange of sum and integral)
      have h_dom_int : Integrable (fun ω => |X (t 0) ω| + |X (proj 0 0) ω - X (t 0) ω| + ∑' K, |Δ K ω|) μ := by
        apply Integrable.add
        apply Integrable.add
        · exact h_t0_int.abs
        · exact (h_proj0_int.sub h_t0_int).abs
        · have h_Δ_abs_int : ∀ K, Integrable (fun ω => |Δ K ω|) μ := fun K => (h_Δ_int K).abs
          have h_Δ_abs_meas : ∀ K, Measurable (fun ω => |Δ K ω|) :=
            fun K => ((hX_meas (proj (K + 1) 0)).sub (hX_meas (proj K 0))).abs
          have h_integrable_tsum : Integrable (fun ω => ∑' K, |Δ K ω|) μ := by
            -- AEStronglyMeasurable via NNReal tsum measurability
            have h_ae_meas : AEStronglyMeasurable (fun ω => ∑' K, |Δ K ω|) μ := by
              let f : ℕ → Ω → NNReal := fun K ω => ‖|Δ K ω|‖₊
              have h_f_meas : ∀ K, Measurable (f K) := fun K => (h_Δ_abs_meas K).nnnorm
              have h_tsum_meas : Measurable (fun ω => ∑' K, f K ω) := Measurable.tsum h_f_meas
              have h_f_eq : ∀ K ω, (f K ω : ℝ) = |Δ K ω| := fun K ω => by
                simp only [f, coe_nnnorm, Real.norm_eq_abs, abs_abs]
              have h_eq : (fun ω => ∑' K, |Δ K ω|) = (fun ω => ∑' K, (f K ω : ℝ)) := by
                ext ω; exact tsum_congr (fun K => (h_f_eq K ω).symm)
              have h_meas' : Measurable (fun ω => ∑' K, (f K ω : ℝ)) := by
                have := NNReal.continuous_coe.measurable.comp h_tsum_meas
                convert this using 1; ext ω; exact NNReal.coe_tsum.symm
              rw [h_eq]; exact h_meas'.aestronglyMeasurable
            have h_finite : HasFiniteIntegral (fun ω => ∑' K, |Δ K ω|) μ := by
              rw [hasFiniteIntegral_iff_norm]
              have h_bound : ∫⁻ ω, ENNReal.ofReal ‖∑' K, |Δ K ω|‖ ∂μ ≤
                             ∑' K, ∫⁻ ω, ENNReal.ofReal |Δ K ω| ∂μ := by
                calc ∫⁻ ω, ENNReal.ofReal ‖∑' K, |Δ K ω|‖ ∂μ
                    = ∫⁻ ω, ENNReal.ofReal (∑' K, |Δ K ω|) ∂μ := by
                      apply lintegral_congr; intro ω
                      rw [Real.norm_eq_abs, abs_of_nonneg (tsum_nonneg (fun K => abs_nonneg _))]
                  _ ≤ ∫⁻ ω, ∑' K, ENNReal.ofReal |Δ K ω| ∂μ := by
                      apply lintegral_mono; intro ω
                      by_cases h : Summable (fun K => |Δ K ω|)
                      · exact le_of_eq (ENNReal.ofReal_tsum_of_nonneg (fun _ => abs_nonneg _) h)
                      · simp [tsum_eq_zero_of_not_summable h]
                  _ = ∑' K, ∫⁻ ω, ENNReal.ofReal |Δ K ω| ∂μ := by
                      apply lintegral_tsum
                      intro K; exact (h_Δ_abs_meas K).ennreal_ofReal.aemeasurable
              calc ∫⁻ ω, ENNReal.ofReal ‖∑' K, |Δ K ω|‖ ∂μ
                  ≤ ∑' K, ∫⁻ ω, ENNReal.ofReal |Δ K ω| ∂μ := h_bound
                _ = ∑' K, ENNReal.ofReal (∫ ω, |Δ K ω| ∂μ) := by
                    congr 1; ext K
                    rw [ofReal_integral_eq_lintegral_ofReal (h_Δ_abs_int K)
                        (ae_of_all μ (fun ω => abs_nonneg _))]
                _ < ⊤ := by
                    have h_nonneg : ∀ K, 0 ≤ ∫ ω, |Δ K ω| ∂μ := fun K => integral_nonneg (fun _ => abs_nonneg _)
                    rw [← ENNReal.ofReal_tsum_of_nonneg h_nonneg h_sum_Δ_summable]
                    exact ENNReal.ofReal_lt_top
            exact ⟨h_ae_meas, h_finite⟩
          exact h_integrable_tsum

      exact h_dom_int.mono' h_g_meas.aestronglyMeasurable h_g_bound_ae

    -- Define Z_K = Y_K - g (non-negative shifted sequence)
    let Z : ℕ → Ω → ℝ := fun K ω => Y K ω - g ω
    have h_Z_nonneg : ∀ K ω, 0 ≤ Z K ω := fun K ω => sub_nonneg.mpr (h_Y_ge_g K ω)
    have h_Z_meas : ∀ K, Measurable (Z K) := by
      intro K
      have h_Y_meas : Measurable (Y K) := by
        convert Finset.measurable_sup' (hnet_nonempty K) (fun u _ => hX_meas u)
        ext ω; exact (Finset.sup'_apply (hnet_nonempty K) X ω).symm
      have h_g_meas : Measurable g := Measurable.iInf (fun J => hX_meas (proj J 0))
      exact h_Y_meas.sub h_g_meas
    have h_Z_int : ∀ K, 1 ≤ K → Integrable (Z K) μ := fun K hK => (h_Y_int K hK).sub h_g_int

    -- Apply Fatou for non-negative Z_K: ∫ liminf Z_K ≤ liminf ∫ Z_K
    have h_fatou_Z : ∫ ω, Filter.liminf (fun K => Z K ω) Filter.atTop ∂μ ≤
        Filter.liminf (fun K => ∫ ω, Z K ω ∂μ) Filter.atTop := by

      let g' : ℕ → Ω → ℝ≥0∞ := fun K ω => ENNReal.ofReal (Z K ω)
      have hg'_meas : ∀ K, Measurable (g' K) := fun K =>
        ENNReal.measurable_ofReal.comp (h_Z_meas K)
      have h_fatou_enn : ∫⁻ ω, Filter.liminf (fun K => g' K ω) Filter.atTop ∂μ ≤
          Filter.liminf (fun K => ∫⁻ ω, g' K ω ∂μ) Filter.atTop :=
        lintegral_liminf_le hg'_meas

      -- ∫ Z_K bounded above by B = 6√2·σ·C₁ + |∫g|
      have h_Z_int_bound : ∃ B : ℝ, 0 ≤ B ∧ ∀ K, 1 ≤ K → ∫ ω, Z K ω ∂μ ≤ B := by
        have h_dyadic_bdd : ∃ C₁ : ℝ, 0 < C₁ ∧ ∀ K, 1 ≤ K → dyadicRHS_real s D (K + 1) ≤ C₁ := by
          use 4 * entropyIntegral s D + 1
          constructor
          · have h_int_nonneg := entropyIntegral_nonneg (s := s) (D := D)
            linarith
          · intro K hK
            have := dyadicRHS_le_four_times_entropyIntegral hs hD (Nat.le_add_left 1 K) hfinite
            linarith
        obtain ⟨C₁, hC₁_pos, hC₁_bound⟩ := h_dyadic_bdd
        -- Use B = (11√2) * σ * C₁ + |∫ g|
        use (6 * Real.sqrt 2) * σ * C₁ + |∫ ω, g ω ∂μ|
        constructor
        · apply add_nonneg
          · apply mul_nonneg (mul_nonneg (by positivity : (0:ℝ) ≤ 6 * Real.sqrt 2) (le_of_lt hσ))
            exact le_of_lt hC₁_pos
          · exact abs_nonneg _
        · intro K hK
          calc ∫ ω, Z K ω ∂μ = ∫ ω, Y K ω ∂μ - ∫ ω, g ω ∂μ :=
                MeasureTheory.integral_sub (h_Y_int K hK) h_g_int
            _ ≤ (6 * Real.sqrt 2) * σ * dyadicRHS_real s D (K + 1) - ∫ ω, g ω ∂μ := by
                apply sub_le_sub_right (h_core_bound K hK)
            _ ≤ (6 * Real.sqrt 2) * σ * C₁ - ∫ ω, g ω ∂μ := by
                apply sub_le_sub_right
                apply mul_le_mul_of_nonneg_left (hC₁_bound K hK)
                apply mul_nonneg (by positivity : (0:ℝ) ≤ 6 * Real.sqrt 2) (le_of_lt hσ)
            _ ≤ (6 * Real.sqrt 2) * σ * C₁ + |∫ ω, g ω ∂μ| := by linarith [neg_abs_le (∫ ω, g ω ∂μ)]

      obtain ⟨B, hB_nonneg, hB_bound⟩ := h_Z_int_bound

      have h_lintegral_eq : ∀ K, 1 ≤ K →
          ∫⁻ ω, g' K ω ∂μ = ENNReal.ofReal (∫ ω, Z K ω ∂μ) := fun K hK => by
        rw [← MeasureTheory.ofReal_integral_eq_lintegral_ofReal
            (h_Z_int K hK)
            (Filter.Eventually.of_forall (h_Z_nonneg K))]

      -- liminf(ofReal(∫ Z_K)) = ofReal(liminf ∫ Z_K) via Monotone.map_liminf_of_continuousAt
      have h_liminf_ofReal : Filter.liminf (fun K => ENNReal.ofReal (∫ ω, Z K ω ∂μ)) Filter.atTop =
          ENNReal.ofReal (Filter.liminf (fun K => ∫ ω, Z K ω ∂μ) Filter.atTop) := by
        have h_nonneg : ∀ K, 0 ≤ ∫ ω, Z K ω ∂μ := fun K => by
          apply MeasureTheory.integral_nonneg
          exact fun ω => h_Z_nonneg K ω
        have h_bdd_above : Filter.IsBoundedUnder (· ≤ ·) Filter.atTop
            (fun K => ∫ ω, Z K ω ∂μ) := by
          use B
          apply Filter.eventually_map.mpr
          filter_upwards [Filter.eventually_ge_atTop 1] with K hK
          exact hB_bound K hK
        have h_bdd_below : Filter.IsBoundedUnder (· ≥ ·) Filter.atTop
            (fun K => ∫ ω, Z K ω ∂μ) := by
          use 0; apply Filter.eventually_map.mpr; exact Filter.Eventually.of_forall h_nonneg
        have hcobdd := h_bdd_above.isCoboundedUnder_ge
        have hbdd := h_bdd_below
        symm; apply Monotone.map_liminf_of_continuousAt
        · exact fun _ _ h => ENNReal.ofReal_le_ofReal h
        · exact ENNReal.continuous_ofReal.continuousAt
        · exact hcobdd
        · exact hbdd

      have h_liminf_eq : Filter.liminf (fun K => ∫⁻ ω, g' K ω ∂μ) Filter.atTop =
          ENNReal.ofReal (Filter.liminf (fun K => ∫ ω, Z K ω ∂μ) Filter.atTop) := by
        have h_eq_eventually : ∀ᶠ K in Filter.atTop,
            ∫⁻ ω, g' K ω ∂μ = ENNReal.ofReal (∫ ω, Z K ω ∂μ) := by
          filter_upwards [Filter.eventually_ge_atTop 1] with K hK
          exact h_lintegral_eq K hK
        rw [Filter.liminf_congr h_eq_eventually]
        exact h_liminf_ofReal

      -- ofReal(liminf f) ≤ liminf(ofReal ∘ f) via monotonicity of ofReal
      have h_ofReal_liminf : ∀ ω, ENNReal.ofReal (Filter.liminf (fun K => Z K ω) Filter.atTop) ≤
          Filter.liminf (fun K => g' K ω) Filter.atTop := fun ω => by
        have h_bdd_below : Filter.IsBoundedUnder (· ≥ ·) Filter.atTop (fun K => Z K ω) := by
          use 0
          rw [Filter.eventually_map]
          exact Filter.Eventually.of_forall (fun K => h_Z_nonneg K ω)
        have h_mono : Monotone ENNReal.ofReal := fun _ _ h => ENNReal.ofReal_le_ofReal h
        by_cases h_cobdd : Filter.IsCoboundedUnder (· ≥ ·) Filter.atTop (fun K => Z K ω)
        · -- Cobounded case: equality via map_liminf_of_continuousAt
          have h_eq := Monotone.map_liminf_of_continuousAt h_mono (fun K => Z K ω)
              ENNReal.continuous_ofReal.continuousAt h_cobdd h_bdd_below
          rw [h_eq]; rfl
        · -- Unbounded case: liminf = 0
          have h_liminf_eq_zero : Filter.liminf (fun K => Z K ω) Filter.atTop = 0 := by
            unfold Filter.liminf Filter.limsInf
            have h_unbdd : ¬BddAbove {a | ∀ᶠ n in Filter.map (fun K => Z K ω) Filter.atTop, a ≤ n} := by
              simp only [Filter.IsCoboundedUnder, Filter.IsCobounded, Filter.eventually_map] at h_cobdd
              intro ⟨b, hb⟩
              apply h_cobdd
              use b
              intro a ha
              exact hb ha
            exact Real.sSup_of_not_bddAbove h_unbdd
          rw [h_liminf_eq_zero]; simp only [ENNReal.ofReal_zero, zero_le]

      have h_lintegral_le : ∫⁻ ω, ENNReal.ofReal (Filter.liminf (fun K => Z K ω) Filter.atTop) ∂μ ≤
          ∫⁻ ω, Filter.liminf (fun K => g' K ω) Filter.atTop ∂μ := lintegral_mono h_ofReal_liminf

      have h_liminf_Z_meas : Measurable (fun ω => Filter.liminf (fun K => Z K ω) Filter.atTop) := by
        apply Measurable.liminf; exact fun K => h_Z_meas K

      -- liminf Z ≥ 0 a.e. (coboundedness from Y's finite EReal liminf)
      have h_liminf_Z_nonneg : ∀ᵐ ω ∂μ, 0 ≤ Filter.liminf (fun K => Z K ω) Filter.atTop := by
        filter_upwards [h_Y_liminf_ne_top] with ω hω_ne_top
        have hbdd_Y := h_Y_bdd_below ω
        have hcobdd_Y : Filter.IsCoboundedUnder (· ≥ ·) Filter.atTop (fun K => Y K ω) := by
          let L := Filter.liminf (fun K => (Y K ω : EReal)) Filter.atTop
          have hL_ne_top : L ≠ ⊤ := hω_ne_top
          have hL_ne_bot : L ≠ ⊥ := by
            obtain ⟨M, hM⟩ := hbdd_Y.eventually_ge
            have h_M_le_L : (M : EReal) ≤ L := by
              apply Filter.le_liminf_of_le
              · exact Filter.isCobounded_ge_of_top
              · filter_upwards [hM] with K hK
                exact EReal.coe_le_coe_iff.mpr hK
            intro hL_bot
            rw [hL_bot] at h_M_le_L
            exact EReal.coe_ne_bot M (le_bot_iff.mp h_M_le_L)
          use L.toReal
          intro a ha
          have h_a_le_L : (a : EReal) ≤ L := by
            apply Filter.le_liminf_of_le
            · exact Filter.isCobounded_ge_of_top
            · filter_upwards [ha] with K hK
              exact EReal.coe_le_coe_iff.mpr hK
          have hL_eq : L = (L.toReal : EReal) := (EReal.coe_toReal hL_ne_top hL_ne_bot).symm
          rw [hL_eq] at h_a_le_L
          exact EReal.coe_le_coe_iff.mp h_a_le_L
        -- Coboundedness for Z follows from Y (Z = Y - g)
        have hcobdd_Z : Filter.IsCoboundedUnder (· ≥ ·) Filter.atTop (fun K => Z K ω) := by
          obtain ⟨b, hb⟩ := hcobdd_Y
          use b - g ω
          intro a ha
          have ha' : ∀ᶠ K in Filter.atTop, a + g ω ≤ Y K ω := by
            have ha_atTop : ∀ᶠ K in Filter.atTop, a ≤ Z K ω := Filter.eventually_map.mp ha
            filter_upwards [ha_atTop] with K hK
            linarith [show Z K ω = Y K ω - g ω from rfl]
          have ha'_map : ∀ᶠ x in Filter.map (fun K => Y K ω) Filter.atTop, x ≥ a + g ω :=
            Filter.eventually_map.mpr ha'
          linarith [hb (a + g ω) ha'_map]
        apply Filter.le_liminf_of_le hcobdd_Z
        exact Filter.Eventually.of_forall (fun K => h_Z_nonneg K ω)

      have h_lhs_eq : ∫ ω, Filter.liminf (fun K => Z K ω) Filter.atTop ∂μ =
          (∫⁻ ω, ENNReal.ofReal (Filter.liminf (fun K => Z K ω) Filter.atTop) ∂μ).toReal :=
        integral_eq_lintegral_of_nonneg_ae h_liminf_Z_nonneg h_liminf_Z_meas.aestronglyMeasurable
      rw [h_lhs_eq]

      -- Chain: ∫⁻ ofReal(liminf Z) ≤ ∫⁻ liminf g' ≤ liminf ∫⁻ g' = ofReal(liminf ∫ Z)
      have h_chain := calc
        ∫⁻ ω, ENNReal.ofReal (Filter.liminf (fun K => Z K ω) Filter.atTop) ∂μ
          ≤ ∫⁻ ω, Filter.liminf (fun K => g' K ω) Filter.atTop ∂μ := h_lintegral_le
        _ ≤ Filter.liminf (fun K => ∫⁻ ω, g' K ω ∂μ) Filter.atTop := h_fatou_enn
        _ = ENNReal.ofReal (Filter.liminf (fun K => ∫ ω, Z K ω ∂μ) Filter.atTop) := h_liminf_eq

      have h_liminf_int_nonneg : 0 ≤ Filter.liminf (fun K => ∫ ω, Z K ω ∂μ) Filter.atTop := by
        apply Filter.le_liminf_of_le
        · use B; intro a ha
          have ha' : ∀ᶠ K in Filter.atTop, ∫ ω, Z K ω ∂μ ≥ a := Filter.eventually_map.mp ha
          have h_evt_B : ∀ᶠ K in Filter.atTop, ∫ ω, Z K ω ∂μ ≤ B := by
            filter_upwards [Filter.eventually_ge_atTop 1] with K hK
            exact hB_bound K hK
          have h_and := Filter.Eventually.and ha' h_evt_B
          obtain ⟨K, hK_ge, hK_le⟩ := h_and.exists
          exact le_trans hK_ge hK_le
        · exact Filter.Eventually.of_forall (fun K =>
            MeasureTheory.integral_nonneg (fun ω => h_Z_nonneg K ω))

      calc (∫⁻ ω, ENNReal.ofReal (Filter.liminf (fun K => Z K ω) Filter.atTop) ∂μ).toReal
        _ ≤ (ENNReal.ofReal (Filter.liminf (fun K => ∫ ω, Z K ω ∂μ) Filter.atTop)).toReal := by
            apply ENNReal.toReal_mono
            · exact ENNReal.ofReal_ne_top
            · exact h_chain
        _ = Filter.liminf (fun K => ∫ ω, Z K ω ∂μ) Filter.atTop := by
            rw [ENNReal.toReal_ofReal h_liminf_int_nonneg]

    -- liminf Z_K = liminf Y_K - g (g is constant in K) via liminf_add_const
    have h_liminf_Z : ∀ᵐ ω ∂μ, Filter.liminf (fun K => Z K ω) Filter.atTop =
        Filter.liminf (fun K => Y K ω) Filter.atTop - g ω := by
      filter_upwards [h_Y_liminf_ne_top] with ω hω_ne_top
      have hbdd := h_Y_bdd_below ω
      -- Coboundedness from finite EReal liminf
      have hcobdd : Filter.IsCoboundedUnder (· ≥ ·) Filter.atTop (fun K => Y K ω) := by
        let L := Filter.liminf (fun K => (Y K ω : EReal)) Filter.atTop
        have hL_ne_top : L ≠ ⊤ := hω_ne_top
        have hL_ne_bot : L ≠ ⊥ := by
          obtain ⟨M, hM⟩ := hbdd.eventually_ge
          have h_M_le_L : (M : EReal) ≤ L := by
            apply Filter.le_liminf_of_le
            · exact Filter.isCobounded_ge_of_top
            · filter_upwards [hM] with K hK
              exact EReal.coe_le_coe_iff.mpr hK
          intro hL_bot
          rw [hL_bot] at h_M_le_L
          exact EReal.coe_ne_bot M (le_bot_iff.mp h_M_le_L)
        -- Step 4: Since L ≠ ⊤ and L ≠ ⊥, use L.toReal as the witness
        use L.toReal
        intro a ha
        -- ha : ∀ᶠ K in atTop, a ≤ Y K ω
        -- Need to show: L.toReal ≥ a
        have h_a_le_L : (a : EReal) ≤ L := by
          apply Filter.le_liminf_of_le
          · exact Filter.isCobounded_ge_of_top
          · filter_upwards [ha] with K hK
            exact EReal.coe_le_coe_iff.mpr hK
        -- Since L = (L.toReal : EReal) (by coe_toReal), we have a ≤ L.toReal
        have hL_eq : L = (L.toReal : EReal) := (EReal.coe_toReal hL_ne_top hL_ne_bot).symm
        rw [hL_eq] at h_a_le_L
        exact EReal.coe_le_coe_iff.mp h_a_le_L
      -- Apply liminf_add_const with constant = -g ω
      have h := liminf_add_const Filter.atTop (fun K => Y K ω) (-g ω) hcobdd hbdd
      -- h : liminf (fun K => Y K ω + (-g ω)) = liminf (fun K => Y K ω) + (-g ω)
      simp only [sub_eq_add_neg] at h ⊢
      exact h

    -- Unshift Fatou: from ∫ liminf Z_K ≤ liminf ∫ Z_K, derive ∫ liminf Y_K ≤ liminf ∫ Y_K
    have h_fatou_Y : ∫ ω, Filter.liminf (fun K => Y K ω) Filter.atTop ∂μ ≤
        Filter.liminf (fun K => ∫ ω, Y K ω ∂μ) Filter.atTop := by

      -- LHS: ∫ liminf Y_K = ∫ liminf Z_K + ∫ g (since liminf Y = liminf Z + g a.e.)
      have h_lhs : ∫ ω, Filter.liminf (fun K => Y K ω) Filter.atTop ∂μ =
          ∫ ω, Filter.liminf (fun K => Z K ω) Filter.atTop ∂μ + ∫ ω, g ω ∂μ := by
        have h_eq_ae : ∀ᵐ ω ∂μ, Filter.liminf (fun K => Y K ω) Filter.atTop =
            Filter.liminf (fun K => Z K ω) Filter.atTop + g ω := by
          filter_upwards [h_liminf_Z] with ω hω
          linarith
        calc ∫ ω, Filter.liminf (fun K => Y K ω) Filter.atTop ∂μ
          = ∫ ω, (Filter.liminf (fun K => Z K ω) Filter.atTop + g ω) ∂μ := by
              apply MeasureTheory.integral_congr_ae
              filter_upwards [h_eq_ae] with ω hω
              exact hω
          _ = ∫ ω, Filter.liminf (fun K => Z K ω) Filter.atTop ∂μ + ∫ ω, g ω ∂μ := by
              apply MeasureTheory.integral_add _ h_g_int
              -- Integrability of liminf Z_K
              -- Step 1: liminf Z is measurable
              have h_meas : Measurable (fun ω => Filter.liminf (fun K => Z K ω) Filter.atTop) :=
                Measurable.liminf (fun K => h_Z_meas K)
              -- Step 2: liminf Z is nonnegative a.e.
              have h_nonneg_ae : 0 ≤ᵐ[μ] (fun ω => Filter.liminf (fun K => Z K ω) Filter.atTop) := by
                filter_upwards [h_liminf_Z] with ω hω
                show 0 ≤ Filter.liminf (fun K => Z K ω) Filter.atTop
                by_cases hcobdd : Filter.IsCoboundedUnder (· ≥ ·) Filter.atTop (fun K => Z K ω)
                · apply Filter.le_liminf_of_le hcobdd
                  exact Filter.Eventually.of_forall (fun K => h_Z_nonneg K ω)
                · -- Unbounded case: liminf = 0 in ℝ
                  unfold Filter.liminf Filter.limsInf
                  have h_unbdd : ¬BddAbove {a | ∀ᶠ n in Filter.map (fun K => Z K ω) Filter.atTop, a ≤ n} := by
                    simp only [Filter.IsCoboundedUnder, Filter.IsCobounded, Filter.eventually_map] at hcobdd
                    intro ⟨b, hb⟩; apply hcobdd; use b; intro a ha; exact hb ha
                  rw [Real.sSup_of_not_bddAbove h_unbdd]
              -- Step 3: lintegral is finite - use Fatou and bounds on ∫ Z_K
              have h_lintegral_finite : ∫⁻ ω, ENNReal.ofReal (Filter.liminf (fun K => Z K ω) Filter.atTop) ∂μ ≠ ⊤ := by
                let g' : ℕ → Ω → ℝ≥0∞ := fun K ω => ENNReal.ofReal (Z K ω)
                have hg'_meas : ∀ K, Measurable (g' K) := fun K =>
                  ENNReal.measurable_ofReal.comp (h_Z_meas K)
                have h_fatou_enn : ∫⁻ ω, Filter.liminf (fun K => g' K ω) Filter.atTop ∂μ ≤
                    Filter.liminf (fun K => ∫⁻ ω, g' K ω ∂μ) Filter.atTop :=
                  MeasureTheory.lintegral_liminf_le hg'_meas
                -- Bound: use dyadicRHS converges → bounded
                -- Direct bound: dyadicRHS ≤ 4 * entropyIntegral (using upstream lemma)
                have h_dyadic_bdd : ∃ B : ℝ, 0 ≤ B ∧ ∀ K, 1 ≤ K → dyadicRHS_real s D (K + 1) ≤ B := by
                  use 4 * entropyIntegral s D + 1
                  constructor
                  · have h_int_nonneg := entropyIntegral_nonneg (s := s) (D := D)
                    linarith
                  · intro K hK
                    have := dyadicRHS_le_four_times_entropyIntegral hs hD (Nat.le_add_left 1 K) hfinite
                    linarith
                obtain ⟨C, hC_nonneg, hC_bound⟩ := h_dyadic_bdd
                -- Include |∫ g| in bound since ∫ g might be negative
                let B := (6 * Real.sqrt 2) * σ * C + |∫ ω, g ω ∂μ|
                have hB_bound : ∀ K, 1 ≤ K → ∫ ω, Z K ω ∂μ ≤ B := fun K hK => by
                  have h_Y_bound := h_core_bound K hK
                  -- Key: ∫ Z = ∫ Y - ∫ g, and -∫ g ≤ |∫ g|
                  calc ∫ ω, Z K ω ∂μ = ∫ ω, Y K ω ∂μ - ∫ ω, g ω ∂μ :=
                      MeasureTheory.integral_sub (h_Y_int K hK) h_g_int
                    _ ≤ ∫ ω, Y K ω ∂μ + |∫ ω, g ω ∂μ| := by linarith [neg_abs_le (∫ ω, g ω ∂μ)]
                    _ ≤ (6 * Real.sqrt 2) * σ * dyadicRHS_real s D (K + 1) + |∫ ω, g ω ∂μ| := by linarith
                    _ ≤ B := by
                        simp only [B]
                        have h1 : dyadicRHS_real s D (K + 1) ≤ C := hC_bound K hK
                        have h2 : (6 * Real.sqrt 2) * σ * dyadicRHS_real s D (K + 1) ≤ (6 * Real.sqrt 2) * σ * C := by
                          apply mul_le_mul_of_nonneg_left h1
                          positivity
                        linarith
                have h_lintegral_eq : ∀ K, 1 ≤ K → ∫⁻ ω, g' K ω ∂μ = ENNReal.ofReal (∫ ω, Z K ω ∂μ) :=
                  fun K hK => by
                    have h_nonneg_K : 0 ≤ᵐ[μ] Z K := Filter.Eventually.of_forall (h_Z_nonneg K)
                    symm
                    exact MeasureTheory.ofReal_integral_eq_lintegral_ofReal (h_Z_int K hK) h_nonneg_K
                have h_lintegral_bound : ∀ K, 1 ≤ K → ∫⁻ ω, g' K ω ∂μ ≤ ENNReal.ofReal B := fun K hK => by
                  rw [h_lintegral_eq K hK]
                  exact ENNReal.ofReal_le_ofReal (hB_bound K hK)
                have h_liminf_bound : Filter.liminf (fun K => ∫⁻ ω, g' K ω ∂μ) Filter.atTop ≤ ENNReal.ofReal B := by
                  apply Filter.liminf_le_of_le
                  · exact ⟨0, Filter.Eventually.of_forall (fun _ => zero_le)⟩
                  · intro b hb
                    have h_evt : ∀ᶠ K in Filter.atTop, ∫⁻ ω, g' K ω ∂μ ≤ ENNReal.ofReal B := by
                      filter_upwards [Filter.eventually_ge_atTop 1] with K hK
                      exact h_lintegral_bound K hK
                    have h_and := Filter.Eventually.and hb h_evt
                    obtain ⟨K, hK_ge, hK_le⟩ := h_and.exists
                    exact le_trans hK_ge hK_le
                have h_ofReal_liminf_le : ∀ ω, ENNReal.ofReal (Filter.liminf (fun K => Z K ω) Filter.atTop) ≤
                    Filter.liminf (fun K => g' K ω) Filter.atTop := fun ω => by
                  by_cases hcobdd : Filter.IsCoboundedUnder (· ≥ ·) Filter.atTop (fun K => Z K ω)
                  · have h_bdd_below : Filter.IsBoundedUnder (· ≥ ·) Filter.atTop (fun K => Z K ω) :=
                      Filter.isBoundedUnder_of_eventually_ge
                        (Filter.Eventually.of_forall (fun K => h_Z_nonneg K ω))
                    have h_mono : Monotone ENNReal.ofReal := fun _ _ h => ENNReal.ofReal_le_ofReal h
                    have h_eq := Monotone.map_liminf_of_continuousAt h_mono (fun K => Z K ω)
                        ENNReal.continuous_ofReal.continuousAt hcobdd h_bdd_below
                    rw [h_eq]; rfl
                  · unfold Filter.liminf Filter.limsInf
                    have h_unbdd : ¬BddAbove {a | ∀ᶠ n in Filter.map (fun K => Z K ω) Filter.atTop, a ≤ n} := by
                      simp only [Filter.IsCoboundedUnder, Filter.IsCobounded, Filter.eventually_map] at hcobdd
                      intro ⟨b, hb⟩; apply hcobdd; use b; intro a ha; exact hb ha
                    rw [Real.sSup_of_not_bddAbove h_unbdd]
                    simp only [ENNReal.ofReal_zero, zero_le]
                intro h_eq_top
                have h1 : ∫⁻ ω, ENNReal.ofReal (Filter.liminf (fun K => Z K ω) Filter.atTop) ∂μ ≤
                    ∫⁻ ω, Filter.liminf (fun K => g' K ω) Filter.atTop ∂μ :=
                  MeasureTheory.lintegral_mono h_ofReal_liminf_le
                have h2 : ∫⁻ ω, Filter.liminf (fun K => g' K ω) Filter.atTop ∂μ ≤
                    Filter.liminf (fun K => ∫⁻ ω, g' K ω ∂μ) Filter.atTop := h_fatou_enn
                have h3 : Filter.liminf (fun K => ∫⁻ ω, g' K ω ∂μ) Filter.atTop ≤ ENNReal.ofReal B :=
                  h_liminf_bound
                have h_chain : ∫⁻ ω, ENNReal.ofReal (Filter.liminf (fun K => Z K ω) Filter.atTop) ∂μ ≤
                    ENNReal.ofReal B := le_trans h1 (le_trans h2 h3)
                rw [h_eq_top] at h_chain
                exact absurd (top_unique h_chain).symm ENNReal.top_ne_ofReal
              exact (MeasureTheory.lintegral_ofReal_ne_top_iff_integrable
                h_meas.aestronglyMeasurable h_nonneg_ae).mp h_lintegral_finite

      -- RHS: liminf ∫ Z_K = liminf ∫ Y_K - ∫ g via liminf_add_const
      have h_rhs : Filter.liminf (fun K => ∫ ω, Z K ω ∂μ) Filter.atTop =
          Filter.liminf (fun K => ∫ ω, Y K ω ∂μ) Filter.atTop - ∫ ω, g ω ∂μ := by
        have h_int_eq : ∀ᶠ K in Filter.atTop, ∫ ω, Z K ω ∂μ = ∫ ω, Y K ω ∂μ - ∫ ω, g ω ∂μ := by
          filter_upwards [Filter.eventually_ge_atTop 1] with K hK
          exact MeasureTheory.integral_sub (h_Y_int K hK) h_g_int
        have h_bdd_below : Filter.IsBoundedUnder (· ≥ ·) Filter.atTop (fun K => ∫ ω, Y K ω ∂μ) := by
          use ∫ ω, g ω ∂μ
          rw [Filter.eventually_map, Filter.eventually_atTop]
          exact ⟨1, fun K hK => MeasureTheory.integral_mono_ae h_g_int (h_Y_int K hK)
            (Filter.Eventually.of_forall (h_Y_ge_g K))⟩
        have h_bdd_above : Filter.IsBoundedUnder (· ≤ ·) Filter.atTop (fun K => ∫ ω, Y K ω ∂μ) := by
          use (6 * Real.sqrt 2) * σ * (4 * entropyIntegral s D)
          rw [Filter.eventually_map, Filter.eventually_atTop]
          use 1
          intro K hK
          have h_dyadic_le := dyadicRHS_le_four_times_entropyIntegral hs hD (Nat.le_add_left 1 K) hfinite
          calc ∫ ω, Y K ω ∂μ ≤ (6 * Real.sqrt 2) * σ * dyadicRHS_real s D (K + 1) := h_core_bound K hK
            _ ≤ (6 * Real.sqrt 2) * σ * (4 * entropyIntegral s D) := by
                apply mul_le_mul_of_nonneg_left h_dyadic_le
                apply mul_nonneg (by positivity : (0:ℝ) ≤ 6 * Real.sqrt 2) (le_of_lt hσ)
        have h_cobdd : Filter.IsCoboundedUnder (· ≥ ·) Filter.atTop (fun K => ∫ ω, Y K ω ∂μ) :=
          h_bdd_above.isCoboundedUnder_ge
        -- Rewrite the h_int_eq to use + (-c) form
        have h_int_eq' : ∀ᶠ K in Filter.atTop, ∫ ω, Z K ω ∂μ = ∫ ω, Y K ω ∂μ + (-(∫ ω, g ω ∂μ)) := by
          filter_upwards [h_int_eq] with K hK
          linarith
        calc Filter.liminf (fun K => ∫ ω, Z K ω ∂μ) Filter.atTop
          = Filter.liminf (fun K => ∫ ω, Y K ω ∂μ + (-(∫ ω, g ω ∂μ))) Filter.atTop := by
              apply Filter.liminf_congr; exact h_int_eq'
          _ = Filter.liminf (fun K => ∫ ω, Y K ω ∂μ) Filter.atTop + (-(∫ ω, g ω ∂μ)) :=
              liminf_add_const Filter.atTop _ _ h_cobdd h_bdd_below
          _ = Filter.liminf (fun K => ∫ ω, Y K ω ∂μ) Filter.atTop - ∫ ω, g ω ∂μ := by ring

      -- Combine: ∫ liminf Y_K = ∫ liminf Z_K + ∫ g ≤ liminf ∫ Z_K + ∫ g = liminf ∫ Y_K
      calc ∫ ω, Filter.liminf (fun K => Y K ω) Filter.atTop ∂μ
        = ∫ ω, Filter.liminf (fun K => Z K ω) Filter.atTop ∂μ + ∫ ω, g ω ∂μ := h_lhs
        _ ≤ Filter.liminf (fun K => ∫ ω, Z K ω ∂μ) Filter.atTop + ∫ ω, g ω ∂μ := by
            linarith [h_fatou_Z]
        _ = (Filter.liminf (fun K => ∫ ω, Y K ω ∂μ) Filter.atTop - ∫ ω, g ω ∂μ) + ∫ ω, g ω ∂μ := by
            rw [h_rhs]
        _ = Filter.liminf (fun K => ∫ ω, Y K ω ∂μ) Filter.atTop := by ring

    -- ∫ sup_n X(t n) ≤ ∫ liminf Y_K via integral_mono_ae
    have h_step1 : ∫ ω, ⨆ n, X (t n) ω ∂μ ≤
        ∫ ω, Filter.liminf (fun K => Y K ω) Filter.atTop ∂μ := by
      have h_int_liminf : Integrable (fun ω => Filter.liminf (fun K => Y K ω) Filter.atTop) μ := by
        have h_eq_ae : ∀ᵐ ω ∂μ, Filter.liminf (fun K => Y K ω) Filter.atTop =
            Filter.liminf (fun K => Z K ω) Filter.atTop + g ω := by
          filter_upwards [h_liminf_Z] with ω hω
          linarith
        have h_int_liminf_Z : Integrable (fun ω => Filter.liminf (fun K => Z K ω) Filter.atTop) μ := by
          have h_meas : Measurable (fun ω => Filter.liminf (fun K => Z K ω) Filter.atTop) :=
            Measurable.liminf (fun K => h_Z_meas K)
          have h_nonneg_ae : 0 ≤ᵐ[μ] (fun ω => Filter.liminf (fun K => Z K ω) Filter.atTop) := by
            filter_upwards [h_liminf_Z] with ω hω
            show 0 ≤ Filter.liminf (fun K => Z K ω) Filter.atTop
            by_cases hcobdd : Filter.IsCoboundedUnder (· ≥ ·) Filter.atTop (fun K => Z K ω)
            · apply Filter.le_liminf_of_le hcobdd
              exact Filter.Eventually.of_forall (fun K => h_Z_nonneg K ω)
            · unfold Filter.liminf Filter.limsInf
              have h_unbdd : ¬BddAbove {a | ∀ᶠ n in Filter.map (fun K => Z K ω) Filter.atTop, a ≤ n} := by
                simp only [Filter.IsCoboundedUnder, Filter.IsCobounded, Filter.eventually_map] at hcobdd
                intro ⟨b, hb⟩; apply hcobdd; use b; intro a ha; exact hb ha
              rw [Real.sSup_of_not_bddAbove h_unbdd]
          have h_lintegral_finite : ∫⁻ ω, ENNReal.ofReal (Filter.liminf (fun K => Z K ω) Filter.atTop) ∂μ ≠ ⊤ := by
            let g' : ℕ → Ω → ℝ≥0∞ := fun K ω => ENNReal.ofReal (Z K ω)
            have hg'_meas : ∀ K, Measurable (g' K) := fun K =>
              ENNReal.measurable_ofReal.comp (h_Z_meas K)
            have h_fatou_enn : ∫⁻ ω, Filter.liminf (fun K => g' K ω) Filter.atTop ∂μ ≤
                Filter.liminf (fun K => ∫⁻ ω, g' K ω ∂μ) Filter.atTop :=
              MeasureTheory.lintegral_liminf_le hg'_meas
            -- Direct bound: dyadicRHS ≤ 4 * entropyIntegral (using upstream lemma)
            have h_dyadic_bdd : ∃ B : ℝ, 0 ≤ B ∧ ∀ K, 1 ≤ K → dyadicRHS_real s D (K + 1) ≤ B := by
              use 4 * entropyIntegral s D + 1
              constructor
              · have h_int_nonneg := entropyIntegral_nonneg (s := s) (D := D)
                linarith
              · intro K hK
                have := dyadicRHS_le_four_times_entropyIntegral hs hD (Nat.le_add_left 1 K) hfinite
                linarith
            obtain ⟨C, hC_nonneg, hC_bound⟩ := h_dyadic_bdd
            let B := (6 * Real.sqrt 2) * σ * C + |∫ ω, g ω ∂μ|
            have hB_bound : ∀ K, 1 ≤ K → ∫ ω, Z K ω ∂μ ≤ B := fun K hK => by
              have h_Y_bound := h_core_bound K hK
              calc ∫ ω, Z K ω ∂μ = ∫ ω, Y K ω ∂μ - ∫ ω, g ω ∂μ :=
                  MeasureTheory.integral_sub (h_Y_int K hK) h_g_int
                _ ≤ ∫ ω, Y K ω ∂μ + |∫ ω, g ω ∂μ| := by linarith [neg_abs_le (∫ ω, g ω ∂μ)]
                _ ≤ (6 * Real.sqrt 2) * σ * dyadicRHS_real s D (K + 1) + |∫ ω, g ω ∂μ| := by linarith
                _ ≤ B := by
                    simp only [B]
                    have h1 : dyadicRHS_real s D (K + 1) ≤ C := hC_bound K hK
                    have h2 : (6 * Real.sqrt 2) * σ * dyadicRHS_real s D (K + 1) ≤ (6 * Real.sqrt 2) * σ * C := by
                      apply mul_le_mul_of_nonneg_left h1; positivity
                    linarith
            have h_lintegral_eq : ∀ K, 1 ≤ K → ∫⁻ ω, g' K ω ∂μ = ENNReal.ofReal (∫ ω, Z K ω ∂μ) :=
              fun K hK => by
                have h_nonneg_K : 0 ≤ᵐ[μ] Z K := Filter.Eventually.of_forall (h_Z_nonneg K)
                symm
                exact MeasureTheory.ofReal_integral_eq_lintegral_ofReal (h_Z_int K hK) h_nonneg_K
            have h_lintegral_bound : ∀ K, 1 ≤ K → ∫⁻ ω, g' K ω ∂μ ≤ ENNReal.ofReal B := fun K hK => by
              rw [h_lintegral_eq K hK]
              exact ENNReal.ofReal_le_ofReal (hB_bound K hK)
            have h_liminf_bound : Filter.liminf (fun K => ∫⁻ ω, g' K ω ∂μ) Filter.atTop ≤ ENNReal.ofReal B := by
              apply Filter.liminf_le_of_le
              · exact ⟨0, Filter.Eventually.of_forall (fun _ => zero_le)⟩
              · intro b hb
                have h_evt : ∀ᶠ K in Filter.atTop, ∫⁻ ω, g' K ω ∂μ ≤ ENNReal.ofReal B := by
                  filter_upwards [Filter.eventually_ge_atTop 1] with K hK
                  exact h_lintegral_bound K hK
                have h_and := Filter.Eventually.and hb h_evt
                obtain ⟨K, hK_ge, hK_le⟩ := h_and.exists
                exact le_trans hK_ge hK_le
            have h_ofReal_liminf_le : ∀ ω, ENNReal.ofReal (Filter.liminf (fun K => Z K ω) Filter.atTop) ≤
                Filter.liminf (fun K => g' K ω) Filter.atTop := fun ω => by
              by_cases hcobdd : Filter.IsCoboundedUnder (· ≥ ·) Filter.atTop (fun K => Z K ω)
              · have h_bdd_below : Filter.IsBoundedUnder (· ≥ ·) Filter.atTop (fun K => Z K ω) :=
                  Filter.isBoundedUnder_of_eventually_ge
                    (Filter.Eventually.of_forall (fun K => h_Z_nonneg K ω))
                have h_mono : Monotone ENNReal.ofReal := fun _ _ h => ENNReal.ofReal_le_ofReal h
                have h_eq := Monotone.map_liminf_of_continuousAt h_mono (fun K => Z K ω)
                    ENNReal.continuous_ofReal.continuousAt hcobdd h_bdd_below
                rw [h_eq]; rfl
              · unfold Filter.liminf Filter.limsInf
                have h_unbdd : ¬BddAbove {a | ∀ᶠ n in Filter.map (fun K => Z K ω) Filter.atTop, a ≤ n} := by
                  simp only [Filter.IsCoboundedUnder, Filter.IsCobounded, Filter.eventually_map] at hcobdd
                  intro ⟨b, hb⟩; apply hcobdd; use b; intro a ha; exact hb ha
                rw [Real.sSup_of_not_bddAbove h_unbdd]
                simp only [ENNReal.ofReal_zero, zero_le]
            intro h_eq_top
            have h1 : ∫⁻ ω, ENNReal.ofReal (Filter.liminf (fun K => Z K ω) Filter.atTop) ∂μ ≤
                ∫⁻ ω, Filter.liminf (fun K => g' K ω) Filter.atTop ∂μ :=
              MeasureTheory.lintegral_mono h_ofReal_liminf_le
            have h2 : ∫⁻ ω, Filter.liminf (fun K => g' K ω) Filter.atTop ∂μ ≤
                Filter.liminf (fun K => ∫⁻ ω, g' K ω ∂μ) Filter.atTop := h_fatou_enn
            have h3 : Filter.liminf (fun K => ∫⁻ ω, g' K ω ∂μ) Filter.atTop ≤ ENNReal.ofReal B :=
              h_liminf_bound
            have h_chain : ∫⁻ ω, ENNReal.ofReal (Filter.liminf (fun K => Z K ω) Filter.atTop) ∂μ ≤
                ENNReal.ofReal B := le_trans h1 (le_trans h2 h3)
            rw [h_eq_top] at h_chain
            exact absurd (top_unique h_chain).symm ENNReal.top_ne_ofReal
          exact (MeasureTheory.lintegral_ofReal_ne_top_iff_integrable
            h_meas.aestronglyMeasurable h_nonneg_ae).mp h_lintegral_finite
        -- liminf Y_K = liminf Z_K + g is integrable
        have h_add := h_int_liminf_Z.add h_g_int
        exact Integrable.congr h_add (Filter.EventuallyEq.symm h_eq_ae)
      have h_int_sup : Integrable (fun ω => ⨆ n, X (t n) ω) μ := by
        -- Use bounds: X(t 0) ≤ sup ≤ liminf Y_K
        have h_aesm : AEStronglyMeasurable (fun ω => ⨆ n, X (t n) ω) μ :=
          (Measurable.iSup (fun n => hX_meas (t n))).aestronglyMeasurable
        -- X(t 0) is integrable
        have h_X_t0_int : Integrable (fun ω => X (t 0) ω) μ := by
          have h := hX_int (t 0) t₀
          simp only [hcenter, sub_zero] at h
          exact h
        -- Use Integrable.mono with bound |X(t 0)| + |liminf Y_K|
        refine MeasureTheory.Integrable.mono (h_X_t0_int.abs.add h_int_liminf.abs) h_aesm ?_
        -- Use h_pointwise which gives ∀ n, X (t n) ω ≤ liminf Y_K ω (stronger than h_pointwise_sup)
        filter_upwards [h_pointwise] with ω hω_pw
        -- hω_pw : ∀ n, X (t n) ω ≤ Filter.liminf (fun K => Y K ω) Filter.atTop
        -- Need: ‖⨆ n, X (t n) ω‖ ≤ ‖|X (t 0) ω| + |liminf Y_K ω|‖
        simp only [Real.norm_eq_abs, Pi.add_apply]
        -- RHS: |⋅| of (|X (t 0) ω| + |liminf Y_K ω|) = |X (t 0) ω| + |liminf Y_K ω| (sum of nonneg)
        have h_rhs_simp : |( |X (t 0) ω| + |Filter.liminf (fun K => Y K ω) Filter.atTop| )| =
            |X (t 0) ω| + |Filter.liminf (fun K => Y K ω) Filter.atTop| :=
          abs_of_nonneg (add_nonneg (abs_nonneg _) (abs_nonneg _))
        rw [h_rhs_simp]
        -- First establish BddAbove using h_pointwise
        have h_bdd_above : BddAbove (Set.range fun n => X (t n) ω) := by
          use Filter.liminf (fun K => Y K ω) Filter.atTop
          rintro _ ⟨n, rfl⟩
          exact hω_pw n
        -- Also get the sup bound
        have hω : ⨆ n, X (t n) ω ≤ Filter.liminf (fun K => Y K ω) Filter.atTop :=
          ciSup_le fun n => hω_pw n
        -- Cases on sign of sup
        by_cases h_sup_nonneg : 0 ≤ ⨆ n, X (t n) ω
        · -- sup ≥ 0, so |sup| = sup ≤ liminf Y_K ≤ |liminf Y_K|
          rw [abs_of_nonneg h_sup_nonneg]
          calc ⨆ n, X (t n) ω ≤ Filter.liminf (fun K => Y K ω) Filter.atTop := hω
            _ ≤ |Filter.liminf (fun K => Y K ω) Filter.atTop| := le_abs_self _
            _ ≤ |X (t 0) ω| + |Filter.liminf (fun K => Y K ω) Filter.atTop| := le_add_of_nonneg_left (abs_nonneg _)
        · -- sup < 0
          push Not at h_sup_nonneg
          -- |sup| = -sup ≤ -X(t 0) since X(t 0) ≤ sup
          rw [abs_of_neg h_sup_nonneg]
          have h_X_t0_le : X (t 0) ω ≤ ⨆ n, X (t n) ω := le_ciSup h_bdd_above 0
          calc -(⨆ n, X (t n) ω) ≤ -X (t 0) ω := neg_le_neg h_X_t0_le
            _ ≤ |X (t 0) ω| := neg_le_abs _
            _ ≤ |X (t 0) ω| + |Filter.liminf (fun K => Y K ω) Filter.atTop| := le_add_of_nonneg_right (abs_nonneg _)
      exact MeasureTheory.integral_mono_ae h_int_sup h_int_liminf h_pointwise_sup

    exact le_trans h_step1 h_fatou_Y

  -- ══════════════════════════════════════════════════════════════════════════════
  -- SECTION 7: UPPER BOUND FOR LIMINF (Lemma 4.10)
  -- liminf((6√2)σ * dyadicRHS) ≤ (12√2)σ * entropyIntegral via δf(δ) → 0.
  -- ══════════════════════════════════════════════════════════════════════════════

  have hδ_pos : ∀ K : ℕ, 0 < D * (2 : ℝ)^(-((K + 1) : ℤ)) := by
    intro K
    apply mul_pos hD
    apply zpow_pos (by norm_num : (0 : ℝ) < 2)

  have h_trunc_tendsto : Filter.Tendsto
      (fun K : ℕ => entropyIntegralTrunc s (D * 2^(-((K + 1) : ℤ))) D)
      Filter.atTop (nhds (entropyIntegral s D)) := by
    have h_base := entropyIntegralTrunc_tendsto_entropyIntegral hD hfinite
    exact h_base.comp (Filter.tendsto_add_atTop_nat 1)

  -- 2δf(δ) → 0 via squeeze with entropyIntegral gap
  have h_tail_tendsto : Filter.Tendsto
      (fun K : ℕ => 2 * (D * (2 : ℝ)^(-(↑K + 1 : ℤ))) * sqrtEntropy (D * (2 : ℝ)^(-(↑K + 1 : ℤ))) s)
      Filter.atTop (nhds 0) := by
    have h_sqrtEntropy_nonneg : ∀ K : ℕ, 0 ≤ sqrtEntropy (D * (2 : ℝ)^(-(↑K + 1 : ℤ))) s :=
      fun K => sqrtEntropy_nonneg _ _
    have h_prod_nonneg : ∀ K : ℕ,
        0 ≤ D * (2 : ℝ)^(-(↑K + 1 : ℤ)) * sqrtEntropy (D * (2 : ℝ)^(-(↑K + 1 : ℤ))) s :=
      fun K => mul_nonneg (le_of_lt (hδ_pos K)) (h_sqrtEntropy_nonneg K)
    have h_integral_gap_tendsto : Filter.Tendsto
        (fun K : ℕ => entropyIntegral s D - entropyIntegralTrunc s (D * (2 : ℝ)^(-(↑K + 1 : ℤ))) D)
        Filter.atTop (nhds 0) := by
      rw [← sub_self (entropyIntegral s D)]
      exact Filter.Tendsto.sub tendsto_const_nhds h_trunc_tendsto
    have h_bound : ∀ K : ℕ,
        D * (2 : ℝ)^(-(↑K + 1 : ℤ)) * sqrtEntropy (D * (2 : ℝ)^(-(↑K + 1 : ℤ))) s ≤
        entropyIntegral s D - entropyIntegralTrunc s (D * (2 : ℝ)^(-(↑K + 1 : ℤ))) D := by
      intro K
      let δ := D * (2 : ℝ)^(-(↑K + 1 : ℤ))
      have hδ_pos' : 0 < δ := hδ_pos K
      have hδ_le_D : δ ≤ D := by
        simp only [δ]
        have h1 : (2 : ℝ) ^ (-((K + 1) : ℤ)) ≤ 1 := zpow_le_one_of_nonpos₀ (by norm_num) (by linarith)
        calc D * (2 : ℝ) ^ (-((K + 1) : ℤ)) ≤ D * 1 := mul_le_mul_of_nonneg_left h1 (le_of_lt hD)
          _ = D := mul_one D
      -- The gap equals the integral over (0, δ]
      have h_gap_eq : entropyIntegral s D - entropyIntegralTrunc s δ D =
          (entropyIntegralENNReal s D - entropyIntegralENNRealTrunc s δ D).toReal := by
        unfold entropyIntegral entropyIntegralTrunc
        have h_trunc_le : entropyIntegralENNRealTrunc s δ D ≤ entropyIntegralENNReal s D := by
          unfold entropyIntegralENNRealTrunc entropyIntegralENNReal
          apply MeasureTheory.lintegral_mono'
          · apply Measure.restrict_mono
            · intro x hx; exact ⟨lt_of_lt_of_le hδ_pos' (le_of_lt hx.1), hx.2⟩
            · exact le_rfl
          · exact fun _ => le_rfl
        rw [ENNReal.toReal_sub_of_le h_trunc_le hfinite]
      -- The ENNReal gap equals the integral over Ioc 0 δ
      have h_ennreal_gap_eq : entropyIntegralENNReal s D - entropyIntegralENNRealTrunc s δ D =
          ∫⁻ eps in Set.Ioc 0 δ, dudleyIntegrand eps s := by
        unfold entropyIntegralENNReal entropyIntegralENNRealTrunc
        have h_union : Set.Ioc 0 D = Set.Ioc 0 δ ∪ Set.Ioc δ D :=
          (Set.Ioc_union_Ioc_eq_Ioc (le_of_lt hδ_pos') hδ_le_D).symm
        have h_disjoint : Disjoint (Set.Ioc 0 δ) (Set.Ioc δ D) :=
          Set.Ioc_disjoint_Ioc_of_le (le_refl δ)
        conv_lhs => rw [h_union]
        rw [MeasureTheory.lintegral_union measurableSet_Ioc h_disjoint]
        rw [ENNReal.add_sub_cancel_right]
        have h_trunc_lt_top : ∫⁻ x in Set.Ioc δ D, dudleyIntegrand x s < ⊤ :=
          entropyIntegralENNRealTrunc_lt_top hδ_pos' hs
        exact ne_top_of_lt h_trunc_lt_top
      -- Lower bound: δ * dudleyIntegrand(δ) ≤ ∫ Ioc 0 δ dudleyIntegrand
      have h_lower : ENNReal.ofReal δ * dudleyIntegrand δ s ≤
          ∫⁻ eps in Set.Ioc 0 δ, dudleyIntegrand eps s := by
        have h_anti : ∀ x ∈ Set.Ioc 0 δ, dudleyIntegrand δ s ≤ dudleyIntegrand x s := by
          intro x hx
          have hx_pos : 0 < x := hx.1
          have hx_le_δ : x ≤ δ := hx.2
          exact dudleyIntegrand_anti_eps_of_totallyBounded hx_pos hx_le_δ hs
        calc ENNReal.ofReal δ * dudleyIntegrand δ s
            = ∫⁻ _ in Set.Ioc 0 δ, dudleyIntegrand δ s := by
              rw [MeasureTheory.lintegral_const,
                  MeasureTheory.Measure.restrict_apply MeasurableSet.univ]
              simp only [Set.univ_inter, Real.volume_Ioc, sub_zero]
              rw [mul_comm]
          _ ≤ ∫⁻ x in Set.Ioc 0 δ, dudleyIntegrand x s := by
              exact MeasureTheory.setLIntegral_mono' measurableSet_Ioc h_anti
      -- Convert to Real
      have h_gap_finite : ∫⁻ eps in Set.Ioc 0 δ, dudleyIntegrand eps s ≠ ⊤ := by
        rw [← h_ennreal_gap_eq]
        exact ENNReal.sub_ne_top hfinite
      calc δ * sqrtEntropy δ s
          = (ENNReal.ofReal δ * dudleyIntegrand δ s).toReal := by
            rw [ENNReal.toReal_mul, ENNReal.toReal_ofReal (le_of_lt hδ_pos')]
            unfold dudleyIntegrand
            rw [ENNReal.toReal_ofReal (sqrtEntropy_nonneg δ s)]
        _ ≤ (∫⁻ eps in Set.Ioc 0 δ, dudleyIntegrand eps s).toReal := by
            apply ENNReal.toReal_mono h_gap_finite h_lower
        _ = entropyIntegral s D - entropyIntegralTrunc s δ D := by
            rw [h_gap_eq, h_ennreal_gap_eq]

    -- First show δ * sqrtEntropy(δ) → 0 using squeeze
    have h_tail_tendsto' : Filter.Tendsto
        (fun K : ℕ => D * (2 : ℝ)^(-(↑K + 1 : ℤ)) * sqrtEntropy (D * (2 : ℝ)^(-(↑K + 1 : ℤ))) s)
        Filter.atTop (nhds 0) := by
      apply squeeze_zero'
      · exact Filter.Eventually.of_forall h_prod_nonneg
      · exact Filter.Eventually.of_forall h_bound
      · exact h_integral_gap_tendsto

    -- Multiply by 2 to get 2 * δ * sqrtEntropy(δ) → 0
    have h2 := Filter.Tendsto.const_mul 2 h_tail_tendsto'
    simp only [mul_zero] at h2
    refine Filter.Tendsto.congr ?_ h2
    intro K
    ring

  -- Upper bound: dyadicRHS ≤ 2*entropyIntegral + 2*δf(δ)
  have h_dyadic_upper : ∀ K, 1 ≤ K → dyadicRHS_real s D (K + 1) ≤
      2 * entropyIntegral s D + 2 * (D * 2^(-((K+1) : ℤ))) * sqrtEntropy (D * 2^(-((K+1) : ℤ))) s := by
    intro K hK
    have h1 := dyadicRHS_le_twice_entropyIntegralTrunc_add_tail hs hD (Nat.le_add_left 1 K)
    have h2 := entropyIntegralTrunc_le_entropyIntegral hD (K + 1) hfinite
    calc dyadicRHS_real s D (K + 1)
        ≤ 2 * entropyIntegralTrunc s (D * 2^(-((K+1) : ℤ))) D +
          2 * (D * 2^(-((K+1) : ℤ))) * sqrtEntropy (D * 2^(-((K+1) : ℤ))) s := h1
      _ ≤ 2 * entropyIntegral s D +
          2 * (D * 2^(-((K+1) : ℤ))) * sqrtEntropy (D * 2^(-((K+1) : ℤ))) s := by
          have : entropyIntegralTrunc s (D * 2^(-((K+1) : ℤ))) D ≤ entropyIntegral s D := h2
          linarith

  -- (11√2)σ * dyadicRHS is bounded above (needed for IsCoboundedUnder)
  have h_rhs_bounded_above : ∃ B : ℝ, ∀ K, 1 ≤ K → (6 * Real.sqrt 2) * σ * dyadicRHS_real s D (K + 1) ≤ B := by
    use (6 * Real.sqrt 2) * σ * (4 * entropyIntegral s D)
    intro K hK
    have h_le := dyadicRHS_le_four_times_entropyIntegral hs hD (Nat.le_add_left 1 K) hfinite
    have h_nonneg : 0 ≤ (6 * Real.sqrt 2) * σ := by
      apply mul_nonneg (by positivity : (0:ℝ) ≤ 6 * Real.sqrt 2) (le_of_lt hσ)
    exact mul_le_mul_of_nonneg_left h_le h_nonneg

  -- Coboundedness for liminf_le_liminf: the function is bounded above
  have h_rhs_cobounded : Filter.IsCoboundedUnder (· ≥ ·) Filter.atTop
      (fun K => (6 * Real.sqrt 2) * σ * dyadicRHS_real s D (K + 1)) := by
    obtain ⟨B, hB⟩ := h_rhs_bounded_above
    refine ⟨B, ?_⟩
    intro a ha
    -- ha : ∀ᶠ x in (atTop.map f), x ≥ a, need to convert to ∀ᶠ K in atTop, f K ≥ a
    rw [Filter.eventually_map] at ha
    rw [Filter.eventually_atTop] at ha
    obtain ⟨N, hN⟩ := ha
    -- At K = max N 1, we have both: function(K) ≥ a and function(K) ≤ B
    have hK : max N 1 ≥ 1 := le_max_right N 1
    have h1 : (6 * Real.sqrt 2) * σ * dyadicRHS_real s D (max N 1 + 1) ≥ a := hN (max N 1) (le_max_left N 1)
    have h2 : (6 * Real.sqrt 2) * σ * dyadicRHS_real s D (max N 1 + 1) ≤ B := hB (max N 1) hK
    linarith

  -- Upper bound on liminf: liminf((6√2)σ*dyadicRHS) ≤ (12√2)σ*entropyIntegral
  have h_liminf_upper : Filter.liminf (fun K => (6 * Real.sqrt 2) * σ * dyadicRHS_real s D (K + 1)) Filter.atTop
      ≤ (12 * Real.sqrt 2) * σ * entropyIntegral s D := by
    -- For each K: (6√2)σ*dyadicRHS ≤ (12√2)σ*entropyIntegral + (12√2)σ*tail(K)
    have h_upper_with_tail : ∀ K, 1 ≤ K → (6 * Real.sqrt 2) * σ * dyadicRHS_real s D (K + 1) ≤
        (12 * Real.sqrt 2) * σ * entropyIntegral s D +
        (12 * Real.sqrt 2) * σ * (D * 2^(-((K+1) : ℤ))) * sqrtEntropy (D * 2^(-((K+1) : ℤ))) s := by
      intro K hK
      have h := h_dyadic_upper K hK
      calc (6 * Real.sqrt 2) * σ * dyadicRHS_real s D (K + 1)
          ≤ (6 * Real.sqrt 2) * σ * (2 * entropyIntegral s D +
            2 * (D * 2^(-((K+1) : ℤ))) * sqrtEntropy (D * 2^(-((K+1) : ℤ))) s) := by
            apply mul_le_mul_of_nonneg_left h
            apply mul_nonneg (by positivity : (0:ℝ) ≤ 6 * Real.sqrt 2) (le_of_lt hσ)
        _ = (12 * Real.sqrt 2) * σ * entropyIntegral s D +
            (12 * Real.sqrt 2) * σ * (D * 2^(-((K+1) : ℤ))) * sqrtEntropy (D * 2^(-((K+1) : ℤ))) s := by ring

    -- The tail tends to 0, so liminf(tail) = 0
    have h_tail_tendsto_scaled : Filter.Tendsto
        (fun K : ℕ => (12 * Real.sqrt 2) * σ * (D * (2 : ℝ)^(-(↑K + 1 : ℤ))) * sqrtEntropy (D * (2 : ℝ)^(-(↑K + 1 : ℤ))) s)
        Filter.atTop (nhds 0) := by
      have h := h_tail_tendsto.const_mul ((6 * Real.sqrt 2) * σ)
      simp only [mul_zero] at h
      refine Filter.Tendsto.congr ?_ h
      intro K; ring

    -- Define the combined function using the same form as h_tail_tendsto_scaled
    let g := fun K : ℕ => (12 * Real.sqrt 2) * σ * entropyIntegral s D +
        (12 * Real.sqrt 2) * σ * (D * 2^(-(↑K + 1 : ℤ))) * sqrtEntropy (D * 2^(-(↑K + 1 : ℤ))) s

    have h_const_plus_vanishing : Filter.Tendsto g Filter.atTop
        (nhds ((12 * Real.sqrt 2) * σ * entropyIntegral s D)) := by
      have h_const : Filter.Tendsto (fun _ : ℕ => (12 * Real.sqrt 2) * σ * entropyIntegral s D)
          Filter.atTop (nhds ((12 * Real.sqrt 2) * σ * entropyIntegral s D)) := tendsto_const_nhds
      have h_sum := Filter.Tendsto.add h_const h_tail_tendsto_scaled
      simp only [add_zero] at h_sum
      exact h_sum

    have h_liminf_eq : Filter.liminf g Filter.atTop = (12 * Real.sqrt 2) * σ * entropyIntegral s D :=
      h_const_plus_vanishing.liminf_eq

    -- (6√2)σ * dyadicRHS is bounded below by 0
    have h_rhs_bounded_below : Filter.IsBoundedUnder (· ≥ ·) Filter.atTop
        (fun K => (6 * Real.sqrt 2) * σ * dyadicRHS_real s D (K + 1)) := by
      use 0
      rw [Filter.eventually_map, Filter.eventually_atTop]
      use 1
      intro K _
      apply mul_nonneg
      · apply mul_nonneg (by positivity : (0:ℝ) ≤ 6 * Real.sqrt 2) (le_of_lt hσ)
      · exact dyadicRHS_real_nonneg hD (K + 1)

    calc Filter.liminf (fun K => (6 * Real.sqrt 2) * σ * dyadicRHS_real s D (K + 1)) Filter.atTop
        ≤ Filter.liminf g Filter.atTop := by
          apply Filter.liminf_le_liminf
          · filter_upwards [Filter.eventually_ge_atTop 1] with K hK
            exact h_upper_with_tail K hK
          · exact h_rhs_bounded_below
          · exact h_const_plus_vanishing.isCoboundedUnder_ge
      _ = (12 * Real.sqrt 2) * σ * entropyIntegral s D := h_liminf_eq

  -- ══════════════════════════════════════════════════════════════════════════════
  -- SECTION 8: INTEGRAL BOUNDED BELOW (Lemma 4.11)
  -- ∫ Y_K ≥ 0 via centering: ∫ X u = 0 for all u ∈ s.
  -- ══════════════════════════════════════════════════════════════════════════════

  have h_int_bounded_below : Filter.IsBoundedUnder (· ≥ ·) Filter.atTop
      (fun K => ∫ ω, Y K ω ∂μ) := by
    have h_int_zero : ∀ u, u ∈ s → ∫ ω, X u ω ∂μ = 0 := by
      intro u hu
      have h := hX_centered u t₀
      simp only [hcenter, sub_zero] at h
      exact h
    use 0
    rw [Filter.eventually_map, Filter.eventually_atTop]
    use 1
    intro K hK
    have h_nets_subset := dn.nets_subset K
    have ⟨u₀, hu₀⟩ := hnet_nonempty K
    have hu₀_in_s : u₀ ∈ s := h_nets_subset hu₀
    have h_int_u₀ : ∫ ω, X u₀ ω ∂μ = 0 := h_int_zero u₀ hu₀_in_s
    calc ∫ ω, Y K ω ∂μ ≥ ∫ ω, X u₀ ω ∂μ := by
          have h_int_u₀' : Integrable (fun ω => X u₀ ω) μ := by
            have h := hX_int u₀ t₀
            simp only [hcenter, sub_zero] at h
            exact h
          apply integral_mono_ae h_int_u₀' (h_Y_int K hK)
          filter_upwards with ω
          exact Finset.le_sup' (fun u => X u ω) hu₀
      _ = 0 := h_int_u₀

  -- ══════════════════════════════════════════════════════════════════════════════
  -- SECTION 9: FINAL ASSEMBLY
  -- ∫ sup X(tₙ) ≤ liminf ∫ Y_K ≤ liminf((6√2)σ*dyadicRHS) ≤ (12√2)σ*entropyIntegral.
  -- ══════════════════════════════════════════════════════════════════════════════

  calc ∫ ω, ⨆ n, X (t n) ω ∂μ
    _ ≤ Filter.liminf (fun K => ∫ ω, Y K ω ∂μ) Filter.atTop := h_fatou
    _ ≤ Filter.liminf (fun K => (6 * Real.sqrt 2) * σ * dyadicRHS_real s D (K + 1)) Filter.atTop := by
        apply Filter.liminf_le_liminf
        · filter_upwards [Filter.eventually_ge_atTop 1] with K hK
          exact h_core_bound K hK
        · exact h_int_bounded_below
        · exact h_rhs_cobounded
    _ ≤ (12 * Real.sqrt 2) * σ * entropyIntegral s D := h_liminf_upper

/-!
## Dudley's Entropy Integral Inequality
-/
theorem dudley {μ : Measure Ω} [IsProbabilityMeasure μ]
    {X : A → Ω → ℝ} {σ : ℝ} (hσ : 0 < σ)
    (hX : IsSubGaussianProcess μ X σ)
    {s : Set A} (hs : TotallyBounded s)
    {D : ℝ} (hD : 0 < D) (hdiam : Metric.diam s ≤ D)
    (t₀ : A) (ht₀ : t₀ ∈ s) (hcenter : ∀ ω, X t₀ ω = 0)
    (hX_meas : ∀ t, Measurable (X t))
    (hX_int_exp : ∀ t s : A, ∀ l : ℝ, Integrable (fun ω => Real.exp (l * (X t ω - X s ω))) μ)
    (hfinite : entropyIntegralENNReal s D ≠ ⊤)
    (hcont : ∀ ω, Continuous (fun (t : ↥s) => X t.1 ω)) :
    ∫ ω, ⨆ t ∈ s, X t ω ∂μ ≤ (12 * Real.sqrt 2) * σ * entropyIntegral s D := by
  -- Derive hsne from t₀ ∈ s
  have hsne : s.Nonempty := ⟨t₀, ht₀⟩

  -- Rewrite biSup as iSup over subtype (handles both bounded and unbounded)
  have h_biSup_eq : ∀ ω, (⨆ t ∈ s, X t ω) = ⨆ (t : ↥s), X t.1 ω := fun ω => by
    have hzero : ∃ t ∈ s, X t ω = 0 := ⟨t₀, ht₀, hcenter ω⟩
    exact biSup_eq_iSup_subtype_real hsne hzero

  -- Use separability to convert to countable supremum
  have h_sep_eq := sup_over_s_eq_iSup_denseSeq_pathwise hs hsne hcont

  -- Combine to get the key equality
  have h_sup_eq : ∀ ω, (⨆ t ∈ s, X t ω) = ⨆ n : ℕ, X (denseSeqInTB hs hsne n).1 ω := by
    intro ω
    rw [h_biSup_eq ω, h_sep_eq ω]

  -- Rewrite the integral using the countable supremum
  have h_int_eq : ∫ ω, ⨆ t ∈ s, X t ω ∂μ = ∫ ω, ⨆ n : ℕ, X (denseSeqInTB hs hsne n).1 ω ∂μ := by
    congr 1
    ext ω
    exact h_sup_eq ω

  rw [h_int_eq]

  -- Apply dudley_chaining_bound_countable
  exact dudley_chaining_bound_countable hσ hX hs hD hdiam t₀ ht₀ hcenter hX_meas
    hX_int_exp hfinite hcont
