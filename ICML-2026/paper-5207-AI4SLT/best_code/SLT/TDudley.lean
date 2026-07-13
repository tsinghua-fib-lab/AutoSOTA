/-
Copyright (c) 2026 Yuanhe Zhang. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Yuanhe Zhang, Jason D. Lee, Fanghui Liu
-/
import SLT.MetricEntropy
import SLT.SubGaussian
import Mathlib.Analysis.Complex.ExponentialBounds
import Mathlib.Analysis.SpecialFunctions.Log.Base
import Mathlib.Tactic.Cases

open scoped BigOperators
open scoped Real
open scoped Nat
open scoped Classical
open scoped Pointwise

noncomputable section

/-
Auxiliary finite-covering API used by the truncated Dudley proof.
-/
open Set Filter MeasureTheory ProbabilityTheory Metric Real Topology


variable {α : Type*} [PseudoMetricSpace α]

namespace TDudley

/-- Internal set-level wrapper around `IsENet`, requiring centers to lie in `s`. -/
def IsSubsetENet (s : Set α) (u : Set α) (δ : ℝ) : Prop :=
  u ⊆ s ∧ ∃ hfin : u.Finite, IsENet hfin.toFinset δ s

lemma IsSubsetENet.of_exists_dist {s u : Set α} {δ : ℝ}
    (hfin : u.Finite) (hsub : u ⊆ s)
    (hcover : ∀ x ∈ s, ∃ y ∈ u, dist x y ≤ δ) :
    IsSubsetENet s u δ := by
  refine ⟨hsub, hfin, ?_⟩
  intro x hx
  obtain ⟨y, hy, hdist⟩ := hcover x hx
  exact mem_iUnion₂.mpr ⟨y, hfin.mem_toFinset.mpr hy, mem_closedBall.mpr hdist⟩

lemma IsSubsetENet.exists_near {s u : Set α} {δ : ℝ}
    (hcover : IsSubsetENet s u δ) {x : α} (hx : x ∈ s) :
    ∃ y ∈ u, dist x y ≤ δ := by
  obtain ⟨hfin, hnet⟩ := hcover.2
  have hx_cover := hnet hx
  rw [mem_iUnion₂] at hx_cover
  obtain ⟨y, hy, hdist⟩ := hx_cover
  exact ⟨y, hfin.mem_toFinset.mp hy, mem_closedBall.mp hdist⟩

/-- Internal Nat-valued minimum size of a finite `δ`-net whose centers lie in `s`.
When the admissible family is nonempty, this is the minimum such cardinality. -/
def subsetENetCard (s : Set α) (δ : ℝ) : ℕ :=
  sInf {n | ∃ u : Finset α, IsSubsetENet s u δ ∧ u.card = n}

/-
Definitions of global oscillation and local oscillation of a function X on a set T.
-/
def globalOsc (X : α → ℝ) (T : Set α) : ℝ :=
  sSup {X x - X y | (x ∈ T) (y ∈ T)}

def localOsc (X : α → ℝ) (T : Set α) (δ : ℝ) : ℝ :=
  sSup {X x - X y | (x ∈ T) (y ∈ T) (_ : dist x y ≤ δ)}

/-
Lemma 1.1: One-step discretization bound.
Fix δ > 0 and let U be a finite centered δ-net of T. Then
sup_{θ, θ' ∈ T} (X_θ - X_θ') ≤ 2 sup_{d(γ, γ') ≤ δ} (X_γ - X_γ') + 2 max_{u ∈ U} |X_u - X_u0|.
-/
lemma one_step_discretization_bound
  {X : α → ℝ} {T : Set α} {δ : ℝ}
  {U : Set α} (hU_fin : U.Finite) (hU_subset : U ⊆ T)
  (hU_cover : IsSubsetENet T U δ)
  (u₀ : α) (hu₀ : u₀ ∈ U) :
  globalOsc X T ≤ 2 * localOsc X T δ + 2 * (⨆ u ∈ U, |X u - X u₀|) := by
  by_contra h_contra;
  
  obtain ⟨θ, θ', hθθ', h_ineq⟩ : ∃ θ θ', θ ∈ T ∧ θ' ∈ T ∧ X θ - X θ' > 2 * localOsc X T δ + 2 * ⨆ u ∈ U, |X u - X u₀| := by
    simp +zetaDelta at *;
    convert exists_lt_of_lt_csSup _ h_contra using 1;
    · exact ⟨ fun ⟨ θ, hθ, x, hx, h ⟩ => ⟨ _, ⟨ θ, hθ, x, hx, rfl ⟩, h ⟩, by rintro ⟨ a, ⟨ θ, hθ, x, hx, rfl ⟩, h ⟩ ; exact ⟨ θ, hθ, x, hx, h ⟩ ⟩;
    · exact ⟨ _, ⟨ u₀, hU_subset hu₀, u₀, hU_subset hu₀, rfl ⟩ ⟩;
  
  obtain ⟨πθ, hπθ⟩ : ∃ πθ ∈ U, dist θ πθ ≤ δ := by
    exact IsSubsetENet.exists_near hU_cover hθθ'
  obtain ⟨πθ', hπθ'⟩ : ∃ πθ' ∈ U, dist θ' πθ' ≤ δ := by
    exact IsSubsetENet.exists_near hU_cover h_ineq.1;
  
  have h_triangle : X θ - X θ' ≤ (X θ - X πθ) + (X πθ - X πθ') + (X πθ' - X θ') := by
    linarith
  have h_local_osc : X θ - X πθ ≤ localOsc X T δ ∧ X πθ' - X θ' ≤ localOsc X T δ := by
    refine' ⟨ le_csSup _ _, le_csSup _ _ ⟩;
    · by_cases h_bdd : BddAbove (Set.image (fun p : α × α => X p.1 - X p.2) {p : α × α | p.1 ∈ T ∧ p.2 ∈ T ∧ dist p.1 p.2 ≤ δ});
      · exact ⟨ h_bdd.choose, by rintro x ⟨ y, hy, z, hz, h, rfl ⟩ ; exact h_bdd.choose_spec ⟨ ( y, z ), ⟨ hy, hz, h ⟩, rfl ⟩ ⟩;
      · unfold globalOsc localOsc at h_contra;
        rw [ Real.sSup_of_not_bddAbove ] at h_contra;
        · rw [ Real.sSup_of_not_bddAbove ] at h_contra;
          · exact False.elim <| h_contra <| add_nonneg ( mul_nonneg zero_le_two <| by norm_num ) <| mul_nonneg zero_le_two <| Real.iSup_nonneg fun _ => Real.iSup_nonneg fun _ => abs_nonneg _;
          · exact fun ⟨ M, hM ⟩ => h_bdd ⟨ M, Set.forall_mem_image.2 fun p hp => hM ⟨ p.1, hp.1, p.2, hp.2.1, hp.2.2, rfl ⟩ ⟩;
        · rintro ⟨M, hM⟩
          exact h_bdd ⟨M, Set.forall_mem_image.2 fun p hp => hM ⟨p.1, hp.1, p.2, hp.2.1, rfl⟩⟩
    · exact ⟨ θ, hθθ', πθ, hU_subset hπθ.1, hπθ.2, rfl ⟩;
    · by_cases h_bdd : BddAbove (Set.image (fun p : α × α => X p.1 - X p.2) {p : α × α | p.1 ∈ T ∧ p.2 ∈ T ∧ dist p.1 p.2 ≤ δ});
      · exact ⟨ h_bdd.choose, by rintro x ⟨ y, hy, z, hz, h, rfl ⟩ ; exact h_bdd.choose_spec ⟨ ( y, z ), ⟨ hy, hz, h ⟩, rfl ⟩ ⟩;
      · unfold globalOsc localOsc at h_contra;
        rw [ Real.sSup_of_not_bddAbove ] at h_contra;
        · rw [ Real.sSup_of_not_bddAbove ] at h_contra;
          · exact False.elim <| h_contra <| add_nonneg ( mul_nonneg zero_le_two <| by norm_num ) <| mul_nonneg zero_le_two <| Real.iSup_nonneg fun _ => Real.iSup_nonneg fun _ => abs_nonneg _;
          · exact fun ⟨ M, hM ⟩ => h_bdd ⟨ M, Set.forall_mem_image.2 fun p hp => hM ⟨ p.1, hp.1, p.2, hp.2.1, hp.2.2, rfl ⟩ ⟩;
        · rintro ⟨M, hM⟩
          exact h_bdd ⟨M, Set.forall_mem_image.2 fun p hp => hM ⟨p.1, hp.1, p.2, hp.2.1, rfl⟩⟩
    · exact ⟨ πθ', hU_subset hπθ'.1, θ', h_ineq.1, by simpa [ dist_comm ] using hπθ'.2, rfl ⟩
  have h_max_abs : X πθ - X πθ' ≤ 2 * ⨆ u ∈ U, |X u - X u₀| := by
    have h_max_abs : |X πθ - X u₀| ≤ ⨆ u ∈ U, |X u - X u₀| ∧ |X πθ' - X u₀| ≤ ⨆ u ∈ U, |X u - X u₀| := by
      have h_bdd_abs : BddAbove (Set.range fun u : α => ⨆ (_ : u ∈ U), |X u - X u₀|) := by
        refine ⟨∑ u ∈ hU_fin.toFinset, |X u - X u₀|, Set.forall_mem_range.2 fun u => ?_⟩
        rw [@ciSup_eq_ite]
        split_ifs with hu
        · exact Finset.single_le_sum (fun u _ => abs_nonneg (X u - X u₀)) (hU_fin.mem_toFinset.mpr hu)
        · exact le_trans (by norm_num) (Finset.sum_nonneg fun u _ => abs_nonneg (X u - X u₀))
      have h_mem_sup (u : α) (hu : u ∈ U) : |X u - X u₀| ≤ ⨆ (_ : u ∈ U), |X u - X u₀| := by
        exact le_ciSup ⟨|X u - X u₀|, Set.forall_mem_range.2 fun _ => le_rfl⟩ hu
      exact ⟨le_trans (h_mem_sup πθ hπθ.1) (le_ciSup h_bdd_abs πθ),
        le_trans (h_mem_sup πθ' hπθ'.1) (le_ciSup h_bdd_abs πθ')⟩
    linarith [ abs_le.mp h_max_abs.1, abs_le.mp h_max_abs.2 ];
  linarith

lemma sub_gaussian_tail_bound {α : Type*} [MeasurableSpace α]
  {X : α → ℝ} {σ_sq : ℝ} {μ : Measure α} [IsFiniteMeasure μ]
  (h_sg : IsSubGaussian X σ_sq μ) (t : ℝ) (ht : 0 ≤ t) :
  μ.real {x | t ≤ X x} ≤ Real.exp (-t^2 / (2 * σ_sq)) := by
  rcases h_sg with ⟨hσ_sq, hX⟩
  convert hX.measure_ge_le ht using 2
  change -t ^ 2 / (2 * σ_sq) = -t ^ 2 / (2 * σ_sq)
  rfl

/-
Bound on the moment generating function of the maximum of sub-Gaussian random variables.
-/
lemma mgf_max_bound {α : Type*} [MeasurableSpace α]
  {M : ℕ} (hM : 0 < M)
  {Y : Fin M → α → ℝ} {σ_sq : ℝ} {μ : Measure α} [IsProbabilityMeasure μ]
  (h_sg : ∀ k, IsSubGaussian (Y k) σ_sq μ) (t : ℝ) :
  ∫ x, Real.exp (t * (⨆ k, Y k x)) ∂μ ≤ M * Real.exp (σ_sq * t^2 / 2) := by
  
  have h_max_exp_bound : ∫ x, Real.exp (t * ⨆ k, Y k x) ∂μ ≤ ∑ k, ∫ x, Real.exp (t * Y k x) ∂μ := by
    rw [ ← MeasureTheory.integral_finsetSum ];
    · refine' MeasureTheory.integral_mono_of_nonneg _ _ _;
      · exact Filter.Eventually.of_forall fun x => Real.exp_nonneg _;
      · refine' MeasureTheory.integrable_finsetSum _ fun i _ => _;
        exact sub_gaussian_integrable (h_sg i) t;
      · filter_upwards [ ] with x;
        
        have h_max_exp : Real.exp (t * ⨆ k, Y k x) ≤ ∑ k, Real.exp (t * Y k x) := by
          have h_max : ∃ k, Y k x = ⨆ k, Y k x := by
            simpa [sSup_range] using
              (IsCompact.sSup_mem
                (isCompact_range <| show Continuous fun k => Y k x from by continuity)
                (Set.nonempty_of_mem <| Set.mem_range_self ⟨0, hM⟩))
          exact le_trans ( by rw [ ← h_max.choose_spec ] ) ( Finset.single_le_sum ( fun k _ => Real.exp_nonneg ( t * Y k x ) ) ( Finset.mem_univ _ ) );
        exact h_max_exp;
    · exact fun k _ => sub_gaussian_integrable ( h_sg k ) t;
  
  have h_sub_gaussian_bound : ∀ k, ∫ x, Real.exp (t * Y k x) ∂μ ≤ Real.exp (σ_sq * t^2 / 2) := by
    intro k
    obtain ⟨hσ_sq, hYk⟩ := h_sg k
    have h_exp_bound : ∫ x, Real.exp (t * Y k x) ∂μ ≤ Real.exp (σ_sq * t^2 / 2) := by
      change ProbabilityTheory.mgf (Y k) μ t ≤ Real.exp (σ_sq * t^2 / 2)
      convert hYk.mgf_le t using 2
      change σ_sq * t ^ 2 / 2 = σ_sq * t ^ 2 / 2
      rfl
    exact h_exp_bound;
  calc
    ∫ x, Real.exp (t * ⨆ k, Y k x) ∂μ
        ≤ ∑ k, ∫ x, Real.exp (t * Y k x) ∂μ := h_max_exp_bound
    _ ≤ ∑ _k : Fin M, Real.exp (σ_sq * t^2 / 2) :=
        Finset.sum_le_sum fun k _ => h_sub_gaussian_bound k
    _ = M * Real.exp (σ_sq * t^2 / 2) := by simp

/-
Lemma 1.2: Metric entropy bound for sub-Gaussian maxima.
Let Y_1, ..., Y_M be mean-zero sub-Gaussian random variables with parameter σ^2.
Then E[max Y_k] ≤ σ * sqrt(2 * log M).
-/
lemma metric_entropy_bound_for_sub_gaussian_maxima
  {α : Type*} [MeasurableSpace α]
  {M : ℕ} (hM : 0 < M)
  {Y : Fin M → α → ℝ} {σ_sq : ℝ} (hσ : 0 ≤ σ_sq)
  {μ : Measure α} [IsProbabilityMeasure μ]
  (h_sg : ∀ k, IsSubGaussian (Y k) σ_sq μ) :
  ∫ x, (⨆ k, Y k x) ∂μ ≤ Real.sqrt (2 * σ_sq * Real.log M) := by
  
  have h_max_bound : ∀ t > 0, ∫ x, (⨆ k, Y k x) ∂μ ≤ (Real.log M) / t + σ_sq * t / 2 := by
    intro t ht
    have h_jensen : Real.exp (t * (∫ x, (⨆ k, Y k x) ∂μ)) ≤ ∫ x, Real.exp (t * (⨆ k, Y k x)) ∂μ := by
      by_cases h_integrable : MeasureTheory.Integrable (fun x => ⨆ k, Y k x) μ;
      · have h_jensen : ConvexOn ℝ (Set.univ : Set ℝ) Real.exp := by
          exact convexOn_exp;
        have h_jensen : Real.exp (∫ x, t * (⨆ k, Y k x) ∂μ) ≤ ∫ x, Real.exp (t * (⨆ k, Y k x)) ∂μ := by
          apply_rules [ h_jensen.map_integral_le ];
          · exact Real.continuousOn_exp;
          · exact isClosed_univ;
          · exact Filter.Eventually.of_forall fun x => Set.mem_univ _;
          · exact h_integrable.const_mul t;
          · have h_integrable_exp : ∀ k, MeasureTheory.Integrable (fun x => Real.exp (t * Y k x)) μ := by
              exact fun k => sub_gaussian_integrable (h_sg k) t;
            refine' MeasureTheory.Integrable.mono' ( MeasureTheory.integrable_finsetSum _ fun k _ => h_integrable_exp k ) _ _;
            exact Finset.univ;
            · exact Real.continuous_exp.comp_aestronglyMeasurable ( h_integrable.aestronglyMeasurable.const_mul _ );
            · filter_upwards [ ] with x;
              rcases M with ( _ | _ | M ) <;> simp_all +decide [ Fin.sum_univ_succ ];
              
              have h_exp_sup : Real.exp (t * ⨆ k, Y k x) ≤ ∑ k : Fin (M + 2), Real.exp (t * Y k x) := by
                have h_exp_sup : ∃ k, Y k x = ⨆ k, Y k x := by
                  exact exists_eq_ciSup_of_finite;
                exact le_trans ( by rw [ ← h_exp_sup.choose_spec ] ) ( Finset.single_le_sum ( fun k _ => Real.exp_nonneg ( t * Y k x ) ) ( Finset.mem_univ _ ) );
              exact h_exp_sup.trans_eq ( by rw [ Fin.sum_univ_succ, Fin.sum_univ_succ ] ; simp +decide );
        simpa only [ MeasureTheory.integral_const_mul ] using h_jensen;
      · have h_integrable : ∀ k, MeasureTheory.Integrable (fun x => Y k x) μ := by
          intro k
          have := h_sg k
          obtain ⟨hσ_sq, h_mgf⟩ := this
          have h_integrable : MeasureTheory.Integrable (fun x => Y k x) μ := by
            have := h_mgf.integrable_exp_mul;
            have := this 1;
            refine' MeasureTheory.Integrable.mono' _ _ _;
            exact fun ω => Real.exp ( Y k ω ) + Real.exp ( -Y k ω );
            · exact MeasureTheory.Integrable.add ( by simpa using this ) ( by simpa using ‹∀ t : ℝ, MeasureTheory.Integrable ( fun ω => Real.exp ( t * Y k ω ) ) μ› ( -1 ) );
            · have := h_mgf.aestronglyMeasurable;
              exact this;
            · filter_upwards [ ] with ω using abs_le.mpr ⟨ by linarith [ Real.add_one_le_exp ( Y k ω ), Real.add_one_le_exp ( -Y k ω ), Real.exp_pos ( Y k ω ), Real.exp_pos ( -Y k ω ) ], by linarith [ Real.add_one_le_exp ( Y k ω ), Real.add_one_le_exp ( -Y k ω ), Real.exp_pos ( Y k ω ), Real.exp_pos ( -Y k ω ) ] ⟩
          exact h_integrable;
        have h_integrable : MeasureTheory.Integrable (fun x => ⨆ k, Y k x) μ := by
          refine' MeasureTheory.Integrable.mono' _ _ _;
          refine' fun x => ∑ k, |Y k x|;
          · exact MeasureTheory.integrable_finsetSum _ fun k _ => MeasureTheory.Integrable.abs ( h_integrable k );
          · have h_integrable : ∀ k, MeasureTheory.AEStronglyMeasurable (fun x => Y k x) μ := by
              exact fun k => ( h_integrable k |> MeasureTheory.Integrable.aestronglyMeasurable );
            have h_integrable : ∀ {f : Fin M → α → ℝ}, (∀ k, MeasureTheory.AEStronglyMeasurable (fun x => f k x) μ) → MeasureTheory.AEStronglyMeasurable (fun x => ⨆ k, f k x) μ := by
              intro f hf;
              choose g hg using hf;
              refine' ⟨ fun x => ⨆ k, g k x, _, _ ⟩;
              · exact (Measurable.iSup fun k => (hg k).1.measurable).stronglyMeasurable;
              · filter_upwards [ MeasureTheory.ae_all_iff.2 fun k => hg k |>.2 ] with x hx using by simp +decide [ hx ] ;
            exact h_integrable ‹_›;
          · filter_upwards [ ] with x;
            refine' abs_le.mpr ⟨ _, _ ⟩;
            · exact le_trans ( neg_le_neg ( Finset.single_le_sum ( fun k _ => abs_nonneg ( Y k x ) ) ( Finset.mem_univ ⟨ 0, hM ⟩ ) ) ) ( le_trans ( neg_le_of_abs_le ( by simp +decide ) ) ( le_ciSup ( Finite.bddAbove_range fun k => Y k x ) ⟨ 0, hM ⟩ ) );
            · convert ciSup_le fun k => show Y k x ≤ ∑ k, |Y k x| from le_trans ( le_abs_self _ ) ( Finset.single_le_sum ( fun a _ => abs_nonneg ( Y a x ) ) ( Finset.mem_univ k ) ) using 1;
              exact ⟨ ⟨ 0, hM ⟩ ⟩;
        contradiction;
    
    have h_mgf_bound : ∫ x, Real.exp (t * (⨆ k, Y k x)) ∂μ ≤ M * Real.exp (σ_sq * t^2 / 2) := by
      convert mgf_max_bound hM h_sg t using 1;
    have := Real.log_le_log ( by positivity ) ( h_jensen.trans h_mgf_bound );
    rw [ Real.log_mul ( by positivity ) ( by positivity ), Real.log_exp, Real.log_exp ] at this ; rw [ div_add_div, le_div_iff₀ ] <;> nlinarith;
  by_cases hσ_pos : 0 < σ_sq;
  · by_cases hM_one : M = 1 <;> simp_all +decide [ div_eq_mul_inv ];
    · contrapose! h_max_bound;
      exact ⟨ ( ∫ x, ⨆ k, Y k x ∂μ ) / σ_sq, div_pos h_max_bound hσ_pos, by ring_nf; nlinarith [ mul_inv_cancel₀ hσ_pos.ne' ] ⟩;
    · convert h_max_bound ( Real.sqrt ( 2 * Real.log M / σ_sq ) ) ( by exact Real.sqrt_pos.mpr ( div_pos ( mul_pos zero_lt_two ( Real.log_pos ( mod_cast lt_of_le_of_ne hM ( Ne.symm hM_one ) ) ) ) hσ_pos ) ) using 1 ; ring_nf ; norm_num [ hσ_pos.le, hσ_pos.ne' ] ; ring_nf ; norm_num [ hσ_pos.le, hσ_pos.ne', hM.ne' ] ;
      have hlog_pos : 0 < Real.log M := by
        exact Real.log_pos (mod_cast lt_of_le_of_ne hM (Ne.symm hM_one))
      have hsqrtσ_pos : 0 < Real.sqrt σ_sq := Real.sqrt_pos.mpr hσ_pos
      have hsqrtlog_pos : 0 < Real.sqrt (Real.log M) := Real.sqrt_pos.mpr hlog_pos
      field_simp [hσ_pos.ne', hsqrtσ_pos.ne', hsqrtlog_pos.ne']
      rw [Real.sq_sqrt (by positivity : (0 : ℝ) ≤ 2), Real.sq_sqrt hσ_pos.le, Real.sq_sqrt hlog_pos.le]
      ring
  · norm_num [ show σ_sq = 0 by linarith ] at *;
    contrapose! h_max_bound;
    exact ⟨ ( Real.log M + 1 ) / ( ∫ x, ⨆ k, Y k x ∂μ ), div_pos ( add_pos_of_nonneg_of_pos ( Real.log_nonneg ( Nat.one_le_cast.2 hM ) ) zero_lt_one ) h_max_bound, by rw [ div_div_eq_mul_div, div_lt_iff₀ ] <;> nlinarith [ Real.log_nonneg ( Nat.one_le_cast.2 hM ) ] ⟩

/-
Bound for the Riemann sum of a non-increasing function with a geometric sequence.
-/
lemma geometric_sum_bound
  {f : ℝ → ℝ} {D : ℝ} (hD : 0 < D)
  (hf : AntitoneOn f (Set.Icc 0 D))
  (L : ℕ) :
  ∑ k ∈ Finset.range L, (D * 2 ^ (-(k : ℤ)) - D * 2 ^ (-(k + 1 : ℤ))) * f (D * 2 ^ (-(k + 1 : ℤ))) ≤
  2 * ∫ x in (D * 2 ^ (-(L + 1 : ℤ)))..(D * 2 ^ (-1 : ℤ)), f x := by
  induction' L with L ih <;> norm_num [ Finset.sum_range_succ, zpow_add₀, zpow_sub₀ ] at *;
  
  have h_split : ∫ x in (D * (1 / 2 * (1 / 2 * (2 ^ L)⁻¹)))..D * (1 / 2), f x = (∫ x in (D * (1 / 2 * (1 / 2 * (2 ^ L)⁻¹)))..D * (1 / 2 * (2 ^ L)⁻¹), f x) + (∫ x in (D * (1 / 2 * (2 ^ L)⁻¹))..D * (1 / 2), f x) := by
    rw [ intervalIntegral.integral_add_adjacent_intervals ] <;> apply_rules [ MeasureTheory.IntegrableOn.intervalIntegrable ];
    · rw [ Set.uIcc_of_le ( by nlinarith [ show ( 0 : ℝ ) ≤ D * ( 1 / 2 * ( 2 ^ L ) ⁻¹ ) by positivity ] ) ];
      exact ( hf.mono ( Set.Icc_subset_Icc ( by positivity ) ( by nlinarith [ show ( 0 : ℝ ) ≤ D * ( 1 / 2 * ( 2 ^ L ) ⁻¹ ) by positivity, inv_le_one_of_one_le₀ ( show ( 1 : ℝ ) ≤ 2 ^ L by exact one_le_pow₀ ( by norm_num ) ) ] ) ) ) |> fun h => h.integrableOn_isCompact ( CompactIccSpace.isCompact_Icc );
    · field_simp;
      exact ( hf.integrableOn_isCompact ( CompactIccSpace.isCompact_Icc ) ) |> fun h => h.mono_set ( by rw [ Set.uIcc_of_le ( by gcongr ; norm_cast ; linarith [ Nat.one_le_pow L 2 zero_lt_two ] ) ] ; exact Set.Icc_subset_Icc ( by positivity ) ( by linarith [ show D / 2 ≤ D by linarith ] ) );
  
  have h_integral_bound : ∫ x in (D * (1 / 2 * (1 / 2 * (2 ^ L)⁻¹)))..D * (1 / 2 * (2 ^ L)⁻¹), f x ≥ (D * (1 / 2 * (2 ^ L)⁻¹) - D * (1 / 2 * (1 / 2 * (2 ^ L)⁻¹))) * f (D * (1 / 2 * (2 ^ L)⁻¹)) := by
    have h_integral_bound : ∀ x ∈ Set.Icc (D * (1 / 2 * (1 / 2 * (2 ^ L)⁻¹))) (D * (1 / 2 * (2 ^ L)⁻¹)), f x ≥ f (D * (1 / 2 * (2 ^ L)⁻¹)) := by
      intro x hx; exact hf ⟨ by nlinarith [ hx.1, show ( 0 : ℝ ) ≤ D * ( 1 / 2 * ( 1 / 2 * ( 2 ^ L ) ⁻¹ ) ) by positivity ], by nlinarith [ hx.2, show ( 0 : ℝ ) ≤ D * ( 1 / 2 * ( 2 ^ L ) ⁻¹ ) by positivity, show ( 2 ^ L : ℝ ) ⁻¹ ≤ 1 by exact inv_le_one_of_one_le₀ <| one_le_pow₀ <| by norm_num ] ⟩ ⟨ by nlinarith [ hx.1, show ( 0 : ℝ ) ≤ D * ( 1 / 2 * ( 1 / 2 * ( 2 ^ L ) ⁻¹ ) ) by positivity ], by nlinarith [ hx.2, show ( 0 : ℝ ) ≤ D * ( 1 / 2 * ( 2 ^ L ) ⁻¹ ) by positivity, show ( 2 ^ L : ℝ ) ⁻¹ ≤ 1 by exact inv_le_one_of_one_le₀ <| one_le_pow₀ <| by norm_num ] ⟩ hx.2;
    refine' le_trans _ ( intervalIntegral.integral_mono_on _ _ _ h_integral_bound ) <;> norm_num;
    · exact mul_le_mul_of_nonneg_left ( mul_le_mul_of_nonneg_left ( mul_le_of_le_one_left ( by positivity ) ( by norm_num ) ) ( by positivity ) ) hD.le;
    · apply_rules [ MeasureTheory.IntegrableOn.intervalIntegrable ];
      rw [ Set.uIcc_of_le ( by nlinarith [ show ( 0 : ℝ ) < D * ( 2 ^ L ) ⁻¹ by positivity ] ) ];
      exact ( hf.mono ( Set.Icc_subset_Icc ( by positivity ) ( by nlinarith [ show ( 2 ^ L : ℝ ) ⁻¹ ≤ 1 by exact inv_le_one_of_one_le₀ ( one_le_pow₀ ( by norm_num ) ) ] ) ) ) |> fun h => h.integrableOn_isCompact ( CompactIccSpace.isCompact_Icc );
  linarith [ inv_pos.mpr ( pow_pos ( zero_lt_two' ℝ ) L ), mul_inv_cancel₀ ( ne_of_gt ( pow_pos ( zero_lt_two' ℝ ) L ) ), pow_pos ( zero_lt_two' ℝ ) L ]

/-
Auxiliary real-valued truncated Dudley integral for the internal Nat-valued cardinality.
-/
def dudleyEntropyIntegral (T : Set α) (δ : ℝ) (D : ℝ) : ℝ :=
  ∫ μ in Set.Icc δ D, Real.sqrt (Real.log (subsetENetCard T μ))

/-
Telescoping sum bound for a sequence.
$|X(\gamma_L) - X(\gamma_1)| \le \sum_{k=0}^{L-2} |X(\gamma_{k+2}) - X(\gamma_{k+1})|$.
-/
lemma telescoping_sum_bound
  {α : Type*}
  {L : ℕ} (hL : 1 ≤ L)
  (gamma : ℕ → α)
  (X : α → ℝ) :
  |X (gamma L) - X (gamma 1)| ≤ ∑ k ∈ Finset.range (L - 1), |X (gamma (k + 2)) - X (gamma (k + 1))| := by
    induction' L with L ih;
    · contradiction;
    · rcases L with ( _ | L ) <;> simp_all +decide [ Finset.sum_range_succ ];
      cases abs_cases ( X ( gamma ( L + 2 ) ) - X ( gamma 1 ) ) <;> cases abs_cases ( X ( gamma ( L + 2 ) ) - X ( gamma ( L + 1 ) ) ) <;> cases abs_cases ( X ( gamma ( L + 1 ) ) - X ( gamma 1 ) ) <;> linarith

/-
Bound for a single step in the chain.
$|X(\gamma_{k+2}) - X(\gamma_{k+1})| \le \sup_{u \in U_{k+2}} |X_u - X_{\pi_{k+1}(u)}|$.
-/
lemma single_step_bound
  {α : Type*}
  {U : ℕ → Set α}
  (hU_finite : ∀ k, (U k).Finite)
  {pi_map : (k : ℕ) → U (k + 1) → U k}
  {X : α → ℝ}
  (gamma : ℕ → α)
  (k : ℕ)
  (h_chain_step : ∃ (h_succ : gamma (k + 2) ∈ U (k + 2)), gamma (k + 1) = pi_map (k + 1) ⟨gamma (k + 2), h_succ⟩)
  :
  |X (gamma (k + 2)) - X (gamma (k + 1))| ≤ ⨆ (u : U (k + 2)), |X u - X (pi_map (k + 1) u)| := by
    refine' le_trans _ ( le_ciSup _ ⟨ gamma ( k + 2 ), h_chain_step.choose ⟩ )
    · simp [h_chain_step.choose_spec]
    · exact Set.Finite.bddAbove <|
        Set.Finite.subset
          (Set.Finite.image (fun u : α × α => |X u.1 - X u.2|)
            (hU_finite (k + 2) |> Set.Finite.prod <| hU_finite (k + 1)))
          (by
            rintro x ⟨u, rfl⟩
            exact ⟨(u, pi_map (k + 1) u), ⟨u.2, (pi_map (k + 1) u).2⟩, rfl⟩)

/-
Bound for a single chain.
$|X(\gamma_L) - X(\gamma_1)| \le \sum_{k=0}^{L-2} \sup_{u \in U_{k+2}} |X_u - X_{\pi_{k+1}(u)}|$.
-/
lemma single_chain_bound
  {α : Type*}
  {L : ℕ} (hL : 1 ≤ L)
  {U : ℕ → Set α}
  (hU_finite : ∀ k, (U k).Finite)
  {pi_map : (k : ℕ) → U (k + 1) → U k}
  {X : α → ℝ}
  (gamma : ℕ → α)
  (h_chain : ∀ k, 1 ≤ k → k < L → ∃ (h_succ : gamma (k + 1) ∈ U (k + 1)), gamma k = pi_map k ⟨gamma (k + 1), h_succ⟩)
  :
  |X (gamma L) - X (gamma 1)| ≤ ∑ k ∈ Finset.range (L - 1), (⨆ (u : U (k + 2)), |X u - X (pi_map (k + 1) u)|) := by
    have := @telescoping_sum_bound α L hL gamma X;
    refine' le_trans this ( Finset.sum_le_sum fun k hk => _ );
    simpa using
      (single_step_bound hU_finite gamma k
        (h_chain (k + 1) (by omega) (Nat.lt_pred_iff.mp (Finset.mem_range.mp hk))))

/-
If a finite covering exists, there exists a minimal finite covering with cardinality equal to the cardinality.
-/
lemma exists_minimal_covering_of_finite_covering {α : Type*} [PseudoMetricSpace α] {T : Set α} {δ : ℝ} (h : ∃ u : Set α, u.Finite ∧ IsSubsetENet T u δ) :
  ∃ u : Set α, u.Finite ∧ IsSubsetENet T u δ ∧ u.ncard = subsetENetCard T δ := by
    obtain ⟨ u, hu ⟩ := Nat.sInf_mem
      (show Set.Nonempty { n : ℕ | ∃ u : Finset α, IsSubsetENet T u δ ∧ u.card = n } from by
        rcases h with ⟨u, hu_finite, hu_cover⟩
        rcases hu_finite.exists_finset_coe with ⟨u', rfl⟩
        exact ⟨_, ⟨u', hu_cover, rfl⟩⟩)
    refine ⟨u, u.finite_toSet, hu.1, ?_⟩
    simpa [subsetENetCard] using hu.2

/-
Existence of the Dudley chain sets and maps.
-/
lemma exists_dudley_chain
  {α : Type*} [PseudoMetricSpace α]
  {T : Set α} {U : Set α}
  (hU_subset : U ⊆ T)
  (hU_finite : U.Finite)
  (D : ℝ) (δ : ℝ)
  (L : ℕ)
  (epsilon : ℕ → ℝ)
  (h_L_def : L = Nat.ceil (Real.logb 2 (D / δ)))
  (h_finite_covering : ∀ m < L, ∃ u, u.Finite ∧ IsSubsetENet T u (epsilon m))
  : ∃ (U_seq : ℕ → Set α) (pi_map : ℕ → α → α),
    (∀ m, (U_seq m).Finite) ∧
    (∀ m, U_seq m ⊆ T) ∧
    (U_seq L = U) ∧
    (∀ m < L, (U_seq m).ncard = subsetENetCard T (epsilon m)) ∧
    (∀ m < L, IsSubsetENet T (U_seq m) (epsilon m)) ∧
    (∀ m < L, ∀ u ∈ U_seq (m + 1), pi_map m u ∈ U_seq m ∧ dist u (pi_map m u) ≤ epsilon m) := by
      have h_chain_construction : ∀ m < L, ∃ U_seq : (m : ℕ) → Set α, (∀ m < L, (U_seq m).Finite ∧ IsSubsetENet T (U_seq m) (epsilon m) ∧ (U_seq m).ncard = subsetENetCard T (epsilon m)) ∧ U_seq m ⊆ T ∧ U_seq L = U := by
        intro m hm;
        have hU_seq : ∀ m < L, ∃ U_seq : Set α, U_seq.Finite ∧ IsSubsetENet T U_seq (epsilon m) ∧ U_seq.ncard = subsetENetCard T (epsilon m) := by
          exact fun m a => exists_minimal_covering_of_finite_covering (h_finite_covering m a);
        choose! U_seq hU_seq using hU_seq;
        use fun n => if n = L then U else U_seq n;
        simp +zetaDelta at *;
        exact ⟨ fun n hn => by rw [ if_neg ( ne_of_lt hn ) ] ; exact hU_seq n hn, by rw [ if_neg ( ne_of_lt hm ) ] ; exact hU_seq m hm |>.2.1.1 ⟩;
      cases' L with L L;
      · exact ⟨ fun _ => U, fun _ => id, fun _ => hU_finite, fun _ => hU_subset, rfl, by norm_num, by norm_num, by norm_num ⟩;
      · obtain ⟨ U_seq, hU_seq₁, hU_seq₂, hU_seq ⟩ := h_chain_construction L ( Nat.lt_succ_self L );
        use fun m => if m < L + 1 then U_seq m else U_seq (L + 1);
        have h_pi_map : ∀ m < L + 1, ∀ u ∈ U_seq (m + 1), ∃ v ∈ U_seq m, dist u v ≤ epsilon m := by
          intro m hm u hu
          have h_cover : IsSubsetENet T (U_seq m) (epsilon m) := by
            exact hU_seq₁ m hm |>.2.1;
          have := IsSubsetENet.exists_near h_cover (by
            by_cases hm' : m + 1 < L + 1;
            · exact hU_seq₁ _ hm' |>.2.1.1 hu;
            · have hm_eq : m = L := by omega
              rw [hm_eq, hU_seq] at hu
              exact hU_subset hu );
          exact this;
        choose! pi_map hpi_map using h_pi_map;
        use fun m u => if hm : m < L + 1 then pi_map m u else u;
        refine ⟨?_, ?_, ?_, ?_, ?_, ?_⟩
        · intro m
          by_cases hm : m < L + 1
          · simpa [Nat.lt_succ_iff, hm] using (hU_seq₁ m hm).1
          · have hm_le : ¬m ≤ L := by simpa [Nat.lt_succ_iff] using hm
            simpa [Nat.lt_succ_iff, hm_le, hU_seq] using hU_finite
        · intro m
          by_cases hm : m < L + 1
          · simpa [Nat.lt_succ_iff, hm] using (hU_seq₁ m hm).2.1.1
          · have hm_le : ¬m ≤ L := by simpa [Nat.lt_succ_iff] using hm
            simpa [Nat.lt_succ_iff, hm_le, hU_seq] using hU_subset
        · simp [hU_seq]
        · intro m hm
          simpa [Nat.lt_succ_iff, hm] using (hU_seq₁ m hm).2.2
        · intro m hm
          simpa [Nat.lt_succ_iff, hm] using (hU_seq₁ m hm).2.1
        · intro m hm u hu
          have hu_seq : u ∈ U_seq (m + 1) := by
            by_cases hm' : m + 1 < L + 1
            · simpa [Nat.lt_succ_iff, hm'] using hu
            · have hm_eq : m = L := by omega
              simpa [Nat.lt_succ_iff, hm_eq, hm'] using hu
          simpa [Nat.lt_succ_iff, hm] using hpi_map m hm u hu_seq

/-
Bound for the expectation of the maximum of absolute values of sub-Gaussian random variables.
-/
lemma sub_gaussian_max_abs_bound
  {Ω : Type*} [MeasurableSpace Ω]
  {ι : Type*} [Fintype ι] [Nonempty ι]
  {Y : ι → Ω → ℝ} {σ_sq : ℝ} (hσ : 0 ≤ σ_sq)
  {μ : MeasureTheory.Measure Ω} [MeasureTheory.IsProbabilityMeasure μ]
  (h_sg : ∀ i, IsSubGaussian (Y i) σ_sq μ) :
  ∫ ω, (⨆ i, |Y i ω|) ∂μ ≤ Real.sqrt (2 * σ_sq * Real.log (2 * Fintype.card ι)) := by
    
    set Z : (ι ⊕ ι) → Ω → ℝ := fun i ω => Sum.elim (fun i => Y i ω) (fun i => -Y i ω) i;
    have hZ_subgaussian : ∀ i, IsSubGaussian (Z i) σ_sq μ := by
      intro i
      rcases i with i | i
      · exact h_sg i
      · rcases h_sg i with ⟨hσi, hYi⟩
        exact ⟨hσi, by
          change HasSubgaussianMGF (-Y i) ⟨σ_sq, hσi⟩ μ
          exact hYi.neg⟩
    
    have h_metric_entropy_bound : ∫ ω, ⨆ i, Z i ω ∂μ ≤ Real.sqrt (2 * σ_sq * Real.log (Fintype.card (ι ⊕ ι))) := by
      convert metric_entropy_bound_for_sub_gaussian_maxima _ _ _;
      any_goals assumption;
      any_goals exact fun i ω => Z ( Fintype.equivFin ( ι ⊕ ι ) |>.symm i ) ω;
      · rw [ ← Equiv.iSup_comp ( Fintype.equivFin ( ι ⊕ ι ) |> Equiv.symm ) ];
      · exact Fintype.card_pos_iff.mpr ⟨ Sum.inl ( Classical.arbitrary ι ) ⟩;
      · exact fun k => hZ_subgaussian _;
    convert h_metric_entropy_bound using 3 <;> norm_num [ two_mul ];
    rw [ @ciSup_eq_of_forall_le_of_forall_lt_exists_gt ];
    · intro i; exact (by
      exact max_le ( le_ciSup ( Finite.bddAbove_range fun i => Z i _ ) ( Sum.inl i ) ) ( le_ciSup ( Finite.bddAbove_range fun i => Z i _ ) ( Sum.inr i ) ));
    · intro w hw;
      contrapose! hw;
      convert ciSup_le _;
      · exact Sum.nonemptyRight;
      · rintro ( i | i ) <;> [ exact le_trans ( le_abs_self _ ) ( hw i ) ; exact le_trans ( neg_le_abs _ ) ( hw i ) ]

/-
Expectation bound for a single step in the chain.
-/
lemma expectation_bound_chain_step
  {α : Type*} [PseudoMetricSpace α]
  {Ω : Type*} [MeasurableSpace Ω] {μ : MeasureTheory.Measure Ω} [MeasureTheory.IsProbabilityMeasure μ]
  {T : Set α}
  {X : Ω → α → ℝ}
  (h_sg : ∀ x ∈ T, ∀ y ∈ T, IsSubGaussian (fun ω => X ω x - X ω y) (dist x y ^ 2) μ)
  (U_curr U_next : Set α)
  (hU_curr_subset : U_curr ⊆ T)
  (hU_next_subset : U_next ⊆ T)
  (hU_next_finite : U_next.Finite)
  (pi_map : α → α)
  (h_pi : ∀ u ∈ U_next, pi_map u ∈ U_curr)
  (eps : ℝ) (h_eps : 0 ≤ eps)
  (h_dist : ∀ u ∈ U_next, dist u (pi_map u) ≤ eps)
  :
  ∫ ω, (⨆ u ∈ U_next, |X ω u - X ω (pi_map u)|) ∂μ ≤ eps * Real.sqrt (2 * Real.log (U_next.ncard)) + eps * Real.sqrt (2 * Real.log 2) := by
    cases' hU_next_finite.exists_finset_coe with F hF;
    by_cases hF_empty : F.Nonempty <;> simp_all +decide;
    · 
      set Y : F → Ω → ℝ := fun u ω => X ω u - X ω (pi_map u);
      
      have h_Y_bound : ∫ ω, (⨆ u : F, |Y u ω|) ∂μ ≤ Real.sqrt (2 * eps^2 * Real.log (2 * F.card)) := by
        have h_Y_bound : ∀ u : F, IsSubGaussian (Y u) (eps^2) μ := by
          intro u
          specialize h_sg u.val (hU_next_subset (hF.subset u.prop)) (pi_map u.val) (hU_curr_subset (h_pi u.val (hF.subset u.prop)));
          refine' ⟨ _, _ ⟩;
          exact sq_nonneg _;
          refine' ⟨ _, fun t => _ ⟩;
          · exact fun t => sub_gaussian_integrable h_sg t;
          · refine' le_trans ( h_sg.choose_spec.2 t ) _;
            exact Real.exp_le_exp.mpr ( by
              gcongr
              exact (sq_le_sq₀ dist_nonneg h_eps).2 (h_dist _ (hF.subset u.2)) );
        convert sub_gaussian_max_abs_bound _ _;
        · exact Eq.symm (Fintype.card_coe F)
        · exact Finset.Nonempty.to_subtype hF_empty
        · exact sq_nonneg eps
        · infer_instance
        · exact fun i => h_Y_bound i
      calc
        ∫ ω, (⨆ u ∈ U_next, |X ω u - X ω (pi_map u)|) ∂μ
            = ∫ ω, (⨆ u : F, |Y u ω|) ∂μ := by
                rw [← hF]
                congr 1 with ω
                have hbdd : BddAbove
                    (Set.range fun u : (↑F : Set α) => |X ω u - X ω (pi_map u)|) := by
                  exact Finite.bddAbove_range _
                have hbot : sSup (∅ : Set ℝ) ≤
                    (⨆ u : (↑F : Set α), |X ω u - X ω (pi_map u)|) := by
                  refine le_trans ?_ (le_ciSup hbdd ⟨hF_empty.choose, hF_empty.choose_spec⟩)
                  simp
                rw [← ciSup_subtype_fun hbdd hbot]
                rfl
        _ ≤ Real.sqrt (2 * eps^2 * Real.log (2 * F.card)) := h_Y_bound
        _ ≤ eps * (Real.sqrt 2 * Real.sqrt (Real.log (U_next.ncard))) +
            eps * (Real.sqrt 2 * Real.sqrt (Real.log 2)) := by
              rw [← hF, Set.ncard_coe_finset]
              have hF_card_pos : 0 < F.card := Finset.card_pos.mpr hF_empty
              have hF_card_ne : (F.card : ℝ) ≠ 0 := by positivity
              have hlogF : 0 ≤ Real.log (F.card : ℝ) := by
                have hone_le_card : (1 : ℝ) ≤ F.card := by
                  exact_mod_cast Nat.succ_le_of_lt hF_card_pos
                exact Real.log_nonneg hone_le_card
              have hlog2 : 0 ≤ Real.log (2 : ℝ) := by positivity
              rw [Real.log_mul (by norm_num : (2 : ℝ) ≠ 0) hF_card_ne]
              apply Real.sqrt_le_iff.mpr
              constructor
              · positivity
              · have hsF := Real.sq_sqrt hlogF
                have hs2 := Real.sq_sqrt hlog2
                have hsqrt2 := Real.sq_sqrt (show 0 ≤ (2 : ℝ) by norm_num)
                have hsq :
                    (eps * (Real.sqrt 2 * Real.sqrt (Real.log (F.card : ℝ))) +
                      eps * (Real.sqrt 2 * Real.sqrt (Real.log 2))) ^ 2 =
                      2 * eps ^ 2 * (Real.log (F.card : ℝ) + Real.log 2) +
                        4 * eps ^ 2 *
                    (Real.sqrt (Real.log (F.card : ℝ)) * Real.sqrt (Real.log 2)) := by
                  ring_nf
                  rw [hsF, hs2, hsqrt2]
                  ring
                rw [hsq]
                have hcross : 0 ≤ 4 * eps ^ 2 *
                    (Real.sqrt (Real.log (F.card : ℝ)) * Real.sqrt (Real.log 2)) := by
                  positivity
                nlinarith
    · simp +decide [ ← hF ];
      positivity

/-
Defines the projection of a point u onto the covering sets U_k via the maps pi_map.
`chain_projection L pi_map u k` is the point in U_k that leads to u in U_L.
-/
def chain_projection {α : Type*} (L : ℕ) (pi_map : ℕ → α → α) (u : α) (k : ℕ) : α :=
  if h : L ≤ k then u
  else
    pi_map k (chain_projection L pi_map u (k + 1))
termination_by L - k
decreasing_by
  omega

/-
The projection of u at step L is u itself.
-/
lemma chain_projection_eq_self {α : Type*} (L : ℕ) (pi_map : ℕ → α → α) (u : α) :
  chain_projection L pi_map u L = u := by
    simp [chain_projection]

/-
The projection at step k is the result of applying pi_map to the projection at step k+1.
-/
lemma chain_projection_step {α : Type*} (L : ℕ) (pi_map : ℕ → α → α) (u : α) (k : ℕ) (hk : k < L) :
  chain_projection L pi_map u k = pi_map k (chain_projection L pi_map u (k + 1)) := by
    rw [chain_projection]
    split_ifs with h
    · exact False.elim (Nat.not_le_of_lt hk h)
    · rfl

/-
The projection of u at step k lies in the set U_k.
-/
lemma chain_projection_mem {α : Type*} {U : ℕ → Set α} (L : ℕ) (pi_map : ℕ → α → α) (u : α)
  (hu : u ∈ U L)
  (h_map : ∀ k < L, ∀ x ∈ U (k + 1), pi_map k x ∈ U k)
  (k : ℕ) (hk : k ≤ L) :
  chain_projection L pi_map u k ∈ U k := by
    
    induction' h : L - k with m ih generalizing k u;
    · have hk_eq : k = L := by omega
      subst hk_eq
      simpa [chain_projection] using hu
    · rw [ Nat.sub_eq_iff_eq_add ] at h <;> try linarith;
      convert h_map k ( by linarith ) _ ( ih u hu ( k + 1 ) ( by linarith ) ( by omega ) ) using 1;
      exact chain_projection_step L pi_map u k ( by linarith )

/-
Telescoping sum bound for the chain projection.
The difference between the value at the start and end of the chain is bounded by the sum of differences along the chain.
-/
lemma chain_telescoping_bound
  {α : Type*}
  {X : α → ℝ}
  {L : ℕ} (hL : 1 ≤ L)
  {pi_map : ℕ → α → α}
  (u : α)
  :
  |X (chain_projection L pi_map u L) - X (chain_projection L pi_map u 1)| ≤
  ∑ k ∈ Finset.range (L - 1), |X (chain_projection L pi_map u (k + 2)) - X (chain_projection L pi_map u (k + 1))| := by
    simpa using
      (telescoping_sum_bound (L := L) hL (gamma := fun k => chain_projection L pi_map u k) (X := X))

/-
Bounds a single step of the chain projection difference by the supremum over the covering set.
Requires U to be finite.
-/
lemma chain_projection_step_bound
  {α : Type*}
  {X : α → ℝ}
  {L : ℕ}
  {U : ℕ → Set α}
  (hU_finite : ∀ k, (U k).Finite)
  {pi_map : ℕ → α → α}
  (u : α) (hu : u ∈ U L)
  (h_chain : ∀ k < L, ∀ x ∈ U (k + 1), pi_map k x ∈ U k)
  (k : ℕ) (hk : k < L - 1)
  :
  |X (chain_projection L pi_map u (k + 2)) - X (chain_projection L pi_map u (k + 1))| ≤
  (⨆ z ∈ U (k + 2), |X z - X (pi_map (k + 1) z)|) := by
    rw [ show chain_projection L pi_map u ( k+2 ) = chain_projection L pi_map u ( k+2 ) from rfl, show chain_projection L pi_map u ( k+1 ) = pi_map ( k+1 ) ( chain_projection L pi_map u ( k+2 ) ) from ?_ ];
    · have h_chain_finite : chain_projection L pi_map u (k + 2) ∈ U (k + 2) := by
        apply chain_projection_mem;
        · assumption;
        · exact h_chain;
        · omega;
      refine' le_trans _ ( le_ciSup _ ( chain_projection L pi_map u ( k + 2 ) ) );
      · exact le_ciSup ⟨|X (chain_projection L pi_map u (k + 2)) - X (pi_map (k + 1) (chain_projection L pi_map u (k + 2)))|,
          Set.forall_mem_range.2 fun _ => le_rfl⟩ h_chain_finite;
      · obtain ⟨ M, hM ⟩ := Set.Finite.bddAbove ( Set.Finite.image ( fun z => |X z - X ( pi_map ( k + 1 ) z ) - 0| ) ( hU_finite ( k + 2 ) ) );
        refine' ⟨ Max.max M 1, Set.forall_mem_range.2 fun z => _ ⟩;
        rw [ @ciSup_eq_ite ]
        split_ifs with hz
        · exact le_trans (by simpa using hM ⟨z, hz, rfl⟩) (le_max_left _ _)
        · exact le_trans (by norm_num) (le_max_right _ _)
    · exact chain_projection_step L pi_map u ( k + 1 ) ( by omega )

/-
Bounds the difference between X(u) and X(projection of u at step 1) by the sum of suprema of differences at each step.
Requires U to be finite.
-/
lemma chain_projection_bound
  {α : Type*}
  {X : α → ℝ}
  {L : ℕ} (hL : 1 ≤ L)
  {U : ℕ → Set α}
  (hU_finite : ∀ k, (U k).Finite)
  {pi_map : ℕ → α → α}
  (u : α) (hu : u ∈ U L)
  (h_chain : ∀ k < L, ∀ x ∈ U (k + 1), pi_map k x ∈ U k)
  :
  |X u - X (chain_projection L pi_map u 1)| ≤
  ∑ k ∈ Finset.range (L - 1), (⨆ z ∈ U (k + 2), |X z - X (pi_map (k + 1) z)|) := by
    calc
      |X u - X (chain_projection L pi_map u 1)|
          = |X (chain_projection L pi_map u L) - X (chain_projection L pi_map u 1)| := by
              rw [chain_projection_eq_self]
      _ ≤ ∑ k ∈ Finset.range (L - 1),
          |X (chain_projection L pi_map u (k + 2)) -
            X (chain_projection L pi_map u (k + 1))| :=
              chain_telescoping_bound hL u
      _ ≤ ∑ k ∈ Finset.range (L - 1),
          (⨆ z ∈ U (k + 2), |X z - X (pi_map (k + 1) z)|) := by
              exact Finset.sum_le_sum fun k hk =>
                chain_projection_step_bound hU_finite u hu h_chain k (Finset.mem_range.mp hk)

/-
Bounds the supremum of the difference between X(u) and X(projection of u at step 1) over all u in U_L.
-/
lemma chain_sup_projection_bound
  {α : Type*}
  {X : α → ℝ}
  {L : ℕ} (hL : 1 ≤ L)
  {U : ℕ → Set α}
  (hU_finite : ∀ k, (U k).Finite)
  {pi_map : ℕ → α → α}
  (h_chain : ∀ k < L, ∀ x ∈ U (k + 1), pi_map k x ∈ U k)
  :
  (⨆ u ∈ U L, |X u - X (chain_projection L pi_map u 1)|) ≤
  ∑ k ∈ Finset.range (L - 1), (⨆ z ∈ U (k + 2), |X z - X (pi_map (k + 1) z)|) := by
    by_contra h_contra;
    obtain ⟨u, hu⟩ : ∃ u ∈ U L, |X u - X (chain_projection L pi_map u 1)| > ∑ k ∈ Finset.range (L - 1), (⨆ z ∈ U (k + 2), |X z - X (pi_map (k + 1) z)|) := by
      push Not at h_contra;
      have h_nonempty : (U L).Nonempty := by
        by_contra h_empty
        rw [Set.not_nonempty_iff_eq_empty.mp h_empty] at h_contra
        simp at h_contra
        exact not_lt_of_ge (Finset.sum_nonneg fun _ _ => Real.iSup_nonneg fun _ => Real.iSup_nonneg fun _ => abs_nonneg _) h_contra
      haveI : Nonempty α := ⟨h_nonempty.some⟩
      haveI : Nonempty (U L) := Set.Nonempty.to_subtype h_nonempty
      let f : U L → ℝ := fun u => |X u - X (chain_projection L pi_map u 1)|
      have h_bdd : BddAbove (Set.range f) := by
        letI : Fintype (U L) := Set.Finite.fintype (hU_finite L)
        exact Finite.bddAbove_range f
      have h_empty_le : sSup (∅ : Set ℝ) ≤ ⨆ u : U L, f u := by
        rw [Real.sSup_empty]
        have h_nonneg : 0 ≤ f ⟨h_nonempty.some, h_nonempty.choose_spec⟩ := by
          simp [f]
        exact le_trans h_nonneg (le_ciSup h_bdd ⟨h_nonempty.some, h_nonempty.choose_spec⟩)
      have hciSup : (⨆ u : U L, f u) = ⨆ u ∈ U L, |X u - X (chain_projection L pi_map u 1)| := by
        simpa [f] using (ciSup_subtype_fun (s := U L) (f := fun u => |X u - X (chain_projection L pi_map u 1)|) h_bdd h_empty_le)
      have h_contra' : ∑ k ∈ Finset.range (L - 1), ⨆ z ∈ U (k + 2), |X z - X (pi_map (k + 1) z)| < ⨆ u : U L, f u := by
        rw [hciSup]
        exact h_contra
      obtain ⟨u, hu⟩ := exists_lt_of_lt_ciSup h_contra'
      exact ⟨u, u.2, hu⟩
    exact hu.2.not_ge ( chain_projection_bound hL hU_finite u hu.1 h_chain )

/-
Bound the difference of a function on a finite set by the supremum of differences.
-/
lemma finite_sup_bound
  {α : Type*}
  {A : Set α} (hA : A.Finite)
  (X : α → ℝ)
  (u v : α) (hu : u ∈ A) (hv : v ∈ A) :
  |X u - X v| ≤ ⨆ x ∈ A, ⨆ y ∈ A, |X x - X y| := by
    have h_finset : ∀ x ∈ A, ∀ y ∈ A, |X x - X y| ≤ ⨆ x ∈ A, ⨆ y ∈ A, |X x - X y| := by
      refine' fun x hx y hy => le_trans _ ( le_ciSup _ x );
      · exact
          le_trans
            (le_trans
              (le_ciSup ⟨|X x - X y|, Set.forall_mem_range.2 fun _ => le_rfl⟩ hy)
              (le_ciSup
                (show BddAbove ( Set.range fun y => ⨆ ( hy : y ∈ A ), |X x - X y| ) from
                  ⟨ ∑ i ∈ hA.toFinset, |X x - X i|,
                    Set.forall_mem_range.2 fun y => by
                      rw [ciSup_eq_ite]
                      split_ifs with hy'
                      · exact Finset.single_le_sum (fun i _ => abs_nonneg (X x - X i)) (hA.mem_toFinset.mpr hy')
                      · exact le_trans (by norm_num) (Finset.sum_nonneg fun i _ => abs_nonneg (X x - X i)) ⟩)
                y))
            (le_ciSup ⟨⨆ y ∈ A, |X x - X y|, Set.forall_mem_range.2 fun _ => le_rfl⟩ hx)
      · 
        have h_finite_range : Set.Finite (Set.range (fun x : α => ⨆ (hx : x ∈ A), ⨆ y ∈ A, |X x - X y|)) := by
          
          have h_finite_range : Set.Finite (Set.image (fun x => ⨆ (hx : x ∈ A), ⨆ y ∈ A, |X x - X y|) A) := by
            exact hA.image _;
          exact Set.Finite.subset ( h_finite_range.union ( Set.finite_singleton 0 ) ) ( Set.range_subset_iff.2 fun x => if hx : x ∈ A then Or.inl ⟨ x, hx, rfl ⟩ else Or.inr <| by simp [hx] );
        exact h_finite_range.bddAbove;
    exact h_finset u hu v hv

/-
Pointwise bound for the difference of a process along a Dudley chain.
-/
lemma dudley_chain_bound_pointwise
  {α : Type*}
  {X : α → ℝ}
  {L : ℕ} (hL : 1 ≤ L)
  {U_seq : ℕ → Set α}
  {pi_map : ℕ → α → α}
  (hU_finite : ∀ m, (U_seq m).Finite)
  (h_chain : ∀ m < L, ∀ u ∈ U_seq (m + 1), pi_map m u ∈ U_seq m)
  (u : α) (hu : u ∈ U_seq L)
  (v : α) (hv : v ∈ U_seq L)
  :
  |X u - X v| ≤
  (⨆ x ∈ U_seq 1, ⨆ y ∈ U_seq 1, |X x - X y|) +
  2 * ∑ k ∈ Finset.range (L - 1), (⨆ z ∈ U_seq (k + 2), |X z - X (pi_map (k + 1) z)|) := by
    
    set u1 := chain_projection L pi_map u 1
    set v1 := chain_projection L pi_map v 1
    have h_triangle : |X u - X v| ≤ |X u - X u1| + |X u1 - X v1| + |X v1 - X v| := by
      cases abs_cases ( X u - X v ) <;> cases abs_cases ( X u - X u1 ) <;> cases abs_cases ( X u1 - X v1 ) <;> cases abs_cases ( X v1 - X v ) <;> linarith;
    have h_chain_bound : ∀ (w : α) (hw : w ∈ U_seq L), |X w - X (chain_projection L pi_map w 1)| ≤ ∑ k ∈ Finset.range (L - 1), (⨆ z ∈ U_seq (k + 2), |X z - X (pi_map (k + 1) z)|) := by
      exact fun w hw => chain_projection_bound hL hU_finite w hw h_chain;
    
    have h_sup_bound : |X u1 - X v1| ≤ ⨆ x ∈ U_seq 1, ⨆ y ∈ U_seq 1, |X x - X y| := by
      apply_rules [ finite_sup_bound ];
      · apply_rules [ chain_projection_mem ];
      · exact chain_projection_mem L pi_map v hv h_chain 1 ( by linarith );
    linarith [ h_chain_bound u hu, h_chain_bound v hv, abs_sub_comm ( X v1 ) ( X v ) ]

/-
Pointwise bound for the difference of a process along a Dudley chain.
-/
lemma dudley_chain_bound_pointwise_v2
  {α : Type*}
  {X : α → ℝ}
  {L : ℕ} (hL : 1 ≤ L)
  {U_seq : ℕ → Set α}
  {pi_map : ℕ → α → α}
  (hU_finite : ∀ m, (U_seq m).Finite)
  (h_chain : ∀ m < L, ∀ u ∈ U_seq (m + 1), pi_map m u ∈ U_seq m)
  (u : α) (hu : u ∈ U_seq L)
  (v : α) (hv : v ∈ U_seq L)
  :
  |X u - X v| ≤
  (⨆ x ∈ U_seq 1, ⨆ y ∈ U_seq 1, |X x - X y|) +
  2 * ∑ k ∈ Finset.range (L - 1), (⨆ z ∈ U_seq (k + 2), |X z - X (pi_map (k + 1) z)|) := by
    convert dudley_chain_bound_pointwise hL hU_finite h_chain u hu v hv using 1

/-
Linearity of expectation applied to the Dudley chain bound.
-/
lemma integral_dudley_chain_bound
  {α : Type*} [PseudoMetricSpace α]
  {Ω : Type*} [MeasurableSpace Ω] {P : Measure Ω} [IsProbabilityMeasure P]
  {X : Ω → α → ℝ}
  {L : ℕ} (hL : 1 ≤ L)
  {U_seq : ℕ → Set α}
  {pi_map : ℕ → α → α}
  (hU_finite : ∀ m, (U_seq m).Finite)
  (h_chain : ∀ m < L, ∀ u ∈ U_seq (m + 1), pi_map m u ∈ U_seq m)
  (h_integrable_1 : Integrable (fun ω => ⨆ x ∈ U_seq 1, ⨆ y ∈ U_seq 1, |X ω x - X ω y|) P)
  (h_integrable_2 : ∀ k < L - 1, Integrable (fun ω => ⨆ z ∈ U_seq (k + 2), |X ω z - X ω (pi_map (k + 1) z)|) P)
  :
  ∫ ω, (⨆ u ∈ U_seq L, ⨆ v ∈ U_seq L, |X ω u - X ω v|) ∂P ≤
  ∫ ω, (⨆ x ∈ U_seq 1, ⨆ y ∈ U_seq 1, |X ω x - X ω y|) ∂P +
  2 * ∑ k ∈ Finset.range (L - 1), ∫ ω, (⨆ z ∈ U_seq (k + 2), |X ω z - X ω (pi_map (k + 1) z)|) ∂P := by
    rw [ ← MeasureTheory.integral_finsetSum ];
    · rw [ ← MeasureTheory.integral_const_mul ];
      rw [ ← MeasureTheory.integral_add ];
      · by_contra h_contra;
        refine' h_contra ( MeasureTheory.integral_mono _ _ fun ω => _ );
        · exact ( by by_contra h; rw [ MeasureTheory.integral_undef h ] at h_contra; exact h_contra <| by exact le_trans ( by norm_num ) <| MeasureTheory.integral_nonneg fun _ => add_nonneg ( by exact Real.iSup_nonneg fun _ => Real.iSup_nonneg fun _ => Real.iSup_nonneg fun _ => Real.iSup_nonneg fun _ => abs_nonneg _ ) <| mul_nonneg zero_le_two <| Finset.sum_nonneg fun _ _ => by exact Real.iSup_nonneg fun _ => Real.iSup_nonneg fun _ => abs_nonneg _ );
        · exact MeasureTheory.Integrable.add h_integrable_1 ( MeasureTheory.Integrable.const_mul ( MeasureTheory.integrable_finsetSum _ fun i hi => h_integrable_2 i ( Finset.mem_range.mp hi ) ) _ );
        · 
          have h_pointwise : ∀ u ∈ U_seq L, ∀ v ∈ U_seq L, |X ω u - X ω v| ≤ (⨆ x ∈ U_seq 1, ⨆ y ∈ U_seq 1, |X ω x - X ω y|) + 2 * ∑ k ∈ Finset.range (L - 1), (⨆ z ∈ U_seq (k + 2), |X ω z - X ω (pi_map (k + 1) z)|) := by
            exact fun u a v a_1 => dudley_chain_bound_pointwise hL hU_finite h_chain u a v a_1;
          by_cases hL_empty : U_seq L = ∅;
          · simp [hL_empty];
            exact add_nonneg ( Real.iSup_nonneg fun _ => Real.iSup_nonneg fun _ => Real.iSup_nonneg fun _ => Real.iSup_nonneg fun _ => abs_nonneg _ ) ( mul_nonneg zero_le_two ( Finset.sum_nonneg fun _ _ => Real.iSup_nonneg fun _ => Real.iSup_nonneg fun _ => abs_nonneg _ ) );
          · obtain ⟨u, hu⟩ : ∃ u, u ∈ U_seq L := by
              exact Set.nonempty_iff_ne_empty.2 hL_empty;
            convert ciSup_le fun v => ?_;
            · exact Nonempty.intro (pi_map L (pi_map L u));
            · rw [ @ciSup_eq_ite ];
              split_ifs with hv;
              · convert ciSup_le fun w => ?_;
                · exact Nonempty.intro (pi_map L (pi_map L u));
                · rw [ @ciSup_eq_ite ];
                  split_ifs with hw <;> [ exact h_pointwise _ hv _ hw; exact le_trans ( by norm_num ) ( add_nonneg ( Real.iSup_nonneg fun _ => Real.iSup_nonneg fun _ => Real.iSup_nonneg fun _ => Real.iSup_nonneg fun _ => abs_nonneg _ ) ( mul_nonneg zero_le_two ( Finset.sum_nonneg fun _ _ => Real.iSup_nonneg fun _ => Real.iSup_nonneg fun _ => abs_nonneg _ ) ) ) ];
              · simp +decide [ Real.sSup_empty ];
                exact add_nonneg ( Real.iSup_nonneg fun _ => Real.iSup_nonneg fun _ => Real.iSup_nonneg fun _ => Real.iSup_nonneg fun _ => abs_nonneg _ ) ( mul_nonneg zero_le_two ( Finset.sum_nonneg fun _ _ => Real.iSup_nonneg fun _ => Real.iSup_nonneg fun _ => abs_nonneg _ ) );
      · exact h_integrable_1;
      · exact MeasureTheory.Integrable.const_mul ( MeasureTheory.integrable_finsetSum _ fun i hi => h_integrable_2 i ( Finset.mem_range.mp hi ) ) _;
    · exact fun k hk => h_integrable_2 k ( Finset.mem_range.mp hk )

/-
Analytical bound for the Dudley sum.
The sum of $2 \epsilon_k f(\epsilon_{k+1})$ is bounded by $8 \int f$.
-/
set_option maxHeartbeats 0 in
lemma dudley_analytical_sum_bound
  {f : ℝ → ℝ}
  {D : ℝ} (hD : 0 < D)
  (hf_antitone : AntitoneOn f (Set.Icc 0 D))
  (L : ℕ) :
  ∑ k ∈ Finset.range L, 2 * D * 2 ^ (-(k : ℝ)) * f (D * 2 ^ (-(k + 1 : ℝ))) ≤
  8 * ∫ x in Set.Icc (D * 2 ^ (-(L + 1 : ℝ))) (D * 2 ^ (-1 : ℝ)), f x := by
    induction' L with L ih;
    · norm_num;
    · 
      have h_split : ∫ x in (Icc (D * 2 ^ (-(L + 2) : ℝ)) (D * 2 ^ (-1 : ℝ))), f x = (∫ x in (Icc (D * 2 ^ (-(L + 2) : ℝ)) (D * 2 ^ (-(L + 1) : ℝ))), f x) + (∫ x in (Icc (D * 2 ^ (-(L + 1) : ℝ)) (D * 2 ^ (-1 : ℝ))), f x) := by
        rw [ MeasureTheory.integral_Icc_eq_integral_Ioc, MeasureTheory.integral_Icc_eq_integral_Ioc, MeasureTheory.integral_Icc_eq_integral_Ioc, ← MeasureTheory.setIntegral_union ] <;> norm_num;
        · rw [ Set.Ioc_union_Ioc_eq_Ioc ] <;> norm_num [ Real.rpow_add, Real.rpow_neg ];
          · exact mul_le_mul_of_nonneg_left ( mul_le_mul_of_nonneg_right ( by norm_num ) ( by positivity ) ) hD.le;
          · exact mul_le_mul_of_nonneg_left ( mul_le_of_le_one_right ( by norm_num ) ( inv_le_one_of_one_le₀ ( one_le_pow₀ ( by norm_num ) ) ) ) hD.le;
        · refine' MeasureTheory.IntegrableOn.mono_set _ _;
          exact Set.Icc 0 D;
          · exact ( hf_antitone.integrableOn_isCompact ( CompactIccSpace.isCompact_Icc ) );
          · exact fun x hx => ⟨ le_trans ( by positivity ) hx.1.le, le_trans hx.2 ( by exact mul_le_of_le_one_right hD.le ( by rw [ Real.rpow_le_one_iff_of_pos ] <;> norm_num ; linarith ) ) ⟩;
        · refine' MeasureTheory.IntegrableOn.mono_set _ ( Set.Ioc_subset_Icc_self );
          refine' MeasureTheory.IntegrableOn.mono_set _ _;
          exact Set.Icc 0 D;
          · exact ( hf_antitone.integrableOn_isCompact ( CompactIccSpace.isCompact_Icc ) );
          · exact Set.Icc_subset_Icc ( by positivity ) ( by nlinarith [ Real.rpow_pos_of_pos zero_lt_two ( -1 + -L : ℝ ) ] );
      
      have h_integral_bound : ∫ x in (Icc (D * 2 ^ (-(L + 2) : ℝ)) (D * 2 ^ (-(L + 1) : ℝ))), f x ≥ (D * 2 ^ (-(L + 1) : ℝ) - D * 2 ^ (-(L + 2) : ℝ)) * f (D * 2 ^ (-(L + 1) : ℝ)) := by
        have h_integral_bound : ∫ x in (Icc (D * 2 ^ (-(L + 2) : ℝ)) (D * 2 ^ (-(L + 1) : ℝ))), f x ≥ ∫ x in (Icc (D * 2 ^ (-(L + 2) : ℝ)) (D * 2 ^ (-(L + 1) : ℝ))), f (D * 2 ^ (-(L + 1) : ℝ)) := by
          refine' MeasureTheory.setIntegral_mono_on _ _ _ _ <;> norm_num;
          · refine' ( hf_antitone.integrableOn_isCompact _ ) |> fun h => h.mono_set _;
            · exact CompactIccSpace.isCompact_Icc;
            · exact Set.Icc_subset_Icc ( by positivity ) ( by exact mul_le_of_le_one_right hD.le ( by rw [ Real.rpow_le_one_iff_of_pos ] <;> norm_num ; linarith ) );
          · intro x hx₁ hx₂; refine' hf_antitone _ _ _ <;> norm_num [ Real.rpow_add, Real.rpow_neg ] at * <;> try nlinarith [ Real.rpow_pos_of_pos zero_lt_two ( -1 + -L : ℝ ), Real.rpow_pos_of_pos zero_lt_two ( -2 + -L : ℝ ) ] ;
            · constructor <;> nlinarith [ inv_pos.mpr ( pow_pos ( zero_lt_two' ℝ ) L ), mul_inv_cancel₀ ( ne_of_gt ( pow_pos ( zero_lt_two' ℝ ) L ) ), pow_le_pow_right₀ ( by norm_num : ( 1 : ℝ ) ≤ 2 ) ( show L ≥ 0 by norm_num ) ];
            · exact ⟨ hD.le, mul_le_of_le_one_right hD.le ( by linarith [ inv_le_one_of_one_le₀ ( one_le_pow₀ ( by norm_num : ( 1 : ℝ ) ≤ 2 ) : ( 1 : ℝ ) ≤ 2 ^ L ) ] ) ⟩;
        convert h_integral_bound using 1 ; norm_num [ Real.rpow_add, Real.rpow_neg ] ; ring_nf;
        exact Or.inl ( mul_le_mul_of_nonneg_left ( by norm_num ) ( by positivity ) );
      norm_num [ Finset.sum_range_succ, Real.rpow_add, Real.rpow_neg ] at * ; ring_nf at * ; linarith!;

/-
If `u` is a centered `δ`-net, it is also a centered `δ'`-net for `δ ≤ δ'`.
-/
lemma IsSubsetENet_mono {α : Type*} [PseudoMetricSpace α] {T : Set α} {u : Set α} {δ δ' : ℝ} (h : δ ≤ δ') (hu : IsSubsetENet T u δ) :
  IsSubsetENet T u δ' := by
    obtain ⟨hfin, hnet⟩ := hu.2
    exact ⟨hu.1, hfin, hnet.mono_eps h⟩

/-
If a finite centered net exists for the smaller radius, `subsetENetCard` is non-increasing.
-/
lemma subsetENetCard_le_of_le {α : Type*} [PseudoMetricSpace α] {T : Set α} {δ δ' : ℝ} (h : δ ≤ δ') (h_ex : ∃ u : Set α, IsSubsetENet T u δ ∧ u.Finite) :
  subsetENetCard T δ' ≤ subsetENetCard T δ := by
    
    set S := fun r => {n : ℕ | ∃ u : Finset α, IsSubsetENet T u r ∧ u.card = n} with hS_def
    have hS_nonempty : S δ ≠ ∅ := by
      rcases h_ex with ⟨ u, hu₁, hu₂ ⟩ ; exact Set.Nonempty.ne_empty ⟨ _, ⟨ hu₂.toFinset, by simpa using hu₁, rfl ⟩ ⟩ ;
    refine' le_csInf _ _;
    · exact Set.nonempty_iff_ne_empty.2 hS_nonempty;
    · rintro n ⟨ u, hu, rfl ⟩;
      exact Nat.sInf_le ⟨ u, IsSubsetENet_mono h hu, rfl ⟩

/-
The centered-net cardinality is antitone on `(0, D]`.
-/
lemma subsetENetCard_antitoneOn_Ioc {α : Type*} [PseudoMetricSpace α] {T : Set α} {D : ℝ}
  (h_finite_cov : ∀ ε > 0, ∃ u : Set α, u.Finite ∧ IsSubsetENet T u ε) :
  AntitoneOn (fun δ => subsetENetCard T δ) (Set.Ioc 0 D) := by
    intro δ hδ δ' hδ' hδδ'';
    obtain ⟨ u, hu, hu' ⟩ := h_finite_cov δ hδ.1;
    convert subsetENetCard_le_of_le hδδ'' ⟨ u, hu', hu ⟩ using 1

/-
Consistent Dudley chain: wrapper around `exists_dudley_chain` that ensures singleton-level
centered nets all use the same center, making `pi_map m u = u` when
`subsetENetCard T (epsilon (m+1)) = 1`.
-/
lemma exists_dudley_chain_consistent
  {α : Type*} [PseudoMetricSpace α]
  {T : Set α} {U : Set α}
  (hU_subset : U ⊆ T)
  (hU_finite : U.Finite)
  (D : ℝ) (hD : 0 < D) (δ : ℝ)
  (L : ℕ)
  (epsilon : ℕ → ℝ)
  (h_eps : ∀ m, epsilon m = D * 2 ^ (-(m : ℝ)))
  (h_L_def : L = Nat.ceil (Real.logb 2 (D / δ)))
  (h_finite_covering : ∀ m < L, ∃ u, u.Finite ∧ IsSubsetENet T u (epsilon m))
  (h_T_nonempty : T.Nonempty)
  (h_finite_cov : ∀ ε > 0, ∃ u : Set α, u.Finite ∧ IsSubsetENet T u ε)
  : ∃ (U_seq : ℕ → Set α) (pi_map : ℕ → α → α),
    (∀ m, (U_seq m).Finite) ∧
    (∀ m, U_seq m ⊆ T) ∧
    (U_seq L = U) ∧
    (∀ m < L, (U_seq m).ncard = subsetENetCard T (epsilon m)) ∧
    (∀ m < L, IsSubsetENet T (U_seq m) (epsilon m)) ∧
    (∀ m < L, ∀ u ∈ U_seq (m + 1), pi_map m u ∈ U_seq m ∧ dist u (pi_map m u) ≤ epsilon m) ∧
    (∀ m, m + 1 < L → subsetENetCard T (epsilon (m + 1)) = 1 →
      ∀ u ∈ U_seq (m + 1), pi_map m u = u) := by
  obtain ⟨U_seq₀, pi_map₀, hUf₀, hUs₀, hUL₀, hUn₀, hUc₀, hUp₀⟩ :=
    exists_dudley_chain hU_subset hU_finite D δ L epsilon h_L_def h_finite_covering
  by_cases h_no_sing : ∀ m, m < L → subsetENetCard T (epsilon m) ≠ 1
  · exact ⟨U_seq₀, pi_map₀, hUf₀, hUs₀, hUL₀, hUn₀, hUc₀, hUp₀,
      fun m hm hN => absurd hN (h_no_sing (m + 1) hm)⟩
  · push Not at h_no_sing
    obtain ⟨m₀, hm₀_lt, hm₀_eq⟩ := h_no_sing
    have h_eps_pos : ∀ m, 0 < epsilon m := fun m => by simp [h_eps]; positivity
    have h_eps_le_D : ∀ m, epsilon m ≤ D := fun m => by
      simp only [h_eps]
      exact mul_le_of_le_one_right hD.le (by
        rw [Real.rpow_le_one_iff_of_pos (by norm_num : (0:ℝ) < 2)]
        left; exact ⟨by linarith, by norm_num⟩)
    have h_eps_antitone : ∀ m₁ m₂, m₁ ≤ m₂ → epsilon m₂ ≤ epsilon m₁ := fun m₁ m₂ h => by
      simp only [h_eps]
      exact mul_le_mul_of_nonneg_left
        (rpow_le_rpow_of_exponent_le (by norm_num : (1:ℝ) ≤ 2)
          (by simp only [neg_le_neg_iff]; exact_mod_cast h))
        hD.le
    set S := (Finset.range L).filter (fun m => subsetENetCard T (epsilon m) = 1)
    have hS_ne : S.Nonempty :=
      ⟨m₀, Finset.mem_filter.mpr ⟨Finset.mem_range.mpr hm₀_lt, hm₀_eq⟩⟩
    set M := S.max' hS_ne
    have hM_mem : M ∈ S := Finset.max'_mem S hS_ne
    have hM_lt : M < L := Finset.mem_range.mp ((Finset.filter_subset _ _) hM_mem)
    have hM_eq : subsetENetCard T (epsilon M) = 1 := (Finset.mem_filter.mp hM_mem).2
    have h_all_sing : ∀ m, m ≤ M → m < L → subsetENetCard T (epsilon m) = 1 := by
      intro m hm hm_lt
      have h_le : subsetENetCard T (epsilon m) ≤ 1 :=
        hM_eq ▸ subsetENetCard_antitoneOn_Ioc h_finite_cov
          ⟨h_eps_pos M, h_eps_le_D M⟩ ⟨h_eps_pos m, h_eps_le_D m⟩ (h_eps_antitone m M hm)
      have h_ge : 0 < subsetENetCard T (epsilon m) := by
        rw [← hUn₀ m hm_lt]
        obtain ⟨t, ht⟩ := h_T_nonempty
        obtain ⟨u, hu, _⟩ := IsSubsetENet.exists_near (hUc₀ m hm_lt) ht
        exact Set.ncard_pos (hUf₀ m) |>.mpr ⟨u, hu⟩
      omega
    have hM_max : ∀ m, M < m → m < L → subsetENetCard T (epsilon m) ≠ 1 := by
      intro m hMm hm hm_eq
      exact absurd (Finset.le_max' S m (Finset.mem_filter.mpr
        ⟨Finset.mem_range.mpr hm, hm_eq⟩)) (not_le.mpr hMm)
    obtain ⟨cstar, hcstar_eq⟩ : ∃ c, U_seq₀ M = {c} :=
      Set.ncard_eq_one.mp (by rw [hUn₀ M hM_lt, hM_eq])
    have hcstar_mem : cstar ∈ U_seq₀ M := hcstar_eq ▸ Set.mem_singleton cstar
    have hcstar_T : cstar ∈ T := hUs₀ M hcstar_mem
    have hcstar_cov : ∀ t ∈ T, dist t cstar ≤ epsilon M := by
      intro t ht
      obtain ⟨y, hy, hd⟩ := IsSubsetENet.exists_near (hUc₀ M hM_lt) ht
      rw [hcstar_eq, Set.mem_singleton_iff] at hy; rwa [hy] at hd
    refine ⟨fun m => if m ≤ M then {cstar} else U_seq₀ m,
            fun m u => if m ≤ M then cstar else pi_map₀ m u,
            ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
    · 
      intro m; dsimp only; split_ifs <;> [exact Set.finite_singleton _; exact hUf₀ m]
    · 
      intro m; dsimp only; split_ifs
      · exact Set.singleton_subset_iff.mpr hcstar_T
      · exact hUs₀ m
    · 
      dsimp only; rw [if_neg (not_le.mpr hM_lt)]; exact hUL₀
    · 
      intro m hm; dsimp only; split_ifs with h
      · rw [Set.ncard_singleton]; exact (h_all_sing m h hm).symm
      · exact hUn₀ m hm
    · 
      intro m hm; dsimp only; split_ifs with h
      · exact IsSubsetENet.of_exists_dist (Set.finite_singleton cstar)
          (Set.singleton_subset_iff.mpr hcstar_T)
          (fun x hx => ⟨cstar, Set.mem_singleton _,
            le_trans (hcstar_cov x hx) (h_eps_antitone m M h)⟩)
      · exact hUc₀ m hm
    · 
      intro m hm u hu; dsimp only at hu ⊢
      by_cases hm_le : m ≤ M
      · simp only [if_pos hm_le]
        constructor
        · exact Set.mem_singleton cstar
        · by_cases hm1_le : m + 1 ≤ M
          · simp only [if_pos hm1_le] at hu
            rw [Set.mem_singleton_iff.mp hu, dist_self]; exact (h_eps_pos m).le
          · simp only [if_neg hm1_le] at hu
            have hm_eq_M : m = M := by omega
            rw [hm_eq_M] at hu ⊢; exact hcstar_cov u (hUs₀ (M + 1) hu)
      · simp only [if_neg hm_le]
        have hm1_gt : ¬(m + 1 ≤ M) := fun h => hm_le (by omega)
        simp only [if_neg hm1_gt] at hu; exact hUp₀ m hm u hu
    · 
      intro m hm hN u hu; dsimp only at hu ⊢
      have hm1_le_M : m + 1 ≤ M := by
        by_contra h; push Not at h; exact hM_max (m + 1) h hm hN
      simp only [if_pos (show m ≤ M by omega), if_pos hm1_le_M] at hu ⊢
      exact (Set.mem_singleton_iff.mp hu).symm

/-
Metric entropy bound for sub-Gaussian maxima indexed by a finite type.
-/
lemma metric_entropy_bound_for_sub_gaussian_maxima_fintype
  {α : Type*} [MeasurableSpace α]
  {ι : Type*} [Fintype ι] [Nonempty ι]
  {Y : ι → α → ℝ} {σ_sq : ℝ} (hσ : 0 ≤ σ_sq)
  {μ : Measure α} [IsProbabilityMeasure μ]
  (h_sg : ∀ i, IsSubGaussian (Y i) σ_sq μ) :
  ∫ x, (⨆ i, Y i x) ∂μ ≤ Real.sqrt (2 * σ_sq * Real.log (Fintype.card ι)) := by
    convert ( metric_entropy_bound_for_sub_gaussian_maxima _ _ _ );
    any_goals assumption;
    rotate_left;
    exact Fintype.card_pos;
    exact fun k => Y ( Fintype.equivFin ι |> Equiv.symm <| k );
    · exact fun k => h_sg _;
    · rw [ ← Equiv.iSup_comp ( Fintype.equivFin ι |> Equiv.symm ) ]

/-
Bound for the expectation of the maximum difference of sub-Gaussian variables.
-/
lemma sub_gaussian_max_diff_bound
  {α : Type*} [MeasurableSpace α]
  {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]
  {X : Ω → α → ℝ}
  {U : Set α} (hU_fin : U.Finite) (hU_nonempty : U.Nonempty)
  {σ_sq : ℝ} (hσ : 0 ≤ σ_sq)
  (h_sg : ∀ u ∈ U, ∀ v ∈ U, IsSubGaussian (fun ω => X ω u - X ω v) σ_sq μ) :
  ∫ ω, (⨆ u ∈ U, ⨆ v ∈ U, X ω u - X ω v) ∂μ ≤ Real.sqrt (2 * σ_sq * Real.log (U.ncard ^ 2)) := by
    
    set ι := {p : α × α | p.1 ∈ U ∧ p.2 ∈ U};
    
    have h_metric_entropy : ∫ ω, (⨆ p : ι, (X ω p.val.1) - (X ω p.val.2)) ∂μ ≤ Real.sqrt (2 * σ_sq * Real.log (ι.ncard)) := by
      have h_card : ∫ ω, (⨆ p : ι, (X ω p.val.1 - X ω p.val.2)) ∂μ ≤ Real.sqrt (2 * σ_sq * Real.log (Nat.card ι)) := by
        have h_finite : Finite ι := by
          exact Set.Finite.to_subtype ( hU_fin.prod hU_fin )
        have h_nonempty : Nonempty ι := by
          exact ⟨ ⟨ hU_nonempty.some, hU_nonempty.some ⟩, hU_nonempty.choose_spec, hU_nonempty.choose_spec ⟩
        convert metric_entropy_bound_for_sub_gaussian_maxima_fintype _ _;
        all_goals try assumption;
        convert Nat.card_eq_fintype_card;
        · exact Set.Finite.fintype ( Set.finite_coe_iff.mp h_finite );
        · exact fun i => h_sg _ i.2.1 _ i.2.2;
      simpa [Nat.card_coe_set_eq] using h_card
    convert h_metric_entropy using 3;
    · convert ( ciSup_eq_of_forall_le_of_forall_lt_exists_gt ?_ ?_ );
      · exact ⟨ hU_nonempty.some ⟩;
      · intro i;
        by_cases hi : i ∈ U <;> simp +decide [ hi ];
        · convert ciSup_le _;
          · exact ⟨ i ⟩;
          · intro x;
            rw [ @ciSup_eq_ite ];
            split_ifs <;> [ exact le_ciSup ( show BddAbove ( Set.range fun p : { p : α × α // p.1 ∈ U ∧ p.2 ∈ U } => X _ p.val.1 - X _ p.val.2 ) from by
                                              exact Set.Finite.bddAbove ( Set.Finite.subset ( hU_fin.prod hU_fin |> Set.Finite.image fun p : α × α => X ‹_› p.1 - X ‹_› p.2 ) fun p hp => by
                                                rcases hp with ⟨q, rfl⟩
                                                exact ⟨q.1, ⟨q.2.1, q.2.2⟩, rfl⟩ ) ) ⟨ ( i, x ), hi, by assumption ⟩ ; exact le_trans ( by norm_num ) ( Real.sSup_nonneg _ fun p hp => by unreachable! ) ];
            refine' le_trans _ ( le_ciSup _ ⟨ ( i, i ), hi, hi ⟩ ) ; norm_num;
            exact Set.Finite.bddAbove ( Set.Finite.subset ( Set.Finite.image ( fun p : α × α => X ‹_› p.1 - X ‹_› p.2 ) ( hU_fin.prod hU_fin ) ) fun p hp => by
              rcases hp with ⟨q, rfl⟩
              exact ⟨q, ⟨q.2.1, q.2.2⟩, rfl⟩ );
        · refine' le_trans _ ( le_ciSup _ ⟨ ( hU_nonempty.some, hU_nonempty.some ), hU_nonempty.choose_spec, hU_nonempty.choose_spec ⟩ ) ; norm_num;
          exact Set.Finite.bddAbove ( Set.Finite.subset ( hU_fin.prod hU_fin |> Set.Finite.image fun p : α × α => X ‹_› p.1 - X ‹_› p.2 ) fun x hx => by
            rcases hx with ⟨q, rfl⟩
            exact ⟨q, ⟨q.2.1, q.2.2⟩, rfl⟩ );
      · intro w hw
        obtain ⟨p, hp⟩ : ∃ p : ι, w < X ‹_› p.val.1 - X ‹_› p.val.2 := by
          contrapose! hw;
          convert ciSup_le _;
          · exact ⟨ ⟨ ⟨ hU_nonempty.some, hU_nonempty.some ⟩, hU_nonempty.choose_spec, hU_nonempty.choose_spec ⟩ ⟩;
          · exact hw;
        use p.val.1;
        rw [ ciSup_pos ];
        · refine' lt_of_lt_of_le hp ( le_trans _ ( le_ciSup _ p.val.2 ) );
          · exact le_ciSup ⟨X ‹_› p.val.1 - X ‹_› p.val.2, Set.forall_mem_range.2 fun _ => le_rfl⟩ p.2.2
          · refine' ⟨ ∑ v ∈ hU_fin.toFinset, |X ‹_› p.val.1 - X ‹_› v|, Set.forall_mem_range.2 fun v => _ ⟩;
            rw [ @ciSup_eq_ite ];
            split_ifs <;> [ exact le_trans ( le_abs_self _ ) ( Finset.single_le_sum ( fun a _ => abs_nonneg ( X ‹_› p.val.1 - X ‹_› a ) ) ( hU_fin.mem_toFinset.mpr ‹_› ) ) ; exact le_trans ( by norm_num ) ( Finset.sum_nonneg fun a _ => abs_nonneg _ ) ];
        · exact p.2.1;
    · rw [ show ι = U ×ˢ U by ext x; simp [ι, Set.mem_prod], Set.ncard_prod ] ; simp +decide [ sq ]

/-
Monotonicity of the sub-Gaussian property with respect to the variance proxy.
-/
lemma IsSubGaussian_mono {α : Type*} [MeasurableSpace α]
  {X : α → ℝ} {σ_sq σ_sq' : ℝ} {μ : MeasureTheory.Measure α}
  (h : IsSubGaussian X σ_sq μ) (hle : σ_sq ≤ σ_sq') :
  IsSubGaussian X σ_sq' μ := by
    obtain ⟨ h₁, h₂ ⟩ := h;
    refine' ⟨ _, _ ⟩;
    linarith;
    constructor;
    · intro t;
      convert h₂.integrable_exp_mul t using 1;
    · intro t;
      exact le_trans ( h₂.mgf_le t ) ( Real.exp_le_exp.mpr ( by
        gcongr
        exact hle ) )

/-
For a symmetric nonempty bounded set of reals, the supremum of the absolute values equals the supremum of the values.
-/
lemma sSup_abs_eq_sSup_of_symmetric
  {S : Set ℝ} (hS_nonempty : S.Nonempty) (hS_bdd : BddAbove S)
  (hS_symm : ∀ x ∈ S, -x ∈ S) :
  sSup (Set.image abs S) = sSup S := by
    rw [ @csSup_eq_of_forall_le_of_forall_lt_exists_gt ];
    · exact hS_nonempty.image _;
    · rintro _ ⟨ x, hx, rfl ⟩ ; cases abs_cases x <;> linarith [ le_csSup hS_bdd hx, le_csSup hS_bdd ( hS_symm x hx ) ] ;
    · intro w hw;
      rcases exists_lt_of_lt_csSup hS_nonempty hw with ⟨ x, hx₁, hx₂ ⟩;
      exact ⟨ |x|, Set.mem_image_of_mem _ hx₁, by cases abs_cases x <;> linarith ⟩

/-
For a finite nonempty set S, the maximum absolute difference of f on S equals the maximum difference of f on S.
-/
lemma sup_abs_diff_eq_sup_diff_finite {α : Type*} {f : α → ℝ} (S : Set α) (hS : S.Finite) (hS_nonempty : S.Nonempty) :
  (⨆ u ∈ S, ⨆ v ∈ S, |f u - f v|) = (⨆ u ∈ S, ⨆ v ∈ S, f u - f v) := by
    
    set D := {d : ℝ | ∃ u ∈ S, ∃ v ∈ S, d = f u - f v} with hD;
    
    have hD_bounded : BddAbove D ∧ BddBelow D := by
      exact ⟨ ⟨ ∑ u ∈ hS.toFinset, |f u| + ∑ u ∈ hS.toFinset, |f u|, by rintro x ⟨ u, hu, v, hv, rfl ⟩ ; cases abs_cases ( f u ) <;> cases abs_cases ( f v ) <;> linarith [ Finset.single_le_sum ( fun a _ => abs_nonneg ( f a ) ) ( hS.mem_toFinset.mpr hu ), Finset.single_le_sum ( fun a _ => abs_nonneg ( f a ) ) ( hS.mem_toFinset.mpr hv ) ] ⟩, ⟨ - ( ∑ u ∈ hS.toFinset, |f u| + ∑ u ∈ hS.toFinset, |f u| ), by rintro x ⟨ u, hu, v, hv, rfl ⟩ ; cases abs_cases ( f u ) <;> cases abs_cases ( f v ) <;> linarith [ Finset.single_le_sum ( fun a _ => abs_nonneg ( f a ) ) ( hS.mem_toFinset.mpr hu ), Finset.single_le_sum ( fun a _ => abs_nonneg ( f a ) ) ( hS.mem_toFinset.mpr hv ) ] ⟩ ⟩;
    
    have hD_abs_eq_D : sSup (Set.image abs D) = sSup D := by
      apply_rules [ sSup_abs_eq_sSup_of_symmetric ];
      · exact ⟨ _, ⟨ hS_nonempty.some, hS_nonempty.choose_spec, hS_nonempty.some, hS_nonempty.choose_spec, rfl ⟩ ⟩;
      · exact hD_bounded.1;
      · rintro _ ⟨ u, hu, v, hv, rfl ⟩ ; exact ⟨ v, hv, u, hu, by ring ⟩;
    convert hD_abs_eq_D using 1;
    · rw [ @csSup_eq_of_forall_le_of_forall_lt_exists_gt ];
      · exact ⟨ _, ⟨ _, ⟨ hS_nonempty.some, hS_nonempty.choose_spec, hS_nonempty.some, hS_nonempty.choose_spec, rfl ⟩, rfl ⟩ ⟩;
      · simp +zetaDelta at *;
        intro a x u hu v hv; subst hv; exact finite_sup_bound hS f x u hu v;
      · intro w hw;
        contrapose! hw;
        convert ciSup_le _;
        · exact ⟨ hS_nonempty.some ⟩;
        · intro x;
          rw [ @ciSup_eq_ite ];
          split_ifs with hx;
          · convert ciSup_le _;
            · exact ⟨ x ⟩;
            · intro y; rw [ @ciSup_eq_ite ] ; split_ifs with h;
              · exact hw _ ⟨_, ⟨x, hx, y, h, rfl⟩, rfl⟩;
              · simp only [Real.sSup_empty]; exact le_trans (abs_nonneg _) (hw _ ⟨_, ⟨hS_nonempty.some, hS_nonempty.choose_spec, hS_nonempty.some, hS_nonempty.choose_spec, rfl⟩, rfl⟩);
          · simp +decide [ Real.sSup_empty ];
            exact le_trans ( abs_nonneg _ ) ( hw _ ⟨ _, ⟨ hS_nonempty.some, hS_nonempty.choose_spec, hS_nonempty.some, hS_nonempty.choose_spec, rfl ⟩, rfl ⟩ );
    · convert ( ciSup_eq_of_forall_le_of_forall_lt_exists_gt _ _ );
      · exact ⟨ hS_nonempty.some ⟩;
      · intro u; by_cases hu : u ∈ S <;> simp +decide [ hu ] ;
        · convert ciSup_le fun v => ?_;
          · exact ⟨ u ⟩;
          · rw [ @ciSup_eq_ite ];
            split_ifs <;> [ exact le_csSup hD_bounded.1 ⟨ u, hu, v, ‹_›, rfl ⟩ ; exact le_trans ( by norm_num ) ( show 0 ≤ sSup D from le_trans ( by norm_num ) ( le_csSup hD_bounded.1 ⟨ u, hu, u, hu, rfl ⟩ ) ) ];
        · exact le_trans ( by norm_num ) ( le_csSup hD_bounded.1 ⟨ hS_nonempty.some, hS_nonempty.choose_spec, hS_nonempty.some, hS_nonempty.choose_spec, rfl ⟩ );
      · intro w hw;
        rcases exists_lt_of_lt_csSup ( show D.Nonempty from ⟨ _, ⟨ hS_nonempty.some, hS_nonempty.choose_spec, hS_nonempty.some, hS_nonempty.choose_spec, rfl ⟩ ⟩ ) hw with ⟨ d, ⟨ u, hu, v, hv, rfl ⟩, hd ⟩;
        use u;
        simp +decide [ hu ];
        refine' lt_of_lt_of_le hd ( le_trans _ ( le_ciSup _ v ) );
        · exact le_ciSup ⟨f u - f v, Set.forall_mem_range.2 fun _ => le_rfl⟩ hv;
        · refine' ⟨ ∑ x ∈ hS.toFinset, |f u - f x|, Set.forall_mem_range.2 fun x => _ ⟩;
          rw [ @ciSup_eq_ite ];
          split_ifs <;> [ exact le_trans ( le_abs_self _ ) ( Finset.single_le_sum ( fun x _ => abs_nonneg ( f u - f x ) ) ( hS.mem_toFinset.mpr ‹_› ) ) ; exact le_trans ( by norm_num ) ( Finset.sum_nonneg fun x _ => abs_nonneg _ ) ]

/-
Bound for the base case of the Dudley chain (U_1).
-/
lemma dudley_base_bound
  {α : Type*} [PseudoMetricSpace α]
  {Ω : Type*} [MeasurableSpace Ω] {μ : MeasureTheory.Measure Ω} [MeasureTheory.IsProbabilityMeasure μ]
  {T : Set α}
  {X : Ω → α → ℝ}
  {D : ℝ} (hD : 0 ≤ D)
  (h_sg : ∀ x ∈ T, ∀ y ∈ T, IsSubGaussian (fun ω => X ω x - X ω y) (dist x y ^ 2) μ)
  (U : Set α) (hU_subset : U ⊆ T) (hU_finite : U.Finite) (hU_nonempty : U.Nonempty)
  (hU_card : U.ncard ≤ subsetENetCard T (D / 2))
  (h_diam : ∀ u ∈ U, ∀ v ∈ U, dist u v ≤ D)
  :
  ∫ ω, (⨆ u ∈ U, ⨆ v ∈ U, |X ω u - X ω v|) ∂μ ≤ 2 * D * Real.sqrt (Real.log (subsetENetCard T (D / 2))) := by
    by_contra h_contra;
    
    have h_sub_gaussian_max_diff_bound : ∫ ω, (⨆ u ∈ U, ⨆ v ∈ U, X ω u - X ω v) ∂μ ≤ Real.sqrt (2 * D^2 * Real.log (U.ncard^2)) := by
      convert sub_gaussian_max_diff_bound hU_finite hU_nonempty _ _ using 1;
      · exact borel α;
      · infer_instance;
      · positivity;
      · exact fun u hu v hv => IsSubGaussian_mono ( h_sg u ( hU_subset hu ) v ( hU_subset hv ) ) ( pow_le_pow_left₀ ( dist_nonneg ) ( h_diam u hu v hv ) 2 );
    refine' h_contra ( le_trans _ ( h_sub_gaussian_max_diff_bound.trans _ ) );
    · 
      have h_abs_eq_sup : ∀ ω, (⨆ u ∈ U, ⨆ v ∈ U, |X ω u - X ω v|) = (⨆ u ∈ U, ⨆ v ∈ U, X ω u - X ω v) := by
        exact fun ω => sup_abs_diff_eq_sup_diff_finite U hU_finite hU_nonempty;
      simp +decide only [h_abs_eq_sup];
      rfl;
    · rw [ Real.sqrt_le_iff ];
      norm_num [ mul_pow ];
      have hU_ncard_pos : 0 < U.ncard := (Set.ncard_pos (hs := hU_finite)).2 hU_nonempty
      have hCover_ncard_pos : 0 < subsetENetCard T (D / 2) := lt_of_lt_of_le hU_ncard_pos hU_card
      have hCover_one_le : 1 ≤ subsetENetCard T (D / 2) := Nat.succ_le_of_lt hCover_ncard_pos
      exact ⟨
        by positivity,
        by
          rw [Real.sq_sqrt (Real.log_nonneg (mod_cast hCover_one_le))]
          nlinarith [Real.log_le_log (Nat.cast_pos.mpr hU_ncard_pos) (Nat.cast_le.mpr hU_card)] ⟩

/-
Inequality for logarithms used in Dudley's bound.
-/
lemma log_inequality_for_dudley (N : ℕ) (hN : 2 ≤ N) :
  Real.sqrt (2 * Real.log (2 * N)) ≤ 2 * Real.sqrt (Real.log N) := by
    rw [ Real.sqrt_le_left ] <;> ring_nf <;> norm_num;
    rw [ Real.sq_sqrt ( Real.log_nonneg ( by norm_cast; linarith ) ) ] ; rw [ Real.log_mul ( by positivity ) ( by positivity ) ] ; ring_nf ; nlinarith [ Real.log_two_gt_d9, Real.log_le_log ( by positivity ) ( show ( N :ℝ ) ≥ 2 by norm_cast ) ] ;

/-
Bound for a single step in the Dudley chain.
-/
lemma dudley_step_bound
  {α : Type*} [PseudoMetricSpace α]
  {Ω : Type*} [MeasurableSpace Ω] {μ : MeasureTheory.Measure Ω} [MeasureTheory.IsProbabilityMeasure μ]
  {T : Set α}
  {X : Ω → α → ℝ}
  (h_sg : ∀ x ∈ T, ∀ y ∈ T, IsSubGaussian (fun ω => X ω x - X ω y) (dist x y ^ 2) μ)
  (U_curr U_next : Set α)
  (hU_curr_subset : U_curr ⊆ T)
  (hU_next_subset : U_next ⊆ T)
  (hU_next_finite : U_next.Finite)
  (pi_map : α → α)
  (h_pi : ∀ u ∈ U_next, pi_map u ∈ U_curr)
  (eps : ℝ) (heps : 0 ≤ eps)
  (h_dist : ∀ u ∈ U_next, dist u (pi_map u) ≤ eps)
  (h_card : U_next.ncard ≤ subsetENetCard T (eps / 2))
  (h_consistent : subsetENetCard T (eps / 2) = 1 → ∀ u ∈ U_next, u = pi_map u)
  :
  ∫ ω, (⨆ u ∈ U_next, |X ω u - X ω (pi_map u)|) ∂μ ≤ 4 * eps * Real.sqrt (Real.log (subsetENetCard T (eps / 2))) := by
    
    have h_exp_bound : ∫ ω, (⨆ u ∈ U_next, |X ω u - X ω (pi_map u)|) ∂μ ≤
      eps * Real.sqrt (2 * Real.log (U_next.ncard)) + eps * Real.sqrt (2 * Real.log 2) := by
        apply_rules [ expectation_bound_chain_step ];
    by_cases h_card_ge_two : 2 ≤ subsetENetCard T (eps / 2);
    · 
      have h_log_ineq : Real.sqrt (2 * Real.log (U_next.ncard)) + Real.sqrt (2 * Real.log 2) ≤ 4 * Real.sqrt (Real.log (subsetENetCard T (eps / 2))) := by
        have h_log_ineq : Real.sqrt (2 * Real.log (U_next.ncard)) ≤ Real.sqrt (2 * Real.log (subsetENetCard T (eps / 2))) := by
          by_cases hU_next_empty : U_next.ncard = 0;
          · simp [hU_next_empty];
          · exact Real.sqrt_le_sqrt <| mul_le_mul_of_nonneg_left ( Real.log_le_log ( Nat.cast_pos.mpr <| Nat.pos_of_ne_zero hU_next_empty ) <| Nat.cast_le.mpr h_card ) zero_le_two;
        have h_log_ineq : Real.sqrt (2 * Real.log 2) ≤ 2 * Real.sqrt (Real.log (subsetENetCard T (eps / 2))) := by
          rw [ Real.sqrt_le_iff ];
          exact ⟨ by positivity, by rw [ mul_pow, Real.sq_sqrt ( Real.log_nonneg ( mod_cast h_card_ge_two.trans' ( by norm_num ) ) ) ] ; nlinarith [ Real.log_pos one_lt_two, Real.log_le_log ( by positivity ) ( show ( subsetENetCard T ( eps / 2 ) : ℝ ) ≥ 2 by norm_cast ) ] ⟩;
        linarith [ show Real.sqrt ( 2 * Real.log ( subsetENetCard T ( eps / 2 ) ) ) ≤ 2 * Real.sqrt ( Real.log ( subsetENetCard T ( eps / 2 ) ) ) by rw [ Real.sqrt_mul ( by positivity ) ] ; nlinarith [ Real.sqrt_nonneg 2, Real.sqrt_nonneg ( Real.log ( subsetENetCard T ( eps / 2 ) ) ), Real.mul_self_sqrt ( show 0 ≤ 2 by positivity ), Real.mul_self_sqrt ( show 0 ≤ Real.log ( subsetENetCard T ( eps / 2 ) ) by exact Real.log_nonneg ( by norm_cast; linarith ) ) ] ];
      nlinarith;
    · interval_cases _ : subsetENetCard T ( eps / 2 ) <;> simp +decide at h_card ⊢;
      · simp_all +decide [ Set.ncard_def ];
      · rw [ MeasureTheory.integral_eq_zero_of_ae ];
        filter_upwards [ ] with;
        by_cases hU_nonempty : U_next.Nonempty
        · haveI : Nonempty α := ⟨hU_nonempty.some⟩
          refine' le_antisymm _ _;
          · apply ciSup_le
            intro u
            rw [ ciSup_eq_ite ]
            split_ifs with hu <;> norm_num
            rw [ ← h_consistent rfl u hu, sub_self ]
          · exact Real.iSup_nonneg fun _ => Real.iSup_nonneg fun _ => abs_nonneg _
        · rw [Set.not_nonempty_iff_eq_empty.mp hU_nonempty]
          simp

/-
Bound for the sum in Dudley's chaining using an integral.
-/
lemma dudley_sum_bound
  {f : ℝ → ℝ}
  {D : ℝ} (hD : 0 < D)
  (hf_antitone : AntitoneOn f (Set.Icc 0 D))
  (L : ℕ) (hL : 1 ≤ L) :
  ∑ k ∈ Finset.range (L - 1), 4 * (D * 2 ^ (-(k + 1 : ℝ))) * f (D * 2 ^ (-(k + 2 : ℝ))) ≤
  16 * ∫ x in Set.Icc (D * 2 ^ (-(L + 1 : ℝ))) (D * 2 ^ (-2 : ℝ)), f x := by
    convert dudley_analytical_sum_bound _ _ _ using 1 <;> norm_num [ mul_assoc, mul_comm, mul_left_comm ];
    rotate_left;
    rotate_left;
    use fun x => f ( x / 2 );
    exact D;
    any_goals exact L - 1;
    · exact hD;
    · exact fun x hx y hy hxy => hf_antitone ⟨ by linarith [ hx.1 ], by linarith [ hx.2 ] ⟩ ⟨ by linarith [ hy.1 ], by linarith [ hy.2 ] ⟩ ( by linarith );
    · norm_num [ Real.rpow_add, Real.rpow_neg ] ; ring_nf;
    · rw [ MeasureTheory.integral_Icc_eq_integral_Ioc, MeasureTheory.integral_Icc_eq_integral_Ioc ] ; rw [ ← intervalIntegral.integral_of_le, ← intervalIntegral.integral_of_le ] <;> norm_num [ Real.rpow_add, Real.rpow_neg ] <;> ring_nf <;> norm_num [ hD.le ] ;
      · cases L <;> norm_num [ pow_succ' ] at * ; ring_nf at *;
      · exact mul_le_of_le_one_right hD.le ( pow_le_one₀ ( by norm_num ) ( by norm_num ) );
      · nlinarith [ pow_le_pow_of_le_one ( by norm_num : ( 0 : ℝ ) ≤ 1 / 2 ) ( by norm_num ) hL ]

lemma dudley_base_bound_integral
  {α : Type*} [PseudoMetricSpace α]
  {T : Set α}
  {D : ℝ} (hD : 0 < D)
  (h_finite_cov : ∀ ε > 0, ∃ u : Set α, u.Finite ∧ IsSubsetENet T u ε) :
  2 * D * Real.sqrt (Real.log (subsetENetCard T (D / 2))) ≤
  8 * ∫ x in Set.Icc (D / 4) (D / 2), Real.sqrt (Real.log (subsetENetCard T x)) := by
    
    set f : ℝ → ℝ := fun x => Real.sqrt (Real.log (subsetENetCard T x))
    have hf_nonneg : ∀ x ∈ Set.Ioc 0 D, 0 ≤ f x := by
      exact fun x hx => Real.sqrt_nonneg _
    have hf_antitone : AntitoneOn f (Set.Ioc 0 D) := by
      have hf_antitone : AntitoneOn (fun δ => subsetENetCard T δ) (Set.Ioc 0 D) := by
        apply_rules [ subsetENetCard_antitoneOn_Ioc ];
      intro x hx y hy hxy;
      simp +zetaDelta at *;
      by_cases h : subsetENetCard T y = 0 <;> by_cases h' : subsetENetCard T x = 0 <;> simp_all +decide;
      · exact False.elim <| by
          simpa [h, h'] using hf_antitone ⟨hx.1, hx.2⟩ ⟨hy.1, hy.2⟩ hxy
      · exact Real.sqrt_le_sqrt ( Real.log_le_log ( Nat.cast_pos.mpr ( Nat.pos_of_ne_zero h ) ) ( Nat.cast_le.mpr ( hf_antitone ⟨ hx.1, hx.2 ⟩ ⟨ hy.1, hy.2 ⟩ hxy ) ) );
    
    have h_integral_bound : ∫ x in Set.Icc (D / 4) (D / 2), f x ≥ ∫ x in Set.Icc (D / 4) (D / 2), f (D / 2) := by
      refine' MeasureTheory.setIntegral_mono_on _ _ _ _ <;> norm_num;
      · 
        have hf_antitone_on : AntitoneOn f (Set.Icc (D / 4) (D / 2)) := by
          exact hf_antitone.mono ( fun x hx => ⟨ by linarith [ hx.1 ], by linarith [ hx.2 ] ⟩ );
        exact ( hf_antitone_on.integrableOn_isCompact ( CompactIccSpace.isCompact_Icc ) );
      · intro x hx₁ hx₂; exact hf_antitone ⟨ by linarith, by linarith ⟩ ⟨ by linarith, by linarith ⟩ ( by linarith ) ;
    have hlen : max (D / 2 - D / 4) 0 = D / 4 := by
      rw [max_eq_left]
      · ring_nf
      · nlinarith [hD]
    have h_const :
        ∫ x in Set.Icc (D / 4) (D / 2), f (D / 2) = (D / 4) * f (D / 2) := by
      rw [MeasureTheory.integral_const]
      simp [hlen]
    rw [h_const] at h_integral_bound
    have h_main : 2 * D * f (D / 2) ≤ 8 * ∫ x in Set.Icc (D / 4) (D / 2), f x := by
      nlinarith
    simpa [f, mul_comm, mul_left_comm, mul_assoc] using h_main

lemma dudley_term_bound
  {f : ℝ → ℝ}
  {D : ℝ} (hD : 0 < D)
  (hf_antitone : AntitoneOn f (Set.Icc 0 D))
  (k : ℕ) :
  4 * (D * 2 ^ (-(k + 1 : ℝ))) * f (D * 2 ^ (-(k + 2 : ℝ))) ≤
  16 * ∫ x in Set.Icc (D * 2 ^ (-(k + 3 : ℝ))) (D * 2 ^ (-(k + 2 : ℝ))), f x := by
    
    have h_int_le : ∫ x in (Set.Icc (D * 2 ^ (-(k + 3 : ℝ))) (D * 2 ^ (-(k + 2 : ℝ)))), f x ≥ ∫ x in (Set.Icc (D * 2 ^ (-(k + 3 : ℝ))) (D * 2 ^ (-(k + 2 : ℝ)))), f (D * 2 ^ (-(k + 2 : ℝ))) := by
      refine' MeasureTheory.setIntegral_mono_on _ _ _ _ <;> norm_num;
      · refine' ( hf_antitone.mono _ ) |> fun h => h.integrableOn_isCompact ( CompactIccSpace.isCompact_Icc ) |> fun h => h.mono_set ( Set.Icc_subset_Icc _ _ );
        exact fun ⦃a⦄ a => a;
        · positivity;
        · exact mul_le_of_le_one_right hD.le ( by rw [ Real.rpow_le_one_iff_of_pos ] <;> norm_num ; linarith );
      · intro x hx₁ hx₂; apply_rules [ hf_antitone ] ; ring_nf at *; norm_num at *;
        · exact ⟨ le_trans ( by positivity ) hx₁, le_trans hx₂ ( mul_le_of_le_one_right ( by positivity ) ( by rw [ Real.rpow_le_one_iff_of_pos ] <;> norm_num ; linarith ) ) ⟩;
        · exact ⟨ by positivity, by exact mul_le_of_le_one_right hD.le ( by rw [ Real.rpow_le_one_iff_of_pos ] <;> norm_num ; linarith ) ⟩;
    norm_num [ Real.rpow_add, Real.rpow_neg ] at *;
    rw [ max_eq_left ] at h_int_le <;> nlinarith [ show 0 < D * ( 1 / 2 * ( 2 ^ k : ℝ ) ⁻¹ ) by positivity, show 0 < D * ( 1 / 8 * ( 2 ^ k : ℝ ) ⁻¹ ) by positivity, inv_pos.mpr ( show 0 < ( 2 ^ k : ℝ ) by positivity ) ]

lemma dudley_step_bound_tight
  {α : Type*} [PseudoMetricSpace α]
  {Ω : Type*} [MeasurableSpace Ω] {μ : MeasureTheory.Measure Ω} [MeasureTheory.IsProbabilityMeasure μ]
  {T : Set α}
  {X : Ω → α → ℝ}
  (h_sg : ∀ x ∈ T, ∀ y ∈ T, IsSubGaussian (fun ω => X ω x - X ω y) (dist x y ^ 2) μ)
  (U_curr U_next : Set α)
  (hU_curr_subset : U_curr ⊆ T)
  (hU_next_subset : U_next ⊆ T)
  (hU_next_finite : U_next.Finite)
  (pi_map : α → α)
  (h_pi : ∀ u ∈ U_next, pi_map u ∈ U_curr)
  (eps : ℝ) (heps : 0 ≤ eps)
  (h_dist : ∀ u ∈ U_next, dist u (pi_map u) ≤ eps)
  (h_card : U_next.ncard ≤ subsetENetCard T (eps / 2))
  (h_consistent : subsetENetCard T (eps / 2) = 1 → ∀ u ∈ U_next, u = pi_map u)
  :
  ∫ ω, (⨆ u ∈ U_next, |X ω u - X ω (pi_map u)|) ∂μ ≤ 2 * eps * Real.sqrt (Real.log (subsetENetCard T (eps / 2))) := by
    by_cases h_cases : subsetENetCard T (eps / 2) = 1;
    · norm_num [ h_cases ];
      refine' MeasureTheory.integral_nonpos fun ω => _;
      by_cases hU_nonempty : U_next.Nonempty
      · haveI : Nonempty α := ⟨hU_nonempty.some⟩
        apply ciSup_le
        intro u
        rw [ciSup_eq_ite]
        split_ifs with hu
        · have h_eq : u = pi_map u := h_consistent h_cases u hu
          have hXu : X ω u = X ω (pi_map u) := congrArg (fun z => X ω z) h_eq
          have h_zero : X ω u - X ω (pi_map u) = 0 := by linarith
          simp [h_zero]
        · norm_num
      · rw [Set.not_nonempty_iff_eq_empty.mp hU_nonempty]
        simp
    · 
      set N := subsetENetCard T (eps / 2);
      
      have h_sub_gaussian_max_abs_bound : ∫ ω, ⨆ u ∈ U_next, |X ω u - X ω (pi_map u)| ∂μ ≤ eps * Real.sqrt (2 * Real.log (2 * U_next.ncard)) := by
        have h_sub_gaussian_max_abs_bound : ∀ u ∈ U_next, IsSubGaussian (fun ω => X ω u - X ω (pi_map u)) (eps ^ 2) μ := by
          intro u hu; specialize h_sg u ( hU_next_subset hu ) ( pi_map u ) ( hU_curr_subset ( h_pi u hu ) ) ; exact (by
          exact IsSubGaussian_mono h_sg ( pow_le_pow_left₀ ( dist_nonneg ) ( h_dist u hu ) 2 ));
        by_cases h : U_next.Nonempty;
        · convert sub_gaussian_max_abs_bound ( show 0 ≤ eps ^ 2 by positivity ) ( show ∀ i : U_next, IsSubGaussian ( fun ω => X ω i - X ω ( pi_map i ) ) ( eps ^ 2 ) μ from fun i => h_sub_gaussian_max_abs_bound i i.2 ) using 1;
          any_goals exact Set.Finite.fintype hU_next_finite;
          · congr! 2;
            convert ( ciSup_eq_of_forall_le_of_forall_lt_exists_gt _ _ );
            · exact ⟨ h.some ⟩;
            · intro i;
              rw [ @ciSup_eq_ite ];
              split_ifs <;> norm_num;
              · apply le_csSup;
                · exact Set.Finite.bddAbove ( Set.Finite.subset ( hU_next_finite.image fun u => |X ‹_› u - X ‹_› ( pi_map u )| ) fun x hx => by
                    rcases hx with ⟨u, rfl⟩
                    exact ⟨u, u.2, rfl⟩ );
                · exact ⟨ ⟨ i, by assumption ⟩, rfl ⟩;
              · exact Real.iSup_nonneg fun _ => abs_nonneg _;
            · intro w hw;
              contrapose! hw;
              convert ciSup_le _;
              · exact ⟨ h.some, h.choose_spec ⟩;
              · exact fun x => le_trans ( by simp +decide [ x.2 ] ) ( hw x );
          · simp +decide [ mul_assoc, mul_comm, Real.sqrt_mul, heps ]
          · exact ⟨ h.some, h.choose_spec ⟩;
        · rw [ show U_next = ∅ by simp [ Set.not_nonempty_iff_eq_empty.mp h ] ] ; norm_num;
      
      have h_log_inequality : Real.sqrt (2 * Real.log (2 * U_next.ncard)) ≤ 2 * Real.sqrt (Real.log N) := by
        have h_log_inequality : Real.sqrt (2 * Real.log (2 * N)) ≤ 2 * Real.sqrt (Real.log N) := by
          by_cases hN : N = 0;
          · simp [hN];
          · have h_log_inequality : Real.sqrt (2 * Real.log (2 * N)) ≤ 2 * Real.sqrt (Real.log N) := by
              have hN_ge_2 : 2 ≤ N := by
                exact Nat.lt_of_le_of_ne ( Nat.pos_of_ne_zero hN ) ( Ne.symm h_cases )
              exact log_inequality_for_dudley N hN_ge_2;
            exact h_log_inequality;
        refine' le_trans _ h_log_inequality;
        by_cases h : U_next.ncard = 0
        · simp [h]
        · have hU_next_ncard_pos : 0 < U_next.ncard := Nat.pos_of_ne_zero h
          have hlog_le : Real.log (2 * U_next.ncard) ≤ Real.log (2 * N) := by
            apply Real.log_le_log
            · norm_cast
              exact Nat.mul_pos zero_lt_two hU_next_ncard_pos
            · norm_cast
              nlinarith
          have hmul_le : 2 * Real.log (2 * U_next.ncard) ≤ 2 * Real.log (2 * N) := by
            nlinarith
          exact Real.sqrt_le_sqrt hmul_le
      nlinarith

lemma dudley_sum_bound_tight
  {f : ℝ → ℝ}
  {D : ℝ} (hD : 0 < D)
  (hf_antitone : AntitoneOn f (Set.Icc 0 D))
  (L : ℕ) (hL : 1 ≤ L) :
  ∑ k ∈ Finset.range (L - 1), 4 * (D * 2 ^ (-(k + 1 : ℝ))) * f (D * 2 ^ (-(k + 2 : ℝ))) ≤
  16 * ∫ x in Set.Icc (D * 2 ^ (-(L + 1 : ℝ))) (D * 2 ^ (-2 : ℝ)), f x := by
    convert dudley_sum_bound hD hf_antitone L hL using 1

/-
Entropy integral monotonicity: integrating √(log N(·,T)) over a larger domain gives a larger value.
-/
lemma dudleyEntropyIntegral_antitone
  {α : Type*} [PseudoMetricSpace α]
  {T : Set α} {D : ℝ}
  (h_finite_cov : ∀ ε > 0, ∃ u : Set α, u.Finite ∧ IsSubsetENet T u ε)
  {δ₁ δ₂ : ℝ} (hδ₁ : 0 < δ₁) (hle : δ₁ ≤ δ₂) :
  dudleyEntropyIntegral T δ₂ D ≤ dudleyEntropyIntegral T δ₁ D := by
  unfold dudleyEntropyIntegral
  apply MeasureTheory.setIntegral_mono_set
  · set f : ℝ → ℝ := fun x => Real.sqrt (Real.log (subsetENetCard T x))
    have hf_antitone : AntitoneOn f (Set.Icc δ₁ D) := by
      intro x hx y hy hxy
      simp +zetaDelta
      by_cases h : subsetENetCard T y = 0 <;> by_cases h' : subsetENetCard T x = 0 <;> simp_all +decide
      · exact False.elim <| by
          simpa [h, h'] using
            subsetENetCard_antitoneOn_Ioc h_finite_cov
              ⟨by linarith [hx.1], by linarith [hx.2]⟩
              ⟨by linarith [hy.1], by linarith [hy.2]⟩
              hxy
      · exact Real.sqrt_le_sqrt (Real.log_le_log (Nat.cast_pos.mpr (Nat.pos_of_ne_zero h)) (Nat.cast_le.mpr (subsetENetCard_antitoneOn_Ioc h_finite_cov ⟨by linarith [hx.1], by linarith [hx.2]⟩ ⟨by linarith [hy.1], by linarith [hy.2]⟩ hxy)))
    exact hf_antitone.integrableOn_isCompact isCompact_Icc
  · exact Filter.Eventually.of_forall fun x => Real.sqrt_nonneg _
  · exact Filter.Eventually.of_forall fun x hx => ⟨by linarith [hx.1], hx.2⟩

/-
Expectation-level one-step discretization: taking expectations of `one_step_discretization_bound`.
-/
lemma expectation_one_step_bound
  {α : Type*} [PseudoMetricSpace α]
  {Ω : Type*} [MeasurableSpace Ω] {P : Measure Ω} [IsProbabilityMeasure P]
  {T : Set α} {X : Ω → α → ℝ} {δ : ℝ}
  {U : Set α} (hU_fin : U.Finite) (hU_subset : U ⊆ T)
  (hU_cover : IsSubsetENet T U δ)
  (u₀ : α) (hu₀ : u₀ ∈ U)
  (h_integrable_global : Integrable (fun ω => globalOsc (X ω) T) P)
  (h_integrable_local : Integrable (fun ω => localOsc (X ω) T δ) P)
  (h_integrable_sup : Integrable (fun ω => ⨆ u ∈ U, |X ω u - X ω u₀|) P) :
  ∫ ω, globalOsc (X ω) T ∂P ≤
    2 * ∫ ω, localOsc (X ω) T δ ∂P +
    2 * ∫ ω, (⨆ u ∈ U, |X ω u - X ω u₀|) ∂P := by
  have h_pointwise : ∀ ω, globalOsc (X ω) T ≤ 2 * localOsc (X ω) T δ + 2 * (⨆ u ∈ U, |X ω u - X ω u₀|) :=
    fun ω => one_step_discretization_bound hU_fin hU_subset hU_cover u₀ hu₀
  have h_rhs_integrable : Integrable (fun ω => 2 * localOsc (X ω) T δ + 2 * (⨆ u ∈ U, |X ω u - X ω u₀|)) P :=
    (h_integrable_local.const_mul 2).add (h_integrable_sup.const_mul 2)
  calc ∫ ω, globalOsc (X ω) T ∂P
      ≤ ∫ ω, (2 * localOsc (X ω) T δ + 2 * (⨆ u ∈ U, |X ω u - X ω u₀|)) ∂P :=
        MeasureTheory.integral_mono h_integrable_global h_rhs_integrable (fun ω => h_pointwise ω)
    _ = 2 * ∫ ω, localOsc (X ω) T δ ∂P + 2 * ∫ ω, (⨆ u ∈ U, |X ω u - X ω u₀|) ∂P := by
        rw [MeasureTheory.integral_add (h_integrable_local.const_mul 2) (h_integrable_sup.const_mul 2),
            MeasureTheory.integral_const_mul, MeasureTheory.integral_const_mul]

/-
Basepoint to pairwise bound: for u₀ ∈ U,
⨆ u ∈ U, |X u - X u₀| ≤ ⨆ u ∈ U, ⨆ v ∈ U, |X u - X v|
-/
lemma basepoint_to_pairwise
  {α : Type*}
  {U : Set α} (hU_fin : U.Finite)
  {X : α → ℝ}
  {u₀ : α} (hu₀ : u₀ ∈ U)
  :
  (⨆ u ∈ U, |X u - X u₀|) ≤ ⨆ u ∈ U, ⨆ v ∈ U, |X u - X v| := by
  have : Nonempty α := ⟨u₀⟩
  apply ciSup_le
  intro u
  rw [@ciSup_eq_ite]
  split_ifs with hu
  · exact finite_sup_bound hU_fin X u u₀ hu hu₀
  · exact le_trans (by norm_num) (Real.iSup_nonneg fun _ => Real.iSup_nonneg fun _ => Real.iSup_nonneg fun _ => Real.iSup_nonneg fun _ => abs_nonneg _)

/-- Finite step suprema of sub-Gaussian increments are integrable. -/
lemma integrable_step_sup_of_subGaussian
  {α : Type*} [PseudoMetricSpace α]
  {Ω : Type*} [MeasurableSpace Ω] {P : Measure Ω} [IsProbabilityMeasure P]
  {T : Set α} {X : Ω → α → ℝ}
  (h_sg : ∀ x ∈ T, ∀ y ∈ T, IsSubGaussian (fun ω => X ω x - X ω y) (dist x y ^ 2) P)
  {V : Set α} (hV_fin : V.Finite) (hV_subset : V ⊆ T)
  (f : α → α) (hf : ∀ z ∈ V, f z ∈ T) :
  Integrable (fun ω => ⨆ z ∈ V, |X ω z - X ω (f z)|) P := by
    cases' hV_fin.exists_finset_coe with F hF
    by_cases hF_nonempty : F.Nonempty
    · set Y : F → Ω → ℝ := fun z ω => X ω z - X ω (f z)
      have hY_int : ∀ z : F, Integrable (Y z) P := by
        intro z
        exact (h_sg z.val (hV_subset (hF.subset z.prop)) (f z.val)
          (hf z.val (hF.subset z.prop))).integrable
      haveI : Nonempty F := ⟨⟨hF_nonempty.choose, hF_nonempty.choose_spec⟩⟩
      have h_int := integrable_ciSup_abs_of_fintype (Y := Y) hY_int
      convert h_int using 1
      rw [← hF]
      congr 1 with ω
      have hbdd : BddAbove
          (Set.range fun z : (↑F : Set α) => |X ω z - X ω (f z)|) := by
        exact Finite.bddAbove_range _
      have hbot : sSup (∅ : Set ℝ) ≤
          (⨆ z : (↑F : Set α), |X ω z - X ω (f z)|) := by
        refine le_trans ?_ (le_ciSup hbdd ⟨hF_nonempty.choose, hF_nonempty.choose_spec⟩)
        simp
      rw [← ciSup_subtype_fun hbdd hbot]
      rfl
    · rw [← hF]
      rw [Finset.not_nonempty_iff_eq_empty.mp hF_nonempty]
      simp

/-- Finite basepoint suprema of sub-Gaussian increments are integrable. -/
lemma integrable_basepoint_sup_of_subGaussian
  {α : Type*} [PseudoMetricSpace α]
  {Ω : Type*} [MeasurableSpace Ω] {P : Measure Ω} [IsProbabilityMeasure P]
  {T : Set α} {X : Ω → α → ℝ}
  (h_sg : ∀ x ∈ T, ∀ y ∈ T, IsSubGaussian (fun ω => X ω x - X ω y) (dist x y ^ 2) P)
  {U : Set α} (hU_fin : U.Finite) (hU_subset : U ⊆ T)
  {u₀ : α} (hu₀ : u₀ ∈ U) :
  Integrable (fun ω => ⨆ u ∈ U, |X ω u - X ω u₀|) P :=
    integrable_step_sup_of_subGaussian h_sg hU_fin hU_subset (fun _ => u₀)
      (fun _ _ => hU_subset hu₀)

/-- Finite pairwise suprema of sub-Gaussian increments are integrable. -/
lemma integrable_pairwise_sup_of_subGaussian
  {α : Type*} [PseudoMetricSpace α]
  {Ω : Type*} [MeasurableSpace Ω] {P : Measure Ω} [IsProbabilityMeasure P]
  {T : Set α} {X : Ω → α → ℝ}
  (h_sg : ∀ x ∈ T, ∀ y ∈ T, IsSubGaussian (fun ω => X ω x - X ω y) (dist x y ^ 2) P)
  {V : Set α} (hV_fin : V.Finite) (hV_subset : V ⊆ T) (hV_nonempty : V.Nonempty) :
  Integrable (fun ω => ⨆ u ∈ V, ⨆ v ∈ V, |X ω u - X ω v|) P := by
    cases' hV_fin.exists_finset_coe with F hF
    have hF_nonempty : F.Nonempty := by
      obtain ⟨v, hv⟩ := hV_nonempty
      exact ⟨v, by simpa [← hF] using hv⟩
    set Y : F × F → Ω → ℝ := fun p ω => X ω p.1 - X ω p.2
    have hY_int : ∀ p : F × F, Integrable (Y p) P := by
      intro p
      exact (h_sg p.1.val (hV_subset (hF.subset p.1.prop))
        p.2.val (hV_subset (hF.subset p.2.prop))).integrable
    haveI : Nonempty (F × F) :=
      ⟨(⟨hF_nonempty.choose, hF_nonempty.choose_spec⟩,
        ⟨hF_nonempty.choose, hF_nonempty.choose_spec⟩)⟩
    have h_int := integrable_ciSup_abs_of_fintype (Y := Y) hY_int
    convert h_int using 1
    rw [← hF]
    congr 1 with ω
    have h_outer_bdd : BddAbove
        (Set.range fun u : (↑F : Set α) =>
          ⨆ v ∈ (↑F : Set α), |X ω u - X ω v|) := by
      exact Finite.bddAbove_range _
    have h_outer_bot : sSup (∅ : Set ℝ) ≤
        (⨆ u : (↑F : Set α), ⨆ v ∈ (↑F : Set α), |X ω u - X ω v|) := by
      simp
      exact Real.iSup_nonneg fun _ => Real.iSup_nonneg fun _ =>
        Real.iSup_nonneg fun _ => abs_nonneg _
    rw [← ciSup_subtype_fun h_outer_bdd h_outer_bot]
    rw [ciSup_prod (f := fun p : F × F => |X ω p.1 - X ω p.2|)
      (by exact Finite.bddAbove_range _)]
    congr 1 with u
    have h_inner_bdd : BddAbove
        (Set.range fun v : (↑F : Set α) => |X ω u - X ω v|) := by
      exact Finite.bddAbove_range _
    have h_inner_bot : sSup (∅ : Set ℝ) ≤
        (⨆ v : (↑F : Set α), |X ω u - X ω v|) := by
      refine le_trans ?_ (le_ciSup h_inner_bdd ⟨hF_nonempty.choose, hF_nonempty.choose_spec⟩)
      simp
    rw [← ciSup_subtype_fun h_inner_bdd h_inner_bot]
    simp

/-
Chaining entropy bound: the expected pairwise sup over a finite cover is bounded by the entropy integral.
This is the core technical lemma combining the chain construction with the base and step bounds.
-/
set_option maxHeartbeats 0 in
lemma chaining_entropy_bound
  {α : Type*} [PseudoMetricSpace α]
  {Ω : Type*} [MeasurableSpace Ω] {P : Measure Ω} [IsProbabilityMeasure P]
  {T : Set α} {X : Ω → α → ℝ}
  {D : ℝ} (hD : 0 < D)
  (h_diam : ∀ x ∈ T, ∀ y ∈ T, dist x y ≤ D)
  (h_sg : ∀ x ∈ T, ∀ y ∈ T, IsSubGaussian (fun ω => X ω x - X ω y) (dist x y ^ 2) P)
  (h_finite_cov : ∀ ε > 0, ∃ u : Set α, u.Finite ∧ IsSubsetENet T u ε)
  (h_T_nonempty : T.Nonempty)
  {δ' : ℝ} (hδ' : 0 < δ') (hδ'D : δ' ≤ D / 4)
  {U : Set α} (hU_fin : U.Finite) (hU_subset : U ⊆ T) (hU_nonempty : U.Nonempty)
  (hU_ncard : U.ncard ≤ subsetENetCard T δ') :
  ∫ ω, (⨆ u ∈ U, ⨆ v ∈ U, |X ω u - X ω v|) ∂P ≤
    16 * dudleyEntropyIntegral T (δ' / 4) D := by
  
  by_cases hU_single : U.ncard ≤ 1
  · 
    have hU1 : U.ncard = 1 := Nat.le_antisymm hU_single
      (Set.ncard_pos hU_fin |>.mpr hU_nonempty)
    obtain ⟨u₀, hu₀_eq⟩ := Set.ncard_eq_one.mp hU1
    refine le_trans ?_ (mul_nonneg (by norm_num : (0:ℝ) ≤ 16) ?_)
    · rw [hu₀_eq]
      refine le_of_eq ?_
      rw [MeasureTheory.integral_eq_zero_of_ae]
      filter_upwards with ω
      apply le_antisymm _ (Real.iSup_nonneg fun _ => Real.iSup_nonneg fun _ =>
        Real.iSup_nonneg fun _ => Real.iSup_nonneg fun _ => abs_nonneg _)
      haveI : Nonempty α := ⟨u₀⟩
      apply ciSup_le; intro u
      rw [ciSup_eq_ite]; split_ifs with hu
      · rw [Set.mem_singleton_iff.mp hu]
        apply ciSup_le; intro v
        rw [ciSup_eq_ite]; split_ifs with hv
        · rw [Set.mem_singleton_iff.mp hv]; simp
        · simp
      · simp
    · exact MeasureTheory.setIntegral_nonneg measurableSet_Icc
        (fun x _ => Real.sqrt_nonneg _)
  
  push Not at hU_single
  
  set L := Nat.ceil (Real.logb 2 (D / δ')) with hL_def
  set eps : ℕ → ℝ := fun m => D * 2 ^ (-(m : ℝ)) with heps_def
  have h_eps_eq : ∀ m : ℕ, eps m = D * 2 ^ (-(m : ℝ)) := fun _ => rfl
  have h_eps_pos : ∀ m : ℕ, 0 < eps m := fun m => by simp [heps_def]; positivity
  
  have hL_ge_1 : 1 ≤ L := by
    rw [hL_def]; rw [Nat.one_le_ceil_iff]
    exact Real.logb_pos (by norm_num) (by rw [lt_div_iff₀ hδ']; linarith)
  have hL_ge_2 : 2 ≤ L := by
    suffices h : (2 : ℝ) ≤ ↑L by exact_mod_cast h
    have h4 : (4 : ℝ) ≤ D / δ' := by rw [le_div_iff₀ hδ']; linarith
    have h_logb : (2 : ℝ) ≤ Real.logb 2 (D / δ') := by
      rw [Real.le_logb_iff_rpow_le (by norm_num : (1:ℝ) < 2) (div_pos hD hδ')]
      norm_num; exact h4
    linarith [Nat.le_ceil (Real.logb 2 (D / δ'))]
  
  have h_fc : ∀ m < L, ∃ u, u.Finite ∧ IsSubsetENet T u (eps m) :=
    fun m _ => h_finite_cov (eps m) (h_eps_pos m)
  
  obtain ⟨U_seq, pi_map, hUf, hUs, hUL, hUn, hUc, hUp, h_chain_consistent⟩ :=
    exists_dudley_chain_consistent hU_subset hU_fin D hD δ' L eps h_eps_eq hL_def h_fc
      h_T_nonempty h_finite_cov
  
  rw [← hUL]
  
  have heps_L_le : eps L ≤ δ' := by
    simp only [heps_def]
    have h1 : Real.logb 2 (D / δ') ≤ (↑L : ℝ) := Nat.le_ceil _
    have h2 : D / δ' ≤ (2 : ℝ) ^ (↑L : ℝ) := by
      calc D / δ' = (2 : ℝ) ^ Real.logb 2 (D / δ') :=
            (Real.rpow_logb (by norm_num) (by norm_num) (div_pos hD hδ')).symm
        _ ≤ (2 : ℝ) ^ (↑L : ℝ) :=
            rpow_le_rpow_of_exponent_le (by norm_num : 1 ≤ (2:ℝ)) h1
    have h3 : D ≤ δ' * (2 : ℝ) ^ (↑L : ℝ) := by
      rw [div_le_iff₀ hδ'] at h2; linarith
    rw [rpow_neg (by norm_num : (0:ℝ) ≤ 2)]
    rwa [mul_inv_le_iff₀ (rpow_pos_of_pos (by norm_num : (0:ℝ) < 2) _)]
  
  have heps_L1_ge : δ' / 4 ≤ eps (L + 1) := by
    simp only [heps_def]
    
    have hL_pred_lt : (↑(L - 1) : ℝ) < Real.logb 2 (D / δ') := by
      rw [← Nat.lt_ceil]; omega
    have hL_pred_cast : (↑(L - 1) : ℝ) = (↑L : ℝ) - 1 := by
      push_cast [Nat.cast_sub hL_ge_1]; ring
    rw [hL_pred_cast] at hL_pred_lt
    
    have hL1_lt : (↑(L + 1) : ℝ) < Real.logb 2 (D / δ') + 2 := by
      push_cast; linarith
    
    have h2L1 : (2 : ℝ) ^ (↑(L + 1) : ℝ) < 4 * (D / δ') := by
      calc (2 : ℝ) ^ (↑(L + 1) : ℝ)
          < (2 : ℝ) ^ (Real.logb 2 (D / δ') + 2) :=
            rpow_lt_rpow_of_exponent_lt (by norm_num) hL1_lt
        _ = (2 : ℝ) ^ Real.logb 2 (D / δ') * (2 : ℝ) ^ (2 : ℝ) := by
            rw [rpow_add (by norm_num : (0:ℝ) < 2)]
        _ = (D / δ') * 4 := by
            rw [Real.rpow_logb (by norm_num) (by norm_num) (div_pos hD hδ')]
            norm_num
        _ = 4 * (D / δ') := by ring
    
    have h_mul : δ' * (2 : ℝ) ^ (↑(L + 1) : ℝ) < 4 * D := by
      have := mul_lt_mul_of_pos_right h2L1 hδ'
      rw [mul_comm ((2:ℝ) ^ _) δ', mul_comm (4 * (D / δ')) δ'] at this
      rwa [mul_comm δ' (4 * (D / δ')), mul_assoc, div_mul_cancel₀ D (ne_of_gt hδ')] at this
    
    rw [rpow_neg (by norm_num : (0:ℝ) ≤ 2)]
    rw [le_mul_inv_iff₀ (rpow_pos_of_pos (by norm_num : (0:ℝ) < 2) _)]
    linarith
  
  have h_chain_mem : ∀ m < L, ∀ u ∈ U_seq (m + 1), pi_map m u ∈ U_seq m :=
    fun m hm u hu => (hUp m hm u hu).1
  have hU1_nonempty : (U_seq 1).Nonempty := by
    by_cases hL1 : L = 1
    · rw [hL1] at hUL; rw [hUL]; exact hU_nonempty
    · have hL2 : 1 < L := by omega
      obtain ⟨t, ht⟩ := h_T_nonempty
      obtain ⟨u, hu, _⟩ := IsSubsetENet.exists_near (hUc 1 hL2) ht
      exact ⟨u, hu⟩
  have h_int_pw : Integrable (fun ω => ⨆ x ∈ U_seq 1, ⨆ y ∈ U_seq 1, |X ω x - X ω y|) P :=
    integrable_pairwise_sup_of_subGaussian h_sg (hUf 1) (hUs 1) hU1_nonempty
  have h_int_step : ∀ k < L - 1, Integrable
      (fun ω => ⨆ z ∈ U_seq (k + 2), |X ω z - X ω (pi_map (k + 1) z)|) P :=
    fun k hk => integrable_step_sup_of_subGaussian h_sg (hUf _) (hUs _) (pi_map (k + 1))
      (fun z hz => hUs (k + 1) (h_chain_mem (k + 1) (by omega) z hz))
  have h_decomp := integral_dudley_chain_bound hL_ge_1 hUf h_chain_mem h_int_pw h_int_step
  
  have h_base_entropy : ∫ ω, (⨆ x ∈ U_seq 1, ⨆ y ∈ U_seq 1, |X ω x - X ω y|) ∂P ≤
      8 * ∫ x in Set.Icc (D / 4) (D / 2), Real.sqrt (Real.log (subsetENetCard T x)) := by
    have h_ncard_1 : (U_seq 1).ncard ≤ subsetENetCard T (D / 2) := by
      rw [hUn 1 (by omega : 1 < L)]
      suffices h : eps 1 = D / 2 by rw [h]
      simp only [heps_def]; push_cast
      rw [rpow_neg (by norm_num : (0:ℝ) ≤ 2), rpow_one]; ring
    have h_diam_1 : ∀ u ∈ U_seq 1, ∀ v ∈ U_seq 1, dist u v ≤ D :=
      fun u hu v hv => h_diam u (hUs 1 hu) v (hUs 1 hv)
    calc ∫ ω, (⨆ x ∈ U_seq 1, ⨆ y ∈ U_seq 1, |X ω x - X ω y|) ∂P
        ≤ 2 * D * Real.sqrt (Real.log (subsetENetCard T (D / 2))) :=
          dudley_base_bound hD.le h_sg (U_seq 1) (hUs 1) (hUf 1) hU1_nonempty h_ncard_1 h_diam_1
      _ ≤ 8 * ∫ x in Set.Icc (D / 4) (D / 2), Real.sqrt (Real.log (subsetENetCard T x)) :=
          dudley_base_bound_integral hD h_finite_cov
  
  have h_step_sum : 2 * ∑ k ∈ Finset.range (L - 1),
      ∫ ω, (⨆ z ∈ U_seq (k + 2), |X ω z - X ω (pi_map (k + 1) z)|) ∂P ≤
      16 * ∫ x in Set.Icc (D * 2 ^ (-(↑L + 1 : ℝ))) (D * 2 ^ (-2 : ℝ)),
      Real.sqrt (Real.log (subsetENetCard T x)) := by
    
    have h_eps_half : ∀ k, eps (k + 1) / 2 = eps (k + 2) := by
      intro k
      simp only [heps_def]
      push_cast
      rw [mul_div_assoc]; congr 1
      conv_rhs => rw [show -(↑k + 2 : ℝ) = -(↑k + 1) - 1 from by push_cast; ring,
                       rpow_sub (by norm_num : (0:ℝ) < 2), rpow_one]
    
    have h_eps_2 : eps 2 = D / 4 := by
      simp only [heps_def]; push_cast
      rw [rpow_neg (by norm_num : (0:ℝ) ≤ 2)]; norm_num; ring
    
    have h_eps_le_D : ∀ m, eps m ≤ D := by
      intro m
      simp only [heps_def]
      exact mul_le_of_le_one_right hD.le (by
        rw [Real.rpow_le_one_iff_of_pos (by norm_num : (0:ℝ) < 2)]
        left; constructor
        · linarith
        · norm_num)
    
    have h_each : ∀ k ∈ Finset.range (L - 1),
        ∫ ω, (⨆ z ∈ U_seq (k + 2), |X ω z - X ω (pi_map (k + 1) z)|) ∂P ≤
        2 * eps (k + 1) * Real.sqrt (Real.log (subsetENetCard T (eps (k + 1) / 2))) := by
      intro k hk_range
      have hk_lt : k < L - 1 := Finset.mem_range.mp hk_range
      apply dudley_step_bound_tight h_sg (U_seq (k + 1)) (U_seq (k + 2))
        (hUs (k + 1)) (hUs (k + 2)) (hUf (k + 2))
      · 
        intro u hu; exact (hUp (k + 1) (by omega) u hu).1
      · 
        exact (h_eps_pos (k + 1)).le
      · 
        intro u hu; exact (hUp (k + 1) (by omega) u hu).2
      · 
        rw [h_eps_half]
        by_cases h_case : k + 2 < L
        · exact le_of_eq (hUn (k + 2) h_case)
        · have hk2_eq : k + 2 = L := by omega
          rw [hk2_eq, hUL]
          exact le_trans hU_ncard (subsetENetCard_antitoneOn_Ioc h_finite_cov
            ⟨h_eps_pos L, h_eps_le_D L⟩ ⟨hδ', le_trans hδ'D (by linarith)⟩ heps_L_le)
      · 
        intro h_eq_1 u hu
        rw [h_eps_half] at h_eq_1
        by_cases h_kL : k + 2 < L
        · 
          exact (h_chain_consistent (k + 1) (by omega) h_eq_1 u hu).symm
        · 
          exfalso
          have hk2_eq : k + 2 = L := by omega
          rw [hk2_eq] at h_eq_1
          have : 2 ≤ subsetENetCard T (eps L) :=
            le_trans hU_single (le_trans hU_ncard (subsetENetCard_antitoneOn_Ioc h_finite_cov
              ⟨h_eps_pos L, h_eps_le_D L⟩ ⟨hδ', le_trans hδ'D (by linarith)⟩ heps_L_le))
          omega
    
    calc 2 * ∑ k ∈ Finset.range (L - 1),
          ∫ ω, (⨆ z ∈ U_seq (k + 2), |X ω z - X ω (pi_map (k + 1) z)|) ∂P
        ≤ 2 * ∑ k ∈ Finset.range (L - 1),
          2 * eps (k + 1) * Real.sqrt (Real.log (subsetENetCard T (eps (k + 1) / 2))) := by
          gcongr with k hk
          exact h_each k hk
      _ = ∑ k ∈ Finset.range (L - 1),
          (4 * eps (k + 1) * Real.sqrt (Real.log (subsetENetCard T (eps (k + 2))))) := by
          rw [Finset.mul_sum]; congr 1; ext k; rw [h_eps_half]; ring
      _ ≤ 16 * ∫ x in Set.Icc (D * 2 ^ (-(↑L + 1 : ℝ))) (D * 2 ^ (-2 : ℝ)),
          Real.sqrt (Real.log (subsetENetCard T x)) := by
        
        set g : ℝ → ℝ := fun x => Real.sqrt (Real.log (subsetENetCard T (max x (eps (L + 1)))))
        
        have hg_antitone : AntitoneOn g (Set.Icc 0 D) := by
          intro x hx y hy hxy; simp only [g]
          have hN_le : subsetENetCard T (max y (eps (L + 1))) ≤
              subsetENetCard T (max x (eps (L + 1))) :=
            subsetENetCard_antitoneOn_Ioc h_finite_cov
              ⟨lt_max_of_lt_right (h_eps_pos _), max_le hx.2 (h_eps_le_D _)⟩
              ⟨lt_max_of_lt_right (h_eps_pos _), max_le hy.2 (h_eps_le_D _)⟩
              (max_le_max_right _ hxy)
          by_cases h0 : subsetENetCard T (max x (eps (L + 1))) = 0
          · have : subsetENetCard T (max y (eps (L + 1))) = 0 :=
              Nat.eq_zero_of_le_zero (h0 ▸ hN_le)
            simp [h0, this]
          by_cases h0' : subsetENetCard T (max y (eps (L + 1))) = 0
          · simp [h0']
          · exact Real.sqrt_le_sqrt (Real.log_le_log
              (Nat.cast_pos.mpr (Nat.pos_of_ne_zero h0'))
              (Nat.cast_le.mpr hN_le))
        
        have h_max_eval : ∀ k, k < L - 1 → eps (L + 1) ≤ eps (k + 2) := by
          intro k hk; simp only [heps_def]
          apply mul_le_mul_of_nonneg_left _ hD.le
          apply rpow_le_rpow_of_exponent_le (by norm_num : (1:ℝ) ≤ 2)
          simp only [neg_le_neg_iff]; push_cast; exact_mod_cast (by omega : k + 2 ≤ L + 1)
        
        have hg_eval : ∀ k, k < L - 1 →
            g (eps (k + 2)) = Real.sqrt (Real.log (subsetENetCard T (eps (k + 2)))) := by
          intro k hk; simp only [g]; congr 1; congr 1; congr 1; congr 1
          exact max_eq_left (h_max_eval k hk)
        
        have h_sum_rw : ∑ k ∈ Finset.range (L - 1),
            (4 * eps (k + 1) * Real.sqrt (Real.log (subsetENetCard T (eps (k + 2))))) =
            ∑ k ∈ Finset.range (L - 1),
            4 * (D * 2 ^ (-(k + 1 : ℝ))) * g (D * 2 ^ (-(k + 2 : ℝ))) :=
          Finset.sum_congr rfl (fun k hk => by
            rw [← hg_eval k (Finset.mem_range.mp hk)]
            simp only [heps_def]; push_cast; rfl)
        rw [h_sum_rw]
        
        have h_sum_bound := dudley_sum_bound_tight hD hg_antitone L hL_ge_1
        
        have hg_eq_on : ∀ x ∈ Set.Icc (D * 2 ^ (-(↑L + 1 : ℝ))) (D * 2 ^ (-2 : ℝ)),
            g x = Real.sqrt (Real.log (subsetENetCard T x)) := by
          intro x hx; simp only [g]; congr 1; congr 1; congr 1; congr 1
          rw [max_eq_left]
          have : D * 2 ^ (-(↑L + 1 : ℝ)) = eps (L + 1) := by
            simp only [heps_def]; push_cast; ring_nf
          linarith [hx.1]
        calc _ ≤ 16 * ∫ x in Set.Icc (D * 2 ^ (-(↑L + 1 : ℝ))) (D * 2 ^ (-2 : ℝ)), g x :=
            h_sum_bound
          _ = 16 * ∫ x in Set.Icc (D * 2 ^ (-(↑L + 1 : ℝ))) (D * 2 ^ (-2 : ℝ)),
              Real.sqrt (Real.log (subsetENetCard T x)) := by
            congr 1; exact MeasureTheory.setIntegral_congr_fun measurableSet_Icc hg_eq_on
  
  calc ∫ ω, (⨆ u ∈ U_seq L, ⨆ v ∈ U_seq L, |X ω u - X ω v|) ∂P
      ≤ ∫ ω, (⨆ x ∈ U_seq 1, ⨆ y ∈ U_seq 1, |X ω x - X ω y|) ∂P +
        2 * ∑ k ∈ Finset.range (L - 1),
        ∫ ω, (⨆ z ∈ U_seq (k + 2), |X ω z - X ω (pi_map (k + 1) z)|) ∂P := h_decomp
    _ ≤ (8 * ∫ x in Set.Icc (D / 4) (D / 2), Real.sqrt (Real.log (subsetENetCard T x))) +
        (16 * ∫ x in Set.Icc (D * 2 ^ (-(↑L + 1 : ℝ))) (D * 2 ^ (-2 : ℝ)),
        Real.sqrt (Real.log (subsetENetCard T x))) := add_le_add h_base_entropy h_step_sum
    _ ≤ 16 * dudleyEntropyIntegral T (δ' / 4) D := by
      
      set f := fun x : ℝ => Real.sqrt (Real.log (subsetENetCard T x)) with hf_def
      
      have h_D4 : D * (2 : ℝ) ^ (-2 : ℝ) = D / 4 := by
        rw [rpow_neg (by norm_num : (0:ℝ) ≤ 2)]; norm_num; ring
      rw [h_D4]
      
      have ha_ge : δ' / 4 ≤ D * 2 ^ (-(↑L + 1 : ℝ)) := by
        have h_eq : D * 2 ^ (-(↑L + 1 : ℝ)) = eps (L + 1) := by
          simp only [heps_def]; push_cast; ring_nf
        rw [h_eq]; exact heps_L1_ge
      
      have ha_le : D * 2 ^ (-(↑L + 1 : ℝ)) ≤ D / 4 := by
        have h_eq : D * 2 ^ (-(↑L + 1 : ℝ)) = eps (L + 1) := by
          simp only [heps_def]; push_cast; ring_nf
        rw [h_eq]
        have : eps (L + 1) ≤ eps L := by
          simp only [heps_def]
          apply mul_le_mul_of_nonneg_left _ hD.le
          apply rpow_le_rpow_of_exponent_le (by norm_num : (1:ℝ) ≤ 2)
          simp only [neg_le_neg_iff]; push_cast; linarith
        linarith [heps_L_le, hδ'D]
      
      have hf_antitone : AntitoneOn f (Set.Icc (δ' / 4) D) := by
        intro x hx y hy hxy
        simp only [f]
        by_cases h : subsetENetCard T y = 0 <;>
          by_cases h' : subsetENetCard T x = 0 <;> simp_all +decide
        · exact False.elim <| by
            simpa [h, h'] using
              subsetENetCard_antitoneOn_Ioc h_finite_cov
                ⟨lt_of_lt_of_le (div_pos hδ' (by norm_num : (0:ℝ) < 4)) hx.1, hx.2⟩
                ⟨lt_of_lt_of_le (div_pos hδ' (by norm_num : (0:ℝ) < 4)) hy.1, hy.2⟩
                hxy
        · exact Real.sqrt_le_sqrt (Real.log_le_log
            (Nat.cast_pos.mpr (Nat.pos_of_ne_zero h))
            (Nat.cast_le.mpr (subsetENetCard_antitoneOn_Ioc h_finite_cov
              ⟨lt_of_lt_of_le (div_pos hδ' (by norm_num : (0:ℝ) < 4)) hx.1, hx.2⟩
              ⟨lt_of_lt_of_le (div_pos hδ' (by norm_num : (0:ℝ) < 4)) hy.1, hy.2⟩
              hxy)))
      
      have hf_int : IntegrableOn f (Set.Icc (δ' / 4) D) :=
        hf_antitone.integrableOn_isCompact isCompact_Icc
      
      have hf_nn : ∀ x : ℝ, 0 ≤ f x := fun x => Real.sqrt_nonneg _
      
      have h_base_nn : 0 ≤ ∫ x in Set.Icc (D / 4) (D / 2), f x ∂ℙ :=
        MeasureTheory.setIntegral_nonneg measurableSet_Icc (fun x _ => hf_nn x)
      
      have h_disj : Disjoint (Set.Ioc (D * 2 ^ (-(↑L + 1 : ℝ))) (D / 4))
          (Set.Ioc (D / 4) (D / 2)) :=
        Set.disjoint_left.mpr fun x hx1 hx2 => absurd hx2.1 (not_lt.mpr hx1.2)
      have h_int_ioc1 : IntegrableOn f
          (Set.Ioc (D * 2 ^ (-(↑L + 1 : ℝ))) (D / 4)) :=
        hf_int.mono_set (Set.Ioc_subset_Icc_self.trans
          (Set.Icc_subset_Icc ha_ge (by linarith : D / 4 ≤ D)))
      have h_int_ioc2 : IntegrableOn f (Set.Ioc (D / 4) (D / 2)) :=
        hf_int.mono_set (Set.Ioc_subset_Icc_self.trans
          (Set.Icc_subset_Icc (by linarith [hδ'D]) (by linarith : D / 2 ≤ D)))
      have h_union := MeasureTheory.setIntegral_union h_disj measurableSet_Ioc
        h_int_ioc1 h_int_ioc2
      rw [Set.Ioc_union_Ioc_eq_Ioc ha_le (by linarith : D / 4 ≤ D / 2)] at h_union
      
      have h_combine :
          ∫ x in Set.Icc (D * 2 ^ (-(↑L + 1 : ℝ))) (D / 4), f x ∂ℙ +
          ∫ x in Set.Icc (D / 4) (D / 2), f x ∂ℙ =
          ∫ x in Set.Icc (D * 2 ^ (-(↑L + 1 : ℝ))) (D / 2), f x ∂ℙ := by
        rw [MeasureTheory.integral_Icc_eq_integral_Ioc,
            MeasureTheory.integral_Icc_eq_integral_Ioc,
            MeasureTheory.integral_Icc_eq_integral_Ioc]
        exact h_union.symm
      
      have h_expand :
          ∫ x in Set.Icc (D * 2 ^ (-(↑L + 1 : ℝ))) (D / 2), f x ∂ℙ ≤
          dudleyEntropyIntegral T (δ' / 4) D := by
        unfold dudleyEntropyIntegral
        exact MeasureTheory.setIntegral_mono_set hf_int
          (Filter.Eventually.of_forall fun x => hf_nn x)
          (Filter.Eventually.of_forall fun x hx =>
            ⟨le_trans ha_ge hx.1, le_trans hx.2 (by linarith)⟩)
      
      nlinarith [h_base_nn, h_combine, h_expand]

theorem truncated_dudley_entropy_bound_aux
  {α : Type*} [PseudoMetricSpace α]
  {Ω : Type*} [MeasurableSpace Ω] {P : Measure Ω} [IsProbabilityMeasure P]
  {T : Set α} {X : Ω → α → ℝ}
  {D : ℝ} (hD : 0 < D) {δ : ℝ} (hδ : 0 < δ) (hδD : δ ≤ D / 4)
  (h_diam : ∀ x ∈ T, ∀ y ∈ T, dist x y ≤ D)
  (h_sg : ∀ x ∈ T, ∀ y ∈ T, IsSubGaussian (fun ω => X ω x - X ω y) (dist x y ^ 2) P)
  (h_finite_cov : ∀ ε > 0, ∃ u : Set α, u.Finite ∧ IsSubsetENet T u ε)
  (h_T_nonempty : T.Nonempty)
  (h_integrable_global : Integrable (fun ω => globalOsc (X ω) T) P)
  (h_integrable_local : Integrable (fun ω => localOsc (X ω) T δ) P) :
  ∫ ω, globalOsc (X ω) T ∂P ≤
    2 * ∫ ω, localOsc (X ω) T δ ∂P + 32 * dudleyEntropyIntegral T (δ / 4) D := by
  obtain ⟨U, hU_fin, hU_cover, hU_ncard_eq⟩ :=
    exists_minimal_covering_of_finite_covering (h_finite_cov δ hδ)
  have hU_subset : U ⊆ T := hU_cover.1
  have hU_nonempty : U.Nonempty := by
    obtain ⟨t, ht⟩ := h_T_nonempty
    obtain ⟨u, hu, _⟩ := IsSubsetENet.exists_near hU_cover ht
    exact ⟨u, hu⟩
  obtain ⟨u₀, hu₀⟩ := hU_nonempty
  have h_integrable_sup := integrable_basepoint_sup_of_subGaussian h_sg hU_fin hU_subset hu₀
  have h_step1 := expectation_one_step_bound hU_fin hU_subset hU_cover u₀ hu₀
    h_integrable_global h_integrable_local h_integrable_sup
  have h_bp : ∀ ω, (⨆ u ∈ U, |X ω u - X ω u₀|) ≤ (⨆ u ∈ U, ⨆ v ∈ U, |X ω u - X ω v|) :=
    fun ω => basepoint_to_pairwise hU_fin hu₀
  have h_integrable_pairwise := integrable_pairwise_sup_of_subGaussian h_sg hU_fin hU_subset ⟨u₀, hu₀⟩
  have h_sup_le_pairwise : ∫ ω, (⨆ u ∈ U, |X ω u - X ω u₀|) ∂P ≤
      ∫ ω, (⨆ u ∈ U, ⨆ v ∈ U, |X ω u - X ω v|) ∂P :=
    MeasureTheory.integral_mono h_integrable_sup h_integrable_pairwise h_bp
  have hU_ncard_le : U.ncard ≤ subsetENetCard T δ := by
    rw [hU_ncard_eq]
  have h_chaining := chaining_entropy_bound hD h_diam h_sg h_finite_cov h_T_nonempty
    hδ hδD hU_fin hU_subset ⟨u₀, hu₀⟩ hU_ncard_le
  linarith

end TDudley

export TDudley (globalOsc localOsc)

lemma TDudley.subsetENetCard_le_coveringNumber_half
  {α : Type*} [PseudoMetricSpace α] {T : Set α}
  (hT_totallyBounded : TotallyBounded T) (h_T_nonempty : T.Nonempty)
  {ε : ℝ} (hε : 0 < ε) :
  (TDudley.subsetENetCard T ε : WithTop ℕ) ≤ _root_.coveringNumber (ε / 2) T := by
  obtain ⟨u, hu_net, hu_subset, hu_card⟩ :=
    exists_enet_subset_from_half hε hT_totallyBounded h_T_nonempty
  have hu_cover : TDudley.IsSubsetENet T (u : Set α) ε := by
    exact TDudley.IsSubsetENet.of_exists_dist u.finite_toSet hu_subset
      (fun x hx => by
        have hx_cover := hu_net hx
        rw [mem_iUnion₂] at hx_cover
        obtain ⟨y, hy_mem, hy_ball⟩ := hx_cover
        exact ⟨y, hy_mem, mem_closedBall.mp hy_ball⟩)
  have h_card_le : TDudley.subsetENetCard T ε ≤ u.card := by
    unfold TDudley.subsetENetCard
    exact Nat.sInf_le ⟨u, hu_cover, rfl⟩
  exact le_trans (by exact_mod_cast h_card_le) hu_card

lemma TDudley.sqrt_log_subsetENetCard_le_sqrtEntropy_half
  {α : Type*} [PseudoMetricSpace α] {T : Set α}
  (hT_totallyBounded : TotallyBounded T) (h_T_nonempty : T.Nonempty)
  {ε : ℝ} (hε : 0 < ε) :
  Real.sqrt (Real.log (TDudley.subsetENetCard T ε)) ≤ sqrtEntropy (ε / 2) T := by
  apply sqrt_log_card_le_sqrtEntropy_of_card_le (eps := ε / 2) (s := T)
  · linarith
  · exact hT_totallyBounded
  · exact TDudley.subsetENetCard_le_coveringNumber_half hT_totallyBounded h_T_nonempty hε

lemma TDudley.dudleyEntropyIntegral_le_sqrtEntropyIntegral_half
  {α : Type*} [PseudoMetricSpace α] {T : Set α}
  (hT_totallyBounded : TotallyBounded T) (h_T_nonempty : T.Nonempty)
  {δ D : ℝ} (hδ : 0 < δ) :
  TDudley.dudleyEntropyIntegral T δ D ≤
    ∫ ε in Set.Icc δ D, sqrtEntropy (ε / 2) T := by
  let f : ℝ → ℝ := fun ε => Real.sqrt (Real.log (TDudley.subsetENetCard T ε))
  let g : ℝ → ℝ := fun ε => sqrtEntropy (ε / 2) T
  have h_finite_cov : ∀ ε > 0, ∃ u : Set α, u.Finite ∧ TDudley.IsSubsetENet T u ε := by
    intro ε hε
    obtain ⟨u, hu_net, hu_subset, _⟩ :=
      exists_enet_subset_from_half hε hT_totallyBounded h_T_nonempty
    refine ⟨u, u.finite_toSet, ?_⟩
    exact TDudley.IsSubsetENet.of_exists_dist u.finite_toSet hu_subset
      (fun x hx => by
        have hx_cover := hu_net hx
        rw [mem_iUnion₂] at hx_cover
        obtain ⟨y, hy_mem, hy_ball⟩ := hx_cover
        exact ⟨y, hy_mem, mem_closedBall.mp hy_ball⟩)
  have hf_antitone : AntitoneOn f (Set.Icc δ D) := by
    intro x hx y hy hxy
    simp only [f]
    by_cases hy_zero : TDudley.subsetENetCard T y = 0 <;>
      by_cases hx_zero : TDudley.subsetENetCard T x = 0 <;> simp_all +decide
    · exact False.elim <| by
        simpa [hy_zero, hx_zero] using
          TDudley.subsetENetCard_antitoneOn_Ioc h_finite_cov
            ⟨lt_of_lt_of_le hδ hx.1, hx.2⟩
            ⟨lt_of_lt_of_le hδ hy.1, hy.2⟩
            hxy
    · exact Real.sqrt_le_sqrt (Real.log_le_log
        (Nat.cast_pos.mpr (Nat.pos_of_ne_zero hy_zero))
        (Nat.cast_le.mpr
          (TDudley.subsetENetCard_antitoneOn_Ioc h_finite_cov
            ⟨lt_of_lt_of_le hδ hx.1, hx.2⟩
            ⟨lt_of_lt_of_le hδ hy.1, hy.2⟩
            hxy)))
  have hg_antitone : AntitoneOn g (Set.Icc δ D) := by
    intro x hx y hy hxy
    simp only [g]
    exact sqrtEntropy_anti_eps_of_totallyBounded
      (by
        have hx_pos : 0 < x := lt_of_lt_of_le hδ hx.1
        linarith)
      (by linarith : x / 2 ≤ y / 2)
      hT_totallyBounded
  have hf_int : IntegrableOn f (Set.Icc δ D) :=
    hf_antitone.integrableOn_isCompact isCompact_Icc
  have hg_int : IntegrableOn g (Set.Icc δ D) :=
    hg_antitone.integrableOn_isCompact isCompact_Icc
  unfold TDudley.dudleyEntropyIntegral
  change ∫ ε in Set.Icc δ D, f ε ≤ ∫ ε in Set.Icc δ D, g ε
  exact MeasureTheory.setIntegral_mono_on hf_int hg_int measurableSet_Icc
    (fun ε hε_mem =>
      TDudley.sqrt_log_subsetENetCard_le_sqrtEntropy_half
        hT_totallyBounded h_T_nonempty (lt_of_lt_of_le hδ hε_mem.1))

/-- Truncated Dudley entropy bound for the global oscillation of a process.

If `0 < δ ≤ D / 4`, pairwise increments on `T` are sub-Gaussian with variance proxy
`dist x y ^ 2`, and `T` is totally bounded, then the expected global oscillation is bounded
by twice the expected local oscillation at scale `δ` plus a truncated integral of the library
metric-entropy integrand `sqrtEntropy`. The half-scale appears because the internal centered
covering proof is compared to the library's more general net cardinality. -/
theorem truncated_dudley_entropy_bound
  {α : Type*} [PseudoMetricSpace α]
  {Ω : Type*} [MeasurableSpace Ω] {P : Measure Ω} [IsProbabilityMeasure P]
  {T : Set α} {X : Ω → α → ℝ}
  {D : ℝ} {δ : ℝ} (hδ : 0 < δ) (hδD : δ ≤ D / 4)
  (h_diam : ∀ x ∈ T, ∀ y ∈ T, dist x y ≤ D)
  (h_sg : ∀ x ∈ T, ∀ y ∈ T, IsSubGaussian (fun ω => X ω x - X ω y) (dist x y ^ 2) P)
  (hT_totallyBounded : TotallyBounded T)
  (h_T_nonempty : T.Nonempty)
  (h_integrable_global : Integrable (fun ω => globalOsc (X ω) T) P)
  (h_integrable_local : Integrable (fun ω => localOsc (X ω) T δ) P) :
  ∫ ω, globalOsc (X ω) T ∂P ≤
    2 * ∫ ω, localOsc (X ω) T δ ∂P +
      32 * ∫ ε in Set.Icc (δ / 4) D, sqrtEntropy (ε / 2) T := by
  have hD : 0 < D := by nlinarith
  have h_finite_cov : ∀ ε > 0, ∃ u : Set α, u.Finite ∧ TDudley.IsSubsetENet T u ε := by
    intro ε hε
    obtain ⟨u, hu_subset, hu_finite, hu_cover⟩ :=
      finite_approx_of_totallyBounded hT_totallyBounded (ε / 2) (by positivity)
    refine ⟨u, hu_finite, ?_⟩
    exact TDudley.IsSubsetENet.of_exists_dist hu_finite hu_subset
      (fun x hx => by
        have hx_cover := hu_cover hx
        rw [mem_iUnion₂] at hx_cover
        obtain ⟨y, hy_mem, hy_ball⟩ := hx_cover
        exact ⟨y, hy_mem, le_trans (le_of_lt (mem_ball.mp hy_ball)) (by linarith)⟩)
  have h_aux := TDudley.truncated_dudley_entropy_bound_aux hD hδ hδD h_diam h_sg h_finite_cov
    h_T_nonempty h_integrable_global h_integrable_local
  have h_entropy :=
    TDudley.dudleyEntropyIntegral_le_sqrtEntropyIntegral_half hT_totallyBounded h_T_nonempty
      (by positivity : 0 < δ / 4) (D := D)
  calc
    ∫ ω, globalOsc (X ω) T ∂P
        ≤ 2 * ∫ ω, localOsc (X ω) T δ ∂P +
          32 * TDudley.dudleyEntropyIntegral T (δ / 4) D := h_aux
    _ ≤ 2 * ∫ ω, localOsc (X ω) T δ ∂P +
          32 * ∫ ε in Set.Icc (δ / 4) D, sqrtEntropy (ε / 2) T := by
        gcongr
