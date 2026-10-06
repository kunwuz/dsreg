import DSReg.Minimality

/-!
# Almost-everywhere supports, footprints and Functional no-cancellation

Throughout, `μ` is a measure on a measurable space `Z`, and
`M, D : Fin p → Fin d → Z → ℝ` are matrix-valued functions. An entry is
*active* when it is not `μ`-a.e. zero, i.e. when it is nonzero on a set of
positive `μ` measure (`not_ae_zero_iff_pos_measure`).

* `supportCount μ M`: the paper's `‖M(·)‖₀,μ`, the number of active entries.
* `footprint μ D i`: the dependency footprint `𝓢ᵢ` of latent `i`.
* `activeSet μ D r`: the active latent set `I_r` of observation row `r`.
* `NoCancellation μ D`: Functional no-cancellation.
* `colSupport μ D U j`: the a.e. support of column `j` of `D(·)U`.

The main results are the union-support lemma (`colSupport_eq_biUnion`) and
its consequence `supportCount_mul_eq_patternCount`: under no-cancellation,
`‖D(·)U‖₀,μ = ∑ⱼ |⋃_{i ∈ 𝓘ⱼ(U)} 𝓢ᵢ|`.
-/

open scoped Classical

noncomputable section

namespace DSReg

open Finset MeasureTheory

variable {d p : ℕ} {Z : Type*} [MeasurableSpace Z] (μ : Measure Z)

/-- "Not `μ`-a.e. zero" is "nonzero on a set of positive `μ` measure", the
paper's wording in the definitions of `‖·‖₀,μ`, `𝓢ᵢ` and `I_r`. This lemma
checks the definitions and is not used in the proofs. -/
lemma not_ae_zero_iff_pos_measure (f : Z → ℝ) :
    ¬ (f =ᵐ[μ] fun _ => 0) ↔ 0 < μ {z | f z ≠ 0} := by
  rw [Filter.EventuallyEq, ae_iff, pos_iff_ne_zero]

/-- The support count `‖M(·)‖₀,μ`: the number of entries `(r, i)` such that
`M r i` is nonzero on a set of positive `μ` measure. -/
def supportCount (M : Fin p → Fin d → Z → ℝ) : ℕ :=
  (univ.filter fun e : Fin p × Fin d => ¬ (M e.1 e.2 =ᵐ[μ] fun _ => 0)).card

/-- Matrix-valued functions that agree `μ`-a.e. have the same support
count. -/
lemma supportCount_congr_ae {M M' : Fin p → Fin d → Z → ℝ}
    (h : ∀ᵐ z ∂μ, ∀ r i, M r i z = M' r i z) :
    supportCount μ M = supportCount μ M' := by
  unfold supportCount
  congr 1
  ext e
  simp only [Finset.mem_filter, Finset.mem_univ, true_and, not_iff_not]
  have he : M e.1 e.2 =ᵐ[μ] M' e.1 e.2 := by
    filter_upwards [h] with z hz
    exact hz e.1 e.2
  exact ⟨fun h0 => he.symm.trans h0, fun h0 => he.trans h0⟩

variable (D : Fin p → Fin d → Z → ℝ)

/-- Dependency footprint `𝓢ᵢ = {r : D r i is nonzero on a set of positive μ
measure}`. -/
def footprint (i : Fin d) : Finset (Fin p) :=
  univ.filter fun r => ¬ (D r i =ᵐ[μ] fun _ => 0)

/-- Active set `I_r = {i : D r i is nonzero on a set of positive μ measure}`. -/
def activeSet (r : Fin p) : Finset (Fin d) :=
  univ.filter fun i => ¬ (D r i =ᵐ[μ] fun _ => 0)

lemma mem_footprint_iff_mem_activeSet {r : Fin p} {i : Fin d} :
    r ∈ footprint μ D i ↔ i ∈ activeSet μ D r := by
  simp [footprint, activeSet]

/-- Functional no-cancellation: for every row `r`, a constant-coefficient
combination of the active derivative functions of that row that vanishes
`μ`-a.e. is trivial. -/
def NoCancellation : Prop :=
  ∀ r : Fin p, ∀ c : Fin d → ℝ,
    ((fun z => ∑ i ∈ activeSet μ D r, c i * D r i z) =ᵐ[μ] fun _ => 0) →
    ∀ i ∈ activeSet μ D r, c i = 0

/-- The a.e. support `supp_μ((D(·)U)_{:,j})` of column `j` of `D(·)U`. -/
def colSupport (U : Matrix (Fin d) (Fin d) ℝ) (j : Fin d) : Finset (Fin p) :=
  univ.filter fun r => ¬ ((fun z => ∑ i, D r i z * U i j) =ᵐ[μ] fun _ => 0)

/-- A finite sum of a.e.-zero functions is a.e. zero. -/
lemma sum_ae_zero {ι : Type*} (s : Finset ι) (f : ι → Z → ℝ)
    (h : ∀ i ∈ s, f i =ᵐ[μ] fun _ => 0) :
    ((fun z => ∑ i ∈ s, f i z) =ᵐ[μ] fun _ => 0) := by
  induction s using Finset.induction_on with
  | empty => simp
  | @insert a s ha ih =>
    have hha := h a (Finset.mem_insert_self a s)
    have hs := ih fun i hi => h i (Finset.mem_insert_of_mem hi)
    filter_upwards [hha, hs] with z hza hzs
    simp only [Finset.sum_insert ha, hza, hzs]
    simp

/-- **Union support**: under Functional no-cancellation, the a.e. support of
column `j` of `D(·)U` is the union of the footprints of the latents mixed
into that column, `⋃_{i ∈ 𝓘ⱼ(U)} 𝓢ᵢ`. -/
theorem colSupport_eq_biUnion (hNC : NoCancellation μ D)
    (U : Matrix (Fin d) (Fin d) ℝ) (j : Fin d) :
    colSupport μ D U j = (mixSet U j).biUnion (footprint μ D) := by
  ext r
  simp only [colSupport, Finset.mem_filter, Finset.mem_univ, true_and,
    Finset.mem_biUnion, mem_mixSet]
  constructor
  · -- if no mixed latent is active at row `r`, the entry vanishes a.e.
    intro hactive
    by_contra hnone
    push_neg at hnone
    apply hactive
    refine sum_ae_zero μ _ _ fun i _ => ?_
    by_cases hUij : U i j = 0
    · simp [hUij]
    · have hfoot := hnone i hUij
      simp only [footprint, Finset.mem_filter, Finset.mem_univ, true_and,
        not_not] at hfoot
      filter_upwards [hfoot] with z hz
      simp [hz]
  · -- if some mixed latent is active at row `r`, no-cancellation keeps the
    -- entry active
    rintro ⟨i₀, hUi₀, hfoot⟩ hzero
    have hi₀active : i₀ ∈ activeSet μ D r :=
      (mem_footprint_iff_mem_activeSet μ D).mp hfoot
    -- the inactive part of the sum vanishes a.e.
    have hinactive : ((fun z => ∑ i ∈ Finset.univ \ activeSet μ D r,
        D r i z * U i j) =ᵐ[μ] fun _ => 0) := by
      refine sum_ae_zero μ _ _ fun i hi => ?_
      simp only [Finset.mem_sdiff, Finset.mem_univ, true_and, activeSet,
        Finset.mem_filter, not_not] at hi
      filter_upwards [hi] with z hz
      simp [hz]
    -- hence the active part vanishes a.e. as well
    have hactive : ((fun z => ∑ i ∈ activeSet μ D r, U i j * D r i z)
        =ᵐ[μ] fun _ => 0) := by
      filter_upwards [hzero, hinactive] with z hz hzin
      have hsdiff : (∑ i ∈ Finset.univ \ activeSet μ D r, D r i z * U i j) +
          ∑ i ∈ activeSet μ D r, D r i z * U i j = ∑ i, D r i z * U i j :=
        Finset.sum_sdiff (Finset.subset_univ _)
      have hval : (∑ i ∈ activeSet μ D r, D r i z * U i j) = 0 := by
        have hz' : (∑ i, D r i z * U i j) = 0 := hz
        have hzin' : (∑ i ∈ Finset.univ \ activeSet μ D r, D r i z * U i j) = 0 :=
          hzin
        rw [hz', hzin'] at hsdiff
        linarith
      calc (∑ i ∈ activeSet μ D r, U i j * D r i z)
          = ∑ i ∈ activeSet μ D r, D r i z * U i j :=
            Finset.sum_congr rfl fun i _ => mul_comm _ _
      _ = 0 := hval
    exact hUi₀ (hNC r (fun i => U i j) hactive i₀ hi₀active)

/-- Under Functional no-cancellation, the support count of `D(·)U` is the
pattern count of the footprints: `‖D(·)U‖₀,μ = ∑ⱼ |⋃_{i ∈ 𝓘ⱼ(U)} 𝓢ᵢ|`. -/
theorem supportCount_mul_eq_patternCount (hNC : NoCancellation μ D)
    (U : Matrix (Fin d) (Fin d) ℝ) :
    supportCount μ (fun r j z => ∑ i, D r i z * U i j) =
      patternCount (footprint μ D) U := by
  unfold supportCount patternCount
  rw [Finset.card_filter, Fintype.sum_prod_type_right]
  refine Finset.sum_congr rfl fun j _ => ?_
  rw [← Finset.card_filter, ← colSupport_eq_biUnion μ D hNC U j]
  rfl

end DSReg

end
