import DSReg.Basic

/-!
# Support minimality: the combinatorial core of the identifiability theorem

For footprints `S : Fin d → Finset (Fin p)` and an orthogonal `U`, the pattern
count is `N(U) = ∑ⱼ |⋃_{i ∈ 𝓘ⱼ(U)} 𝓢ᵢ|`. Under Functional no-cancellation it
equals the support count `‖D(·)U‖₀,μ` (`DSReg.supportCount_mul_eq_patternCount`
in `DSReg/Support.lean`). This file proves:

* `patternCount_one`: the identity attains the diagonal cost `∑ᵢ |𝓢ᵢ|`;
* `isSignedPerm_of_patternCount_le`: under Structural Diversity, an orthogonal
  `U` with `N(U) ≤ ∑ᵢ |𝓢ᵢ|` is a signed permutation (the support-minimality
  lemma of the paper).

The Lean proof of the support-minimality lemma takes a permutation `σ` with
`U (σ j) j ≠ 0` for all `j` from the determinant expansion of the whole
matrix, and uses a footprint of minimal cardinality where the paper uses an
inclusion-minimal one. The hypotheses and the conclusion are those of the
paper.
-/

noncomputable section

namespace DSReg

open Finset

variable {d p : ℕ}

/-- Pattern count `N(U) = ∑ⱼ |⋃_{i ∈ 𝓘ⱼ(U)} 𝓢ᵢ|` of the footprints `S`
under the mixing pattern of `U`. -/
def patternCount (S : Fin d → Finset (Fin p)) (U : Matrix (Fin d) (Fin d) ℝ) : ℕ :=
  ∑ j, ((mixSet U j).biUnion S).card

/-- Structural Diversity: the footprints are pairwise distinct. -/
def StructuralDiversity (S : Fin d → Finset (Fin p)) : Prop :=
  ∀ ⦃i j : Fin d⦄, i ≠ j → S i ≠ S j

/-- The identity attains the diagonal cost `∑ᵢ |𝓢ᵢ|`. -/
lemma patternCount_one (S : Fin d → Finset (Fin p)) :
    patternCount S (1 : Matrix (Fin d) (Fin d) ℝ) = ∑ i, (S i).card := by
  unfold patternCount
  refine Finset.sum_congr rfl fun j _ => ?_
  have : mixSet (1 : Matrix (Fin d) (Fin d) ℝ) j = {j} := by
    ext i
    simp [Matrix.one_apply]
  rw [this, Finset.singleton_biUnion]

/-- **Support minimality**: under Structural Diversity, an orthogonal `U`
whose pattern count does not exceed the diagonal cost is a signed
permutation. -/
theorem isSignedPerm_of_patternCount_le {S : Fin d → Finset (Fin p)}
    (hSD : StructuralDiversity S) {U : Matrix (Fin d) (Fin d) ℝ}
    (hU : IsOrthogonal U) (hle : patternCount S U ≤ ∑ i, (S i).card) :
    IsSignedPerm U := by
  obtain ⟨σ, hσ⟩ := exists_perm_ne_zero hU
  have hmem : ∀ j, σ j ∈ mixSet U j := fun j => mem_mixSet.mpr (hσ j)
  have hsub : ∀ j, S (σ j) ⊆ (mixSet U j).biUnion S :=
    fun j => Finset.subset_biUnion_of_mem S (hmem j)
  -- per-column equality of cardinalities
  have hpt : ∀ j, ((mixSet U j).biUnion S).card = (S (σ j)).card := by
    by_contra hcontra
    push_neg at hcontra
    obtain ⟨j₀, hj₀⟩ := hcontra
    have hstrict : (S (σ j₀)).card < ((mixSet U j₀).biUnion S).card :=
      lt_of_le_of_ne (Finset.card_le_card (hsub j₀)) (Ne.symm hj₀)
    have hlt : ∑ j, (S (σ j)).card < ∑ j, ((mixSet U j).biUnion S).card :=
      Finset.sum_lt_sum (fun j _ => Finset.card_le_card (hsub j))
        ⟨j₀, Finset.mem_univ j₀, hstrict⟩
    rw [Equiv.sum_comp σ fun i => (S i).card] at hlt
    exact absurd hle (not_le.mpr hlt)
  -- per-column equality of sets: every latent mixed into column `j` has its
  -- footprint inside `S (σ j)`
  have hSsub : ∀ j, ∀ k ∈ mixSet U j, S k ⊆ S (σ j) := by
    intro j k hk
    rw [Finset.eq_of_subset_of_card_le (hsub j) (le_of_eq (hpt j))]
    exact Finset.subset_biUnion_of_mem S hk
  -- suppose some column is mixed
  refine isSignedPerm_of_forall_card_le_one hU ?_
  by_contra hmix
  push_neg at hmix
  obtain ⟨jmix, hjmix⟩ := hmix
  set J : Finset (Fin d) := Finset.univ.filter fun j => 2 ≤ (mixSet U j).card with hJ
  have hjmixJ : jmix ∈ J := by
    simp only [hJ, Finset.mem_filter, Finset.mem_univ, true_and]
    omega
  set I : Finset (Fin d) := J.biUnion (mixSet U) with hI
  -- singleton columns have mixing set `{σ ℓ}`
  have hsingleton : ∀ ℓ, ℓ ∉ J → mixSet U ℓ = {σ ℓ} := by
    intro ℓ hℓ
    simp only [hJ, Finset.mem_filter, Finset.mem_univ, true_and, not_le] at hℓ
    have h1 : (mixSet U ℓ).card = 1 :=
      le_antisymm (by omega) (Finset.one_le_card.mpr ⟨σ ℓ, hmem ℓ⟩)
    obtain ⟨a, ha⟩ := Finset.card_eq_one.mp h1
    have : σ ℓ ∈ ({a} : Finset (Fin d)) := ha ▸ hmem ℓ
    rw [Finset.mem_singleton] at this
    rw [ha, this]
  -- rows selected by singleton columns avoid `I`
  have hAvoid : ∀ ℓ, ℓ ∉ J → σ ℓ ∉ I := by
    intro ℓ hℓ hmemI
    simp only [hI, Finset.mem_biUnion] at hmemI
    obtain ⟨j, hjJ, hj⟩ := hmemI
    have hjℓ : j ≠ ℓ := fun h => hℓ (h ▸ hjJ)
    exact (mem_mixSet.mp hj) (row_eq_zero_of_mixSet_singleton hU (hsingleton ℓ hℓ) j hjℓ)
  -- a minimal-cardinality footprint among the rows in `I`
  have hIne : I.Nonempty := by
    refine ⟨σ jmix, ?_⟩
    simp only [hI, Finset.mem_biUnion]
    exact ⟨jmix, hjmixJ, hmem jmix⟩
  obtain ⟨istar, histar, hminimal⟩ := Finset.exists_min_image I (fun i => (S i).card) hIne
  obtain ⟨jstar, hjstar⟩ : ∃ jstar, σ jstar = istar := ⟨σ.symm istar, σ.apply_symm_apply istar⟩
  have hjstarJ : jstar ∈ J := by
    by_contra hnot
    exact hAvoid jstar hnot (hjstar ▸ histar)
  -- a second latent mixed into column `jstar`
  have hcard2 : 1 < (mixSet U jstar).card := by
    have : jstar ∈ Finset.univ.filter fun j => 2 ≤ (mixSet U j).card := hjstarJ
    simp only [Finset.mem_filter] at this
    omega
  obtain ⟨k, hk, hkne⟩ :=
    (Finset.one_lt_card_iff_nontrivial.mp hcard2).exists_ne (σ jstar)
  have hkI : k ∈ I := by
    simp only [hI, Finset.mem_biUnion]
    exact ⟨jstar, hjstarJ, hk⟩
  have hksub : S k ⊆ S istar := hjstar ▸ hSsub jstar k hk
  have hkeq : S k = S istar := Finset.eq_of_subset_of_card_le hksub (hminimal k hkI)
  exact hSD (hjstar ▸ hkne) hkeq

end DSReg

end
