import Mathlib

/-!
# Orthogonal matrices, signed permutations and mixing sets

* `DSReg.IsOrthogonal U`: `U * Uᵀ = 1 ∧ Uᵀ * U = 1`, i.e. `U ∈ O(d)`
  (`isOrthogonal_iff_mem_orthogonalGroup` identifies it with mathlib's
  `Matrix.orthogonalGroup`).
* `DSReg.IsSignedPerm U`: column `j` of `U` has the single nonzero entry
  `s j ∈ {1, -1}`, in row `σ j`, for a permutation `σ`.
* `DSReg.mixSet U j`: the set `𝓘ⱼ(U) = {i : U i j ≠ 0}` of latents mixed into
  column `j` (the notation of the paper's proofs).

The main facts proved here: an orthogonal matrix has a permutation `σ` with
`U (σ j) j ≠ 0` for every `j` (`exists_perm_ne_zero`), and an orthogonal
matrix whose columns each mix at most one latent is a signed permutation
(`isSignedPerm_of_forall_card_le_one`).
-/

noncomputable section

namespace DSReg

open Finset

variable {d : ℕ}

/-- `U` is orthogonal: `U Uᵀ = 1` and `Uᵀ U = 1`. -/
def IsOrthogonal (U : Matrix (Fin d) (Fin d) ℝ) : Prop :=
  U * U.transpose = 1 ∧ U.transpose * U = 1

/-- `IsOrthogonal` is membership in mathlib's orthogonal group `O(d)`. This
lemma checks the definition and is not used in the proofs. -/
lemma isOrthogonal_iff_mem_orthogonalGroup (U : Matrix (Fin d) (Fin d) ℝ) :
    IsOrthogonal U ↔ U ∈ Matrix.orthogonalGroup (Fin d) ℝ := by
  rw [Matrix.mem_orthogonalGroup_iff]
  exact ⟨fun h => h.1, fun h => ⟨h, mul_eq_one_comm.mp h⟩⟩

/-- `U` is a signed permutation matrix: column `j` has the single nonzero
entry `s j ∈ {1, -1}` in row `σ j`. -/
def IsSignedPerm (U : Matrix (Fin d) (Fin d) ℝ) : Prop :=
  ∃ (σ : Equiv.Perm (Fin d)) (s : Fin d → ℝ),
    (∀ j, s j = 1 ∨ s j = -1) ∧ ∀ i j, U i j = if i = σ j then s j else 0

/-- The mixing set `𝓘ⱼ(U) = {i : U i j ≠ 0}` of column `j`. -/
def mixSet (U : Matrix (Fin d) (Fin d) ℝ) (j : Fin d) : Finset (Fin d) :=
  univ.filter fun i => U i j ≠ 0

@[simp] lemma mem_mixSet {U : Matrix (Fin d) (Fin d) ℝ} {i j : Fin d} :
    i ∈ mixSet U j ↔ U i j ≠ 0 := by simp [mixSet]

lemma isOrthogonal_one : IsOrthogonal (1 : Matrix (Fin d) (Fin d) ℝ) := by
  constructor <;> simp

lemma IsOrthogonal.transpose {U : Matrix (Fin d) (Fin d) ℝ} (hU : IsOrthogonal U) :
    IsOrthogonal U.transpose := by
  refine ⟨?_, ?_⟩
  · simpa [Matrix.transpose_transpose] using hU.2
  · simpa [Matrix.transpose_transpose] using hU.1

lemma IsOrthogonal.mul {A B : Matrix (Fin d) (Fin d) ℝ}
    (hA : IsOrthogonal A) (hB : IsOrthogonal B) : IsOrthogonal (A * B) := by
  constructor
  · calc A * B * (A * B).transpose = A * (B * B.transpose) * A.transpose := by
          rw [Matrix.transpose_mul]; simp only [Matrix.mul_assoc]
    _ = 1 := by rw [hB.1, Matrix.mul_one, hA.1]
  · calc (A * B).transpose * (A * B) = B.transpose * (A.transpose * A) * B := by
          rw [Matrix.transpose_mul]; simp only [Matrix.mul_assoc]
    _ = 1 := by rw [hA.2, Matrix.mul_one, hB.2]

/-- Columns of an orthogonal matrix have unit Euclidean norm. -/
lemma IsOrthogonal.col_sq_sum {U : Matrix (Fin d) (Fin d) ℝ} (hU : IsOrthogonal U)
    (j : Fin d) : ∑ i, (U i j) ^ 2 = 1 := by
  have h := congrFun (congrFun hU.2 j) j
  simp only [Matrix.mul_apply, Matrix.transpose_apply, Matrix.one_apply_eq] at h
  calc ∑ i, (U i j) ^ 2 = ∑ i, U i j * U i j := by simp [sq]
  _ = 1 := h

/-- Rows of an orthogonal matrix have unit Euclidean norm. -/
lemma IsOrthogonal.row_sq_sum {U : Matrix (Fin d) (Fin d) ℝ} (hU : IsOrthogonal U)
    (i : Fin d) : ∑ j, (U i j) ^ 2 = 1 := by
  have h := congrFun (congrFun hU.1 i) i
  simp only [Matrix.mul_apply, Matrix.transpose_apply, Matrix.one_apply_eq] at h
  calc ∑ j, (U i j) ^ 2 = ∑ j, U i j * U i j := by simp [sq]
  _ = 1 := h

lemma IsOrthogonal.det_ne_zero {U : Matrix (Fin d) (Fin d) ℝ} (hU : IsOrthogonal U) :
    U.det ≠ 0 := by
  have h1 : U.det * U.transpose.det = 1 := by
    rw [← Matrix.det_mul, hU.1, Matrix.det_one]
  intro h
  rw [h, zero_mul] at h1
  exact zero_ne_one h1

/-- Every column of an orthogonal matrix mixes at least one latent. -/
lemma mixSet_nonempty {U : Matrix (Fin d) (Fin d) ℝ} (hU : IsOrthogonal U)
    (j : Fin d) : (mixSet U j).Nonempty := by
  by_contra h
  rw [Finset.not_nonempty_iff_eq_empty] at h
  have hzero : ∀ i, U i j = 0 := by
    intro i
    by_contra hne
    have : i ∈ mixSet U j := mem_mixSet.mpr hne
    simp [h] at this
  have := hU.col_sq_sum j
  simp [hzero] at this

/-- Determinant expansion: some permutation `σ` picks a nonzero entry
`U (σ j) j` from every column `j`. -/
lemma exists_perm_ne_zero {U : Matrix (Fin d) (Fin d) ℝ} (hU : IsOrthogonal U) :
    ∃ σ : Equiv.Perm (Fin d), ∀ j, U (σ j) j ≠ 0 := by
  by_contra h
  push_neg at h
  apply hU.det_ne_zero
  rw [Matrix.det_apply]
  refine Finset.sum_eq_zero fun σ _ => ?_
  obtain ⟨j, hj⟩ := h σ
  have hprod : (∏ i, U (σ i) i) = 0 := Finset.prod_eq_zero (Finset.mem_univ j) hj
  rw [hprod, smul_zero]

/-- Row completion: if column `j` is supported on the single row `r`, then
row `r` vanishes in every other column. -/
lemma row_eq_zero_of_mixSet_singleton {U : Matrix (Fin d) (Fin d) ℝ}
    (hU : IsOrthogonal U) {j r : Fin d} (hsing : mixSet U j = {r}) :
    ∀ ℓ, ℓ ≠ j → U r ℓ = 0 := by
  have hzero : ∀ i, i ≠ r → U i j = 0 := by
    intro i hi
    by_contra hne
    have : i ∈ mixSet U j := mem_mixSet.mpr hne
    rw [hsing, Finset.mem_singleton] at this
    exact hi this
  have hrj : (U r j) ^ 2 = 1 := by
    have : ∑ i, (U i j) ^ 2 = (U r j) ^ 2 := by
      refine Finset.sum_eq_single r (fun i _ hi => ?_) (fun h => absurd (Finset.mem_univ r) h)
      rw [hzero i hi]; ring
    rw [← this, hU.col_sq_sum j]
  have hsplit : (U r j) ^ 2 + ∑ ℓ ∈ Finset.univ.erase j, (U r ℓ) ^ 2 = 1 :=
    (Finset.add_sum_erase Finset.univ (fun ℓ => (U r ℓ) ^ 2)
      (Finset.mem_univ j)).trans (hU.row_sq_sum r)
  have hrest : ∑ ℓ ∈ Finset.univ.erase j, (U r ℓ) ^ 2 = 0 := by
    rw [hrj] at hsplit; linarith
  intro ℓ hℓ
  have hmem : ℓ ∈ Finset.univ.erase j := Finset.mem_erase.mpr ⟨hℓ, Finset.mem_univ ℓ⟩
  have := (Finset.sum_eq_zero_iff_of_nonneg (fun m _ => sq_nonneg (U r m))).mp hrest ℓ hmem
  exact pow_eq_zero_iff (n := 2) (by norm_num) |>.mp this

/-- If every column of an orthogonal matrix mixes at most one latent, the
matrix is a signed permutation. -/
lemma isSignedPerm_of_forall_card_le_one {U : Matrix (Fin d) (Fin d) ℝ}
    (hU : IsOrthogonal U) (h1 : ∀ j, (mixSet U j).card ≤ 1) : IsSignedPerm U := by
  have hsing : ∀ j, ∃ r, mixSet U j = {r} := by
    intro j
    have hcard : (mixSet U j).card = 1 :=
      le_antisymm (h1 j) (Finset.one_le_card.mpr (mixSet_nonempty hU j))
    exact Finset.card_eq_one.mp hcard
  choose ρ hρ using hsing
  have hinj : Function.Injective ρ := by
    intro j j' hjj'
    by_contra hne
    have hmem : ρ j' ∈ mixSet U j' := by rw [hρ j']; exact Finset.mem_singleton_self _
    have : U (ρ j') j' = 0 := by
      have h0 := row_eq_zero_of_mixSet_singleton hU (hρ j) j' (fun h => hne h.symm)
      rwa [hjj'] at h0
    exact (mem_mixSet.mp hmem) this
  have hbij : Function.Bijective ρ := (Finite.injective_iff_bijective).mp hinj
  refine ⟨Equiv.ofBijective ρ hbij, fun j => U (ρ j) j, ?_, ?_⟩
  · intro j
    have hsq : (U (ρ j) j) ^ 2 = 1 := by
      have hz : ∀ i, i ≠ ρ j → U i j = 0 := by
        intro i hi
        by_contra hne
        have : i ∈ mixSet U j := mem_mixSet.mpr hne
        rw [hρ j, Finset.mem_singleton] at this
        exact hi this
      have : ∑ i, (U i j) ^ 2 = (U (ρ j) j) ^ 2 := by
        refine Finset.sum_eq_single (ρ j) (fun i _ hi => ?_)
          (fun h => absurd (Finset.mem_univ _) h)
        rw [hz i hi]; ring
      rw [← this, hU.col_sq_sum j]
    have hfactor : (U (ρ j) j - 1) * (U (ρ j) j + 1) = 0 := by nlinarith [hsq]
    rcases mul_eq_zero.mp hfactor with h | h
    · left; linarith
    · right; linarith
  · intro i j
    by_cases hij : i = Equiv.ofBijective ρ hbij j
    · simp only [hij]
      rfl
    · rw [if_neg hij]
      by_contra hne
      have : i ∈ mixSet U j := mem_mixSet.mpr hne
      rw [hρ j, Finset.mem_singleton] at this
      exact hij (by simpa [Equiv.ofBijective] using this)

/-- The transpose of a signed permutation is a signed permutation. -/
lemma IsSignedPerm.transpose {U : Matrix (Fin d) (Fin d) ℝ}
    (h : IsSignedPerm U) : IsSignedPerm U.transpose := by
  obtain ⟨σ, s, hs, hentry⟩ := h
  refine ⟨σ.symm, fun j => s (σ.symm j), fun j => hs _, ?_⟩
  intro i j
  rw [Matrix.transpose_apply, hentry]
  by_cases hij : i = σ.symm j
  · have hji : j = σ i := by rw [hij, Equiv.apply_symm_apply]
    rw [if_pos hji, if_pos hij, hij]
  · have hji : ¬ j = σ i := fun hcontra =>
      hij (by rw [hcontra, Equiv.symm_apply_apply])
    rw [if_neg hji, if_neg hij]

/-- Action of a signed permutation on vectors:
`(U z)ᵢ = sᵢ · z_{π(i)}` for a permutation `π` and signs `sᵢ ∈ {1, -1}`. -/
lemma IsSignedPerm.mulVec_eq {U : Matrix (Fin d) (Fin d) ℝ}
    (h : IsSignedPerm U) :
    ∃ (π : Equiv.Perm (Fin d)) (s : Fin d → ℝ),
      (∀ i, s i = 1 ∨ s i = -1) ∧
      ∀ (z : Fin d → ℝ) (i : Fin d), U.mulVec z i = s i * z (π i) := by
  obtain ⟨σ, s, hs, hentry⟩ := h
  refine ⟨σ.symm, fun i => s (σ.symm i), fun i => hs _, fun z i => ?_⟩
  have hexp : U.mulVec z i = ∑ j, U i j * z j := by
    simp [Matrix.mulVec, dotProduct]
  rw [hexp, Finset.sum_eq_single (σ.symm i)]
  · rw [hentry, if_pos (Equiv.apply_symm_apply σ i).symm]
  · intro j _ hj
    rw [hentry]
    have hne : ¬ i = σ j := fun hc => hj (by rw [hc, Equiv.symm_apply_apply])
    rw [if_neg hne, zero_mul]
  · exact fun hmem => absurd (Finset.mem_univ _) hmem

end DSReg

end
