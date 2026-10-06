import DSReg.Identifiability

/-!
# Certificates: non-vacuity and the role of Structural Diversity

Both certificates use the standard Gaussian law `γ = N(0, I₂)` on `ℝ²`, the
law of the latents in the LeJEPA setting, together with the representation
`h = id` and `Q = 1`. For a continuous function, being `γ`-a.e. zero is the
same as vanishing everywhere (`ae_zero_γ`), and every support below is
computed this way.

1. **Nested regime, end to end** (`nested_certificate`). For the nonlinear
   smooth map `g(z) = (z₀, z₁ + z₀³)`, with Jacobian `[[1, 0], [3 z₀², 1]]`,
   the footprints are `𝓢₀ = {0, 1}` and `𝓢₁ = {1}`: nested and distinct.
   Every hypothesis of `DSReg.identifiability` is proved for this instance, a
   minimizer `R` is obtained from `DSReg.exists_minimizer`, and
   `DSReg.identifiability` itself shows that `R` is a signed permutation.
   So the hypotheses of the identifiability theorem are jointly satisfiable
   under the paper's own law, including in the nested regime.

2. **Structural Diversity cannot be dropped** (`boundary_certificate`). For
   `g(z) = z₁ + z₀³` (one observation) both footprints equal `{0}`.
   Every hypothesis of `DSReg.identifiability` holds except Structural
   Diversity, and the rotation `[[3/5, -4/5], [4/5, 3/5]]` minimizes the
   criterion although it is not a signed permutation. The conclusion of the
   identifiability theorem therefore fails without Structural Diversity.
-/

noncomputable section

namespace DSReg.Sanity

open DSReg Finset MeasureTheory ProbabilityTheory ContinuousLinearMap

/-! ## The standard Gaussian law -/

/-- The standard Gaussian law `N(0, I₂)` on `ℝ²`. -/
def γ : Measure (Fin 2 → ℝ) := Measure.pi fun _ => gaussianReal 0 1

/-- `γ` gives positive mass to every nonempty open set. -/
lemma isOpenPosMeasure_γ : γ.IsOpenPosMeasure := by
  have : (gaussianReal 0 1).IsOpenPosMeasure :=
    (gaussianReal_absolutelyContinuous' 0 one_ne_zero).isOpenPosMeasure
  unfold γ
  infer_instance

/-- A continuous function is `γ`-a.e. zero exactly when it vanishes
everywhere. -/
lemma ae_zero_γ {f : (Fin 2 → ℝ) → ℝ} (hf : Continuous f) :
    (f =ᵐ[γ] fun _ => 0) ↔ ∀ z, f z = 0 := by
  have := isOpenPosMeasure_γ
  rw [hf.ae_eq_iff_eq γ continuous_const, funext_iff]

/-- For a continuous Jacobian entry, membership in a footprint under `γ` is
pointwise nonvanishing. -/
lemma mem_footprint_γ {p : ℕ} {D : Fin p → Fin 2 → (Fin 2 → ℝ) → ℝ} {r : Fin p}
    {i : Fin 2} (hD : Continuous (D r i)) :
    r ∈ footprint γ D i ↔ ∃ z, D r i z ≠ 0 := by
  simp [footprint, ae_zero_γ hD]

/-- For a row with continuous entries, a combination that vanishes `γ`-a.e.
vanishes at every point. -/
lemma rowSum_eq_zero {p : ℕ} {D : Fin p → Fin 2 → (Fin 2 → ℝ) → ℝ} {r : Fin p}
    (hD : ∀ i, Continuous (D r i)) {s : Finset (Fin 2)} {c : Fin 2 → ℝ}
    (h : (fun z => ∑ i ∈ s, c i * D r i z) =ᵐ[γ] fun _ => 0) (z : Fin 2 → ℝ) :
    ∑ i ∈ s, c i * D r i z = 0 :=
  (ae_zero_γ (continuous_finset_sum _ fun i _ => continuous_const.mul (hD i))).mp h z

/-! ## Computing dependency Jacobians -/

/-- Read an entry of the dependency Jacobian off the derivative of one
observation coordinate. -/
lemma depJac_eq_of_hasFDerivAt {d p : ℕ} {g : (Fin d → ℝ) → (Fin p → ℝ)}
    {z : Fin d → ℝ} (hg : DifferentiableAt ℝ g z) {r : Fin p}
    {φ : (Fin d → ℝ) →L[ℝ] ℝ} (hφ : HasFDerivAt (fun w => g w r) φ z) (i : Fin d) :
    depJac g r i z = φ (Pi.single i 1) := by
  rw [hφ.unique (hasFDerivAt_pi'.mp hg.hasFDerivAt r)]
  rfl

/-- The derivative of `w ↦ w₁ + w₀³` is `proj₁ + 3 z₀² proj₀`. -/
lemma hasFDerivAt_cubic (z : Fin 2 → ℝ) :
    HasFDerivAt (fun w : Fin 2 → ℝ => w 1 + w 0 ^ 3)
      (proj (R := ℝ) (φ := fun _ : Fin 2 => ℝ) 1 +
        (3 * z 0 ^ 2) • proj (R := ℝ) (φ := fun _ : Fin 2 => ℝ) 0) z := by
  convert (hasFDerivAt_apply (𝕜 := ℝ) (1 : Fin 2) z).add
    ((hasFDerivAt_apply (𝕜 := ℝ) (0 : Fin 2) z).pow 3) using 1
  ext v
  simp

/-! ## Certificate 1: the nested regime -/

/-- The nonlinear observation map `g(z) = (z₀, z₁ + z₀³)`. -/
def gN : (Fin 2 → ℝ) → (Fin 2 → ℝ) := fun z => ![z 0, z 1 + z 0 ^ 3]

lemma hasFDerivAt_gN_zero (z : Fin 2 → ℝ) :
    HasFDerivAt (fun w => gN w 0) (proj (R := ℝ) (φ := fun _ : Fin 2 => ℝ) 0) z := by
  simpa [gN] using hasFDerivAt_apply (𝕜 := ℝ) (0 : Fin 2) z

lemma hasFDerivAt_gN_one (z : Fin 2 → ℝ) :
    HasFDerivAt (fun w => gN w 1)
      (proj (R := ℝ) (φ := fun _ : Fin 2 => ℝ) 1 +
        (3 * z 0 ^ 2) • proj (R := ℝ) (φ := fun _ : Fin 2 => ℝ) 0) z := by
  simpa [gN] using hasFDerivAt_cubic z

lemma differentiableAt_gN (z : Fin 2 → ℝ) : DifferentiableAt ℝ gN z :=
  differentiableAt_pi.mpr (Fin.forall_fin_two.mpr
    ⟨(hasFDerivAt_gN_zero z).differentiableAt, (hasFDerivAt_gN_one z).differentiableAt⟩)

/-- The dependency Jacobian of `gN` is `[[1, 0], [3 z₀², 1]]`. -/
lemma depJac_gN (z : Fin 2 → ℝ) (r i : Fin 2) :
    depJac gN r i z = !![1, 0; 3 * z 0 ^ 2, 1] r i := by
  have h0 : depJac gN 0 i z = !![1, 0; 3 * z 0 ^ 2, 1] 0 i := by
    rw [depJac_eq_of_hasFDerivAt (differentiableAt_gN z) (hasFDerivAt_gN_zero z)]
    fin_cases i <;> simp
  have h1 : depJac gN 1 i z = !![1, 0; 3 * z 0 ^ 2, 1] 1 i := by
    rw [depJac_eq_of_hasFDerivAt (differentiableAt_gN z) (hasFDerivAt_gN_one z)]
    fin_cases i <;> simp
  fin_cases r
  exacts [h0, h1]

lemma continuous_depJac_gN (r i : Fin 2) : Continuous (depJac gN r i) := by
  rw [show depJac gN r i = fun z => !![1, 0; 3 * z 0 ^ 2, 1] r i from
    funext fun z => depJac_gN z r i]
  fin_cases r <;> fin_cases i <;> simp <;> fun_prop

/-- The footprints of `gN` are `𝓢₀ = {0, 1}` and `𝓢₁ = {1}`. -/
theorem footprint_gN :
    footprint γ (depJac gN) 0 = {0, 1} ∧ footprint γ (depJac gN) 1 = {1} := by
  constructor <;>
  · ext r
    rw [mem_footprint_γ (continuous_depJac_gN _ _)]
    simp only [depJac_gN]
    -- the entry `3 z₀²` is nonzero at `z = (1, 1)`
    fin_cases r <;> simp <;> exact ⟨1, by simp⟩

lemma activeSet_gN_zero : activeSet γ (depJac gN) 0 = {0} := by
  ext i
  rw [← mem_footprint_iff_mem_activeSet]
  fin_cases i <;> simp [footprint_gN.1, footprint_gN.2]

lemma activeSet_gN_one : activeSet γ (depJac gN) 1 = {0, 1} := by
  ext i
  rw [← mem_footprint_iff_mem_activeSet]
  fin_cases i <;> simp [footprint_gN.1, footprint_gN.2]

theorem noCancellation_gN : NoCancellation γ (depJac gN) := by
  intro r c hsum i hi
  have h0 := rowSum_eq_zero (continuous_depJac_gN r) hsum 0
  have h1 := rowSum_eq_zero (continuous_depJac_gN r) hsum 1
  fin_cases r
  · simp only [Fin.zero_eta, activeSet_gN_zero, Finset.mem_singleton,
      Finset.sum_singleton, depJac_gN] at hi h0
    subst hi
    simpa using h0
  · simp only [Fin.mk_one, activeSet_gN_one, depJac_gN] at hi h0 h1
    rw [Finset.sum_pair (by decide)] at h0 h1
    simp at h0 h1
    fin_cases i
    · simp only [Fin.zero_eta]; linarith
    · simpa using h0

theorem structuralDiversity_gN : StructuralDiversity (footprint γ (depJac gN)) := by
  have h01 : footprint γ (depJac gN) 0 ≠ footprint γ (depJac gN) 1 := by
    rw [footprint_gN.1, footprint_gN.2]
    decide
  intro i j hij
  fin_cases i <;> fin_cases j
  exacts [absurd rfl hij, h01, h01.symm, absurd rfl hij]

/-- **Certificate 1.** For the nonlinear `gN` under the standard Gaussian law
`γ`, with `h = id` and `Q = 1`, the footprints are nested and distinct, a
minimizer `R` of the support criterion exists, and `DSReg.identifiability`
applies to it: `R` is a signed permutation and recovers the individual
latents. -/
theorem nested_certificate :
    footprint γ (depJac gN) 0 = {0, 1} ∧ footprint γ (depJac gN) 1 = {1} ∧
    ∃ R : Matrix (Fin 2) (Fin 2) ℝ, IsOrthogonal R ∧
      (∀ R', IsOrthogonal R' → criterion γ gN id 1 R ≤ criterion γ gN id 1 R') ∧
      IsSignedPerm R ∧
      ∃ (π : Equiv.Perm (Fin 2)) (s : Fin 2 → ℝ), (∀ i, s i = 1 ∨ s i = -1) ∧
        ∀ (z : Fin 2 → ℝ) (i : Fin 2), R.mulVec (id z) i = s i * z (π i) := by
  obtain ⟨R, hR, hmin⟩ := exists_minimizer γ gN id 1
  obtain ⟨hRQ, hrec⟩ := identifiability (ae_of_all _ differentiableAt_gN)
    isOrthogonal_one (fun z => by simp) structuralDiversity_gN noCancellation_gN hR hmin
  rw [Matrix.mul_one] at hRQ
  exact ⟨footprint_gN.1, footprint_gN.2, R, hR, hmin, hRQ, hrec⟩

/-! ## Certificate 2: Structural Diversity cannot be dropped -/

/-- One observation, `g(z) = z₁ + z₀³`. Both latents have footprint `{0}`. -/
def gB : (Fin 2 → ℝ) → (Fin 1 → ℝ) := fun z => ![z 1 + z 0 ^ 3]

lemma hasFDerivAt_gB (z : Fin 2 → ℝ) :
    HasFDerivAt (fun w => gB w 0)
      (proj (R := ℝ) (φ := fun _ : Fin 2 => ℝ) 1 +
        (3 * z 0 ^ 2) • proj (R := ℝ) (φ := fun _ : Fin 2 => ℝ) 0) z := by
  simpa [gB] using hasFDerivAt_cubic z

lemma differentiableAt_gB (z : Fin 2 → ℝ) : DifferentiableAt ℝ gB z :=
  differentiableAt_pi.mpr fun r => by
    rw [Subsingleton.elim r 0]
    exact (hasFDerivAt_gB z).differentiableAt

/-- The dependency Jacobian of `gB` is the row `[3 z₀², 1]`. -/
lemma depJac_gB (z : Fin 2 → ℝ) (r : Fin 1) (i : Fin 2) :
    depJac gB r i z = ![3 * z 0 ^ 2, 1] i := by
  rw [Subsingleton.elim r 0,
    depJac_eq_of_hasFDerivAt (differentiableAt_gB z) (hasFDerivAt_gB z)]
  fin_cases i <;> simp

lemma continuous_depJac_gB (r : Fin 1) (i : Fin 2) : Continuous (depJac gB r i) := by
  rw [show depJac gB r i = fun z => ![3 * z 0 ^ 2, 1] i from
    funext fun z => depJac_gB z r i]
  fin_cases i <;> simp <;> fun_prop

lemma footprint_gB (i : Fin 2) : footprint γ (depJac gB) i = {0} := by
  ext r
  rw [mem_footprint_γ (continuous_depJac_gB r i), Subsingleton.elim r 0]
  simp only [depJac_gB, Finset.mem_singleton, iff_true]
  fin_cases i
  · exact ⟨1, by simp⟩
  · exact ⟨0, by simp⟩

lemma activeSet_gB (r : Fin 1) : activeSet γ (depJac gB) r = {0, 1} := by
  ext i
  rw [← mem_footprint_iff_mem_activeSet, footprint_gB, Subsingleton.elim r 0]
  fin_cases i <;> simp

theorem noCancellation_gB : NoCancellation γ (depJac gB) := by
  intro r c hsum i hi
  have h0 := rowSum_eq_zero (continuous_depJac_gB r) hsum 0
  have h1 := rowSum_eq_zero (continuous_depJac_gB r) hsum 1
  simp only [activeSet_gB, depJac_gB] at h0 h1
  rw [Finset.sum_pair (by decide)] at h0 h1
  simp at h0 h1
  fin_cases i
  · simp only [Fin.zero_eta]; linarith
  · simpa using h0

theorem not_structuralDiversity_gB : ¬ StructuralDiversity (footprint γ (depJac gB)) :=
  fun hSD => hSD (show (0 : Fin 2) ≠ 1 by decide) (by rw [footprint_gB, footprint_gB])

/-- With equal footprints, every orthogonal `R` pays the same criterion. -/
lemma criterion_gB {R : Matrix (Fin 2) (Fin 2) ℝ} (hR : IsOrthogonal R) :
    criterion γ gB id 1 R = 2 := by
  rw [criterion_eq_patternCount (ae_of_all _ differentiableAt_gB) isOrthogonal_one
    (fun z => by simp) noCancellation_gB hR, Matrix.transpose_one, Matrix.one_mul]
  have hcol : ∀ j, (mixSet R.transpose j).biUnion (footprint γ (depJac gB)) = {0} := by
    intro j
    obtain ⟨k, hk⟩ := mixSet_nonempty hR.transpose j
    ext r
    simp only [Finset.mem_biUnion, footprint_gB, Finset.mem_singleton]
    exact ⟨fun ⟨_, _, hr⟩ => hr, fun hr => ⟨k, hk, hr⟩⟩
  unfold patternCount
  simp only [hcol, Finset.card_singleton, Finset.sum_const, Finset.card_univ,
    Fintype.card_fin, smul_eq_mul, mul_one]

/-- A rotation with all entries nonzero. -/
def U35 : Matrix (Fin 2) (Fin 2) ℝ := !![3/5, -4/5; 4/5, 3/5]

lemma isOrthogonal_U35 : IsOrthogonal U35 := by
  constructor <;>
  · ext i j
    fin_cases i <;> fin_cases j <;>
      simp [U35, Matrix.mul_apply, Fin.sum_univ_two] <;> norm_num

theorem not_isSignedPerm_U35 : ¬ IsSignedPerm U35 := by
  rintro ⟨σ, s, hs, hentry⟩
  have h00 := hentry 0 0
  have h10 := hentry 1 0
  by_cases h0 : (0 : Fin 2) = σ 0
  · have h1 : ¬ (1 : Fin 2) = σ 0 := fun h1 => absurd (h0.trans h1.symm) (by decide)
    rw [if_neg h1] at h10
    norm_num [U35] at h10
  · rw [if_neg h0] at h00
    norm_num [U35] at h00

/-- **Certificate 2.** With `gB` under `γ`, `h = id` and `Q = 1`, every
hypothesis of `DSReg.identifiability` holds except Structural Diversity, the
rotation `U35` minimizes the support criterion over `O(2)`, and `U35 * Q` is
not a signed permutation. -/
theorem boundary_certificate :
    (∀ᵐ z ∂γ, DifferentiableAt ℝ gB z) ∧
    IsOrthogonal (1 : Matrix (Fin 2) (Fin 2) ℝ) ∧
    (∀ z, id z = (1 : Matrix (Fin 2) (Fin 2) ℝ).mulVec z) ∧
    NoCancellation γ (depJac gB) ∧
    ¬ StructuralDiversity (footprint γ (depJac gB)) ∧
    IsOrthogonal U35 ∧
    (∀ R', IsOrthogonal R' → criterion γ gB id 1 U35 ≤ criterion γ gB id 1 R') ∧
    ¬ IsSignedPerm (U35 * 1) :=
  ⟨ae_of_all _ differentiableAt_gB, isOrthogonal_one, fun z => by simp,
    noCancellation_gB, not_structuralDiversity_gB, isOrthogonal_U35,
    fun R' hR' => by rw [criterion_gB isOrthogonal_U35, criterion_gB hR'],
    by rw [Matrix.mul_one]; exact not_isSignedPerm_U35⟩

end DSReg.Sanity

end
