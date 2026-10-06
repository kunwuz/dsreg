import DSReg.Bridge

/-!
# Component-wise identifiability without reconstruction

* `DSReg.exists_minimizer`: the support criterion has a minimizer over `O(d)`.
* `DSReg.identifiability`: the main identifiability theorem of the paper.
  Every minimizer `R` of the support criterion makes `R Q` a signed
  permutation, and the candidate latents `z̃ = R h(z)` satisfy
  `z̃ᵢ = sᵢ z_{π(i)}`.
* `DSReg.identifiability_ae`: the same conclusion when the LeJEPA premise
  `h(z) = Q z` holds `μ`-almost everywhere, the form in which the linear
  identifiability theorem of Klindt, LeCun and Balestriero (2026) provides it.

The proof follows the paper. The dependency bridge (`dependency_bridge`)
turns the criterion into `‖D(·) Qᵀ Rᵀ‖₀,μ`, the union-support lemma
(`colSupport_eq_biUnion`) turns that into the pattern count
`∑ⱼ |⋃_{i ∈ 𝓘ⱼ} 𝓢ᵢ|`, and the support-minimality lemma
(`isSignedPerm_of_patternCount_le`) applies, comparing `R` with the competitor
`R' = Qᵀ`, for which `Qᵀ R'ᵀ = 1`.
-/

noncomputable section

namespace DSReg

open MeasureTheory

variable {d p : ℕ}

/-- By the dependency bridge and under Functional no-cancellation, the support
criterion at an orthogonal `R` is the pattern count of the footprints at
`U = Qᵀ Rᵀ`. -/
theorem criterion_eq_patternCount {μ : Measure (Fin d → ℝ)}
    {g : (Fin d → ℝ) → (Fin p → ℝ)} {h : (Fin d → ℝ) → (Fin d → ℝ)}
    {Q R : Matrix (Fin d) (Fin d) ℝ}
    (hg : ∀ᵐ z ∂μ, DifferentiableAt ℝ g z)
    (hQ : IsOrthogonal Q) (hh : ∀ z, h z = Q.mulVec z)
    (hNC : NoCancellation μ (depJac g)) (hR : IsOrthogonal R) :
    criterion μ g h Q R =
      patternCount (footprint μ (depJac g)) (Q.transpose * R.transpose) := by
  rw [criterion_eq_supportCount hg hQ hh hR,
    supportCount_mul_eq_patternCount μ (depJac g) hNC]

/-- **Component-wise identifiability without reconstruction**, the main
identifiability theorem of the paper. Let `x = g(z)` with `g` differentiable
`μ`-a.e., let the representation satisfy the LeJEPA premise `h(z) = Q z` with
`Q ∈ O(d)`, and assume Structural Diversity and Functional no-cancellation
for the dependency Jacobian `D = ∂x/∂z`. Then for every minimizer `R ∈ O(d)`
of the support criterion, `R Q` is a signed permutation, and there are a
permutation `π` and signs `sᵢ ∈ {1, -1}` with `z̃ᵢ = sᵢ z_{π(i)}` for the
candidate latents `z̃ = R h(z)`. The recovery holds for every `z`, which is
stronger than the paper's almost-sure statement. -/
theorem identifiability {μ : Measure (Fin d → ℝ)}
    {g : (Fin d → ℝ) → (Fin p → ℝ)} {h : (Fin d → ℝ) → (Fin d → ℝ)}
    {Q R : Matrix (Fin d) (Fin d) ℝ}
    (hg : ∀ᵐ z ∂μ, DifferentiableAt ℝ g z)
    (hQ : IsOrthogonal Q) (hh : ∀ z, h z = Q.mulVec z)
    (hSD : StructuralDiversity (footprint μ (depJac g)))
    (hNC : NoCancellation μ (depJac g))
    (hR : IsOrthogonal R)
    (hmin : ∀ R', IsOrthogonal R' → criterion μ g h Q R ≤ criterion μ g h Q R') :
    IsSignedPerm (R * Q) ∧
      ∃ (π : Equiv.Perm (Fin d)) (s : Fin d → ℝ), (∀ i, s i = 1 ∨ s i = -1) ∧
        ∀ (z : Fin d → ℝ) (i : Fin d), R.mulVec (h z) i = s i * z (π i) := by
  -- compare `R` with the competitor `Qᵀ`, whose induced matrix is `Qᵀ Q = 1`
  have hle : patternCount (footprint μ (depJac g)) (Q.transpose * R.transpose) ≤
      ∑ i, (footprint μ (depJac g) i).card := by
    have hcomp := hmin Q.transpose hQ.transpose
    rwa [criterion_eq_patternCount hg hQ hh hNC hR,
      criterion_eq_patternCount hg hQ hh hNC hQ.transpose,
      Matrix.transpose_transpose, hQ.2, patternCount_one] at hcomp
  have hU : IsSignedPerm (Q.transpose * R.transpose) :=
    isSignedPerm_of_patternCount_le hSD (hQ.transpose.mul hR.transpose) hle
  have hRQ : IsSignedPerm (R * Q) := by
    have ht := hU.transpose
    rwa [Matrix.transpose_mul, Matrix.transpose_transpose,
      Matrix.transpose_transpose] at ht
  refine ⟨hRQ, ?_⟩
  obtain ⟨π, s, hs, hact⟩ := hRQ.mulVec_eq
  exact ⟨π, s, hs, fun z i => by rw [candLatent_eq hh, hact]⟩

/-- The identifiability theorem with the LeJEPA premise in the form the
linear identifiability theorem of Klindt, LeCun and Balestriero (2026)
provides it, `h(z) = Q z` for `μ`-a.e. `z`. The conclusion then holds
`μ`-almost surely, as in the paper's definition of recovering individual
latents. -/
theorem identifiability_ae {μ : Measure (Fin d → ℝ)}
    {g : (Fin d → ℝ) → (Fin p → ℝ)} {h : (Fin d → ℝ) → (Fin d → ℝ)}
    {Q R : Matrix (Fin d) (Fin d) ℝ}
    (hg : ∀ᵐ z ∂μ, DifferentiableAt ℝ g z)
    (hQ : IsOrthogonal Q) (hh : ∀ᵐ z ∂μ, h z = Q.mulVec z)
    (hSD : StructuralDiversity (footprint μ (depJac g)))
    (hNC : NoCancellation μ (depJac g))
    (hR : IsOrthogonal R)
    (hmin : ∀ R', IsOrthogonal R' → criterion μ g h Q R ≤ criterion μ g h Q R') :
    IsSignedPerm (R * Q) ∧
      ∃ (π : Equiv.Perm (Fin d)) (s : Fin d → ℝ), (∀ i, s i = 1 ∨ s i = -1) ∧
        ∀ᵐ z ∂μ, ∀ i, R.mulVec (h z) i = s i * z (π i) := by
  -- pass to the representative `z ↦ Q z`, which has the same criterion
  have hmin' : ∀ R', IsOrthogonal R' →
      criterion μ g Q.mulVec Q R ≤ criterion μ g Q.mulVec Q R' := fun R' hR' => by
    rw [← criterion_congr_ae hh, ← criterion_congr_ae hh]
    exact hmin R' hR'
  obtain ⟨hRQ, π, s, hs, hrec⟩ :=
    identifiability hg hQ (fun _ => rfl) hSD hNC hR hmin'
  refine ⟨hRQ, π, s, hs, ?_⟩
  filter_upwards [hh] with z hz
  rw [hz]
  exact hrec z

/-- Minimizers of the support criterion over `O(d)` exist: the criterion
takes values in `ℕ`. No hypothesis on `μ`, `g`, `h` or `Q` is needed. -/
theorem exists_minimizer (μ : Measure (Fin d → ℝ))
    (g : (Fin d → ℝ) → (Fin p → ℝ)) (h : (Fin d → ℝ) → (Fin d → ℝ))
    (Q : Matrix (Fin d) (Fin d) ℝ) :
    ∃ R : Matrix (Fin d) (Fin d) ℝ, IsOrthogonal R ∧
      ∀ R', IsOrthogonal R' → criterion μ g h Q R ≤ criterion μ g h Q R' := by
  classical
  have hex : ∃ n, ∃ R, IsOrthogonal R ∧ criterion μ g h Q R = n :=
    ⟨_, 1, isOrthogonal_one, rfl⟩
  obtain ⟨R, hR, hRn⟩ := Nat.find_spec hex
  refine ⟨R, hR, fun R' hR' => ?_⟩
  rw [hRn]
  exact Nat.find_min' hex ⟨R', hR', rfl⟩

end DSReg

end
