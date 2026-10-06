import DSReg.Support

/-!
# The observation model, the dependency bridge and the criterion

The paper's objects, defined from its data:

* latents `z : Fin d → ℝ` with law `μ`, an arbitrary measure on `Fin d → ℝ`;
* the observation map `g : (Fin d → ℝ) → (Fin p → ℝ)`, `x = g(z)`;
* the dependency Jacobian `D(z) = ∂x/∂z`, entry `(r, i)`:
  `depJac g r i z = (fderiv ℝ g z (Pi.single i 1)) r`;
* the representation `h : (Fin d → ℝ) → (Fin d → ℝ)` and, for a rotation `R`,
  the candidate latents `z̃ = R h(z)`, written `R.mulVec (h z)`;
* the observations as a function of the candidate latents,
  `candObs g Q R = fun w => g ((R Q)ᵀ w)`. Under the LeJEPA premise
  `h(z) = Q z` it returns the observation, `candObs g Q R (R h(z)) = g(z)`
  (`candObs_apply_cand`), and it is the only map that does
  (`candObs_unique`), so it is the map `z̃ ↦ x` of the paper;
* the candidate Jacobian `∂x/∂z̃` at `z̃ = R h(z)`, entry `(r, j)`:
  `candJac g h Q R r j z = (fderiv ℝ (candObs g Q R) (R h(z)) (Pi.single j 1)) r`;
* the support criterion, `criterion μ g h Q R = ‖∂x/∂z̃‖₀,μ`, the number of
  entries `(r, j)` of the candidate Jacobian that are not `μ`-a.e. zero.

`g` is not assumed differentiable everywhere, continuous or measurable. Its
Jacobian entries are Borel measurable in any case (`measurable_depJac`), so
the paper's assumption of a measurable Jacobian holds automatically. Where a
function is not differentiable, mathlib sets `fderiv` to zero; under the
hypothesis that `g` is differentiable `μ`-a.e. these points form a `μ`-null
set and do not affect any support.

Main results: `dependency_bridge_at` (the dependency bridge at a single
point, in matrix form), `dependency_bridge` (the dependency bridge: for
`μ`-a.e. `z`, `∂x/∂z̃ = D(z) Qᵀ Rᵀ`) and `criterion_eq_supportCount`
(`‖∂x/∂z̃‖₀,μ = ‖D(·) Qᵀ Rᵀ‖₀,μ`).
-/

noncomputable section

namespace DSReg

open MeasureTheory

variable {d p : ℕ}

/-! ## The dependency Jacobian -/

/-- Entry `(r, i)` of the dependency Jacobian `D(z) = ∂x/∂z` of `x = g(z)`. -/
def depJac (g : (Fin d → ℝ) → (Fin p → ℝ)) (r : Fin p) (i : Fin d)
    (z : Fin d → ℝ) : ℝ :=
  fderiv ℝ g z (Pi.single i 1) r

/-- Every entry of the dependency Jacobian is Borel measurable, for every `g`,
because `fderiv` of any function is. The paper's assumption of a measurable
Jacobian therefore always holds. This lemma checks the definition and is not
used in the proofs. -/
lemma measurable_depJac (g : (Fin d → ℝ) → (Fin p → ℝ)) (r : Fin p) (i : Fin d) :
    Measurable (depJac g r i) :=
  (measurable_pi_apply r).comp (measurable_fderiv_apply_const ℝ g (Pi.single i 1))

/-- The matrix of `fderiv ℝ g z` in the standard bases is `D(z)`. -/
lemma toMatrix'_fderiv (g : (Fin d → ℝ) → (Fin p → ℝ)) (z : Fin d → ℝ) :
    LinearMap.toMatrix' (fderiv ℝ g z : (Fin d → ℝ) →ₗ[ℝ] (Fin p → ℝ)) =
      Matrix.of fun r i => depJac g r i z := by
  ext r i
  rw [LinearMap.toMatrix'_apply]
  rfl

/-- At a point where `g` is differentiable, its derivative is multiplication
by the matrix `D(z)`. -/
lemma hasFDerivAt_depJac {g : (Fin d → ℝ) → (Fin p → ℝ)} {z : Fin d → ℝ}
    (hz : DifferentiableAt ℝ g z) :
    HasFDerivAt g
      ((Matrix.of fun r i => depJac g r i z).mulVecLin.toContinuousLinearMap) z := by
  rw [← toMatrix'_fderiv, ← Matrix.toLin'_apply', Matrix.toLin'_toMatrix']
  convert hz.hasFDerivAt using 1

/-- **Dependency bridge at a point** (the chain-rule step of the dependency
bridge, in matrix form): if `g` has Jacobian matrix `D` at `z` and `M` is
orthogonal, then `z̃ ↦ g(Mᵀ z̃)` has Jacobian matrix `D Mᵀ` at `z̃ = M z`. -/
theorem dependency_bridge_at {g : (Fin d → ℝ) → (Fin p → ℝ)}
    {D : Matrix (Fin p) (Fin d) ℝ} {z : Fin d → ℝ}
    (hg : HasFDerivAt g D.mulVecLin.toContinuousLinearMap z)
    {M : Matrix (Fin d) (Fin d) ℝ} (hM : IsOrthogonal M) :
    HasFDerivAt (fun w => g (M.transpose.mulVec w))
      ((D * M.transpose).mulVecLin.toContinuousLinearMap) (M.mulVec z) := by
  -- the linear substitution and its derivative
  have hlin : HasFDerivAt (fun w : Fin d → ℝ => M.transpose.mulVec w)
      (M.transpose.mulVecLin.toContinuousLinearMap) (M.mulVec z) :=
    (M.transpose.mulVecLin.toContinuousLinearMap).hasFDerivAt
  -- the substitution returns to the base point: `Mᵀ (M z) = z`
  have hpoint : M.transpose.mulVec (M.mulVec z) = z := by
    rw [Matrix.mulVec_mulVec, hM.2, Matrix.one_mulVec]
  have hg' : HasFDerivAt g (D.mulVecLin.toContinuousLinearMap)
      (M.transpose.mulVec (M.mulVec z)) := by
    rwa [hpoint]
  -- chain rule, and the composed derivative is the matrix product
  have hder : (D.mulVecLin.toContinuousLinearMap).comp
      (M.transpose.mulVecLin.toContinuousLinearMap)
      = (D * M.transpose).mulVecLin.toContinuousLinearMap := by
    ext w
    simp [Matrix.mulVecLin_mul]
  rw [← hder]
  exact hg'.comp (M.mulVec z) hlin

/-! ## Candidate latents, candidate observations and the criterion -/

/-- The observations as a function of the candidate latents `z̃ = R h(z)`
under the premise `h(z) = Q z`: `x(z̃) = g((R Q)ᵀ z̃)`. -/
def candObs (g : (Fin d → ℝ) → (Fin p → ℝ)) (Q R : Matrix (Fin d) (Fin d) ℝ) :
    (Fin d → ℝ) → (Fin p → ℝ) :=
  fun w => g ((R * Q).transpose.mulVec w)

/-- Entry `(r, j)` of the candidate Jacobian `∂x/∂z̃`, evaluated at the
candidate latents `z̃ = R h(z)` of `z`. -/
def candJac (g : (Fin d → ℝ) → (Fin p → ℝ)) (h : (Fin d → ℝ) → (Fin d → ℝ))
    (Q R : Matrix (Fin d) (Fin d) ℝ) (r : Fin p) (j : Fin d) (z : Fin d → ℝ) : ℝ :=
  fderiv ℝ (candObs g Q R) (R.mulVec (h z)) (Pi.single j 1) r

/-- The support criterion `‖∂x/∂z̃‖₀,μ` with `z̃ = R h(z)`: the number of
entries `(r, j)` of `∂x/∂z̃` that are nonzero on a set of positive `μ`
measure. -/
def criterion (μ : Measure (Fin d → ℝ)) (g : (Fin d → ℝ) → (Fin p → ℝ))
    (h : (Fin d → ℝ) → (Fin d → ℝ)) (Q R : Matrix (Fin d) (Fin d) ℝ) : ℕ :=
  supportCount μ (candJac g h Q R)

section Premise

variable {g : (Fin d → ℝ) → (Fin p → ℝ)} {h : (Fin d → ℝ) → (Fin d → ℝ)}
  {Q R : Matrix (Fin d) (Fin d) ℝ}

/-- Under the premise `h z = Q z`, the candidate latents are `z̃ = (R Q) z`. -/
lemma candLatent_eq (hh : ∀ z, h z = Q.mulVec z) (z : Fin d → ℝ) :
    R.mulVec (h z) = (R * Q).mulVec z := by
  rw [hh, Matrix.mulVec_mulVec]

/-- `candObs` returns the observation: `x(R h(z)) = g(z)` for every `z`.
Together with `candObs_unique`, this justifies the definition of `candObs`.
It is used only to prove `candObs_unique`, and neither lemma is used in the
other proofs. -/
theorem candObs_apply_cand (hQ : IsOrthogonal Q) (hR : IsOrthogonal R)
    (hh : ∀ z, h z = Q.mulVec z) (z : Fin d → ℝ) :
    candObs g Q R (R.mulVec (h z)) = g z := by
  simp only [candObs, candLatent_eq hh, Matrix.mulVec_mulVec, (hR.mul hQ).2,
    Matrix.one_mulVec]

/-- Uniqueness of `x(z̃)`: any map `G` with `G(R h(z)) = g(z)` for all `z` is
`candObs g Q R`. So `candObs` is determined by `g`, `h` and `R`. -/
theorem candObs_unique (hQ : IsOrthogonal Q) (hR : IsOrthogonal R)
    (hh : ∀ z, h z = Q.mulVec z) {G : (Fin d → ℝ) → (Fin p → ℝ)}
    (hG : ∀ z, G (R.mulVec (h z)) = g z) :
    G = candObs g Q R := by
  funext w
  have hw : R.mulVec (h ((R * Q).transpose.mulVec w)) = w := by
    rw [candLatent_eq hh, Matrix.mulVec_mulVec, (hR.mul hQ).1, Matrix.one_mulVec]
  rw [← hw, hG, candObs_apply_cand hQ hR hh]

/-- **Dependency bridge.** Assume `h z = Q z` with `Q` orthogonal, `R`
orthogonal, and `g` differentiable `μ`-a.e. Then for `μ`-a.e.
`z`, the observations are differentiable as a function of the candidate
latents at `z̃ = R h(z)`, and `∂x/∂z̃ = D(z) Qᵀ Rᵀ` entrywise. -/
theorem dependency_bridge {μ : Measure (Fin d → ℝ)}
    (hg : ∀ᵐ z ∂μ, DifferentiableAt ℝ g z)
    (hQ : IsOrthogonal Q) (hh : ∀ z, h z = Q.mulVec z) (hR : IsOrthogonal R) :
    ∀ᵐ z ∂μ, DifferentiableAt ℝ (candObs g Q R) (R.mulVec (h z)) ∧
      ∀ r j, candJac g h Q R r j z =
        ∑ i, depJac g r i z * (Q.transpose * R.transpose) i j := by
  filter_upwards [hg] with z hz
  -- the chain rule at `z`, with `M = R Q` and `z̃ = (R Q) z = R h(z)`
  have hb : HasFDerivAt (candObs g Q R)
      ((Matrix.of (fun r i => depJac g r i z) *
        (R * Q).transpose).mulVecLin.toContinuousLinearMap) (R.mulVec (h z)) := by
    rw [candLatent_eq hh]
    exact dependency_bridge_at (hasFDerivAt_depJac hz) (hR.mul hQ)
  refine ⟨hb.differentiableAt, fun r j => ?_⟩
  rw [candJac, hb.fderiv, LinearMap.coe_toContinuousLinearMap',
    Matrix.mulVecLin_apply, Matrix.mulVec_single_one, Matrix.transpose_mul]
  simp [Matrix.mul_apply]

/-- The criterion in post-bridge form: `‖∂x/∂z̃‖₀,μ = ‖D(·) Qᵀ Rᵀ‖₀,μ`. -/
theorem criterion_eq_supportCount {μ : Measure (Fin d → ℝ)}
    (hg : ∀ᵐ z ∂μ, DifferentiableAt ℝ g z)
    (hQ : IsOrthogonal Q) (hh : ∀ z, h z = Q.mulVec z) (hR : IsOrthogonal R) :
    criterion μ g h Q R = supportCount μ
      (fun r j z => ∑ i, depJac g r i z * (Q.transpose * R.transpose) i j) := by
  refine supportCount_congr_ae μ ?_
  filter_upwards [dependency_bridge hg hQ hh hR] with z hz
  exact hz.2

end Premise

/-- Changing the representation on a `μ`-null set does not change the
criterion. -/
lemma criterion_congr_ae {μ : Measure (Fin d → ℝ)} {g : (Fin d → ℝ) → (Fin p → ℝ)}
    {h h' : (Fin d → ℝ) → (Fin d → ℝ)} (hh' : ∀ᵐ z ∂μ, h z = h' z)
    (Q R : Matrix (Fin d) (Fin d) ℝ) :
    criterion μ g h Q R = criterion μ g h' Q R := by
  refine supportCount_congr_ae μ ?_
  filter_upwards [hh'] with z hz
  intro r j
  simp only [candJac, hz]

end DSReg

end
