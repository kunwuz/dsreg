# Lean formalization of the identifiability theorem

This directory contains a Lean 4 proof of the main theorem of the paper, component-wise identifiability without reconstruction. The statement is built from the paper's own objects: the observation map `g`, its dependency Jacobian, the representation `h`, the candidate latents `R h(z)`, the observations as a function of the candidate latents, and the support count of their Jacobian. The proofs contain no `sorry` and use no axioms beyond `propext`, `Classical.choice` and `Quot.sound`, the three that mathlib itself relies on.

## Building

The toolchain is pinned in `lean-toolchain` (Lean 4.28.0), and mathlib is pinned to `v4.28.0` in `lakefile.lean` and `lake-manifest.json`.

```bash
lake exe cache get   # download prebuilt mathlib
lake build           # build the DSReg library
```

## Checking axioms

The sources contain no `sorry`, `admit`, `axiom`, `native_decide`, `implemented_by`, `extern`, `unsafe` or `opaque`. To confirm that the results rest only on the standard axioms, save the following as `Check.lean` next to `lakefile.lean` and run `lake env lean Check.lean`.

```lean
import DSReg

#print axioms DSReg.identifiability
#print axioms DSReg.identifiability_ae
#print axioms DSReg.exists_minimizer
#print axioms DSReg.Sanity.nested_certificate
#print axioms DSReg.Sanity.boundary_certificate
```

Each line should report `[propext, Classical.choice, Quot.sound]`. Every other declaration in the `DSReg` namespace depends on these three axioms or a subset of them.

## Files

| File | Content |
|---|---|
| `DSReg/Basic.lean` | Orthogonal matrices, signed permutation matrices, mixing sets `𝓘ⱼ(U)` |
| `DSReg/Minimality.lean` | Pattern count, Structural Diversity, the support-minimality lemma |
| `DSReg/Support.lean` | Support count `‖·‖₀,μ`, footprints, Functional no-cancellation, the union-support lemma |
| `DSReg/Bridge.lean` | Dependency Jacobian, candidate observations and their Jacobian, the dependency bridge, the support criterion |
| `DSReg/Identifiability.lean` | The identifiability theorem and the existence of minimizers |
| `DSReg/Sanity.lean` | Two certificates on concrete instances, described below |

## Correspondence with the paper

All names are in the namespace `DSReg`. The union-support lemma and the support-minimality lemma are in the appendix of the paper.

| Paper | Lean |
|---|---|
| Latents `z ∈ ℝᵈ` with law `μ` | `z : Fin d → ℝ` and `μ : Measure (Fin d → ℝ)`, an arbitrary measure |
| `x = g(z)`, with `g` differentiable `μ`-a.e. | `g : (Fin d → ℝ) → (Fin p → ℝ)` with the hypothesis `∀ᵐ z ∂μ, DifferentiableAt ℝ g z` |
| Dependency Jacobian `D(z) = ∂x/∂z` | `depJac g r i z := fderiv ℝ g z (Pi.single i 1) r`, the matrix of `fderiv ℝ g z` (`toMatrix'_fderiv`) |
| `‖M(·)‖₀,μ` | `supportCount μ M`, the number of entries that are not `μ`-a.e. zero |
| Footprint `𝓢ᵢ` | `footprint μ (depJac g) i` |
| Active set `I_r` | `activeSet μ (depJac g) r` |
| Structural Diversity | `StructuralDiversity (footprint μ (depJac g))` |
| Functional no-cancellation | `NoCancellation μ (depJac g)` |
| `O(d)` | `IsOrthogonal` |
| LeJEPA premise `h(z) = Qz` with `Q ∈ O(d)` | hypotheses `∀ z, h z = Q.mulVec z` (or `∀ᵐ z ∂μ, h z = Q.mulVec z`) and `IsOrthogonal Q` |
| Candidate latents `z̃ = R h(z)` | `R.mulVec (h z)` |
| `x` as a function of `z̃` | `candObs g Q R := fun w => g ((R * Q).transpose.mulVec w)` |
| `∂x/∂z̃` at `z̃ = R h(z)` | `candJac g h Q R r j z := fderiv ℝ (candObs g Q R) (R.mulVec (h z)) (Pi.single j 1) r` |
| Dependency bridge | `dependency_bridge` (`μ`-a.e.), derived from `dependency_bridge_at` (the chain rule at one point, in matrix form) |
| Support criterion `‖∂x/∂z̃‖₀,μ` | `criterion μ g h Q R := supportCount μ (candJac g h Q R)`, minimized over `R` with `IsOrthogonal R` |
| Minimizers over `O(d)` exist | `exists_minimizer` |
| Union-support lemma | `colSupport_eq_biUnion`, and its consequence `supportCount_mul_eq_patternCount` |
| Support-minimality lemma | `isSignedPerm_of_patternCount_le` |
| Signed permutation | `IsSignedPerm` |
| Definition of recovering individual latents | second conjunct of the conclusions of `identifiability` and `identifiability_ae` |
| Component-wise identifiability without reconstruction | `identifiability`, and `identifiability_ae` for the almost-sure premise |

The candidate observation map is written in closed form with `Q`, because under the premise the map `z ↦ R h(z)` is the linear bijection `z ↦ (RQ) z`. `candObs_apply_cand` shows that `candObs g Q R (R h(z)) = g(z)` for every `z`, and `candObs_unique` shows that any map `G` with `G(R h(z)) = g(z)` for every `z` equals `candObs g Q R`. The map is therefore fixed by `g`, `h` and `R`, and it is the map `z̃ ↦ x` of the paper.

Five lemmas in this section are not used in the proofs of the main results or of the certificates. `candObs_apply_cand` and `candObs_unique` justify the definition of `candObs`, and the first is used only to prove the second. `isOrthogonal_iff_mem_orthogonalGroup` identifies `IsOrthogonal` with mathlib's `Matrix.orthogonalGroup`. `not_ae_zero_iff_pos_measure` shows that "not `μ`-a.e. zero" is the paper's "nonzero on a set of positive `μ` measure". `measurable_depJac` shows that every entry of `depJac g` is Borel measurable, for every `g`.

## The main results

```lean
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
        ∀ (z : Fin d → ℝ) (i : Fin d), R.mulVec (h z) i = s i * z (π i)

theorem exists_minimizer (μ : Measure (Fin d → ℝ))
    (g : (Fin d → ℝ) → (Fin p → ℝ)) (h : (Fin d → ℝ) → (Fin d → ℝ))
    (Q : Matrix (Fin d) (Fin d) ℝ) :
    ∃ R : Matrix (Fin d) (Fin d) ℝ, IsOrthogonal R ∧
      ∀ R', IsOrthogonal R' → criterion μ g h Q R ≤ criterion μ g h Q R'
```

`identifiability_ae` has the same hypotheses as `identifiability` except that the premise is `∀ᵐ z ∂μ, h z = Q.mulVec z`, and its recovery clause holds for `μ`-a.e. `z`. It changes `h` on a `μ`-null set to the representative `z ↦ Qz`, which leaves the criterion unchanged (`criterion_congr_ae`), and then applies `identifiability`. The paper's proof of the identifiability theorem begins with the same step.

The proof of `identifiability` follows the paper. The dependency bridge rewrites the criterion as `‖D(·) Qᵀ Rᵀ‖₀,μ`, the union-support lemma turns this into the pattern count `∑ⱼ |⋃_{i ∈ 𝓘ⱼ} 𝓢ᵢ|`, and the support-minimality lemma applies after comparing `R` with the competitor `Qᵀ`, whose induced matrix is the identity. The Lean proof of the support-minimality lemma takes the permutation it needs from the determinant expansion of the whole matrix and uses a footprint of minimal cardinality where the paper uses an inclusion-minimal one.

## Certificates

`DSReg/Sanity.lean` checks the theorem on two instances. Both use the standard Gaussian law `γ = N(0, I₂)` on `ℝ²`, which is the law of the latents in the LeJEPA setting, together with `h = id` and `Q = 1`. Every Jacobian entry in these instances is continuous, and a continuous function is `γ`-a.e. zero exactly when it vanishes everywhere (`ae_zero_γ`), so each support is computed pointwise.

`nested_certificate` takes the nonlinear map `g(z) = (z₀, z₁ + z₀³)`, whose Jacobian is `[[1, 0], [3z₀², 1]]`. Its footprints are `𝓢₀ = {0, 1}` and `𝓢₁ = {1}`, which are distinct and nested. Every hypothesis of `identifiability` is proved for this instance, a minimizer `R` is obtained from `exists_minimizer`, and `identifiability` itself concludes that `R` is a signed permutation. The hypotheses of the theorem are therefore jointly satisfiable under the paper's own law, in the nested regime as well.

`boundary_certificate` takes the single observation `g(z) = z₁ + z₀³`, for which both latents have footprint `{0}`. Every hypothesis of `identifiability` holds except Structural Diversity, and the rotation `[[3/5, -4/5], [4/5, 3/5]]` minimizes the criterion without being a signed permutation. Without Structural Diversity the conclusion of the theorem fails.

## Scope

The only input that is not proved here is the LeJEPA premise. The linear identifiability theorem of Klindt, LeCun and Balestriero (2026) gives `h(z) = Qz` almost surely for some `Q ∈ O(d)`. It enters as the hypotheses `hQ` and `hh`, for every `z` in `identifiability` (the form stated in the dependency bridge) and `μ`-almost everywhere in `identifiability_ae` (the form their theorem provides).

Everything else in the statement of the identifiability theorem is defined from `g`, `h`, `μ`, `Q` and `R` and proved. `Q` is an argument of `candObs` and `criterion` because the map `z̃ ↦ x` is written in closed form with it. Under the premise of `identifiability` this adds no freedom: column `j` of `Q` is `h(eⱼ)`, so `Q` is determined by `h`. Under the almost-sure premise of `identifiability_ae`, `Q` is determined by `h` whenever every nonempty open set has positive `μ` measure, as for the standard Gaussian. For a degenerate `μ`, such as a point mass, several `Q` can satisfy the almost-sure premise, and the criterion is then taken relative to the given one.

The law `μ` and the map `g` are more general than in the paper. The law `μ` is any measure, and the standard Gaussian of the LeJEPA setting is one case. The map `g` needs no continuity, no measurability and no differentiability outside a `μ`-null set. The paper also assumes that the Jacobian is measurable, and in Lean this holds for every `g` (`measurable_depJac`). Where `g` is not differentiable, mathlib sets `fderiv` to zero, and these points form a `μ`-null set under the hypothesis on `g`. The recovery `z̃ᵢ = sᵢ z_{π(i)}` in `identifiability` holds for every `z`, which implies the almost-sure recovery required by the paper's definition of recovering individual latents.
