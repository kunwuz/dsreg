import Lake
open Lake DSL

package dsreg where
  leanOptions := #[
    ⟨`autoImplicit, false⟩
  ]

@[default_target]
lean_lib DSReg where

require "leanprover-community" / "mathlib" @ git "v4.28.0"
