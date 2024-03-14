use crate::poly::{ Coeff, ExtendedLagrangeCoeff, LagrangeCoeff, Polynomial };
use crate::plonk::{ plonky2::GenericConfig2, circuit::{Any, Column} };

// pub(crate) mod keygen;
pub(crate) mod prover;
// pub(crate) mod verifier;

/// The verifying key for a single permutation argument.
#[derive(Clone, Debug)]
pub(crate) struct VerifyingKey<G: GenericConfig2> {
    commitments: Vec<G::F>,
}

/// The proving key for a single permutation argument.
#[derive(Clone, Debug)]
pub(crate) struct ProvingKey<G: GenericConfig2> {
    permutations: Vec<Polynomial<G::F, LagrangeCoeff>>,
    polys: Vec<Polynomial<G::F, Coeff>>,
    pub(super) cosets: Vec<Polynomial<G::F, ExtendedLagrangeCoeff>>,
}
