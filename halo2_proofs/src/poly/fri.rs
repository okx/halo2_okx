//! This module contains an implementation of the FRI commitment opening
//! scheme described in the paper.
//!

mod prover;
mod verifier;

pub use prover::create_proof;
pub use verifier::verify_proof;

use crate::poly::{
        Coeff, Polynomial, commitment
    };

use crate::plonk::plonky2::GenericConfig2;



/// A polynomial query at a point
#[derive(Debug, Clone)]
pub struct ProverQuery<'a, G: GenericConfig2> {
    /// point at which polynomial is queried
    pub point: G::F,
    /// coefficients of polynomial
    pub poly: &'a Polynomial<G::F, Coeff>,
    /// blinding factor of polynomial
    pub blind: commitment::Blind<G::F>,
}
