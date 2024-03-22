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

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct FriConfig {
    /// `rate = 2^{-rate_bits}`.
    pub rate_bits: usize,

    /// Height of Merkle tree caps.
    pub cap_height: usize,

    pub proof_of_work_bits: u32,

    /// Number of query rounds to perform.
    pub num_query_rounds: usize,
}



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
