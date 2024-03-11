//! This module contains an implementation of the FRI commitment opening
//! scheme described in the paper.
//!

mod prover;
mod verifier;

pub use prover::create_proof;
pub use verifier::verify_proof;
