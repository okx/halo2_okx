use crate::plonk::plonky2::{GenericConfig2, VerifyingKey};
use std::marker::PhantomData;

mod prover;
mod verifier;

/// A vanishing argument.
pub(crate) struct Argument<G: GenericConfig2> {
    _marker: PhantomData<G>,
}
