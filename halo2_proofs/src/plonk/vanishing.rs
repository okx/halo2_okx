use std::marker::PhantomData;

use crate::plonk::config::GenericConfig;

mod prover;
mod verifier;

/// A vanishing argument.
pub(crate) struct Argument<C: GenericConfig> {
    _marker: PhantomData<C>,
}
