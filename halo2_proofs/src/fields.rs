//! This module contains goldilocks field and its extension

#[macro_use]
mod macros;

mod goldilocks_field;

pub use goldilocks_field::*;

///
pub trait Field64: ff::Field {
    ///
    const ORDER: u64;

    ///
    fn to_canonical_u64(&self) -> u64;

    ///
    fn from_canonical_u64(n: u64) -> Self;

    ///
    fn from_noncanonical_u64(n: u64) -> Self;

    ///
    fn to_noncanonical_u64(&self) -> u64;
}
