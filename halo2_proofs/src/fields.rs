//! This module contains goldilocks field and its extension

mod goldilocks_field;

pub use goldilocks_field::*;

///
pub trait Field64: ff::Field {
    ///
    fn to_canonical_u64(&self) -> u64;

    ///
    fn from_canonical_u64(n: u64) -> Self;
}
