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

    /// Returns `n % Self::characteristic()`. May be cheaper than from_noncanonical_u128 when we know
    /// that `n < 2 ** 96`.
    #[inline]
    fn from_noncanonical_u96((n_lo, n_hi): (u64, u32)) -> Self {
        // Default implementation.
        let n: u128 = ((n_hi as u128) << 64) + (n_lo as u128);
        Self::from_noncanonical_u128(n)
    }

    /// Returns `n % Self::characteristic()`.
    fn from_noncanonical_u128(n: u128) -> Self;

    /// Equivalent to *self + x * y, but may be cheaper.
    #[inline]
    fn multiply_accumulate(&self, x: Self, y: Self) -> Self {
        // Default implementation.
        *self + x * y
    }

    /// Equivalent to *self + Self::from_canonical_u64(rhs), but may be cheaper. The caller must
    /// ensure that 0 <= rhs < Self::ORDER. The function may return incorrect results if this
    /// precondition is not met. It is marked unsafe for this reason.
    // TODO: Move to `Field`.
    #[inline]
    unsafe fn add_canonical_u64(&self, rhs: u64) -> Self {
        // Default implementation.
        *self + Self::from_canonical_u64(rhs)
    }

    /// # Safety
    /// Equivalent to *self - Self::from_canonical_u64(rhs), but may be cheaper. The caller must
    /// ensure that 0 <= rhs < Self::ORDER. The function may return incorrect results if this
    /// precondition is not met. It is marked unsafe for this reason.
    // TODO: Move to `Field`.
    #[inline]
    unsafe fn sub_canonical_u64(&self, rhs: u64) -> Self {
        // Default implementation.
        *self - Self::from_canonical_u64(rhs)
    }
}
