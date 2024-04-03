use std::io;

use crate::maybe_rayon::*;
use core::hint::unreachable_unchecked;
use pasta_curves::arithmetic::CurveAffine;

pub(crate) trait CurveRead: CurveAffine {
    /// Reads a compressed element from the buffer and attempts to parse it
    /// using `from_bytes`.
    fn read<R: io::Read>(reader: &mut R) -> io::Result<Self> {
        let mut compressed = Self::Repr::default();
        reader.read_exact(compressed.as_mut())?;
        Option::from(Self::from_bytes(&compressed))
            .ok_or_else(|| io::Error::new(io::ErrorKind::Other, "invalid point encoding in proof"))
    }
}

impl<C: CurveAffine> CurveRead for C {}

/// Computes `log_2(n)`, panicking if `n` is not a power of two.
pub fn log2_strict(n: usize) -> usize {
    let res = n.trailing_zeros();
    assert!(n.wrapping_shr(res) == 1, "Not a power of two: {n}");
    // Tell the optimizer about the semantics of `log2_strict`. i.e. it can replace `n` with
    // `1 << res` and vice versa.
    assume(n == 1 << res);
    res as usize
}

#[inline(always)]
pub fn assume(p: bool) {
    debug_assert!(p);
    if !p {
        unsafe {
            unreachable_unchecked();
        }
    }
}

pub fn transpose<T: Send + Sync + Copy>(matrix: &[Vec<T>]) -> Vec<Vec<T>> {
    let len = matrix[0].len();
    (0..len)
        .into_par_iter()
        .map(|i| matrix.iter().map(|row| row[i]).collect())
        .collect()
}

// Ensure that SMALL_ARR_SIZE >= 4 * BIG_T_SIZE.
const BIG_T_SIZE: usize = 1 << 14;
const SMALL_ARR_SIZE: usize = 1 << 16;
pub fn reverse_index_bits_in_place<T>(arr: &mut [T]) {
    let n = arr.len();
    let lb_n = log2_strict(n);
    //TODO(fix and optimize it)
    todo!()
}
