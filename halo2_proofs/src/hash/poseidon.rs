use crate::fields::Field64;
use crate::hash::{GenericHashOut, Hasher, PlonkyPermutation};

///
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct PoseidonHash;

pub struct HashOut;

impl<F: Field64> GenericHashOut<F> for HashOut {
    ///
    fn to_bytes(&self) -> Vec<u8> {}

    ///
    fn from_bytes(bytes: &[u8]) -> Self {}

    ///
    fn to_vec(&self) -> Vec<F> {}
}

pub struct PoseidonPermutation;

impl<T: Field64> PlonkyPermutation<T> for PoseidonPermutation {}

impl<F: Field64> Hasher<F> for PoseidonHash {
    const HASH_SIZE: usize = 10;
    type Hash = HashOut;

    type Permutation = PoseidonPermutation;

    fn hash_no_pad(input: &[F]) -> Self::Hash {}
}
