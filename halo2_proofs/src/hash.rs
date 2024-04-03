//! This module contains Hash function

use crate::fields::Field64;
use core::fmt::Debug;
use ff::Field;

///
pub mod poseidon;
///
pub mod poseidon_goldilocks;

pub use poseidon::*;

///
pub trait GenericHashOut<F: Field>: Copy + Clone + Debug + Eq + PartialEq + Send + Sync {
    ///
    fn to_bytes(&self) -> Vec<u8>;

    ///
    fn from_bytes(bytes: &[u8]) -> Self;

    ///
    fn to_vec(&self) -> Vec<F>;

    ///
    fn from_vec(elements: Vec<F>) -> Self;
}

///
pub trait PlonkyPermutation<T: Copy + Default>:
    AsRef<[T]> + Copy + Debug + Default + Eq + Sync + Send
{
    ///
    const RATE: usize;
    ///
    const WIDTH: usize;

    ///
    fn new<I: IntoIterator<Item = T>>(iter: I) -> Self;

    ///
    fn set_from_slice(&mut self, elts: &[T], start_idx: usize);

    ///
    fn set_from_iter<I: IntoIterator<Item = T>>(&mut self, elts: I, start_idx: usize);

    ///
    fn permute(&mut self);

    ///
    fn squeeze(&self) -> &[T];
}

///
pub trait Hasher<F: Field64>: Sized + Copy + Debug + Eq + PartialEq {
    ///
    const HASH_SIZE: usize;

    ///
    type Hash: GenericHashOut<F>;

    ///
    type Permutation: PlonkyPermutation<F>;

    ///
    fn hash_no_pad(input: &[F]) -> Self::Hash;

    ///
    fn hash_pad(input: &[F]) -> Self::Hash {
        let mut padded_input = input.to_vec();
        padded_input.push(F::ONE);
        while (padded_input.len() + 1) % Self::Permutation::WIDTH != 0 {
            padded_input.push(F::ZERO);
        }
        padded_input.push(F::ONE);
        Self::hash_no_pad(&padded_input)
    }

    ///
    fn hash_or_noop(inputs: &[F]) -> Self::Hash {
        if inputs.len() * 8 < Self::HASH_SIZE {
            let mut inputs_bytes = vec![0u8; Self::HASH_SIZE];
            for i in 0..inputs.len() {
                inputs_bytes[i * 8..(i + 1) * 8]
                    .copy_from_slice(&inputs[i].to_canonical_u64().to_le_bytes());
            }
            Self::Hash::from_bytes(&inputs_bytes)
        } else {
            Self::hash_no_pad(inputs)
        }
    }

    ///
    fn two_to_one(left: Self::Hash, right: Self::Hash) -> Self::Hash;
}
