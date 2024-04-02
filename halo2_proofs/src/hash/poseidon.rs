use crate::fields::Field64;
use crate::hash::{GenericHashOut, Hasher, PlonkyPermutation};
use ff::Field;

///
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct PoseidonHash;

///
pub const NUM_HASH_OUT_ELTS: usize = 4;

///
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct HashOut<F: Field> {
    ///
    pub elements: [F; NUM_HASH_OUT_ELTS],
}

impl<F: Field64> GenericHashOut<F> for HashOut<F> {
    ///
    fn to_bytes(&self) -> Vec<u8> {
        self.elements
            .into_iter()
            .flat_map(|x| x.to_canonical_u64().to_le_bytes())
            .collect()
    }

    ///
    fn from_bytes(bytes: &[u8]) -> Self {
        HashOut {
            elements: bytes
                .chunks(8)
                .take(NUM_HASH_OUT_ELTS)
                .map(|x| F::from_canonical_u64(u64::from_le_bytes(x.try_into().unwrap())))
                .collect::<Vec<_>>()
                .try_into()
                .unwrap(),
        }
    }

    ///
    fn to_vec(&self) -> Vec<F> {
        self.elements.to_vec()
    }

    // TODO: Switch to a TryFrom impl.
    fn from_vec(elements: Vec<F>) -> Self {
        debug_assert!(elements.len() == NUM_HASH_OUT_ELTS);
        Self {
            elements: elements.try_into().unwrap(),
        }
    }
}

///
pub const SPONGE_RATE: usize = 8;
///
pub const SPONGE_CAPACITY: usize = 4;
///
pub const SPONGE_WIDTH: usize = SPONGE_RATE + SPONGE_CAPACITY;

///
#[derive(Copy, Clone, Default, Debug, PartialEq)]
pub struct PoseidonPermutation<T> {
    pub(super) state: [T; SPONGE_WIDTH],
}

impl<T: Eq> Eq for PoseidonPermutation<T> {}

impl<T> AsRef<[T]> for PoseidonPermutation<T> {
    fn as_ref(&self) -> &[T] {
        &self.state
    }
}

impl<T: Field64> PlonkyPermutation<T> for PoseidonPermutation<T> {
    const RATE: usize = SPONGE_RATE;
    const WIDTH: usize = SPONGE_WIDTH;

    fn new<I: IntoIterator<Item = T>>(elts: I) -> Self {
        let mut perm = Self {
            state: [T::default(); SPONGE_WIDTH],
        };
        perm.set_from_iter(elts, 0);
        perm
    }

    fn set_from_slice(&mut self, elts: &[T], start_idx: usize) {
        self.state[start_idx..].copy_from_slice(elts);
    }

    fn set_from_iter<I: IntoIterator<Item = T>>(&mut self, elts: I, start_idx: usize) {
        for (s, e) in self.state[start_idx..].iter_mut().zip(elts) {
            *s = e;
        }
    }

    fn squeeze(&self) -> &[T] {
        &self.state[..Self::RATE]
    }

    fn permute(&mut self) {
        self.permute();
    }
}

impl<F: Field64> Hasher<F> for PoseidonHash {
    const HASH_SIZE: usize = 10;
    type Hash = HashOut<F>;

    type Permutation = PoseidonPermutation<F>;

    fn hash_no_pad(input: &[F]) -> Self::Hash {
        let mut perm = Self::Permutation::new(core::iter::repeat(F::ZERO));

        for input_chunk in input.chunks(Self::Permutation::RATE) {
            perm.set_from_slice(input_chunk, 0);
            perm.permute();
        }
        let mut outputs = Vec::new();
        loop {
            for &item in perm.squeeze() {
                outputs.push(item);
                if outputs.len() == NUM_HASH_OUT_ELTS {
                    return HashOut::from_vec(outputs);
                }
            }
            perm.permute();
        }
    }

    fn two_to_one(left: Self::Hash, right: Self::Hash) -> Self::Hash {
        debug_assert_eq!(left.elements.len(), NUM_HASH_OUT_ELTS);
        debug_assert_eq!(right.elements.len(), NUM_HASH_OUT_ELTS);
        debug_assert!(Self::Permutation::RATE >= NUM_HASH_OUT_ELTS);

        let mut perm = Self::Permutation::new(core::iter::repeat(F::ZERO));
        perm.set_from_slice(&left.elements, 0);
        perm.set_from_slice(&right.elements, NUM_HASH_OUT_ELTS);

        perm.permute();
        HashOut {
            elements: perm.squeeze()[..NUM_HASH_OUT_ELTS].try_into().unwrap(),
        }
    }
}
