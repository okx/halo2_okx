use core::fmt::Debug;

use crate::fields::{Field64, GoldilocksField};
use crate::hash::{Hasher, PoseidonHash};
use crate::poly::commitment::{Commitment, MerkleCommitment};

///
pub trait GenericConfig: Debug + Clone + Copy + Sync + Sized + Send + Eq + PartialEq {
    ///
    type Scalar: ff::WithSmallOrderMulGroup<3> + Field64 + Ord;

    /// hash used for proving system
    type Hasher: Hasher<Self::Scalar>;

    /// commitment
    type Commitment: Commitment<Self::Scalar, Self::Hasher>;
}

/// Configuration using Poseidon over the Goldilocks field.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct PoseidonGoldilocksConfig;
impl GenericConfig for PoseidonGoldilocksConfig {
    type Scalar = GoldilocksField;
    type Hasher = PoseidonHash;
    type Commitment = MerkleCommitment<GoldilocksField, PoseidonHash>;
}
