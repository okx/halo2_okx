use crate::fields::Field64;
use core::fmt::Debug;

use crate::hash::Hasher;

// use crate::poly::merkle_tree::MerkleCap;

///
pub trait Commitment<F: Field64, H: Hasher<F>>: Debug + Clone + Copy + Sync + Sized {}

///
pub trait GenericConfig: Debug + Clone + Copy + Sync + Sized + Send + Eq + PartialEq {
    ///
    type Scalar: ff::WithSmallOrderMulGroup<3> + Field64 + Ord;

    /// hash used for proving system
    type Hasher: Hasher<Self::Scalar>;

    /// commitment
    type Commitment: Commitment<Self::Scalar, Self::Hasher>;
}
