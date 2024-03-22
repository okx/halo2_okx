use core::fmt::Debug;

use plonky2::hash::hash_types::RichField;
use plonky2::hash::merkle_tree::MerkleCap;
use plonky2::plonk::config::{Hasher, AlgebraicHasher};
use plonky2::field::extension::{Extendable, FieldExtension};
use plonky2::hash::poseidon::PoseidonHash;

use field::goldilocks_field::GoldilocksField;

use crate::plonk::ConstraintSystem;
use crate::poly::{Coeff, EvaluationDomain, ExtendedLagrangeCoeff, LagrangeCoeff, Polynomial};
use crate::poly::fri::FriConfig;

mod prover;
mod verifier;
mod permutation;
mod vanishing;
mod transcript;
mod keygen;

pub use prover::*;
pub use verifier::*;
pub use transcript::*;
pub use keygen::*;

/// These are the public parameters for the polynomial commitment scheme.
#[derive(Clone, Debug)]
pub struct Params {
    pub(crate) k: u32,
    pub(crate) n: u64,
    pub(crate) fri_config: FriConfig,
}

/// This is a verifying key which allows for the verification of proofs for a
/// particular circuit.
#[derive(Clone, Debug)]
pub struct VerifyingKey<C: GenericConfig> {
    domain: EvaluationDomain<C::Scalar>,
    // only use single merkle tree to handle both of fixed and permutation polys
    // constants_sigmas_cap: MerkleCap<G::F, G::Hasher>,
    permutation: permutation::VerifyingKey<C>,
    cs: ConstraintSystem<C::Scalar>,
    /// Cached maximum degree of `cs` (which doesn't change after construction).
    cs_degree: usize,
    /// The representative of this `VerifyingKey` in transcripts.
    transcript_repr: <C::Hasher as Hasher<C::Scalar>>::Hash,
}

/// This is a proving key which allows for the creation of proofs for a
/// particular circuit.
#[derive(Clone, Debug)]
pub struct ProvingKey<C: GenericConfig> {
    vk: VerifyingKey<C>,
    l0: Polynomial<C::Scalar, ExtendedLagrangeCoeff>,
    l_blind: Polynomial<C::Scalar, ExtendedLagrangeCoeff>,
    l_last: Polynomial<C::Scalar, ExtendedLagrangeCoeff>,
    // constants
    fixed_values: Vec<Polynomial<C::Scalar, LagrangeCoeff>>,
    fixed_polys: Vec<Polynomial<C::Scalar, Coeff>>,
    fixed_cosets: Vec<Polynomial<C::Scalar, ExtendedLagrangeCoeff>>,

    // permutation
    permutation: permutation::ProvingKey<C>,
}

/// reimplement new Generic configuration trait.
pub trait GenericConfig:
      Debug + Clone + Sync + Sized + Send + Eq + PartialEq
  {
      /// Main field.
      type Scalar: ff::WithSmallOrderMulGroup<3> + Ord;

      type Hasher: Hasher<Self::Scalar>;
  }

/// Configuration using Poseidon over the Goldilocks field.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct PoseidonGoldilocksConfig;

impl GenericConfig for PoseidonGoldilocksConfig {
    type Scalar = GoldilocksField;
    // type FE = QuadraticExtension<Self::F>;
    // type Hasher = PoseidonHash;
    // type InnerHasher = PoseidonHash;
}
