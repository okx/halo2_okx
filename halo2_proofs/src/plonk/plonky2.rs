use core::fmt::Debug;

use plonky2::hash::hash_types::RichField;
use plonky2::hash::merkle_tree::MerkleCap;
use plonky2::plonk::config::{Hasher, AlgebraicHasher};
use plonky2::field::extension::{Extendable, FieldExtension};

use crate::plonk::ConstraintSystem;
use crate::poly::{Coeff, EvaluationDomain, ExtendedLagrangeCoeff, LagrangeCoeff, Polynomial};

mod prover;
mod verifier;
mod permutation;
mod vanishing;
mod transcript;

pub use prover::*;
pub use verifier::*;
pub use transcript::*;

/// This is a verifying key which allows for the verification of proofs for a
/// particular circuit.
#[derive(Clone, Debug)]
pub struct VerifyingKey<G: GenericConfig2> {
    domain: EvaluationDomain<G::F>,
    // only use single merkle tree to handle both of fixed and permutation polys
    constants_sigmas_cap: MerkleCap<G::F, G::Hasher>,
    cs: ConstraintSystem<G::F>,
    /// Cached maximum degree of `cs` (which doesn't change after construction).
    cs_degree: usize,
    /// The representative of this `VerifyingKey` in transcripts.
    transcript_repr: <G::Hasher as Hasher<G::F>>::Hash,
}

// pub(crate) struct PermutationProvingKey<F: Scalar<D>, const D: usize> {
    // permutations: Vec<Polynomial<F, LagrangeCoeff>>,
    // polys: Vec<Polynomial<F, Coeff>>,
    // pub(super) cosets: Vec<Polynomial<F, ExtendedLagrangeCoeff>>,
// }

/// This is a proving key which allows for the creation of proofs for a
/// particular circuit.
#[derive(Clone, Debug)]
pub struct ProvingKey<G: GenericConfig2> {
    vk: VerifyingKey<G>,
    l0: Polynomial<G::F, ExtendedLagrangeCoeff>,
    l_blind: Polynomial<G::F, ExtendedLagrangeCoeff>,
    l_last: Polynomial<G::F, ExtendedLagrangeCoeff>,
    // constants
    fixed_values: Vec<Polynomial<G::F, LagrangeCoeff>>,
    fixed_polys: Vec<Polynomial<G::F, Coeff>>,
    fixed_cosets: Vec<Polynomial<G::F, ExtendedLagrangeCoeff>>,

    // permutation
    permutation: permutation::ProvingKey<G>,
}

/// reimplement new Generic configuration trait.
pub trait GenericConfig<const D: usize>:
      Debug + Clone + Sync + Sized + Send + Eq + PartialEq
  {
      /// Main field.
      type F: RichField + Extendable<D, Extension = Self::FE> + ff::WithSmallOrderMulGroup<3>;
      /// Field extension of degree D of the main field.
      type FE: FieldExtension<D, BaseField = Self::F>;
      /// Hash function used for building Merkle trees.
      type Hasher: Hasher<Self::F>;
      /// Algebraic hash function used for the challenger and hashing public inputs.
      type InnerHasher: AlgebraicHasher<Self::F>;
  }

pub trait GenericConfig2: GenericConfig<2>{}
