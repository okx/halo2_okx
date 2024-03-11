use ff::Field;

use plonky2::field::extension::Extendable;
use plonky2::hash::hash_types::RichField;
use plonky2::hash::merkle_tree::MerkleCap;
use plonky2::plonk::config::{GenericConfig, Hasher};

use crate::plonk::ConstraintSystem;
use crate::poly::{Coeff, EvaluationDomain, ExtendedLagrangeCoeff, LagrangeCoeff, Polynomial};

mod prover;
mod verifier;
mod permutation;
// mod vanishing;

pub use prover::*;
pub use verifier::*;

pub trait Scalar<const D: usize>:
    RichField + Extendable<D> + ff::WithSmallOrderMulGroup<3>
{
}

/// This is a verifying key which allows for the verification of proofs for a
/// particular circuit.
#[derive(Clone, Debug)]
pub struct VerifyingKey<F: Scalar<D>, C: GenericConfig<D, F = F>, const D: usize> {
    domain: EvaluationDomain<F>,
    // only use single merkle tree to handle both of fixed and permutation polys
    constants_sigmas_cap: MerkleCap<C::F, C::Hasher>,
    cs: ConstraintSystem<F>,
    /// Cached maximum degree of `cs` (which doesn't change after construction).
    cs_degree: usize,
    /// The representative of this `VerifyingKey` in transcripts.
    transcript_repr: <<C as GenericConfig<D>>::Hasher as Hasher<C::F>>::Hash,
}

pub(crate) struct PermutationProvingKey<F: Scalar<D>, const D: usize> {
    permutations: Vec<Polynomial<F, LagrangeCoeff>>,
    polys: Vec<Polynomial<F, Coeff>>,
    pub(super) cosets: Vec<Polynomial<F, ExtendedLagrangeCoeff>>,
}

/// This is a proving key which allows for the creation of proofs for a
/// particular circuit.
#[derive(Clone, Debug)]
pub struct ProvingKey<F: Scalar<D>, C: GenericConfig<D, F = F>, const D: usize> {
    vk: VerifyingKey<F, C, D>,
    l0: Polynomial<F, ExtendedLagrangeCoeff>,
    l_blind: Polynomial<F, ExtendedLagrangeCoeff>,
    l_last: Polynomial<F, ExtendedLagrangeCoeff>,
    // constants
    fixed_values: Vec<Polynomial<F, LagrangeCoeff>>,
    fixed_polys: Vec<Polynomial<F, Coeff>>,
    fixed_cosets: Vec<Polynomial<F, ExtendedLagrangeCoeff>>,

    // permutation
    permutation: PermutationProvingKey<F, D>,
}
