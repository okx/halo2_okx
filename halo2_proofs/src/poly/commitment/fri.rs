use super::{super::domain::EvaluationDomain, MerkleTree};
use super::{Coeff, LagrangeCoeff, Polynomial};
use crate::helpers::{log2_strict, reverse_index_bits_in_place, transpose};
use crate::plonk::GenericConfig;

/// config for FRI
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct FriConfig {
    /// `rate = 2^{-rate_bits}`.
    pub rate_bits: usize,

    /// Height of Merkle tree caps.
    pub cap_height: usize,

    ///
    pub proof_of_work_bits: u32,

    ///
    pub reduction_strategy: FriReductionStrategy,

    /// Number of query rounds to perform.
    pub num_query_rounds: usize,

    ///
    pub hiding: bool,
}

/// FRI parameters, including generated parameters which are specific to an instance size, in
/// contrast to `FriConfig` which is user-specified and independent of instance size.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct FriParams {
    /// User-specified FRI configuration.
    pub config: FriConfig,

    /// The degree of the purported codeword, measured in bits.
    pub degree_bits: usize,

    /// The arity of each FRI reduction step, expressed as the log2 of the actual arity.
    /// For example, `[3, 2, 1]` would describe a FRI reduction tree with 8-to-1 reduction, then
    /// a 4-to-1 reduction, then a 2-to-1 reduction. After these reductions, the reduced polynomial
    /// is sent directly.
    pub reduction_arity_bits: Vec<usize>,
}

/// A method for deciding what arity to use at each reduction layer.
#[derive(Debug, Clone, Eq, PartialEq)]
pub enum FriReductionStrategy {
    /// Specifies the exact sequence of arities (expressed in bits) to use.
    Fixed(Vec<usize>),

    /// `ConstantArityBits(arity_bits, final_poly_bits)` applies reductions of arity `2^arity_bits`
    /// until the polynomial degree is less than or equal to `2^final_poly_bits` or until any further
    /// `arity_bits`-reduction makes the last FRI tree have height less than `cap_height`.
    /// This tends to work well in the recursive setting, as it avoids needing multiple configurations
    /// of gates used in FRI verification, such as `InterpolationGate`.
    ConstantArityBits(usize, usize),

    /// `MinSize(opt_max_arity_bits)` searches for an optimal sequence of reduction arities, with an
    /// optional max `arity_bits`. If this proof will have recursive proofs on top of it, a max
    /// `arity_bits` of 3 is recommended.
    MinSize(Option<usize>),
}

impl FriReductionStrategy {
    /// The arity of each FRI reduction step, expressed as the log2 of the actual arity.
    pub fn reduction_arity_bits(
        &self,
        mut degree_bits: usize,
        rate_bits: usize,
        cap_height: usize,
        num_queries: usize,
    ) -> Vec<usize> {
        match self {
            FriReductionStrategy::Fixed(reduction_arity_bits) => reduction_arity_bits.to_vec(),
            &FriReductionStrategy::ConstantArityBits(arity_bits, final_poly_bits) => {
                let mut result = Vec::new();
                while degree_bits > final_poly_bits
                    && degree_bits + rate_bits - arity_bits >= cap_height
                {
                    result.push(arity_bits);
                    assert!(degree_bits >= arity_bits);
                    degree_bits -= arity_bits;
                }
                result.shrink_to_fit();
                result
            }
            FriReductionStrategy::MinSize(opt_max_arity_bits) => {
                min_size_arity_bits(degree_bits, rate_bits, num_queries, *opt_max_arity_bits)
            }
        }
    }
}

fn min_size_arity_bits(
    _degree_bits: usize,
    _rate_bits: usize,
    _num_queries: usize,
    _opt_max_arity_bits: Option<usize>,
) -> Vec<usize> {
    todo!()
}

///
#[derive(Debug, Clone)]
pub struct FriOracle<C: GenericConfig> {
    ///
    pub polys: Vec<Polynomial<C::Scalar, Coeff>>,
    ///
    pub merkle_tree: MerkleTree<C::Scalar, C::Hasher>,
    ///
    pub degree_log: usize,
    ///
    pub rate_bits: usize,
    ///
    pub blinding: bool,
}

impl<C: GenericConfig> FriOracle<C> {
    ///
    pub fn from_coeffs(
        polys: Vec<Polynomial<C::Scalar, Coeff>>,
        rate_bits: usize,
        blinding: bool,
        cap_height: usize,
        domain: &EvaluationDomain<C::Scalar>,
    ) -> Self {
        let degree = polys[0].len();
        let lde_values = domain.lde_values(&polys, rate_bits, blinding);

        let mut leaves = transpose(&lde_values);
        reverse_index_bits_in_place(&mut leaves);
        let merkle_tree = MerkleTree::new(leaves, cap_height);

        Self {
            polys,
            merkle_tree,
            degree_log: log2_strict(degree),
            rate_bits,
            blinding,
        }
    }
}
