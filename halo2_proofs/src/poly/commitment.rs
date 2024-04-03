//! This module contains an implementation of the polynomial commitment scheme
//! described in the [Halo][halo] paper.
//!
//! [halo]: https://eprint.iacr.org/2019/1021

use super::{Coeff, LagrangeCoeff, Polynomial};
use crate::fields::Field64;
use crate::hash::Hasher;

use ff::Field;
use std::marker::PhantomData;
use std::ops::{Add, AddAssign, Mul, MulAssign};

mod fri;
mod merkle_tree;
mod prover;
mod verifier;

use crate::plonk::config::GenericConfig;
pub use fri::*;
pub use merkle_tree::*;
pub use prover::create_proof;
pub use verifier::verify_proof;

use std::io;

///
#[derive(Clone, Debug)]
pub struct Config {
    ///
    pub k: u32,
    ///
    pub fri_config: FriConfig,
}

impl Default for Config {
    fn default() -> Self {
        Self::standard_recursion_config()
    }
}

impl Config {
    /// A typical recursion config, without zero-knowledge, targeting ~100 bit security.
    pub fn standard_recursion_config() -> Self {
        Self {
            k: 22,
            fri_config: FriConfig {
                hiding: true,
                rate_bits: 3,
                cap_height: 4,
                proof_of_work_bits: 16,
                reduction_strategy: FriReductionStrategy::ConstantArityBits(4, 5),
                num_query_rounds: 28,
            },
        }
    }
}

/// These are the public parameters for the polynomial commitment scheme.
#[derive(Clone, Debug)]
pub struct Params<C: GenericConfig> {
    pub(crate) k: u32,
    pub(crate) n: u64,
    pub(crate) fri_params: FriParams,

    _marker: PhantomData<C>,
}

impl<C: GenericConfig> Params<C> {
    /// Initializes parameters for the curve, given a random oracle to draw
    /// points from.
    pub fn new(config: Config) -> Self {
        let k = config.k;
        // This is usually a limitation on the curve, but we also want 32-bit
        // architectures to be supported.
        assert!(k < 32);

        // In src/arithmetic/fields.rs we ensure that usize is at least 32 bits.

        let n: u64 = 1 << k;

        let fri_config = config.fri_config;
        let degree_bits = k as usize;

        let reduction_arity_bits = fri_config.reduction_strategy.reduction_arity_bits(
            degree_bits,
            fri_config.rate_bits,
            fri_config.cap_height,
            fri_config.num_query_rounds,
        );
        let fri_params = FriParams {
            config: fri_config,
            degree_bits: k as usize,
            reduction_arity_bits,
        };

        Params {
            k,
            n,
            fri_params,
            _marker: PhantomData,
        }
    }

    /// This commits to a polynomial using its evaluations over the $2^k$ size
    /// evaluation domain. The commitment will be blinded by the blinding factor
    /// `r`.
    pub fn commit_lagrange_batch(
        &self,
        polys: &Vec<Polynomial<C::Scalar, LagrangeCoeff>>,
        r: Blind<C::Scalar>,
    ) -> C::Commitment {
        todo!()
    }

    ///
    pub fn commit(
        &self,
        polys: &Polynomial<C::Scalar, Coeff>,
        r: Blind<C::Scalar>,
    ) -> C::Commitment {
        todo!()
    }

    ///
    pub fn commit_batch(
        &self,
        polys: &Vec<Polynomial<C::Scalar, Coeff>>,
        r: Blind<C::Scalar>,
    ) -> C::Commitment {
        todo!()
    }

    ///
    pub fn commit_lagrange(
        &self,
        polys: &Polynomial<C::Scalar, LagrangeCoeff>,
        r: Blind<C::Scalar>,
    ) -> C::Commitment {
        todo!()
    }

    /// Writes params to a buffer.
    pub fn write<W: io::Write>(&self, writer: &mut W) -> io::Result<()> {
        writer.write_all(&self.k.to_le_bytes())?;
        Ok(())
    }

    /// Reads params from a buffer.
    pub fn read<R: io::Read>(reader: &mut R) -> io::Result<Self> {
        let mut k = [0u8; 4];
        reader.read_exact(&mut k[..])?;
        let k = u32::from_le_bytes(k);

        let n: u64 = 1 << k;

        // TODO(fix it)
        let fri_params = FriParams {
            config: FriConfig {
                rate_bits: 0,
                proof_of_work_bits: 0,
                cap_height: 0,
                hiding: false,
                num_query_rounds: 0,
                reduction_strategy: FriReductionStrategy::ConstantArityBits(0, 0),
            },
            degree_bits: k as usize,
            reduction_arity_bits: vec![],
        };

        Ok(Params {
            k,
            n,
            fri_params,
            _marker: PhantomData,
        })
    }
}

/// Wrapper type around a blinding factor.
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub struct Blind<F>(pub F);

impl<F: Field> Default for Blind<F> {
    fn default() -> Self {
        Blind(F::ONE)
    }
}

impl<F: Field> Add for Blind<F> {
    type Output = Self;

    fn add(self, rhs: Blind<F>) -> Self {
        Blind(self.0 + rhs.0)
    }
}

impl<F: Field> Mul for Blind<F> {
    type Output = Self;

    fn mul(self, rhs: Blind<F>) -> Self {
        Blind(self.0 * rhs.0)
    }
}

impl<F: Field> AddAssign for Blind<F> {
    fn add_assign(&mut self, rhs: Blind<F>) {
        self.0 += rhs.0;
    }
}

impl<F: Field> MulAssign for Blind<F> {
    fn mul_assign(&mut self, rhs: Blind<F>) {
        self.0 *= rhs.0;
    }
}

impl<F: Field> AddAssign<F> for Blind<F> {
    fn add_assign(&mut self, rhs: F) {
        self.0 += rhs;
    }
}

impl<F: Field> MulAssign<F> for Blind<F> {
    fn mul_assign(&mut self, rhs: F) {
        self.0 *= rhs;
    }
}

#[test]
fn test_commit_lagrange_epaffine() {
    const K: u32 = 6;

    use rand_core::OsRng;

    use crate::fields::GoldilocksField;
    use crate::plonk::PoseidonGoldilocksConfig;
    let config = Config::default();
    config.k = K;
    let params = Params::<PoseidonGoldilocksConfig>::new(config);
    let domain = super::EvaluationDomain::new(1, K);

    let mut a = domain.empty_lagrange();

    for (i, a) in a.iter_mut().enumerate() {
        *a = GoldilocksField::from(i as u64);
    }

    let b = domain.lagrange_to_coeff(a.clone());

    let alpha = Blind(GoldilocksField::random(OsRng));

    assert_eq!(params.commit(&b, alpha), params.commit_lagrange(&a, alpha));
}

#[test]
fn test_commit_lagrange_eqaffine() {
    const K: u32 = 6;

    use rand_core::OsRng;

    use crate::fields::GoldilocksField;
    use crate::plonk::PoseidonGoldilocksConfig;
    let params = Params::<PoseidonGoldilocksConfig>::new(K);
    let domain = super::EvaluationDomain::new(1, K);

    let mut a = domain.empty_lagrange();

    for (i, a) in a.iter_mut().enumerate() {
        *a = GoldilocksField::from(i as u64);
    }

    let b = domain.lagrange_to_coeff(a.clone());

    let alpha = Blind(GoldilocksField::random(OsRng));

    assert_eq!(params.commit(&b, alpha), params.commit_lagrange(&a, alpha));
}

#[test]
fn test_opening_proof() {
    const K: u32 = 6;

    // use ff::Field;
    // use rand_core::OsRng;

    // use super::{
    // commitment::{Blind, Params},
    // EvaluationDomain,
    // };
    // use crate::arithmetic::eval_polynomial;
    // use crate::pasta::{EpAffine, Fq};
    // use crate::transcript::{
    // Challenge255, PoseidonRead, PoseidonWrite, Transcript, TranscriptRead, TranscriptWrite,
    // };

    // let rng = OsRng;
    todo!();

    // let params = Params::<EpAffine>::new(K);
    // let mut params_buffer = vec![];
    // params.write(&mut params_buffer).unwrap();
    // let params: Params<EpAffine> = Params::read::<_>(&mut &params_buffer[..]).unwrap();

    // let domain = EvaluationDomain::new(1, K);

    // let mut px = domain.empty_coeff();

    // for (i, a) in px.iter_mut().enumerate() {
    // *a = Fq::from(i as u64);
    // }

    // let blind = Blind(Fq::random(rng));

    // let p = params.commit(&px, blind).to_affine();

    // let mut transcript = PoseidonWrite::<Vec<u8>, EpAffine, Challenge255<EpAffine>>::init(vec![]);
    // transcript.write_point(p).unwrap();
    // let x = transcript.squeeze_challenge_scalar::<()>();
    // // Evaluate the polynomial
    // let v = eval_polynomial(&px, *x);
    // transcript.write_scalar(v).unwrap();

    // let (proof, ch_prover) = {
    // create_proof(&params, rng, &mut transcript, &px, blind, *x).unwrap();
    // let ch_prover = transcript.squeeze_challenge();
    // (transcript.finalize(), ch_prover)
    // };

    // // Verify the opening proof
    // let mut transcript = PoseidonRead::<&[u8], EpAffine, Challenge255<EpAffine>>::init(&proof[..]);
    // let p_prime = transcript.read_point().unwrap();
    // assert_eq!(p, p_prime);
    // let x_prime = transcript.squeeze_challenge_scalar::<()>();
    // assert_eq!(*x, *x_prime);
    // let v_prime = transcript.read_scalar().unwrap();
    // assert_eq!(v, v_prime);

    // let mut commitment_msm = params.empty_msm();
    // commitment_msm.append_term(Field::ONE, p);
    // let guard = verify_proof(&params, commitment_msm, &mut transcript, *x, v).unwrap();
    // let ch_verifier = transcript.squeeze_challenge();
    // assert_eq!(*ch_prover, *ch_verifier);

    // // Test guard behavior prior to checking another proof
    // {
    // // Test use_challenges()
    // let msm_challenges = guard.clone().use_challenges();
    // assert!(msm_challenges.eval());
}

///
pub trait Commitment<F: Field64, H: Hasher<F>>: std::fmt::Debug + Clone + Sync + Sized {}
