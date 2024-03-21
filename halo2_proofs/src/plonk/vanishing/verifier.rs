use std::iter;

use ff::Field;

use crate::{
    plonk::{Error, VerifyingKey},
    poly::{commitment::Params, multiopen::VerifierQuery},
    transcript::{read_n_commitments, EncodedChallenge, TranscriptRead},
};

use super::super::{ChallengeX, ChallengeY};
use super::{Argument, GenericConfig};

pub struct Committed<C: GenericConfig> {
    random_poly_commitment: C::Commitment,
}

pub struct Constructed<C: GenericConfig> {
    h_commitments: Vec<C::Commitment>,
    final_commitment: C::Commitment,
    random_poly_commitment: C::Commitment,
}

pub struct PartiallyEvaluated<C: GenericConfig> {
    h_commitments: Vec<C::Commitment>,
    final_commitment: C::Commitment,
    random_poly_commitment: C::Commitment,
    random_eval: C::Scalar,
}

pub struct Evaluated<C: GenericConfig> {
    h_commitment: C::Commitment,
    random_poly_commitment: C::Commitment,
    expected_h_eval: C::Scalar,
    random_eval: C::Scalar,
}

impl<C: GenericConfig> Argument<C> {
    pub(in crate::plonk) fn read_commitments_before_y<
        E: EncodedChallenge<C>,
        T: TranscriptRead<C, E>,
    >(
        transcript: &mut T,
    ) -> Result<Committed<C>, Error> {
        let random_poly_commitment = transcript.read_commitment()?;

        Ok(Committed {
            random_poly_commitment,
        })
    }
}

impl<C: GenericConfig> Committed<C> {
    pub(in crate::plonk) fn read_commitments_after_y<
        E: EncodedChallenge<C>,
        T: TranscriptRead<C, E>,
    >(
        self,
        vk: &VerifyingKey<C>,
        transcript: &mut T,
    ) -> Result<Constructed<C>, Error> {
        // Obtain a commitment to h(X) in the form of multiple pieces of degree n - 1
        let h_commitments = read_n_commitments(transcript, vk.domain.get_quotient_poly_degree())?;
        let final_commitment = transcript.read_commitment()?;

        Ok(Constructed {
            h_commitments,
            final_commitment,
            random_poly_commitment: self.random_poly_commitment,
        })
    }
}

impl<C: GenericConfig> Constructed<C> {
    pub(in crate::plonk) fn evaluate_after_x<E: EncodedChallenge<C>, T: TranscriptRead<C, E>>(
        self,
        transcript: &mut T,
    ) -> Result<PartiallyEvaluated<C>, Error> {
        let random_eval = transcript.read_scalar()?;

        Ok(PartiallyEvaluated {
            h_commitments: self.h_commitments,
            final_commitment: self.final_commitment,
            random_poly_commitment: self.random_poly_commitment,
            random_eval,
        })
    }
}

impl<C: GenericConfig> PartiallyEvaluated<C> {
    pub(in crate::plonk) fn verify(
        self,
        params: &Params<C>,
        expressions: impl Iterator<Item = C::Scalar>,
        y: ChallengeY<C>,
        xn: C::Scalar,
    ) -> Evaluated<C> {
        let expected_h_eval = expressions.fold(C::Scalar::ZERO, |h_eval, v| h_eval * &*y + &v);
        let expected_h_eval = expected_h_eval * ((xn - C::Scalar::ONE).invert().unwrap());

        // cannot calculate the commitment for the final poly, it must be given by prover
        // let h_commitment =
        // self.h_commitments
        // .iter()
        // .rev()
        // .fold(params.empty_msm(), |mut acc, commitment| {
        // acc.scale(xn);
        // acc.append_term(C::Scalar::ONE, *commitment);
        // acc
        // });

        Evaluated {
            expected_h_eval,
            h_commitment: self.final_commitment,
            random_poly_commitment: self.random_poly_commitment,
            random_eval: self.random_eval,
        }
    }
}

impl<C: GenericConfig> Evaluated<C> {
    pub(in crate::plonk) fn queries<'r>(
        &'r self,
        x: ChallengeX<C>,
    ) -> impl Iterator<Item = VerifierQuery<'r, C>> + Clone {
        iter::empty()
            .chain(Some(VerifierQuery::new_commitment(
                &self.h_commitment,
                *x,
                self.expected_h_eval,
            )))
            .chain(Some(VerifierQuery::new_commitment(
                &self.random_poly_commitment,
                *x,
                self.random_eval,
            )))
    }
}
