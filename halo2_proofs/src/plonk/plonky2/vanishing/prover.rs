use std::iter;

use ff::Field;
use group::Curve;
use rand_core::RngCore;

use super::Argument;
use crate::{
    arithmetic::{eval_polynomial, CurveAffine},
    plonk::{ChallengeX, ChallengeY, Error},
    poly::{
        self,
        commitment::{Blind, Params},
        fri::ProverQuery,
        Coeff, EvaluationDomain, ExtendedLagrangeCoeff, Polynomial,
    },
    transcript::{EncodedChallenge, TranscriptWrite},
};
use crate::plonk::plonky2::{
    GenericConfig2,
};
use plonky2::iop::challenger::Challenger;

pub(in crate::plonk) struct Committed<G: GenericConfig2> {
    random_poly: Polynomial<G::F, Coeff>,
    random_blind: Blind<G::F>,
}

pub(in crate::plonk) struct Constructed<G: GenericConfig2> {
    h_pieces: Vec<Polynomial<G::F, Coeff>>,
    h_blinds: Vec<Blind<G::F>>,
    committed: Committed<G>,
}

pub(in crate::plonk) struct Evaluated<G: GenericConfig2> {
    h_poly: Polynomial<G::F, Coeff>,
    h_blind: Blind<G::F>,
    committed: Committed<G>,
}

impl<G: GenericConfig2> Argument<G> {
    pub(in crate::plonk) fn commit<R: RngCore>(
        domain: &EvaluationDomain<G::F>,
        mut rng: R,
        challenger: &mut Challenger<G::F, G::Hasher>,
        // transcript: &mut T,
    ) -> Result<Committed<G>, Error> {
        // Sample a random polynomial of degree n - 1
        let mut random_poly = domain.empty_coeff();
        for coeff in random_poly.iter_mut() {
            *coeff = G::F::random(&mut rng);
        }
        // Sample a random blinding factor
        let random_blind = Blind(G::F::random(rng));

        // Commit
        // let c = params.commit(&random_poly, random_blind).to_affine();
        // transcript.write_point(c)?;

        Ok(Committed {
            random_poly,
            random_blind,
        })
    }
}

impl<G: GenericConfig2> Committed<G> {
    #[allow(clippy::too_many_arguments)]
    pub(in crate::plonk) fn construct<
        // E: EncodedChallenge<C>,
        Ev: Copy + Send + Sync,
        R: RngCore,
        // T: TranscriptWrite<C, E>,
    >(
        self,
        domain: &EvaluationDomain<G::F>,
        evaluator: poly::Evaluator<Ev, G::F, ExtendedLagrangeCoeff>,
        expressions: impl Iterator<Item = poly::Ast<Ev, G::F, ExtendedLagrangeCoeff>>,
        y: G::F,
        mut rng: R,
        challenger: &mut Challenger<G::F, G::Hasher>,
        // transcript: &mut T,
    ) -> Result<Constructed<G>, Error> {
        // Evaluate the h(X) polynomial's constraint system expressions for the constraints provided
        let h_poly = poly::Ast::distribute_powers(expressions, y); // Fold the gates together with the y challenge
        let h_poly = evaluator.evaluate(&h_poly, domain); // Evaluate the h(X) polynomial

        // Divide by t(X) = X^{params.n} - 1.
        let h_poly = domain.divide_by_vanishing_poly(h_poly);

        // Obtain final h(X) polynomial
        let h_poly = domain.extended_to_coeff(h_poly);

        // Split h(X) up into pieces
        let h_pieces = h_poly
            .chunks_exact(domain.n as usize)
            .map(|v| domain.coeff_from_vec(v.to_vec()))
            .collect::<Vec<_>>();
        drop(h_poly);
        let h_blinds: Vec<_> = h_pieces
            .iter()
            .map(|_| Blind(G::F::random(&mut rng)))
            .collect();

        // Compute commitments to each h(X) piece
        // let h_commitments_projective: Vec<_> = h_pieces
            // .iter()
            // .zip(h_blinds.iter())
            // .map(|(h_piece, blind)| params.commit(h_piece, *blind))
            // .collect();
        // let mut h_commitments = vec![C::identity(); h_commitments_projective.len()];
        // C::Curve::batch_normalize(&h_commitments_projective, &mut h_commitments);
        // let h_commitments = h_commitments;

        // Hash each h(X) piece
        // for c in h_commitments.iter() {
            // challenger.observe_element(c)
            // // transcript.write_point(*c)?;
        // }

        Ok(Constructed {
            h_pieces,
            h_blinds,
            committed: self,
        })
    }
}

impl<G: GenericConfig2> Constructed<G> {
    pub(in crate::plonk) fn evaluate(
        self,
        x: G::F,
        xn: G::F,
        domain: &EvaluationDomain<G::F>,
        // transcript: &mut T,
    ) -> Result<Evaluated<G>, Error> {
        let h_poly = self
            .h_pieces
            .iter()
            .rev()
            .fold(domain.empty_coeff(), |acc, eval| acc * xn + eval);

        let h_blind = self
            .h_blinds
            .iter()
            .rev()
            .fold(Blind(G::F::ZERO), |acc, eval| acc * Blind(xn) + *eval);

        let random_eval = eval_polynomial(&self.committed.random_poly, x);
        // transcript.write_scalar(random_eval)?;

        Ok(Evaluated {
            h_poly,
            h_blind,
            committed: self.committed,
        })
    }
}

impl<G: GenericConfig2> Evaluated<G> {
    pub(in crate::plonk) fn open(
        &self,
        x: G::F,
    ) -> impl Iterator<Item = ProverQuery<'_, G>> + Clone {
        iter::empty()
            .chain(Some(ProverQuery {
                point: x,
                poly: &self.h_poly,
                blind: self.h_blind,
            }))
            .chain(Some(ProverQuery {
                point: x,
                poly: &self.committed.random_poly,
                blind: self.committed.random_blind,
            }))
    }
}
