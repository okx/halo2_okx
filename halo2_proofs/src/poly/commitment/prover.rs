use rand_core::RngCore;

use super::super::{Coeff, Polynomial};
use super::{Blind, Params};
use crate::plonk::GenericConfig;
use crate::transcript::{EncodedChallenge, TranscriptWrite};

use std::io;

/// Create a polynomial commitment opening proof for the polynomial defined
/// by the coefficients `px`, the blinding factor `blind` used for the
/// polynomial commitment, and the point `x` that the polynomial is
/// evaluated at.
///
/// This function will panic if the provided polynomial is too large with
/// respect to the polynomial commitment parameters.
///
/// **Important:** This function assumes that the provided `transcript` has
/// already seen the common inputs: the polynomial commitment P, the claimed
/// opening v, and the point x. It's probably also nice for the transcript
/// to have seen the elliptic curve description and the URS, if you want to
/// be rigorous.
pub fn create_proof<
    C: GenericConfig,
    E: EncodedChallenge<C>,
    R: RngCore,
    T: TranscriptWrite<C, E>,
>(
    _params: &Params<C>,
    mut _rng: R,
    _transcript: &mut T,
    _p_poly: &Polynomial<C::Scalar, Coeff>,
    _p_blind: Blind<C::Scalar>,
    _x_3: C::Scalar,
) -> io::Result<()> {
    Ok(())
}
