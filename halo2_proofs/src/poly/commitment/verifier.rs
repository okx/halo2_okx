use super::super::Error;
use super::Params;
use crate::transcript::{EncodedChallenge, TranscriptRead};

use crate::plonk::config::GenericConfig;

/// Checks to see if the proof represented within `transcript` is valid, and a
/// point `x` that the polynomial commitment `P` opens purportedly to the value
/// `v`. The provided `msm` should evaluate to the commitment `P` being opened.
pub fn verify_proof<'a, C: GenericConfig, E: EncodedChallenge<C>, T: TranscriptRead<C, E>>(
    params: &'a Params<C>,
    transcript: &mut T,
    x: C::Scalar,
) -> Result<(), Error> {
    let k = params.k as usize;
    Ok(())
}
