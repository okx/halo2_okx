use plonky2::iop::challenger::Challenger;
use crate::plonk::plonky2::GenericConfig2;
use std::io::{Read, Write};
use ff::FromUniformBytes;
use std::io;


/// We will replace BLAKE2b with an algebraic hash function in a later version.
#[derive(Debug, Clone)]
pub struct PoseidonRead<R: Read, G: GenericConfig2 > {
    challenger: Challenger<G::F, G::Hasher>,
    reader: R,
}

pub struct PoseidonWrite<W: Write, G: GenericConfig2 > {
    challenger: Challenger<G::F, G::Hasher>,
    reader: W,
}


impl<R: Read, G: GenericConfig2, > PoseidonRead<R, G> {
    /// Initialize a transcript given an input buffer.
    pub fn init(reader: R) -> Self {
        PoseidonRead {
            challenger: Challenger::<G::F, G::Hasher>::new(),
            // state: Blake2bParams::new()
                // .hash_length(64)
                // .personal(b"Halo2-Transcript")
                // .to_state(),
            reader,
        }
    }
}

// impl<R: Read, G: GenericConfig2> TranscriptRead<G>
    // for PoseidonRead<R, G>
// where
    // G::F: FromUniformBytes<64>,
// {
    // fn read_point(&mut self) -> io::Result<G::Hasher> {
        // // let mut compressed = C::Repr::default();
        // // self.reader.read_exact(compressed.as_mut())?;
        // // let point: C = Option::from(C::from_bytes(&compressed)).ok_or_else(|| {
            // // io::Error::new(io::ErrorKind::Other, "invalid point encoding in proof")
        // // })?;
        // // self.common_point(point)?;

        // Ok(point)
    // }

    // fn read_scalar(&mut self) -> io::Result<C::Scalar> {
        // let mut data = <C::Scalar as PrimeField>::Repr::default();
        // self.reader.read_exact(data.as_mut())?;
        // let scalar: C::Scalar = Option::from(C::Scalar::from_repr(data)).ok_or_else(|| {
            // io::Error::new(
                // io::ErrorKind::Other,
                // "invalid field element encoding in proof",
            // )
        // })?;
        // self.common_scalar(scalar)?;

        // Ok(scalar)
    // }
// }

// impl<R: Read, C: CurveAffine> Transcript<C, Challenge255<C>> for Blake2bRead<R, C, Challenge255<C>>
// where
    // C::Scalar: FromUniformBytes<64>,
// {
    // fn squeeze_challenge(&mut self) -> Challenge255<C> {
        // self.state.update(&[BLAKE2B_PREFIX_CHALLENGE]);
        // let hasher = self.state.clone();
        // let result: [u8; 64] = hasher.finalize().as_bytes().try_into().unwrap();
        // Challenge255::<C>::new(&result)
    // }

    // fn common_point(&mut self, point: C) -> io::Result<()> {
        // self.state.update(&[BLAKE2B_PREFIX_POINT]);
        // let coords: Coordinates<C> = Option::from(point.coordinates()).ok_or_else(|| {
            // io::Error::new(
                // io::ErrorKind::Other,
                // "cannot write points at infinity to the transcript",
            // )
        // })?;
        // self.state.update(coords.x().to_repr().as_ref());
        // self.state.update(coords.y().to_repr().as_ref());

        // Ok(())
    // }

    // fn common_scalar(&mut self, scalar: C::Scalar) -> io::Result<()> {
        // self.state.update(&[BLAKE2B_PREFIX_SCALAR]);
        // self.state.update(scalar.to_repr().as_ref());

        // Ok(())
    // }
// }
