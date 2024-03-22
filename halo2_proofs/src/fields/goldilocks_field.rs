use ff::{Field, PrimeField, WithSmallOrderMulGroup};

///
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct GoldilocksField;

impl GoldilocksField {}

impl Field for GoldilocksField {}

impl PrimeField for GoldilocksField {}

impl WithSmallOrderMulGroup<3> for GoldilocksField {
    const ZETA: GoldilocksField = GoldilocksField;
}
