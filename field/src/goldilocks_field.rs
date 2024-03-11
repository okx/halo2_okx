use core::ops::{Add, Mul, Neg, Sub};
use ff::{Field, WithSmallOrderMulGroup};
use plonky2_field::goldilocks_field::GoldilocksField as GoldilocksFieldInner;
use plonky2_field::types::{Field as Plonky2_Field, Field64, PrimeField as Plonky2_PrimeField};
use rand::RngCore;

use subtle::{Choice, ConditionallySelectable, ConstantTimeEq, CtOption};

#[derive(Copy, Clone, Default, Debug)]
pub struct GoldilocksField(pub GoldilocksFieldInner);

impl From<bool> for GoldilocksField {
    fn from(bit: bool) -> GoldilocksField {
        if bit {
            GoldilocksField::one()
        } else {
            GoldilocksField::zero()
        }
    }
}

impl From<u64> for GoldilocksField {
    fn from(val: u64) -> GoldilocksField {
        GoldilocksField(GoldilocksFieldInner::from_noncanonical_u64(val))
    }
}

impl PartialEq for GoldilocksField {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.ct_eq(other).unwrap_u8() == 1
    }
}

impl Eq for GoldilocksField {}

impl core::cmp::Ord for GoldilocksField {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.0 .0.cmp(&other.0 .0)
    }
}

impl core::cmp::PartialOrd for GoldilocksField {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl ConditionallySelectable for GoldilocksField {
    fn conditional_select(a: &Self, b: &Self, choice: Choice) -> Self {
        GoldilocksField(GoldilocksFieldInner::from_noncanonical_u64(
            u64::conditional_select(&a.0 .0, &b.0 .0, choice),
        ))
    }
}

impl<'a> Neg for &'a GoldilocksField {
    type Output = GoldilocksField;

    #[inline]
    fn neg(self) -> GoldilocksField {
        GoldilocksField(-self.0)
    }
}

impl Neg for GoldilocksField {
    type Output = GoldilocksField;

    #[inline]
    fn neg(self) -> GoldilocksField {
        -&self
    }
}

impl<'a, 'b> Sub<&'b GoldilocksField> for &'a GoldilocksField {
    type Output = GoldilocksField;

    #[inline]
    fn sub(self, rhs: &'b GoldilocksField) -> GoldilocksField {
        GoldilocksField(self.0 - rhs.0)
    }
}

impl<'a, 'b> Add<&'b GoldilocksField> for &'a GoldilocksField {
    type Output = GoldilocksField;

    #[inline]
    fn add(self, rhs: &'b GoldilocksField) -> GoldilocksField {
        GoldilocksField(self.0 + rhs.0)
    }
}

impl<'a, 'b> Mul<&'b GoldilocksField> for &'a GoldilocksField {
    type Output = GoldilocksField;

    #[inline]
    fn mul(self, rhs: &'b GoldilocksField) -> GoldilocksField {
        GoldilocksField(self.0 * rhs.0)
    }
}

///////////////////////////////////////////////////////
impl_binops_additive!(GoldilocksField, GoldilocksField);
impl_binops_multiplicative!(GoldilocksField, GoldilocksField);

impl ConstantTimeEq for GoldilocksField {
    fn ct_eq(&self, other: &Self) -> Choice {
        self.0 .0.ct_eq(&other.0 .0)
    }
}

impl<T: ::core::borrow::Borrow<GoldilocksField>> ::core::iter::Sum<T> for GoldilocksField {
    fn sum<I: Iterator<Item = T>>(iter: I) -> Self {
        iter.fold(Self::ZERO, |acc, item| acc + item.borrow())
    }
}

impl<T: ::core::borrow::Borrow<GoldilocksField>> ::core::iter::Product<T> for GoldilocksField {
    fn product<I: Iterator<Item = T>>(iter: I) -> Self {
        iter.fold(Self::ONE, |acc, item| acc * item.borrow())
    }
}

impl GoldilocksField {
    /// Returns zero, the additive identity.
    #[inline]
    pub const fn zero() -> GoldilocksField {
        GoldilocksField(GoldilocksFieldInner::ZERO)
    }

    /// Returns one, the multiplicative identity.
    #[inline]
    pub const fn one() -> GoldilocksField {
        GoldilocksField(GoldilocksFieldInner::ONE)
    }

    /// Converts from an integer represented in little endian
    /// into its (congruent) `Fp` representation.
    pub const fn from_raw(val: u64) -> Self {
        GoldilocksField(GoldilocksFieldInner(val))
    }
}

impl ff::Field for GoldilocksField {
    const ZERO: Self = Self::zero();
    const ONE: Self = Self::one();

    fn random(mut rng: impl RngCore) -> Self {
        use rand::Rng;
        GoldilocksField(GoldilocksFieldInner::from_canonical_u64(
            rng.gen_range(0..GoldilocksFieldInner::ORDER),
        ))
    }

    fn double(&self) -> Self {
        self * self
    }

    #[inline(always)]
    fn square(&self) -> Self {
        use plonky2_field::ops::Square;
        GoldilocksField(self.0.square())
    }

    fn sqrt_ratio(num: &Self, div: &Self) -> (Choice, Self) {
        (0.into(), GoldilocksField::ZERO)
        // #[cfg(feature = "sqrt-table")]
        // {
        // FP_TABLES.sqrt_ratio(num, div)
        // }

        // #[cfg(not(feature = "sqrt-table"))]
        // ff::helpers::sqrt_ratio_generic(num, div)
    }

    #[cfg(feature = "sqrt-table")]
    fn sqrt_alt(&self) -> (Choice, Self) {
        FP_TABLES.sqrt_alt(self)
    }

    /// Computes the square root of this element, if it exists.
    fn sqrt(&self) -> CtOption<Self> {
        let res = self.0.sqrt().map(GoldilocksField);
        CtOption::new(res.unwrap(), 1.into())
        // #[cfg(feature = "sqrt-table")]
        // {
        // let (is_square, res) = FP_TABLES.sqrt_alt(self);
        // CtOption::new(res, is_square)
        // }

        // #[cfg(not(feature = "sqrt-table"))]
        // ff::helpers::sqrt_tonelli_shanks(self, &T_MINUS1_OVER2)
    }

    /// Computes the multiplicative inverse of this element,
    /// failing if the element is zero.
    fn invert(&self) -> CtOption<Self> {
        let tmp = self.pow_vartime(&[
            0x992d30ecffffffff,
            0x224698fc094cf91b,
            0x0,
            0x4000000000000000,
        ]);

        CtOption::new(tmp, !self.ct_eq(&Self::zero()))
    }

    fn pow_vartime<S: AsRef<[u64]>>(&self, exp: S) -> Self {
        let mut res = Self::one();
        let mut found_one = false;
        for e in exp.as_ref().iter().rev() {
            for i in (0..64).rev() {
                if found_one {
                    res = res.square();
                }

                if ((*e >> i) & 1) == 1 {
                    found_one = true;
                    res *= self;
                }
            }
        }
        res
    }
}

// const MODULUS: GoldilocksField = GoldilocksField(GoldilocksFieldInner(0xFFFFFFFF00000001));
const GENERATOR: GoldilocksField =
    GoldilocksField(GoldilocksFieldInner::MULTIPLICATIVE_GROUP_GENERATOR);

const S: u32 = 32;
/// GENERATOR^t where t * 2^s + 1 = p
/// with t odd. In other words, this
/// is a 2^s root of unity.
const ROOT_OF_UNITY: GoldilocksField =
    GoldilocksField(GoldilocksFieldInner::POWER_OF_TWO_GENERATOR);

/// GENERATOR^{2^s} where t * 2^s + 1 = p
/// with t odd. In other words, this
/// is a t root of unity.
const DELTA: GoldilocksField = GoldilocksField(GoldilocksFieldInner(12275445934081160404));

impl ff::PrimeField for GoldilocksField {
    type Repr = [u8; 8];

    const MODULUS: &'static str = "0xffffffff00000001";
    // TODO(fix)
    const TWO_INV: Self = GoldilocksField::ZERO;

    const NUM_BITS: u32 = 64;
    const CAPACITY: u32 = 63;
    const MULTIPLICATIVE_GENERATOR: Self = GENERATOR;
    const S: u32 = S;
    const ROOT_OF_UNITY: Self = ROOT_OF_UNITY;
    const ROOT_OF_UNITY_INV: Self = GoldilocksField::from_raw(0);
    const DELTA: Self = DELTA;

    fn from_u128(v: u128) -> Self {
        GoldilocksField(GoldilocksFieldInner::from_noncanonical_u128(v))
    }

    fn from_repr(repr: Self::Repr) -> CtOption<Self> {
        let tmp = u64::from_le_bytes(repr[0..8].try_into().unwrap());
        let tmp = GoldilocksField(GoldilocksFieldInner::from_canonical_u64(tmp));

        CtOption::new(tmp, Choice::from(1))
    }

    fn to_repr(&self) -> Self::Repr {
        let mut res = [0; 8];
        res[0..8].copy_from_slice(&self.0 .0.to_le_bytes());
        res
    }

    fn is_odd(&self) -> Choice {
        Choice::from(self.to_repr()[0] & 1)
    }
}

impl WithSmallOrderMulGroup<3> for GoldilocksField {
    const ZETA: Self = GoldilocksField::from_raw(18446744065119617025);
}

#[cfg(test)]
mod tests {}
