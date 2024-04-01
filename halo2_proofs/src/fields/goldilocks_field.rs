use super::Field64;
use num::bigint::BigUint;

impl Field64 for GoldilocksField {
    const ORDER: u64 = 0xFFFFFFFF00000001;

    #[inline]
    fn to_canonical_u64(&self) -> u64 {
        let mut c = self.0;
        // We only need one condition subtraction, since 2 * ORDER would not fit in a u64.
        if c >= Self::ORDER {
            c -= Self::ORDER;
        }
        c
    }

    #[inline(always)]
    fn from_canonical_u64(n: u64) -> Self {
        debug_assert!(n < Self::ORDER);
        Self(n)
    }

    #[inline]
    fn from_noncanonical_u64(n: u64) -> Self {
        Self(n)
    }

    #[inline]
    fn to_noncanonical_u64(&self) -> u64 {
        self.0
    }
}

use core::fmt;
use core::ops::{Add, Mul, Neg, Sub};

use ff::{Field, FromUniformBytes, PrimeField, WithSmallOrderMulGroup};
use rand::RngCore;
use subtle::{Choice, ConditionallySelectable, ConstantTimeEq, CtOption};

#[cfg(feature = "bits")]
use ff::{FieldBits, PrimeFieldBits};

const EPSILON: u64 = (1 << 32) - 1;
/// A field selected to have fast reduction.
///
/// Its order is 2^64 - 2^32 + 1.
/// ```ignore
/// P = 2**64 - EPSILON
///   = 2**64 - 2**32 + 1
///   = 2**32 * (2**32 - 1) + 1
/// ```
#[derive(Clone, Copy, Eq)]
#[repr(transparent)]
pub struct GoldilocksField(pub u64);

impl fmt::Debug for GoldilocksField {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let tmp = self.to_repr();
        write!(f, "0x")?;
        for &b in tmp.iter().rev() {
            write!(f, "{:02x}", b)?;
        }
        Ok(())
    }
}

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
        GoldilocksField::from_raw(val)
    }
}

impl ConstantTimeEq for GoldilocksField {
    fn ct_eq(&self, other: &Self) -> Choice {
        self.0.ct_eq(&other.0)
    }
}

impl PartialEq for GoldilocksField {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.ct_eq(other).unwrap_u8() == 1
    }
}

impl core::cmp::Ord for GoldilocksField {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        let left = self.to_repr();
        let right = other.to_repr();
        left.iter()
            .zip(right.iter())
            .rev()
            .find_map(|(left_byte, right_byte)| match left_byte.cmp(right_byte) {
                core::cmp::Ordering::Equal => None,
                res => Some(res),
            })
            .unwrap_or(core::cmp::Ordering::Equal)
    }
}

impl core::cmp::PartialOrd for GoldilocksField {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl ConditionallySelectable for GoldilocksField {
    fn conditional_select(a: &Self, b: &Self, choice: Choice) -> Self {
        GoldilocksField(u64::conditional_select(&a.0, &b.0, choice))
    }
}

/// Constant representing the modulus
/// p = 0xFFFFFFFF00000001
const MODULUS: GoldilocksField = GoldilocksField(GoldilocksField::ORDER);

impl<'a> Neg for &'a GoldilocksField {
    type Output = GoldilocksField;

    #[inline]
    fn neg(self) -> GoldilocksField {
        self.neg()
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
        self.sub(rhs)
    }
}

impl<'a, 'b> Add<&'b GoldilocksField> for &'a GoldilocksField {
    type Output = GoldilocksField;

    #[inline]
    fn add(self, rhs: &'b GoldilocksField) -> GoldilocksField {
        self.add(rhs)
    }
}

impl<'a, 'b> Mul<&'b GoldilocksField> for &'a GoldilocksField {
    type Output = GoldilocksField;

    #[inline]
    fn mul(self, rhs: &'b GoldilocksField) -> GoldilocksField {
        self.mul(rhs)
    }
}

impl_binops_additive!(GoldilocksField, GoldilocksField);
impl_binops_multiplicative!(GoldilocksField, GoldilocksField);

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

/// INV = -(p^{-1} mod 2^64) mod 2^64
// const INV: u64 = 0x992d30ecffffffff;

/// `GENERATOR = 5 mod p` is a generator of the `p - 1` order multiplicative
/// subgroup, or in other words a primitive root of the field.
const GENERATOR: GoldilocksField = GoldilocksField::from_raw(7);

const S: u32 = 32;

/// GENERATOR^t where t * 2^s + 1 = p
/// with t odd. In other words, this
/// is a 2^s root of unity.
const ROOT_OF_UNITY: GoldilocksField = GoldilocksField::from_raw(1753635133440165772);

/// GENERATOR^{2^s} where t * 2^s + 1 = p
/// with t odd. In other words, this
/// is a t root of unity.
const DELTA: GoldilocksField = GoldilocksField::from_raw(12275445934081160404);

impl Default for GoldilocksField {
    #[inline]
    fn default() -> Self {
        Self::zero()
    }
}

impl GoldilocksField {
    const NEG_ONE: Self = Self(Self::ORDER - 1);

    /// Returns zero, the additive identity.
    #[inline]
    pub const fn zero() -> GoldilocksField {
        GoldilocksField(0)
    }

    /// Returns one, the multiplicative identity.
    #[inline]
    pub const fn one() -> GoldilocksField {
        GoldilocksField(1)
    }

    /// Doubles this field element.
    #[inline]
    pub const fn double(&self) -> GoldilocksField {
        // TODO: This can be achieved more efficiently with a bitshift.
        self.add(self)
    }

    /// Converts from an integer represented in little endian
    /// into its (congruent) `Fp` representation.
    pub const fn from_raw(val: u64) -> Self {
        Self(val)
    }

    fn exp_power_of_2(&self, power_log: usize) -> Self {
        let mut res = *self;
        for _ in 0..power_log {
            res = res.square();
        }
        res
    }

    /// Squares this element.
    #[cfg_attr(not(feature = "uninline-portable"), inline)]
    pub fn square(&self) -> GoldilocksField {
        self.mul(self)
    }

    /// Multiplies `rhs` by `self`, returning the result.
    #[cfg_attr(not(feature = "uninline-portable"), inline)]
    pub fn mul(&self, rhs: &Self) -> Self {
        reduce128((self.0 as u128) * (rhs.0 as u128))
    }

    /// Subtracts `rhs` from `self`, returning the result.
    #[cfg_attr(not(feature = "uninline-portable"), inline)]
    pub const fn sub(&self, rhs: &Self) -> Self {
        let (diff, under) = self.0.overflowing_sub(rhs.0);
        let (mut diff, under) = diff.overflowing_sub((under as u64) * EPSILON);
        if under {
            // NB: self.0 < EPSILON - 1 && rhs.0 > Self::ORDER is necessary but not sufficient for
            // double-underflow.
            // This assume does two things:
            //  1. If compiler knows that either self.0 >= EPSILON - 1 or rhs.0 <= ORDER, then it
            //     can skip this check.
            //  2. Hints to the compiler how rare this double-underflow is (thus handled better
            //     with a branch).
            // assume(self.0 < EPSILON - 1 && rhs.0 > Self::ORDER);
            // branch_hint();
            diff -= EPSILON; // Cannot underflow.
        }
        Self(diff)
    }

    /// Adds `rhs` to `self`, returning the result.
    #[cfg_attr(not(feature = "uninline-portable"), inline)]
    pub const fn add(&self, rhs: &Self) -> Self {
        let (sum, over) = self.0.overflowing_add(rhs.0);
        let (mut sum, over) = sum.overflowing_add((over as u64) * EPSILON);
        if over {
            // NB: self.0 > Self::ORDER && rhs.0 > Self::ORDER is necessary but not sufficient for
            // double-overflow.
            // This assume does two things:
            //  1. If compiler knows that either self.0 or rhs.0 <= ORDER, then it can skip this
            //     check.
            //  2. Hints to the compiler how rare this double-overflow is (thus handled better with
            //     a branch).
            // assume(self.0 > Self::ORDER && rhs.0 > Self::ORDER);
            // branch_hint();
            sum += EPSILON; // Cannot overflow.
        }
        Self(sum)
    }

    /// Negates `self`.
    #[cfg_attr(not(feature = "uninline-portable"), inline)]
    pub fn neg(&self) -> Self {
        if self.is_zero().into() {
            Self::ZERO
        } else {
            Self(Self::ORDER - self.to_canonical_u64())
        }
    }

    fn to_canonical_biguint(&self) -> BigUint {
        self.to_canonical_u64().into()
    }

    fn is_quadratic_residue(&self) -> bool {
        if self.is_zero().into() {
            return true;
        }
        // This is based on Euler's criterion.
        let power = Self::NEG_ONE.to_canonical_biguint() / 2u8;
        let exp = self.exp_biguint(&power);
        if exp == Self::ONE {
            return true;
        }
        if exp == Self::NEG_ONE {
            return false;
        }
        panic!("Unreachable")
    }

    fn exp_biguint(&self, power: &BigUint) -> Self {
        let mut result = Self::ONE;
        for &digit in power.to_u64_digits().iter().rev() {
            result = result.exp_power_of_2(64);
            result *= self.exp_u64(digit);
        }
        result
    }

    fn exp_u64(&self, power: u64) -> Self {
        let mut current = *self;
        let mut product = Self::ONE;

        for j in 0..bits_u64(power) {
            if (power >> j & 1) != 0 {
                product *= current;
            }
            current = current.square();
        }
        product
    }

    #[inline]
    fn is_zero(&self) -> bool {
        *self == Self::ZERO
    }

    #[inline]
    fn is_one(&self) -> bool {
        *self == Self::ONE
    }
}

impl From<GoldilocksField> for [u8; 8] {
    fn from(value: GoldilocksField) -> [u8; 8] {
        value.to_repr()
    }
}

impl<'a> From<&'a GoldilocksField> for [u8; 8] {
    fn from(value: &'a GoldilocksField) -> [u8; 8] {
        value.to_repr()
    }
}

impl ff::Field for GoldilocksField {
    const ZERO: Self = Self::zero();
    const ONE: Self = Self::one();

    fn random(mut rng: impl RngCore) -> Self {
        use rand::Rng;
        Self::from_canonical_u64(rng.gen_range(0..Self::ORDER))
    }

    fn double(&self) -> Self {
        self.double()
    }

    #[inline(always)]
    fn square(&self) -> Self {
        self.square()
    }

    fn sqrt_ratio(num: &Self, div: &Self) -> (Choice, Self) {
        ff::helpers::sqrt_ratio_generic(num, div)
    }

    #[cfg(feature = "sqrt-table")]
    fn sqrt_alt(&self) -> (Choice, Self) {
        FP_TABLES.sqrt_alt(self)
    }

    /// Computes the square root of this element, if it exists.
    fn sqrt(&self) -> CtOption<Self> {
        if self.is_zero() {
            CtOption::new(*self, Choice::from(1))
        } else if self.is_quadratic_residue() {
            let t = (Self::ORDER - BigUint::from(1u32)) / (BigUint::from(2u32).pow(Self::S));
            let mut z = Self::ROOT_OF_UNITY;
            let mut w = self.exp_biguint(&((t - BigUint::from(1u32)) / BigUint::from(2u32)));
            let mut x = w * *self;
            let mut b = x * w;

            let mut v = Self::S as usize;

            while !b.is_one() {
                let mut k = 0usize;
                let mut b2k = b;
                while !b2k.is_one() {
                    b2k = b2k * b2k;
                    k += 1;
                }
                let j = v - k - 1;
                w = z;
                for _ in 0..j {
                    w = w * w;
                }

                z = w * w;
                b *= z;
                x *= w;
                v = k;
            }
            CtOption::new(x, Choice::from(1))
        } else {
            CtOption::new(*self, Choice::from(0))
        }
    }

    /// Computes the multiplicative inverse of this element,
    /// failing if the element is zero.
    fn invert(&self) -> CtOption<Self> {
        // compute base^(P - 2) using 72 multiplications
        // The exponent P - 2 is represented in binary as:
        // 0b1111111111111111111111111111111011111111111111111111111111111111

        // compute base^11
        let t2 = self.square() * *self;

        // compute base^111
        let t3 = t2.square() * *self;

        // compute base^111111 (6 ones)
        // repeatedly square t3 3 times and multiply by t3
        let t6 = exp_acc::<3>(t3, t3);

        // compute base^111111111111 (12 ones)
        // repeatedly square t6 6 times and multiply by t6
        let t12 = exp_acc::<6>(t6, t6);

        // compute base^111111111111111111111111 (24 ones)
        // repeatedly square t12 12 times and multiply by t12
        let t24 = exp_acc::<12>(t12, t12);

        // compute base^1111111111111111111111111111111 (31 ones)
        // repeatedly square t24 6 times and multiply by t6 first. then square t30 and
        // multiply by base
        let t30 = exp_acc::<6>(t24, t6);
        let t31 = t30.square() * *self;

        // compute base^111111111111111111111111111111101111111111111111111111111111111
        // repeatedly square t31 32 times and multiply by t31
        let t63 = exp_acc::<32>(t31, t31);

        // compute base^1111111111111111111111111111111011111111111111111111111111111111
        CtOption::new(t63.square() * *self, !self.ct_eq(&Self::zero()))
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

impl ff::PrimeField for GoldilocksField {
    type Repr = [u8; 8];

    const MODULUS: &'static str =
        "0x40000000000000000000000000000000224698fc094cf91b992d30ed00000001";
    // TODO(fix it)
    const TWO_INV: Self = GoldilocksField::from_raw(0);
    const NUM_BITS: u32 = 64;
    const CAPACITY: u32 = 63;
    const MULTIPLICATIVE_GENERATOR: Self = GENERATOR;
    const S: u32 = S;
    const ROOT_OF_UNITY: Self = ROOT_OF_UNITY;
    // TODO(fix it)
    const ROOT_OF_UNITY_INV: Self = GoldilocksField::from_raw(0);
    const DELTA: Self = DELTA;

    fn from_u128(v: u128) -> Self {
        reduce128(v)
    }

    fn from_repr(repr: Self::Repr) -> CtOption<Self> {
        let mut tmp = GoldilocksField(0);

        tmp.0 = u64::from_le_bytes(repr[0..8].try_into().unwrap());

        // Try to subtract the modulus
        let (_, borrow) = tmp.0.overflowing_sub(MODULUS.0);

        // If the element is smaller than MODULUS then the
        // subtraction will underflow, producing a borrow value
        // of 0xffff...ffff. Otherwise, it'll be zero.
        let is_some = (borrow as u8) & 1;

        CtOption::new(tmp, Choice::from(is_some))
    }

    fn to_repr(&self) -> Self::Repr {
        // Turn into canonical form by computing

        let mut res = [0; 8];
        res[0..8].copy_from_slice(&self.to_canonical_u64().to_le_bytes());

        res
    }

    fn is_odd(&self) -> Choice {
        Choice::from(self.to_repr()[0] & 1)
    }
}

#[cfg(all(feature = "bits", not(target_pointer_width = "64")))]
type ReprBits = [u32; 8];

#[cfg(all(feature = "bits", target_pointer_width = "64"))]
type ReprBits = [u64; 4];

#[cfg(feature = "bits")]
#[cfg_attr(docsrs, doc(cfg(feature = "bits")))]
impl PrimeFieldBits for GoldilocksField {
    type ReprBits = ReprBits;

    fn to_le_bits(&self) -> FieldBits<Self::ReprBits> {
        let bytes = self.to_repr();

        #[cfg(not(target_pointer_width = "64"))]
        let limbs = [
            u32::from_le_bytes(bytes[0..4].try_into().unwrap()),
            u32::from_le_bytes(bytes[4..8].try_into().unwrap()),
            u32::from_le_bytes(bytes[8..12].try_into().unwrap()),
            u32::from_le_bytes(bytes[12..16].try_into().unwrap()),
            u32::from_le_bytes(bytes[16..20].try_into().unwrap()),
            u32::from_le_bytes(bytes[20..24].try_into().unwrap()),
            u32::from_le_bytes(bytes[24..28].try_into().unwrap()),
            u32::from_le_bytes(bytes[28..32].try_into().unwrap()),
        ];

        #[cfg(target_pointer_width = "64")]
        let limbs = [
            u64::from_le_bytes(bytes[0..8].try_into().unwrap()),
            u64::from_le_bytes(bytes[8..16].try_into().unwrap()),
            u64::from_le_bytes(bytes[16..24].try_into().unwrap()),
            u64::from_le_bytes(bytes[24..32].try_into().unwrap()),
        ];

        FieldBits::new(limbs)
    }

    fn char_le_bits() -> FieldBits<Self::ReprBits> {
        #[cfg(not(target_pointer_width = "64"))]
        {
            FieldBits::new(MODULUS_LIMBS_32)
        }

        #[cfg(target_pointer_width = "64")]
        FieldBits::new(MODULUS.0)
    }
}

// #[cfg(feature = "sqrt-table")]
// lazy_static! {
// // The perfect hash parameters are found by `squareroottab.sage` in zcash/pasta.
// #[cfg_attr(docsrs, doc(cfg(feature = "sqrt-table")))]
// static ref FP_TABLES: SqrtTables<Fp> = SqrtTables::new(0x11BE, 1098);
// }

// impl SqrtTableHelpers for Fp {
// fn pow_by_t_minus1_over2(&self) -> Self {
// let sqr = |x: Fp, i: u32| (0..i).fold(x, |x, _| x.square());

// let r10 = self.square();
// let r11 = r10 * self;
// let r110 = r11.square();
// let r111 = r110 * self;
// let r1001 = r111 * r10;
// let r1101 = r111 * r110;
// let ra = sqr(*self, 129) * self;
// let rb = sqr(ra, 7) * r1001;
// let rc = sqr(rb, 7) * r1101;
// let rd = sqr(rc, 4) * r11;
// let re = sqr(rd, 6) * r111;
// let rf = sqr(re, 3) * r111;
// let rg = sqr(rf, 10) * r1001;
// let rh = sqr(rg, 5) * r1001;
// let ri = sqr(rh, 4) * r1001;
// let rj = sqr(ri, 3) * r111;
// let rk = sqr(rj, 4) * r1001;
// let rl = sqr(rk, 5) * r11;
// let rm = sqr(rl, 4) * r111;
// let rn = sqr(rm, 4) * r11;
// let ro = sqr(rn, 6) * r1001;
// let rp = sqr(ro, 5) * r1101;
// let rq = sqr(rp, 4) * r11;
// let rr = sqr(rq, 7) * r111;
// let rs = sqr(rr, 3) * r11;
// rs.square() // rt
// }

// fn get_lower_32(&self) -> u32 {
// // TODO: don't reduce, just hash the Montgomery form. (Requires rebuilding perfect hash table.)
// let tmp = Fp::montgomery_reduce(self.0[0], self.0[1], self.0[2], self.0[3], 0, 0, 0, 0);

// tmp.0[0] as u32
// }
// }

impl WithSmallOrderMulGroup<3> for GoldilocksField {
    const ZETA: Self = GoldilocksField::from_raw(18446744065119617025);
}

impl FromUniformBytes<8> for GoldilocksField {
    /// Converts a 512-bit little endian integer into
    /// a `Fp` by reducing by the modulus.
    fn from_uniform_bytes(bytes: &[u8; 8]) -> GoldilocksField {
        GoldilocksField::from_raw(u64::from_le_bytes(bytes[0..8].try_into().unwrap()))
    }
}

/// Fast addition modulo ORDER for x86-64.
/// This function is marked unsafe for the following reasons:
///   - It is only correct if x + y < 2**64 + ORDER = 0x1ffffffff00000001.
///   - It is only faster in some circumstances. In particular, on x86 it overwrites both inputs in
///     the registers, so its use is not recommended when either input will be used again.
#[inline(always)]
#[cfg(target_arch = "x86_64")]
unsafe fn add_no_canonicalize_trashing_input(x: u64, y: u64) -> u64 {
    let res_wrapped: u64;
    let adjustment: u64;
    core::arch::asm!(
        "add {0}, {1}",
        // Trick. The carry flag is set iff the addition overflowed.
        // sbb x, y does x := x - y - CF. In our case, x and y are both {1:e}, so it simply does
        // {1:e} := 0xffffffff on overflow and {1:e} := 0 otherwise. {1:e} is the low 32 bits of
        // {1}; the high 32-bits are zeroed on write. In the end, we end up with 0xffffffff in {1}
        // on overflow; this happens be EPSILON.
        // Note that the CPU does not realize that the result of sbb x, x does not actually depend
        // on x. We must write the result to a register that we know to be ready. We have a
        // dependency on {1} anyway, so let's use it.
        "sbb {1:e}, {1:e}",
        inlateout(reg) x => res_wrapped,
        inlateout(reg) y => adjustment,
        options(pure, nomem, nostack),
    );
    // assume(x != 0 || (res_wrapped == y && adjustment == 0));
    // assume(y != 0 || (res_wrapped == x && adjustment == 0));
    // Add EPSILON == subtract ORDER.
    // Cannot overflow unless the assumption if x + y < 2**64 + ORDER is incorrect.
    res_wrapped + adjustment
}

#[inline(always)]
#[cfg(not(target_arch = "x86_64"))]
const unsafe fn add_no_canonicalize_trashing_input(x: u64, y: u64) -> u64 {
    let (res_wrapped, carry) = x.overflowing_add(y);
    // Below cannot overflow unless the assumption if x + y < 2**64 + ORDER is incorrect.
    res_wrapped + EPSILON * (carry as u64)
}

/// Squares the base N number of times and multiplies the result by the tail value.
#[inline(always)]
fn exp_acc<const N: usize>(base: GoldilocksField, tail: GoldilocksField) -> GoldilocksField {
    base.exp_power_of_2(N) * tail
}

/// Reduces to a 64-bit value. The result might not be in canonical form; it could be in between the
/// field order and `2^64`.
#[inline]
fn reduce128(x: u128) -> GoldilocksField {
    let (x_lo, x_hi) = split(x); // This is a no-op
    let x_hi_hi = x_hi >> 32;
    let x_hi_lo = x_hi & EPSILON;

    let (mut t0, borrow) = x_lo.overflowing_sub(x_hi_hi);
    if borrow {
        // branch_hint(); // A borrow is exceedingly rare. It is faster to branch.
        t0 -= EPSILON; // Cannot underflow.
    }
    let t1 = x_hi_lo * EPSILON;
    let t2 = unsafe { add_no_canonicalize_trashing_input(t0, t1) };
    GoldilocksField(t2)
}

#[inline]
const fn split(x: u128) -> (u64, u64) {
    (x as u64, (x >> 64) as u64)
}

#[inline]
const fn bits_u64(n: u64) -> usize {
    (64 - n.leading_zeros()) as usize
}

#[cfg(feature = "gpu")]
impl ec_gpu::GpuName for GoldilocksField {
    fn name() -> alloc::string::String {
        ec_gpu::name!()
    }
}

#[cfg(feature = "gpu")]
impl ec_gpu::GpuField for GoldilocksField {
    fn one() -> alloc::vec::Vec<u32> {
        crate::fields::u64_to_u32(&R.0[..])
    }

    fn r2() -> alloc::vec::Vec<u32> {
        crate::fields::u64_to_u32(&R2.0[..])
    }

    fn modulus() -> alloc::vec::Vec<u32> {
        crate::fields::u64_to_u32(&MODULUS.0[..])
    }
}

// #[test]
// fn test_inv() {
// // Compute -(r^{-1} mod 2^64) mod 2^64 by exponentiating
// // by totient(2**64) - 1

// let mut inv = 1u64;
// for _ in 0..63 {
// inv = inv.wrapping_mul(inv);
// inv = inv.wrapping_mul(MODULUS.0);
// }
// inv = inv.wrapping_neg();

// assert_eq!(inv, INV);
// }

#[test]
fn test_sqrt() {
    // NB: TWO_INV is standing in as a "random" field element
    let v = (GoldilocksField::TWO_INV).square().sqrt().unwrap();
    assert!(v == GoldilocksField::TWO_INV || (-v) == GoldilocksField::TWO_INV);
}

#[test]
fn test_sqrt_32bit_overflow() {
    assert!((GoldilocksField::from(5)).sqrt().is_none().unwrap_u8() == 1);
}

#[test]
fn test_sqrt_ratio_and_alt() {
    // (true, sqrt(num/div)), if num and div are nonzero and num/div is a square in the field
    let num = (GoldilocksField::TWO_INV).square();
    let div = GoldilocksField::from(25);
    let div_inverse = div.invert().unwrap();
    let expected = GoldilocksField::TWO_INV * GoldilocksField::from(7).invert().unwrap();
    let (is_square, v) = GoldilocksField::sqrt_ratio(&num, &div);
    assert!(bool::from(is_square));
    assert!(v == expected || (-v) == expected);

    let (is_square_alt, v_alt) = GoldilocksField::sqrt_alt(&(num * div_inverse));
    assert!(bool::from(is_square_alt));
    assert!(v_alt == v);

    // (false, sqrt(ROOT_OF_UNITY * num/div)), if num and div are nonzero and num/div is a nonsquare in the field
    let num = num * GoldilocksField::ROOT_OF_UNITY;
    let expected = GoldilocksField::TWO_INV
        * GoldilocksField::ROOT_OF_UNITY
        * GoldilocksField::from(5).invert().unwrap();
    let (is_square, v) = GoldilocksField::sqrt_ratio(&num, &div);
    assert!(!bool::from(is_square));
    assert!(v == expected || (-v) == expected);

    let (is_square_alt, v_alt) = GoldilocksField::sqrt_alt(&(num * div_inverse));
    assert!(!bool::from(is_square_alt));
    assert!(v_alt == v);

    // (true, 0), if num is zero
    let num = GoldilocksField::zero();
    let expected = GoldilocksField::zero();
    let (is_square, v) = GoldilocksField::sqrt_ratio(&num, &div);
    assert!(bool::from(is_square));
    assert!(v == expected);

    let (is_square_alt, v_alt) = GoldilocksField::sqrt_alt(&(num * div_inverse));
    assert!(bool::from(is_square_alt));
    assert!(v_alt == v);

    // (false, 0), if num is nonzero and div is zero
    let num = (GoldilocksField::TWO_INV).square();
    let div = GoldilocksField::zero();
    let expected = GoldilocksField::zero();
    let (is_square, v) = GoldilocksField::sqrt_ratio(&num, &div);
    assert!(!bool::from(is_square));
    assert!(v == expected);
}

#[test]
fn test_zeta() {
    assert_eq!(
        format!("{:?}", GoldilocksField::ZETA),
        "0x12ccca834acdba712caad5dc57aab1b01d1f8bd237ad31491dad5ebdfdfe4ab9"
    );

    let a = GoldilocksField::ZETA;
    assert!(a != GoldilocksField::one());
    let b = a * a;
    assert!(b != GoldilocksField::one());
    let c = b * a;
    assert!(c == GoldilocksField::one());
}

#[test]
fn test_root_of_unity() {
    assert_eq!(
        GoldilocksField::ROOT_OF_UNITY.pow_vartime(&[1 << GoldilocksField::S, 0, 0, 0]),
        GoldilocksField::one()
    );
}

#[test]
fn test_inv_root_of_unity() {
    assert_eq!(
        GoldilocksField::ROOT_OF_UNITY_INV,
        GoldilocksField::ROOT_OF_UNITY.invert().unwrap()
    );
}

#[test]
fn test_inv_2() {
    assert_eq!(
        GoldilocksField::TWO_INV,
        GoldilocksField::from(2).invert().unwrap()
    );
}

#[test]
fn test_delta() {
    assert_eq!(
        GoldilocksField::DELTA,
        GENERATOR.pow(&[1u64 << GoldilocksField::S, 0, 0, 0])
    );
    assert_eq!(
        GoldilocksField::DELTA,
        GoldilocksField::MULTIPLICATIVE_GENERATOR.pow(&[1u64 << GoldilocksField::S, 0, 0, 0])
    );
}

#[cfg(not(target_pointer_width = "64"))]
#[test]
fn consistent_modulus_limbs() {
    for (a, &b) in MODULUS
        .0
        .iter()
        .flat_map(|&limb| {
            Some(limb as u32)
                .into_iter()
                .chain(Some((limb >> 32) as u32))
        })
        .zip(MODULUS_LIMBS_32.iter())
    {
        assert_eq!(a, b);
    }
}
