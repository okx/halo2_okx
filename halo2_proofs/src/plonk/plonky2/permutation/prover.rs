use group::ff::{BatchInvert, Field, PrimeField};
use crate::plonk::plonky2::GenericConfig2;
use rand_core::RngCore;
use std::iter::{self, ExactSizeIterator};
use plonky2::iop::challenger::Challenger;

use super::super::super::{circuit::Any, ChallengeBeta, ChallengeGamma, ChallengeX};
use super::{Argument, ProvingKey};
use crate::{
    arithmetic::{eval_polynomial, parallelize},
    plonk::{self, Error},
    poly::{
        self,
        commitment::{Blind, Params},
        fri::ProverQuery,
        Coeff, ExtendedLagrangeCoeff, LagrangeCoeff, Polynomial, Rotation,
    },
    transcript::{EncodedChallenge, TranscriptWrite},
};

pub struct CommittedSet<G: GenericConfig2, Ev> {
    permutation_product_poly: Polynomial<G::F, Coeff>,
    permutation_product_coset: poly::AstLeaf<Ev, ExtendedLagrangeCoeff>,
    permutation_product_blind: Blind<G::F>,
}

pub(crate) struct Committed<G: GenericConfig2, Ev> {
    sets: Vec<CommittedSet<G, Ev>>
}

pub struct ConstructedSet<G: GenericConfig2 > {
    permutation_product_poly: Polynomial<G::F, Coeff>,
    permutation_product_blind: Blind<G::F>,
}

pub(crate) struct Constructed<G: GenericConfig2> {
    sets: Vec<ConstructedSet<G>>
}

pub(crate) struct Evaluated<G: GenericConfig2> {
    constructed: Constructed<G>,
}

impl Argument {
    #[allow(clippy::too_many_arguments)]
    pub(in crate::plonk) fn commit<
        G: GenericConfig2,
        Ev: Copy + Send + Sync,
        R: RngCore
    >(
        &self,
        pk: &plonk::plonky2::ProvingKey<G>,
        pkey: &ProvingKey<G>,
        advice: &[Polynomial<G::F, LagrangeCoeff>],
        fixed: &[Polynomial<G::F, LagrangeCoeff>],
        instance: &[Polynomial<G::F, LagrangeCoeff>],
        beta: G::F,
        gamma: G::F,
        evaluator: &mut poly::Evaluator<Ev, G::F, ExtendedLagrangeCoeff>,
        mut rng: R,
        transcript: &mut Challenger<G::F, G::Hasher>,
    ) -> Result<Committed<G, Ev>, Error> {
        let domain = &pk.vk.domain;

        // How many columns can be included in a single permutation polynomial?
        // We need to multiply by z(X) and (1 - (l_last(X) + l_blind(X))). This
        // will never underflow because of the requirement of at least a degree
        // 3 circuit for the permutation argument.
        assert!(pk.vk.cs_degree >= 3);
        let chunk_len = pk.vk.cs_degree - 2;
        let blinding_factors = pk.vk.cs.blinding_factors();

        // Each column gets its own delta power.
        let mut deltaomega = <G::F as ff::Field>::ONE;

        // Track the "last" value from the previous column set
        let mut last_z = <G::F as ff::Field>::ONE;

        let mut sets = vec![];

        for (columns, permutations) in self
            .columns
            .chunks(chunk_len)
            .zip(pkey.permutations.chunks(chunk_len))
        {
            // Goal is to compute the products of fractions
            //
            // (p_j(\omega^i) + \delta^j \omega^i \beta + \gamma) /
            // (p_j(\omega^i) + \beta s_j(\omega^i) + \gamma)
            //
            // where p_j(X) is the jth column in this permutation,
            // and i is the ith row of the column.

            let mut modified_values = vec![<G::F as ff::Field>::ONE; domain.n as usize];

            // Iterate over each column of the permutation
            for (&column, permuted_column_values) in columns.iter().zip(permutations.iter()) {
                let values = match column.column_type() {
                    Any::Advice => advice,
                    Any::Fixed => fixed,
                    Any::Instance => instance,
                };
                parallelize(&mut modified_values, |modified_values, start| {
                    for ((modified_values, value), permuted_value) in modified_values
                        .iter_mut()
                        .zip(values[column.index()][start..].iter())
                        .zip(permuted_column_values[start..].iter())
                    {
                        *modified_values *= &(beta * permuted_value + &gamma + value);
                    }
                });
            }

            // Invert to obtain the denominator for the permutation product polynomial
            modified_values.batch_invert();

            // Iterate over each column again, this time finishing the computation
            // of the entire fraction by computing the numerators
            for &column in columns.iter() {
                let omega = domain.get_omega();
                let values = match column.column_type() {
                    Any::Advice => advice,
                    Any::Fixed => fixed,
                    Any::Instance => instance,
                };
                parallelize(&mut modified_values, |modified_values, start| {
                    let mut deltaomega = deltaomega * &omega.pow_vartime([start as u64, 0, 0, 0]);
                    for (modified_values, value) in modified_values
                        .iter_mut()
                        .zip(values[column.index()][start..].iter())
                    {
                        // Multiply by p_j(\omega^i) + \delta^j \omega^i \beta
                        *modified_values *= &(deltaomega * &beta + &gamma + value);
                        deltaomega *= &omega;
                    }
                });
                deltaomega *= &G::F::DELTA;
            }

            // The modified_values vector is a vector of products of fractions
            // of the form
            //
            // (p_j(\omega^i) + \delta^j \omega^i \beta + \gamma) /
            // (p_j(\omega^i) + \beta s_j(\omega^i) + \gamma)
            //
            // where i is the index into modified_values, for the jth column in
            // the permutation

            // Compute the evaluations of the permutation product polynomial
            // over our domain, starting with z[0] = 1
            let mut z = vec![last_z];
            for row in 1..(domain.n as usize) {
                let mut tmp = z[row - 1];

                tmp *= &modified_values[row - 1];
                z.push(tmp);
            }
            let mut z = domain.lagrange_from_vec(z);
            // Set blinding factors
            for z in &mut z[domain.n as usize - blinding_factors..] {
                *z = G::F::random(&mut rng);
            }
            // Set new last_z
            last_z = z[domain.n as usize - (blinding_factors + 1)];

            let blind = Blind(G::F::random(&mut rng));

            // let permutation_product_commitment_projective = params.commit_lagrange(&z, blind);
            let permutation_product_blind = blind;
            let z = domain.lagrange_to_coeff(z);
            let permutation_product_poly = z.clone();

            let permutation_product_coset =
                evaluator.register_poly(domain.coeff_to_extended(z.clone()));

            // let permutation_product_commitment =
                // permutation_product_commitment_projective.to_affine();

            // Hash the permutation product commitment
            // transcript.write_point(permutation_product_commitment)?;

            sets.push(CommittedSet {
                permutation_product_poly,
                permutation_product_coset,
                permutation_product_blind,
            });
        }

        Ok(Committed { sets })
    }
}

impl<G: GenericConfig2, Ev: Copy + Send + Sync> Committed<G, Ev> {
    #[allow(clippy::too_many_arguments)]
    pub(in crate::plonk) fn construct<'a>(
        self,
        pk: &'a plonk::plonky2::ProvingKey<G>,
        p: &'a Argument,
        advice_cosets: &'a [poly::AstLeaf<Ev, ExtendedLagrangeCoeff>],
        fixed_cosets: &'a [poly::AstLeaf<Ev, ExtendedLagrangeCoeff>],
        instance_cosets: &'a [poly::AstLeaf<Ev, ExtendedLagrangeCoeff>],
        permutation_cosets: &'a [poly::AstLeaf<Ev, ExtendedLagrangeCoeff>],
        l0: poly::AstLeaf<Ev, ExtendedLagrangeCoeff>,
        l_blind: poly::AstLeaf<Ev, ExtendedLagrangeCoeff>,
        l_last: poly::AstLeaf<Ev, ExtendedLagrangeCoeff>,
        beta: G::F,
        gamma: G::F,
    ) -> (
        Constructed<G>,
        impl Iterator<Item = poly::Ast<Ev, G::F, ExtendedLagrangeCoeff>> + 'a,
    ) {
        let chunk_len = pk.vk.cs_degree - 2;
        let blinding_factors = pk.vk.cs.blinding_factors();
        let last_rotation = Rotation(-((blinding_factors + 1) as i32));

        let constructed = Constructed {
            sets: self
                .sets
                .iter()
                .map(|set| ConstructedSet {
                    permutation_product_poly: set.permutation_product_poly.clone(),
                    permutation_product_blind: set.permutation_product_blind,
                })
                .collect()
        };

        let expressions = iter::empty()
            // Enforce only for the first set.
            // l_0(X) * (1 - z_0(X)) = 0
            .chain(
                self.sets
                    .first()
                    .map(|first_set| (poly::Ast::one() - first_set.permutation_product_coset) * l0),
            )
            // Enforce only for the last set.
            // l_last(X) * (z_l(X)^2 - z_l(X)) = 0
            .chain(self.sets.last().map(|last_set| {
                ((poly::Ast::from(last_set.permutation_product_coset)
                    * last_set.permutation_product_coset)
                    - last_set.permutation_product_coset)
                    * l_last
            }))
            // Except for the first set, enforce.
            // l_0(X) * (z_i(X) - z_{i-1}(\omega^(last) X)) = 0
            .chain(
                self.sets
                    .iter()
                    .skip(1)
                    .zip(self.sets.iter())
                    .map(|(set, last_set)| {
                        (poly::Ast::from(set.permutation_product_coset)
                            - last_set
                                .permutation_product_coset
                                .with_rotation(last_rotation))
                            * l0
                    })
                    .collect::<Vec<_>>(),
            )
            // And for all the sets we enforce:
            // (1 - (l_last(X) + l_blind(X))) * (
            //   z_i(\omega X) \prod_j (p(X) + \beta s_j(X) + \gamma)
            // - z_i(X) \prod_j (p(X) + \delta^j \beta X + \gamma)
            // )
            .chain(
                self.sets
                    .into_iter()
                    .zip(p.columns.chunks(chunk_len))
                    .zip(permutation_cosets.chunks(chunk_len))
                    .enumerate()
                    .map(move |(chunk_index, ((set, columns), cosets))| {
                        let mut left = poly::Ast::<_, G::F, _>::from(
                            set.permutation_product_coset
                                .with_rotation(Rotation::next()),
                        );
                        for (values, permutation) in columns
                            .iter()
                            .map(|&column| match column.column_type() {
                                Any::Advice => &advice_cosets[column.index()],
                                Any::Fixed => &fixed_cosets[column.index()],
                                Any::Instance => &instance_cosets[column.index()],
                            })
                            .zip(cosets.iter())
                        {
                            left *= poly::Ast::<_, G::F, _>::from(*values)
                                + (poly::Ast::ConstantTerm(beta) * poly::Ast::from(*permutation))
                                + poly::Ast::ConstantTerm(gamma);
                        }

                        let mut right = poly::Ast::from(set.permutation_product_coset);
                        let mut current_delta = beta
                            * &(G::F::DELTA.pow_vartime([(chunk_index * chunk_len) as u64]));
                        for values in columns.iter().map(|&column| match column.column_type() {
                            Any::Advice => &advice_cosets[column.index()],
                            Any::Fixed => &fixed_cosets[column.index()],
                            Any::Instance => &instance_cosets[column.index()],
                        }) {
                            right *= poly::Ast::from(*values)
                                + poly::Ast::LinearTerm(current_delta)
                                + poly::Ast::ConstantTerm(gamma);
                            current_delta *= &G::F::DELTA;
                        }

                        (left - right) * (poly::Ast::one() - (poly::Ast::from(l_last) + l_blind))
                    }),
            );

        (constructed, expressions)
    }
}

impl<G: GenericConfig2> super::ProvingKey<G> {
    pub(in crate::plonk) fn open(
        &self,
        x: G::F,
    ) -> impl Iterator<Item = ProverQuery<'_, G>> + Clone {
        self.polys.iter().map(move |poly| ProverQuery {
            point: x,
            poly,
            blind: Blind::default(),
        })
    }

    pub(in crate::plonk) fn evaluate(
        &self,
        x: G::F,
        challenger: &mut Challenger<G::F, G::Hasher>,
        // transcript: &mut T,
    ) -> Result<(), Error> {
        // Hash permutation evals
        for eval in self.polys.iter().map(|poly| eval_polynomial(poly, x)) {
            // transcript.write_scalar(eval)?;
            challenger.observe_element(eval);
        }

        Ok(())
    }
}

impl<G: GenericConfig2> Constructed<G> {
    pub(in crate::plonk) fn evaluate(
        self,
        pk: &plonk::plonky2::ProvingKey<G>,
        x: G::F,
        challenger: &mut Challenger<G::F, G::Hasher>,
    ) -> Result<Evaluated<G>, Error> {
        let domain = &pk.vk.domain;
        let blinding_factors = pk.vk.cs.blinding_factors();

        {
            let mut sets = self.sets.iter();

            while let Some(set) = sets.next() {
                let permutation_product_eval = eval_polynomial(&set.permutation_product_poly, x);

                let permutation_product_next_eval = eval_polynomial(
                    &set.permutation_product_poly,
                    domain.rotate_omega(x, Rotation::next()),
                );

                // Hash permutation product evals
                for eval in iter::empty()
                    .chain(Some(&permutation_product_eval))
                    .chain(Some(&permutation_product_next_eval))
                {
                    challenger.observe_element(*eval);
                }

                // If we have any remaining sets to process, evaluate this set at omega^u
                // so we can constrain the last value of its running product to equal the
                // first value of the next set's running product, chaining them together.
                if sets.len() > 0 {
                    let permutation_product_last_eval = eval_polynomial(
                        &set.permutation_product_poly,
                        domain.rotate_omega(x, Rotation(-((blinding_factors + 1) as i32))),
                    );

                    challenger.observe_element(permutation_product_last_eval);
                }
            }
        }

        Ok(Evaluated { constructed: self })
    }
}

impl<G: GenericConfig2> Evaluated<G> {
    pub(in crate::plonk) fn open<'a>(
        &'a self,
        pk: &'a plonk::plonky2::ProvingKey<G>,
        x: G::F,
    ) -> impl Iterator<Item = ProverQuery<'a, G>> + Clone {
        let blinding_factors = pk.vk.cs.blinding_factors();
        let x_next = pk.vk.domain.rotate_omega(x, Rotation::next());
        let x_last = pk
            .vk
            .domain
            .rotate_omega(x, Rotation(-((blinding_factors + 1) as i32)));

        iter::empty()
            .chain(self.constructed.sets.iter().flat_map(move |set| {
                iter::empty()
                    // Open permutation product commitments at x and \omega x
                    .chain(Some(ProverQuery {
                        point: x,
                        poly: &set.permutation_product_poly,
                        blind: set.permutation_product_blind,
                    }))
                    .chain(Some(ProverQuery {
                        point: x_next,
                        poly: &set.permutation_product_poly,
                        blind: set.permutation_product_blind,
                    }))
            }))
            // Open it at \omega^{last} x for all but the last set. This rotation is only
            // sensical for the first row, but we only use this rotation in a constraint
            // that is gated on l_0.
            .chain(
                self.constructed
                    .sets
                    .iter()
                    .rev()
                    .skip(1)
                    .flat_map(move |set| {
                        Some(ProverQuery {
                            point: x_last,
                            poly: &set.permutation_product_poly,
                            blind: set.permutation_product_blind,
                        })
                    }),
            )
    }
}
