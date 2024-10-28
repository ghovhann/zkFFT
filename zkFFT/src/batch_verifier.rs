use halo2_proofs::{arithmetic::{CurveAffine, Field}, transcript::{EncodedChallenge, TranscriptRead}};
use pasta_curves::arithmetic::CurveExt;
use rand_core::OsRng;
use crate::utils::*;

fn challenge_products<C: CurveAffine>(challenges: &[(C::Scalar, C::Scalar)]) -> Vec<C::Scalar> {
    let mut products = vec![C::Scalar::ONE; 1 << challenges.len()];

    if !challenges.is_empty() {
      products[0] = challenges[0].1;
      products[1] = challenges[0].0;

      for (j, challenge) in challenges.iter().enumerate().skip(1) {
        let mut slots = (1 << (j + 1)) - 1;
        while slots > 0 {
          products[slots] = products[slots / 2] * challenge.0;
          products[slots - 1] = products[slots / 2] * challenge.1;

          slots = slots.saturating_sub(2);
        }
      }

      for product in &products {
        debug_assert!(!bool::from(product.is_zero()));
      }
    }
    products
}

pub fn verify<C: CurveAffine, E: EncodedChallenge<C>, T: TranscriptRead<C, E>>(
    transcript: &mut Vec<&mut T>,
    generators_g: Vec<C>,
    generator_g: Vec<C>,
    generator_h: C,
    witness_b: Vec<Vec<C::Scalar>>,
    p: Vec<P<C>>,
) {
    let rng = OsRng;
    let s = C::Scalar::random(rng);

    let batch_size = p.len();
    let mut P_terms = vec![];
    for i in 0..batch_size {
        let tmp = match p[i].clone() {
            P::Point(point) => vec![(C::Scalar::ONE, point)],
            P::Terms(terms) => terms,
        };
        P_terms.push(tmp);
    }

    let mut multiexp_var: Vec<(<<C as CurveAffine>::CurveExt as CurveExt>::ScalarExt, C)> = vec![];
    let mut r_answer = Vec::with_capacity(batch_size);
    let mut delta_answer = Vec::with_capacity(batch_size);
    let mut generator_g_exp = vec![C::Scalar::ZERO; generator_g.len()];
    let mut generators_g_exp = vec![C::Scalar::ZERO; generators_g.len()];

    for proof_index in 0..batch_size {     
        let mut challenges = vec![];

        let mut g_bold = generators_g.clone();
        let mut b: Vec<Vec<<<C as CurveAffine>::CurveExt as CurveExt>::ScalarExt>> = witness_b.clone();
        while g_bold.len() > 1 {
            let (b1, b2): (Vec<Vec<_>>, Vec<Vec<_>>) =
                b.into_iter().map(split_vector_in_half).unzip();
            let (g_bold1, g_bold2) = split_vector_in_half(g_bold.clone());
    
            let L = transcript[proof_index].read_point().unwrap();
            let R = transcript[proof_index].read_point().unwrap();
    
            let (e, inv_e, e_square, inv_e_square);
            (e, inv_e, e_square, inv_e_square, g_bold) =
                next_G_H_v(transcript[proof_index], g_bold1, g_bold2);
            challenges.push((e, inv_e));
        
            b = vec![];
            for (b1_i, b2_i) in b1.iter().zip(b2.iter()) {
                let tmp1: Vec<C::Scalar> = b1_i.into_iter().map(|x| *x * inv_e).collect();
                let tmp2: Vec<C::Scalar> = b2_i.into_iter().map(|x| *x * e).collect();
                b.push(tmp1.iter().zip(tmp2.iter()).map(|(&a, &b)| a + b).collect());
            }
    
            P_terms[proof_index].push((e_square, L));
            P_terms[proof_index].push((inv_e_square, R));
        }
        let product_cache = challenge_products::<C>(&challenges);

        let A = transcript[proof_index].read_point().unwrap();
    
        let e = transcript_e_v(transcript[proof_index]);
        r_answer.push(transcript[proof_index].read_scalar().unwrap());
        delta_answer.push(transcript[proof_index].read_scalar().unwrap());

        let s_exp = s.pow_vartime([proof_index as u64]);
        for (scalar, base) in P_terms[proof_index].iter() {
            multiexp_var.push((*scalar * -e * s_exp, base.clone()));
        }

        multiexp_var.push((-s_exp, A));

        for i in 0..generators_g.len() {
            generators_g_exp[i] += r_answer[proof_index] * s_exp * product_cache[i];
        }

        for i in 0..generator_g.len() {
            generator_g_exp[i] += r_answer[proof_index] * s_exp * b[i][0];
        }
    }

    for i in 0..generators_g.len() {
        multiexp_var.push((generators_g_exp[i], generators_g[i].clone()));
    }

    for i in 0..generator_g.len() {
        multiexp_var.push((generator_g_exp[i], generator_g[i]));
    }

    let new_delta_answer = scale_and_sum::<C>(delta_answer, s);
    multiexp_var.push((new_delta_answer, generator_h));

    assert_eq!(multiexp(&P::Terms(multiexp_var)), C::identity());
}