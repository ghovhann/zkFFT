use std::sync::{Arc, Mutex};
use rayon::prelude::*;

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

pub fn verify<C: CurveAffine, E: EncodedChallenge<C>, T: TranscriptRead<C, E> + Clone + Sync>(
    transcript: &mut Vec<T>,  
    generators_g: Vec<C>,
    generator_g: Vec<C>,
    generator_h: C,
    witness_b: Vec<Vec<C::Scalar>>,
    p: Vec<P<C>>,
) {
    let rng = OsRng;
    let s = C::Scalar::random(rng);

    let batch_size = p.len();
    let P_terms = Arc::new(Mutex::new(
        p.iter()
            .map(|item| match item {
                P::Point(point) => vec![(C::Scalar::ONE, point.clone())],
                P::Terms(terms) => terms.clone(),
            })
            .collect::<Vec<Vec<(C::Scalar, C)>>>(),
    ));

    let multiexp_var = Arc::new(Mutex::new(vec![]));
    let generators_g_exp = Arc::new(Mutex::new(vec![C::Scalar::ZERO; generators_g.len()]));
    let generator_g_exp = Arc::new(Mutex::new(vec![C::Scalar::ZERO; generator_g.len()]));
    
    let delta_answer = Arc::new(Mutex::new(Vec::with_capacity(batch_size)));

    (0..batch_size).into_par_iter().for_each(|proof_index| {
        let mut local_transcript = transcript[proof_index].clone();
        let mut challenges = vec![];
        let mut g_bold = generators_g.clone();
        let mut b: Vec<Vec<<<C as CurveAffine>::CurveExt as CurveExt>::ScalarExt>> = witness_b.clone();
        
        while g_bold.len() > 1 {
            let (b1, b2): (Vec<Vec<_>>, Vec<Vec<_>>) = b.into_iter().map(split_vector_in_half).unzip();
            let (g_bold1, g_bold2) = split_vector_in_half(g_bold.clone());

            let L = local_transcript.read_point().unwrap();
            let R = local_transcript.read_point().unwrap();

            let (e, inv_e, e_square, inv_e_square, new_g_bold) =
                next_G_H_v(&mut local_transcript, g_bold1, g_bold2);
            g_bold = new_g_bold;
            
            challenges.push((e, inv_e));

            b = b1.iter()
                .zip(b2.iter())
                .map(|(b1_i, b2_i)| {
                    let tmp1: Vec<C::Scalar> = b1_i.iter().map(|x| *x * inv_e).collect();
                    let tmp2: Vec<C::Scalar> = b2_i.iter().map(|x| *x * e).collect();
                    tmp1.iter().zip(tmp2.iter()).map(|(&a, &b)| a + b).collect()
                })
                .collect();

            {
                let mut p_terms = P_terms.lock().unwrap();
                p_terms[proof_index].push((e_square, L));
                p_terms[proof_index].push((inv_e_square, R));
            }
        }

        let product_cache = challenge_products::<C>(&challenges);
        let A = local_transcript.read_point().unwrap();

        let e = transcript_e_v(&mut local_transcript);
        let r = local_transcript.read_scalar().unwrap();
        let delta = local_transcript.read_scalar().unwrap();

        {
            let mut delta_answer_lock = delta_answer.lock().unwrap();
            delta_answer_lock.push(delta);
        }

        let s_exp = s.pow_vartime([proof_index as u64]);
        let mut local_multiexp = vec![];

        {
            let p_terms = P_terms.lock().unwrap();
            for (scalar, base) in &p_terms[proof_index] {
                local_multiexp.push((*scalar * -e * s_exp, base.clone()));
            }
        }
        local_multiexp.push((-s_exp, A));

        let mut local_generators_g_exp = vec![C::Scalar::ZERO; generators_g.len()];
        let mut local_generator_g_exp = vec![C::Scalar::ZERO; generator_g.len()];
        
        for i in 0..generators_g.len() {
            local_generators_g_exp[i] += r * s_exp * product_cache[i];
        }
        for i in 0..generator_g.len() {
            local_generator_g_exp[i] += r * s_exp * b[i][0];
        }

        {
            let mut multiexp_lock = multiexp_var.lock().unwrap();
            multiexp_lock.extend(local_multiexp);
        }
        {
            let mut generators_g_lock = generators_g_exp.lock().unwrap();
            for (i, val) in local_generators_g_exp.iter().enumerate() {
                generators_g_lock[i] += *val;
            }
        }
        {
            let mut generator_g_lock = generator_g_exp.lock().unwrap();
            for (i, val) in local_generator_g_exp.iter().enumerate() {
                generator_g_lock[i] += *val;
            }
        }
    });

    let multiexp_var = &mut *multiexp_var.lock().unwrap();
    for i in 0..generators_g.len() {
        multiexp_var.push((generators_g_exp.lock().unwrap()[i], generators_g[i].clone()));
    }

    for i in 0..generator_g.len() {
        multiexp_var.push((generator_g_exp.lock().unwrap()[i], generator_g[i]));
    }

    let new_delta_answer = scale_and_sum::<C>(delta_answer.lock().unwrap().clone(), s);
    multiexp_var.push((new_delta_answer, generator_h));

    assert_eq!(multiexp(&P::Terms(multiexp_var.clone())), C::identity());
}