use halo2_proofs::{arithmetic::{CurveAffine, Field}, transcript::{EncodedChallenge, TranscriptRead}};
use pasta_curves::arithmetic::CurveExt;
use crate::utils::*;

pub fn verify<C: CurveAffine, E: EncodedChallenge<C>, T: TranscriptRead<C, E>>(
    transcript: &mut T,
    generators_g: Vec<C>,
    generator_g: Vec<C>,
    generator_h: C,
    witness_b: Vec<Vec<C::Scalar>>,
    p: P<C>,
) {
    let mut P_terms = match p {
        P::Point(point) => vec![(C::Scalar::ONE, point)],
        P::Terms(terms) => terms,
    };

    let mut g_bold = generators_g.clone();
    let mut b: Vec<Vec<<<C as CurveAffine>::CurveExt as CurveExt>::ScalarExt>> = witness_b.clone();

    while g_bold.len() > 1 {
        let (b1, b2): (Vec<Vec<_>>, Vec<Vec<_>>) =
            b.into_iter().map(split_vector_in_half).unzip();
        let (g_bold1, g_bold2) = split_vector_in_half(g_bold.clone());

        let L = transcript.read_point().unwrap();
        let R = transcript.read_point().unwrap();

        let (e, inv_e, e_square, inv_e_square);
        (e, inv_e, e_square, inv_e_square, g_bold) =
            next_G_H_v(transcript, g_bold1, g_bold2);
    
        b = vec![];
        for (b1_i, b2_i) in b1.iter().zip(b2.iter()) {
            let tmp1: Vec<C::Scalar> = b1_i.into_iter().map(|x| *x * inv_e).collect();
            let tmp2: Vec<C::Scalar> = b2_i.into_iter().map(|x| *x * e).collect();
            b.push(tmp1.iter().zip(tmp2.iter()).map(|(&a, &b)| a + b).collect());
        }

        P_terms.push((e_square, L));
        P_terms.push((inv_e_square, R));
    }

    let A = transcript.read_point().unwrap();

    let e = transcript_e_v(transcript);
    let r_answer = transcript.read_scalar().unwrap();
    let delta_answer = transcript.read_scalar().unwrap();
    
    let mut multiexp_var = P_terms;
    for (scalar, _) in multiexp_var.iter_mut() {
        *scalar *= -e;
    }

    for i in 0..g_bold.len() {
        multiexp_var.push((r_answer, g_bold[i].clone()));
    }

    multiexp_var.push((-C::Scalar::ONE, A));

    let mut i = 0;
    for g_i in generator_g.iter() {
        multiexp_var.push((r_answer * b[i][0], *g_i));
        i += 1;
    }
    multiexp_var.push((delta_answer, generator_h));

    assert_eq!(multiexp(&P::Terms(multiexp_var)), C::identity());
}