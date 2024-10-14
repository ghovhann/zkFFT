use halo2_proofs::{arithmetic::{CurveAffine, Field}, transcript::{EncodedChallenge, TranscriptWrite}};
use rand_core::OsRng;
use crate::utils::*;

pub fn prove<C: CurveAffine, E: EncodedChallenge<C>, T: TranscriptWrite<C, E>>(
    transcript: &mut T,
    witness: Vec<WipWitness<C>>,
    generators_g: Vec<C>,
    generator_g: Vec<C>,
    generator_h: C,
    committments: Vec<C>,
) -> WipProof<C> {
    let rng = OsRng;

    for i in 0..committments.len() {
        transcript.write_point(committments[i]).unwrap();
    }
    let s = transcript_e(transcript);

    let mut g_bold = generators_g.clone();

    let witness_a = witness
        .iter() // Borrow the witness
        .map(|w| w.a.clone()) // Extract references to the scalars from each struct
        .collect();
    let mut a = scale_and_sum_vectors::<C>(witness_a, s);
    let mut b = witness[0].b.clone(); //domain should be the same for all of them
    let witness_alpha = witness
    .into_iter()
    .map(|w| w.alpha) // Extract the scalars from each struct
    .collect();
    let mut alpha = scale_and_sum::<C>(witness_alpha, s);

    for b_i in b.iter() {
        assert_eq!(a.len(), b_i.len());
    }

    // // From here on, g_bold.len() is used as n
    assert_eq!(g_bold.len(), a.len());

    let mut L_vec: Vec<C> = vec![];
    let mut R_vec: Vec<C> = vec![];

    // // else n > 1 case from figure 1
    while g_bold.len() > 1 {
        let b_clone = b.clone();
        let (a1, a2) = split_vector_in_half(a.clone());
        let (b1, b2): (Vec<Vec<_>>, Vec<Vec<_>>) =
            b_clone.into_iter().map(split_vector_in_half).unzip();
        let (g_bold1, g_bold2) = split_vector_in_half(g_bold.clone());

        let n_hat = g_bold1.len();
        assert_eq!(a1.len(), n_hat);
        assert_eq!(a2.len(), n_hat);
        for b_i in b1.iter() {
            assert_eq!(n_hat, b_i.len());
        }
        for b_i in b2.iter() {
            assert_eq!(n_hat, b_i.len());
        }
        assert_eq!(g_bold1.len(), n_hat);
        assert_eq!(g_bold2.len(), n_hat);

        let d_l = C::Scalar::random(rng);
        let d_r = C::Scalar::random(rng);

        let mut c_l: Vec<C::Scalar> = vec![];
        let mut c_r: Vec<C::Scalar> = vec![];

        for b_i in b2.iter() {
            let tmp = inner_product::<C>(&a1, b_i);
            c_l.push(tmp);
        }
        for b_i in b1.iter() {
            let tmp = inner_product::<C>(&a2, b_i);
            c_r.push(tmp);
        }

        let mut L_terms: Vec<(C::Scalar, C)> = a1
            .iter()
            .copied()
            .zip(g_bold2.iter().copied())
            .chain(c_l.iter().copied().zip(generator_g.iter().copied()))
            .collect::<Vec<_>>();
        L_terms.push((d_l, generator_h));
        let L = multiexp(&P::Terms(L_terms));
        L_vec.push(L);
        transcript.write_point(L).unwrap();

        let mut R_terms: Vec<(C::Scalar, C)> = a2
            .iter()
            .copied()
            .zip(g_bold1.iter().copied())
            .chain(c_r.iter().copied().zip(generator_g.iter().copied()))
            .collect::<Vec<_>>();
        R_terms.push((d_r, generator_h));
        let R = multiexp(&P::Terms(R_terms));
        R_vec.push(R);
        transcript.write_point(R).unwrap();

        let (e, inv_e, e_square, inv_e_square);
        (e, inv_e, e_square, inv_e_square, g_bold) =
            next_G_H(transcript, g_bold1, g_bold2);

        let tmp1: Vec<C::Scalar> = a1.into_iter().map(|x| x * e).collect();
        let tmp2: Vec<C::Scalar> = a2.into_iter().map(|x| x * inv_e).collect();
        a = tmp1.iter().zip(tmp2.iter()).map(|(&a, &b)| a + b).collect();

        b = vec![];
        for (b1_i, b2_i) in b1.iter().zip(b2.iter()) {
            let tmp1: Vec<C::Scalar> = b1_i.into_iter().map(|x| *x * inv_e).collect();
            let tmp2: Vec<C::Scalar> = b2_i.into_iter().map(|x| *x * e).collect();
            b.push(tmp1.iter().zip(tmp2.iter()).map(|(&a, &b)| a + b).collect());
        }

        alpha += (d_l * e_square) + (d_r * inv_e_square);

        debug_assert_eq!(g_bold.len(), a.len());
        for b_i in b.iter() {
            debug_assert_eq!(g_bold.len(), b_i.len());
        }
    }

    // // n == 1 case from figure 1
    assert_eq!(g_bold.len(), 1);
    assert_eq!(a.len(), 1);
    for b_i in b.iter() {
        assert_eq!(b_i.len(), 1);
    }

    let r = C::Scalar::random(rng);
    let delta = C::Scalar::random(rng);

    let mut g_terms: Vec<(C::Scalar, C)> = vec![];
    for (g_i, b_i) in generator_g.iter().zip(b.iter()) {
        g_terms.push(((r * b_i[0]), *g_i))
    }

    let mut A_terms: Vec<(C::Scalar, C)> = vec![(r, g_bold[0]), (delta, generator_h)];
    A_terms.extend(g_terms);
    let A: C = multiexp(&P::Terms(A_terms));
    transcript.write_point(A).unwrap();

    let e = transcript_e(transcript);
    let r_answer = r + (a[0] * e);
    let delta_answer = delta + (alpha * e);
    transcript.write_scalar(r_answer).unwrap();
    transcript.write_scalar(delta_answer).unwrap();

    WipProof {
        L: L_vec,
        R: R_vec,
        A,
        r_answer,
        delta_answer,
    }
}