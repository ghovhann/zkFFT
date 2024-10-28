use ff::Field;
use group::Curve;
use halo2_proofs::arithmetic::{CurveAffine, best_multiexp};
use halo2_proofs::transcript::{ChallengeScalar, EncodedChallenge, TranscriptWrite, TranscriptRead};
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};

#[derive(Clone)]
pub struct WipWitness<C: CurveAffine> {
    pub a: Vec<C::Scalar>,
    pub b: Vec<Vec<C::Scalar>>,
    pub alpha: C::Scalar,
}

#[derive(Debug, Clone)]
pub struct WipProof<C: CurveAffine> {
    pub L: Vec<C>,
    pub R: Vec<C>,
    pub A: C,
    pub r_answer: C::Scalar,
    pub delta_answer: C::Scalar,
}

#[derive(PartialEq, Debug, Clone)]
pub enum P<C: CurveAffine> {
    Point(C),
    Terms(Vec<(C::Scalar, C)>),
}

pub fn scale_and_sum_vectors<C: CurveAffine>(vectors: Vec<Vec<C::Scalar>>, scalar: C::Scalar) -> Vec<C::Scalar> {
    // Initialize a result vector with zeros having the same length as the first vector
    let mut result = vec![C::Scalar::ZERO; vectors[0].len()];

    for (index, vector) in vectors.into_iter().enumerate() {
        let power = scalar.pow_vartime([index as u64]); // Calculate the scalar raised to the power of the index
        for (i, elem) in vector.into_iter().enumerate() {
            result[i] += elem * power; // Multiply each element by the power and add to the result
        }
    }

    result
}

pub fn scale_and_sum<C: CurveAffine>(scalars: Vec<C::Scalar>, scalar: C::Scalar) -> C::Scalar {
    let mut sum = C::Scalar::ZERO; // Initialize the sum to zero

    for (index, value) in scalars.into_iter().enumerate() {
        let power = scalar.pow_vartime([index as u64]); // Raise the scalar to the power of the index
        sum += value * power; // Multiply and accumulate
    }

    sum // Return the final sum
}

pub fn scale_and_sum_points<C: CurveAffine>(elements: Vec<C>, s: C::Scalar) -> C {
    elements
        .into_iter()
        .enumerate() // Get index and element
        .map(|(index, elem)| {
            let power = s.pow_vartime([index as u64]); // Calculate s raised to the power of the index
            elem * power // Scale the element (point) by s^index
        })
        .fold(C::identity(), |acc, scaled_elem| (acc + scaled_elem.into()).into()) // Sum the scaled elements
}

pub fn inner_product<C: CurveAffine>(a: &[C::Scalar], b: &[C::Scalar]) -> C::Scalar {
    assert_eq!(a.len(), b.len());

    let mut acc = C::Scalar::from(0);
    for (a, b) in a.iter().zip(b.iter()) {
        acc += (*a) * (*b);
    }

    acc
}

pub fn multiexp<C: CurveAffine>(p: &P<C>) -> C {
    match p {
        P::Point(p) => *p,
        P::Terms(v) => {
            let (coeffs, bases): (Vec<C::Scalar>, Vec<C>) = v.into_iter().cloned().unzip();
            let mut new_bases = Vec::with_capacity(bases.len());
            for b in bases.iter() {
                let tmp = Into::<C::CurveExt>::into(*b)
                    .to_affine()
                    .coordinates()
                    .expect("Couldn't get coordinates of a point");
                new_bases.push(
                    C::from_xy(*tmp.x(), *tmp.y())
                        .expect("Couldn't construct point from coordinates"),
                );
            }
            best_multiexp::<C>(&coeffs, &new_bases).into()
        }
    }
}

pub fn split_vector_in_half<T: Clone>(vec: Vec<T>) -> (Vec<T>, Vec<T>) {
    let mid = vec.len() / 2 + vec.len() % 2; // calculate midpoint, extra element goes into the first half if odd length
    let (first_half, second_half) = vec.split_at(mid);
    (first_half.to_vec(), second_half.to_vec()) // convert slices to vectors
}

pub fn transcript_e<C: CurveAffine, E: EncodedChallenge<C>, T: TranscriptWrite<C, E>>(
    transcript: &mut T,
) -> C::Scalar {
    let e: ChallengeScalar<C, T> = transcript.squeeze_challenge_scalar();
    if bool::from(e.is_zero()) {
        panic!("zero challenge in WIP round");
    }
    *e
}

pub fn transcript_e_v<C: CurveAffine, E: EncodedChallenge<C>, T: TranscriptRead<C, E>>(
    transcript: &mut T,
) -> C::Scalar {
    let e: ChallengeScalar<C, T> = transcript.squeeze_challenge_scalar();
    if bool::from(e.is_zero()) {
        panic!("zero challenge in WIP round");
    }
    *e
}

pub fn next_G_H<C: CurveAffine + Clone + Send + Sync, E: EncodedChallenge<C>, T: TranscriptWrite<C, E>>(
    transcript: &mut T,
    g_bold1: Vec<C>,
    g_bold2: Vec<C>,
) -> (
    C::Scalar,
    C::Scalar,
    C::Scalar,
    C::Scalar,
    Vec<C>,
) {
    assert_eq!(g_bold1.len(), g_bold2.len());

    let e = transcript_e(transcript);
    let inv_e = e.invert().unwrap();

    let new_g_bold: Vec<C> = g_bold1.into_par_iter()
        .zip(g_bold2.into_par_iter())
        .map(|(g1, g2)| {
            let tmp: P<C> = P::Terms(vec![(inv_e.clone(), g1), (e.clone(), g2)]);
            multiexp(&tmp)
        })
        .collect();

    let e_square = e.square();
    let inv_e_square = inv_e.square();

    (e, inv_e, e_square, inv_e_square, new_g_bold)
}


pub fn next_G_H_v<C: CurveAffine + Clone + Send + Sync, E: EncodedChallenge<C>, T: TranscriptRead<C, E>>(
    transcript: &mut T,
    g_bold1: Vec<C>,
    g_bold2: Vec<C>,
) -> (
    C::Scalar,
    C::Scalar,
    C::Scalar,
    C::Scalar,
    Vec<C>,
) {
    assert_eq!(g_bold1.len(), g_bold2.len());

    let e = transcript_e_v(transcript);
    let inv_e = e.invert().unwrap();

    // Parallelize processing of g_bold1 and g_bold2
    let new_g_bold: Vec<C> = g_bold1.into_par_iter()
        .zip(g_bold2.into_par_iter())
        .map(|(g1, g2)| {
            let tmp: P<C> = P::Terms(vec![(inv_e.clone(), g1), (e.clone(), g2)]);
            multiexp(&tmp)
        })
        .collect();

    let e_square = e.square();
    let inv_e_square = inv_e.square();

    (e, inv_e, e_square, inv_e_square, new_g_bold)
}