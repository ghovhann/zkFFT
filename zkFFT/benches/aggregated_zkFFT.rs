use criterion::{criterion_group, criterion_main, Criterion};
use zkFFT::{utils::*, aggregated_verifier, aggregated_prover};
use ff::PrimeField;
use group::Group;
use halo2_proofs::{
    arithmetic::{CurveAffine, Field},
    transcript::{Blake2bRead, Blake2bWrite, Challenge255},
};
use pasta_curves::{arithmetic::CurveExt, pallas, Ep};
use rand::Rng;

pub fn gens<C: CurveAffine>(k: u64, n: u64) -> (Vec<C>, Vec<C>, C) {
    let mut gens_g = vec![];
    let mut gen_g = vec![];
    let hasher = C::CurveExt::hash_to_curve("GENERATORS");

    for _ in 0..n {
        let mut my_array: [u8; 11] = [0; 11];
        let mut rng = rand::thread_rng();
        for i in 0..11 {
            my_array[i] = rng.gen();
        }
        let c = hasher(&my_array);
        gens_g.push(c);
    }

    for _ in 0..k {
        let mut my_array: [u8; 11] = [0; 11];
        let mut rng = rand::thread_rng();
        for i in 0..11 {
            my_array[i] = rng.gen();
        }
        let c = hasher(&my_array);
        gen_g.push(c);
    }

    let mut my_array: [u8; 11] = [0; 11];
    let mut rng = rand::thread_rng();
    for i in 0..11 {
        my_array[i] = rng.gen();
    }
    let c = hasher(&my_array);
    let gen_h = c;

    (
        gens_g.into_iter().map(|ep| ep.into()).collect(),
        gen_g.into_iter().map(|ep| ep.into()).collect(),
        gen_h.into(),
    )
}

fn benchmark_prove(c: &mut Criterion) {
    let mut criterion = Criterion::default().sample_size(10);

    let k = 2048;
    let n = k;
    let batch_size = 100;
    let mut a = Vec::with_capacity(n);
    let mut b = Vec::with_capacity(k);

    for _ in 0..n {
        let mut rng = rand::thread_rng();
        a.push(pallas::Scalar::from(rng.gen::<u64>()));
    }

    let omega = pallas::Scalar::ROOT_OF_UNITY;

    for i in 0..k {
        let mut tmp = vec![];
        for j in 0..n {
            tmp.push(omega.pow_vartime([(j*i) as u64, 0, 0, 0]));        
        }
        b.push(tmp.clone());
    }

    let w = WipWitness {
        a,
        b,
        alpha: pallas::Scalar::from(5),
    };

    let (gens_g, gen_g, gen_h) = gens::<pallas::Affine>(k as u64, n as u64);
    let mut ip = vec![];
    for i in 0..k {
        ip.push(inner_product::<pallas::Affine>(&w.a, &w.b[i]))
    }

    let mut commit = Ep::identity();
    for i in 0..n {
        commit += gens_g[i] * w.a[i];
    }

    for i in 0..k {
        commit += gen_g[i] * ip[i];
    }
    commit += gen_h * w.alpha;

    let transcript = Blake2bWrite::<_, pallas::Affine, Challenge255<_>>::init(vec![]);
    // Benchmark the `prove` function
    criterion.bench_function("Aggregated Prove Benchmark", |b| {
        b.iter(|| {
            let mut transcript_clone = transcript.clone();  // Clone the transcript to reuse in the benchmark
            aggregated_prover::prove(
                &mut transcript_clone,
                vec![w.clone(); batch_size],
                gens_g.clone().into_iter().map(|ep| ep.into()).collect(),
                gen_g.clone().into_iter().map(|ep| ep.into()).collect(),
                gen_h.into(),
                vec![commit.into(); batch_size]
            );
        });
    });
}

fn benchmark_verify(c: &mut Criterion) {
    let mut criterion = Criterion::default().sample_size(10);

    let k = 2048;
    let n = k;
    let batch_size = 100;
    let mut a = Vec::with_capacity(n);
    let mut b = Vec::with_capacity(k);

    for _ in 0..n {
        let mut rng = rand::thread_rng();
        a.push(pallas::Scalar::from(rng.gen::<u64>()));
    }

    let omega = pallas::Scalar::ROOT_OF_UNITY;

    for i in 0..k {
        let mut tmp = vec![];
        for j in 0..n {
            tmp.push(omega.pow_vartime([(j*i) as u64, 0, 0, 0]));        
        }
        b.push(tmp.clone());
    }

    let w = WipWitness {
        a,
        b,
        alpha: pallas::Scalar::from(5),
    };

    let (gens_g, gen_g, gen_h) = gens::<pallas::Affine>(k as u64, n as u64);
    let mut ip = vec![];
    for i in 0..k {
        ip.push(inner_product::<pallas::Affine>(&w.a, &w.b[i]))
    }

    let mut commit = Ep::identity();
    for i in 0..n {
        commit += gens_g[i] * w.a[i];
    }

    for i in 0..k {
        commit += gen_g[i] * ip[i];
    }
    commit += gen_h * w.alpha;

    let mut transcript = Blake2bWrite::<_, pallas::Affine, Challenge255<_>>::init(vec![]);
    aggregated_prover::prove(
        &mut transcript,
        vec![w.clone(); batch_size],
        gens_g.clone().into_iter().map(|ep| ep.into()).collect(),
        gen_g.clone().into_iter().map(|ep| ep.into()).collect(),
        gen_h.into(),
        vec![commit.into(); batch_size]
    );

    let proof_bytes = transcript.finalize();
    println!("Proof size is {}kb", (proof_bytes.len()) as f64 / 1024.);
    let transcript = Blake2bRead::<_, pallas::Affine, Challenge255<_>>::init(&*proof_bytes);

    // Benchmark the `verify` function
    criterion.bench_function("Aggregated Verify Benchmark", |b| {
        b.iter(|| {
            let mut transcript_clone = transcript.clone();  // Clone the transcript to reuse in the benchmark
            aggregated_verifier::verify(
                &mut transcript_clone,
                gens_g.clone().into_iter().map(|ep| ep.into()).collect(),
                gen_g.clone().into_iter().map(|ep| ep.into()).collect(),
                gen_h.into(),
                w.b.clone(),
                batch_size
            );
        });
    });
}

// Group benchmarks together
criterion_group!(aggregated_zkFFT, benchmark_prove, benchmark_verify);
criterion_main!(aggregated_zkFFT);
