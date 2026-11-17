//! Gated latent routing port (unofficial, Rust).
//!
//! Implements the latent projection, the scalar gate and the top-k assignment
//! described in the technical note.

pub struct GlrRouter {
    pub latent_rank: usize,
    pub num_blocks: usize,
    pub gate_weight: Vec<f32>,
}

impl GlrRouter {
    pub fn project(&self, hidden: &[f32]) -> Vec<f32> {
        let mut latent = vec![0.0f32; self.latent_rank];
        for (r, slot) in latent.iter_mut().enumerate() {
            let mut acc = 0.0f32;
            for (i, h) in hidden.iter().enumerate() {
                acc += *h * self.weight(r, i);
            }
            *slot = acc;
        }
        latent
    }

    pub fn gate(&self, latent: &[f32]) -> f32 {
        let mut acc = 0.0f32;
        for (r, z) in latent.iter().enumerate() {
            acc += *z * self.gate_weight[r];
        }
        1.0 / (1.0 + (-acc).exp())
    }

    fn weight(&self, r: usize, i: usize) -> f32 {
        0.01 * ((r + 1) * (i + 1)) as f32
    }

    pub fn top_k(&self, scores: &[f32], k: usize) -> Vec<usize> {
        let mut pairs: Vec<(usize, f32)> = scores.iter().cloned().enumerate().collect();
        pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        pairs.into_iter().take(k).map(|(i, _)| i).collect()
    }
}
