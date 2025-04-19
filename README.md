# Low Rank Generative Models

This is the codebase for my MSc Dissertation at Imperial, focusing on exploring the application of low-rank methods to generative models (such as Diffusion).

# File Structure

The `notebooks` folder contains exploratory Jupyter Notebook code.

The `src` folder contains training and architecture code.

# TODO
- [ ] Add EMA (Exponential Moving Average)
- [ ] Add FID (Fréchet Inception Distance) and BPD (Bits Per Dimension) metrics
- [ ] Add TensorBoard for better logging
- [ ] Implement post-hoc low-rank compression
- [ ] Implement low-rank training from scratch
- [ ] Extend to EDM2 structure

# Current Problems to Solve
- [ ] The sampling stage creates a VRAM bottleneck, leading to out-of-memory errors.
- [ ] The introduction of a VAE (Variational Autoencoder) into the pipeline causes the loss to become extremely high and unstable.
