# Low Rank Generative Models

This is the codebase for my MSc Dissertation at Imperial, focusing on exploring the application of low-rank methods to generative models (such as Diffusion).

# File Structure

The `notebooks` folder contains exploratory Jupyter Notebook code.

The `src` folder contains training and architecture code.

# TODO
- [ ] Add better category based traning config management
- [x] Add EMA 
- [x] Add FID metric
- [ ] Add BPD metric
- [ ] Add TensorBoard for better logging
- [x] Implement post-hoc low-rank compression
- [x] Implement low-rank training from scratch
- [ ] Extend to EDM2 structure

# Current Problems to Solve
- [x] The base model cannot achieve good FID performance. (Now around 80 FID, aim for FID below 20)
- [x] The sampling stage creates a VRAM bottleneck, leading to out-of-memory errors.
- [ ] The introduction of a VAE into the pipeline causes the loss to become extremely high and unstable.
