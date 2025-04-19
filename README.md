# Low Rank Generative Models

This is the codebase of my MSc Dissertation at Imperial, focusing on exploring the applicatin of low rank method on generative models (such as Diffusion). 

# File Structure

`notesbooks` folder contains exploratary jupyter notebook code.

`src` folder contains training and architecture code.

# TODO
- [ ] Add EMA
- [ ] Add FID and BPD metrics
- [ ] Add tensorborad for better logging
- [ ] Implemented post-hoc low rank compression
- [ ] Implmenet low rank training from scratch
- [ ] Extend to EDM2 structure


# Current Problem to Solve
- [ ] Sampling stage create VRAM bottleneck, makes out of VRAM problem. 
- [ ] Introduction of VAE in pipeline makes loss extremely high and unstable.