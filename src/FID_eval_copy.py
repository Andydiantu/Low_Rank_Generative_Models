from eval import Eval
from preprocessing import create_dataloader
from diffusers import DiTPipeline
from DiT import create_model, create_noise_scheduler
from vae import SD_VAE, DummyAutoencoderKL
from config import TrainingConfig
from pathlib import Path
import torch
# from accelerate import Accelerator
import argparse
from low_rank_compression import low_rank_layer_replacement, TimestepConditionedWrapper


def evaluate_fid(config, pipeline):
        test_dataloader = create_dataloader("nielsr/CelebA-faces", "train", config)
        eval = Eval(test_dataloader, config)
        metrics = eval.compute_metrics(pipeline, num_samples=5000)
        print(f"Metrics: FID={metrics['fid']:.4f}, Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}")
        del pipeline

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluate_path", type=str, required=True)
    args = parser.parse_args()

    config = TrainingConfig()
    config.noise_scheduler = "DDIM"
    config.num_inference_steps = 100

    evaluate_path = "logs/" + args.evaluate_path
    print(evaluate_path)
    evaluate_path = Path(__file__).parent.parent / evaluate_path 
    print(evaluate_path)
    model = create_model(config)


    # Move model to CUDA
    model = model.cuda()
    config.low_rank_pretraining = True

    if config.low_rank_pretraining:
        model = low_rank_layer_replacement(model, percentage=config.low_rank_rank, config=config)
        print(f"number of parameters in model after compression is: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    model.load_state_dict(torch.load(evaluate_path))
    print(f"Loaded model from {evaluate_path}")
    

    # model = TimestepConditionedWrapper(model, config)
    # print("Enabled timestep-conditioned rank scheduling")
    # print(f"  Schedule: {config.rank_schedule}")
    # print(f"  Min ratio: {config.rank_min_ratio}")
    # print(f"  Max timesteps: {config.num_training_steps}")

    noise_scheduler = create_noise_scheduler(config)
    vae = SD_VAE() if config.vae else DummyAutoencoderKL()
    
    # Move VAE to CUDA
    vae = vae.cuda()

    pipeline = DiTPipeline(
                    transformer=model,
                    scheduler=noise_scheduler,
                    vae=vae
                )
    # Move pipeline to CUDA
    pipeline = pipeline.to("cuda")
    
    config.eval_dataset_size = 5000
    config.eval_batch_size = 128
    config.cfg_enabled = False
    print(f"Evaluating FID for {config.eval_dataset_size} images")

    for i in [0.5]:
        print(f"Evaluating FID for CFG scale {i+1}")
        config.guidance_scale = i+1
        print(config)
        evaluate_fid(config, pipeline)
        print(f"Evaluating FID for CFG scale {i+1} done")
        print("--------------------------------")



if __name__ == "__main__":
    main()
