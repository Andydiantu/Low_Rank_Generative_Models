from eval import Eval
from preprocessing import create_dataloader
from diffusers import DiTPipeline
from DiT import create_model, create_noise_scheduler, print_model_settings, print_noise_scheduler_settings
from vae import SD_VAE, DummyAutoencoderKL
from config import TrainingConfig
from pathlib import Path
import torch
from accelerate import Accelerator
import argparse


def evaluate_fid(config, pipeline):
        test_dataloader = create_dataloader("uoft-cs/cifar10", "test", config)
        eval = Eval(test_dataloader, config)
        fid_score = eval.compute_metrics(pipeline)
        print(f"FID Score: {fid_score}")
        del pipeline

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluate_path", type=str, required=True)
    args = parser.parse_args()

    config = TrainingConfig()

    evaluate_path = "logs/" + args.evaluate_path + "/model.pt"
    print(evaluate_path)
    evaluate_path = Path(__file__).parent.parent / evaluate_path 
    print(evaluate_path)
    model = create_model(config)

    model.load_state_dict(torch.load(evaluate_path))
    print(f"Loaded model from {evaluate_path}")
    
    # Move model to CUDA
    model = model.cuda()

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
    
    config.eval_dataset_size = 10000
    config.eval_batch_size = 512
    print(f"Evaluating FID for {config.eval_dataset_size} images")
    evaluate_fid(config, pipeline)


if __name__ == "__main__":
    main()
