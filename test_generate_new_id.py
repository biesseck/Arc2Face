from diffusers import (
    StableDiffusionPipeline,
    UNet2DConditionModel,
    DPMSolverMultistepScheduler,
)

from arc2face import CLIPTextModelWrapper, project_face_embs

import os
import sys
import torch
from insightface.app import FaceAnalysis
from PIL import Image
import numpy as np
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path-input", type=str, default="assets/examples/joacquin.png")
    parser.add_argument("--similarity", type=float, default=0.7)
    parser.add_argument("--num-samples", type=int, default=4)
    args = parser.parse_args()
    return args


def rotate_embedding_by_cosine_similarity(v1: torch.Tensor, cosine_similarity: float) -> torch.Tensor:
    v1 = torch.squeeze(v1)
    if not (0.0 <= cosine_similarity <= 1.0):
        raise ValueError("Cosine similarity must be between 0.0 and 1.0.")
    if v1.dim() != 1 or v1.size(0) != 512:
        raise ValueError("Input tensor must be 512-dimensional (1D tensor).")
    
    theta = torch.acos(torch.tensor(cosine_similarity, device=v1.device))
    
    if torch.isclose(theta, torch.tensor(0.0).to(torch.float32)):
        return v1.clone()
    
    v1_norm = torch.linalg.norm(v1).to(torch.float32)
    if torch.isclose(v1_norm, torch.tensor(0.0).to(torch.float32)):
        return v1.clone()
        
    u1 = v1 / v1_norm

    random_vector = torch.randn_like(v1)
    
    projection_onto_u1 = torch.dot(random_vector, u1) * u1
    
    u2_raw = random_vector - projection_onto_u1
    
    u2_norm = torch.linalg.norm(u2_raw).to(torch.float32)
    
    if torch.isclose(u2_norm, torch.tensor(0.0).to(torch.float32)):
        raise RuntimeError("Failed to generate a non-collinear random vector. Try running again.")

    u2 = u2_raw / u2_norm
    u1_prime = (u1 * torch.cos(theta)) + (u2 * torch.sin(theta))
    v1_prime = u1_prime * v1_norm
    v1_prime = torch.unsqueeze(v1_prime, 0)
    return v1_prime



if __name__ == '__main__':

    args = parse_arguments()
    assert os.path.isfile(args.path_input), f"Error: file not found \'{args.path_input}\'"

    # Arc2Face is built upon SD1.5
    # The repo below can be used instead of the now deprecated 'runwayml/stable-diffusion-v1-5'
    base_model = 'stable-diffusion-v1-5/stable-diffusion-v1-5'

    encoder = CLIPTextModelWrapper.from_pretrained(
        'models', subfolder="encoder", torch_dtype=torch.float16
    )

    unet = UNet2DConditionModel.from_pretrained(
        'models', subfolder="arc2face", torch_dtype=torch.float16
    )

    pipeline = StableDiffusionPipeline.from_pretrained(
        base_model,
        text_encoder=encoder,
        unet=unet,
        torch_dtype=torch.float16,
        safety_checker=None
    )

    # You can use any SD-compatible schedulers and steps, just like with Stable Diffusion.
    # By default, we use DPMSolverMultistepScheduler with 25 steps, which produces very good
    # results in just a few seconds.
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline = pipeline.to('cuda')

    # Pick an image and extract the ID-embedding:
    app = FaceAnalysis(name='antelopev2', root='./', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))

    img = np.array(Image.open(args.path_input))[:,:,::-1]

    faces = app.get(img)   # detect face

    if len(faces) == 0:   # no face detected
        raise Exception(f'No face detected in image: \'{args.path_input}\'')
    else:
        faces = sorted(faces, key=lambda x:(x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]))[-1]  # select largest face (if more than one detected)
        src_id_emb = torch.tensor(faces['embedding'], dtype=torch.float16)[None].cuda()

        # Generate new identity embedding
        new_id_emb = rotate_embedding_by_cosine_similarity(src_id_emb, args.similarity)

        new_id_emb = new_id_emb/torch.norm(new_id_emb, dim=1, keepdim=True)   # normalize embedding
        new_id_emb = project_face_embs(pipeline, new_id_emb)    # pass through the encoder

        # Generate images:
        print(f'Generating {args.num_samples} new images...')
        images = pipeline(prompt_embeds=new_id_emb, num_inference_steps=25, guidance_scale=3.0, num_images_per_prompt=args.num_samples).images
        output_folder = f"{os.path.splitext(args.path_input)[0]}_newId_sim={args.similarity}"
        os.makedirs(output_folder, exist_ok=True)
        for i, img in enumerate(images):
            output_img_name = os.path.splitext(os.path.basename(args.path_input))[0]
            path_output_img = os.path.join(output_folder, f"{output_img_name}_newID_newSample_{i}.png")
            print(f"Saving output img: \'{path_output_img}\'")
            img.save(path_output_img)

