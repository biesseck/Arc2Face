import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
import numpy as np
import random
from PIL import Image
from insightface.app import FaceAnalysis
import face_alignment

# Local project imports
from diffusers import StableDiffusionPipeline, UNet2DConditionModel, DPMSolverMultistepScheduler
from arc2face import CLIPTextModelWrapper, project_face_embs, image_align, ReferenceAdapter
from arc2face.exp_utils import ExpressionEncoder, run_smirk

def main():
    parser = argparse.ArgumentParser(description="Arc2Face Expression Transfer CLI")
    
    # Required Arguments
    parser.add_argument("--id_image", type=str, required=True, help="Path to the identity face image")
    parser.add_argument("--exp_image", type=str, required=True, help="Path to the expression reference image")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Directory to save results")
    
    # Model Parameters
    parser.add_argument("--ref_scale", type=float, default=1.0, help="Reference Adapter scale (0.0 to 1.0)")
    parser.add_argument("--use_ref", action="store_true", help="Enable Reference Adapter (preserves background/pose)")
    parser.add_argument("--steps", type=int, default=25, help="Number of inference steps")
    parser.add_argument("--guidance", type=float, default=3.0, help="Guidance scale")
    parser.add_argument("--num_images", type=int, default=1, help="Number of images to generate")
    parser.add_argument("--exp_scale", type=float, default=1.0, help="Expression Adapter scale")
    parser.add_argument("--seed", type=int, default=None, help="Manual seed")

    args = parser.parse_args()

    # Setup Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    print(f"Using device: {device}")

    # Initialization
    app = FaceAnalysis(name='antelopev2', root='./', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(256, 256))
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False, device=device)

    # Load Pipeline
    base_model = 'stable-diffusion-v1-5/stable-diffusion-v1-5'
    encoder = CLIPTextModelWrapper.from_pretrained('models', subfolder="encoder", torch_dtype=dtype)
    unet = UNet2DConditionModel.from_pretrained('models', subfolder="arc2face", torch_dtype=dtype)
    
    pipeline = StableDiffusionPipeline.from_pretrained(
        base_model, text_encoder=encoder, unet=unet, torch_dtype=dtype, safety_checker=None
    )
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline = pipeline.to(device)

    # Load Adapters
    pipeline.load_ip_adapter("models", subfolder="exp_adapter", weight_name="exp_adapter.bin", image_encoder_folder=None)
    pipeline.load_lora_weights("models/ref_adapter", weight_name="pytorch_lora_weights.safetensors", adapter_name="ref")

    ref_unet = UNet2DConditionModel.from_pretrained('models', subfolder="arc2face", torch_dtype=dtype).to(device)
    ref_adapter_w = ReferenceAdapter(ref_unet, mode="write")
    ref_adapter_r = ReferenceAdapter(pipeline.unet, mode="read", cfg=True)

    smirk_encoder = ExpressionEncoder(n_exp=50).to(device)
    checkpoint = torch.load('models/smirk/SMIRK_em1.pt')
    checkpoint_encoder = {k.replace('smirk_encoder.expression_encoder.', ''): v for k, v in checkpoint.items() if 'smirk_encoder.expression_encoder.' in k}
    smirk_encoder.load_state_dict(checkpoint_encoder)
    smirk_encoder.eval()

    # Image Processing Logic
    pipeline.set_ip_adapter_scale(args.exp_scale)
    pipeline.set_adapters("ref", args.ref_scale if args.use_ref else 0.0)

    img = Image.open(args.id_image)
    if args.use_ref:
        face_landmarks, _, bboxes = fa.get_landmarks(np.array(img), return_bboxes=True)
        if face_landmarks is None:
            print("Error: Face detection failed on ID image.")
            return
        lmks = face_landmarks[np.argmax([(b[2]-b[0])*(b[3]-b[1]) for b in bboxes])] if len(face_landmarks) > 1 else face_landmarks[0]
        img = image_align(img, lmks, output_size=512)

    faces = app.get(np.array(img)[:,:,::-1])
    if not faces:
        print("Error: Could not extract ID embedding.")
        return
    
    faces = sorted(faces, key=lambda x:(x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]))[-1]
    id_emb = torch.tensor(faces['embedding'], dtype=dtype)[None].to(device)
    id_emb = id_emb/torch.norm(id_emb, dim=1, keepdim=True)
    id_emb = project_face_embs(pipeline, id_emb)

    # Expression Extraction
    exp_img = np.array(Image.open(args.exp_image))
    outputs = run_smirk(smirk_encoder, exp_img, device=device)
    exp_embs = torch.cat([outputs['expression_params'], outputs['eyelid_params'], outputs['jaw_params']], dim=1).to(dtype=dtype)
    exp_adapter_embeds = torch.cat([torch.zeros_like(exp_embs[:,None,:]), exp_embs[:,None,:]], dim=0)
    exp_adapter_embeds = exp_adapter_embeds.repeat_interleave(repeats=args.num_images, dim=0)

    seed = args.seed if args.seed is not None else random.randint(0, np.iinfo(np.int32).max)
    generator = torch.Generator(device=device).manual_seed(seed)

    if args.use_ref:
        ref_img = (torch.tensor(np.array(img), dtype=dtype).to(device).permute(2,0,1)/255)*2-1
        ref_img = torch.stack([ref_img, ref_img]).repeat_interleave(repeats=args.num_images, dim=0)
        ref_img = pipeline.vae.encode(ref_img).latent_dist.sample() * pipeline.vae.config.scaling_factor
        encoder_hidden_states = torch.cat([id_emb, id_emb], dim=0).repeat_interleave(repeats=args.num_images, dim=0)
        ref_unet(ref_img, torch.zeros(ref_img.size(0), device=ref_img.device).long(), encoder_hidden_states, return_dict=False)
        ref_adapter_r.update(ref_adapter_w)

    print(f"Starting inference with seed {seed}...")
    images = pipeline(
        prompt_embeds=id_emb.repeat_interleave(repeats=args.num_images, dim=0),
        ip_adapter_image_embeds=[exp_adapter_embeds],
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        generator=generator
    ).images

    # Cleanup and Save
    ref_adapter_r.clear()
    ref_adapter_w.clear()

    os.makedirs(args.output_dir, exist_ok=True)
    for i, image in enumerate(images):
        save_path = os.path.join(args.output_dir, f"output_id={os.path.splitext(os.path.basename(args.id_image))[0]}_exp={os.path.splitext(os.path.basename(args.exp_image))[0]}_{i}.png")
        image.save(save_path)
        print(f"Saved: {save_path}")

if __name__ == "__main__":
    main()