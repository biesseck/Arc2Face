import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import insightface
from insightface.app import FaceAnalysis
from PIL import Image
import numpy as np
import argparse
import random
import re

from diffusers import (
    StableDiffusionPipeline,
    UNet2DConditionModel,
    DPMSolverMultistepScheduler,
)

from arc2face import CLIPTextModelWrapper, project_face_embs


def parse_list_arg(arg_string):
    try:
        values = [float(item.strip().strip('[').strip(']')) for item in arg_string.split(',')]
        return values
    except ValueError:
        raise argparse.ArgumentTypeError("List values must be floats separated by commas, e.g., '0.5,0.69'")

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subj-path",           type=str, default="/nobackup3/bjgbiesseck/CASIA-Webface/imgs_crops_112x112_FACE_EMBEDDINGS_R100_WebFace42M_ArcFace/0")
    parser.add_argument("--similarity-range",    type=parse_list_arg, default=[0.5,0.69], required=True, help='A list of float values separated by commas, e.g., 0.5,0.69 or [0.5,0.69]')
    parser.add_argument("--num-samples-by-id",   type=int, default=10)
    parser.add_argument("--batch",               type=int, default=10)
    parser.add_argument("--num-inference-steps", type=int, default=25)
    parser.add_argument("--output-path",         type=str, default="./output_transfer_style")
    args = parser.parse_args()
    return args


def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


def get_all_files_in_path(folder_path, file_extension=['.jpg','.jpeg','.png', '.npy', '.pt'], pattern='', ignore_pattern=''):
    file_list = []
    for root, _, files in os.walk(folder_path):
        for filename in files:
            path_file = os.path.join(root, filename)
            for ext in file_extension:
                if pattern in path_file and path_file.lower().endswith(ext.lower()):
                    if not ignore_pattern or not ignore_pattern in path_file:
                        file_list.append(path_file)
                    # print(f'Found files: {len(file_list)}', end='\r')
    # print()
    file_list = natural_sort(file_list)
    return file_list


def get_arc2face_model():
    base_model = 'stable-diffusion-v1-5/stable-diffusion-v1-5'
    encoder = CLIPTextModelWrapper.from_pretrained(
        '../models', subfolder="encoder", torch_dtype=torch.float16
    )

    unet = UNet2DConditionModel.from_pretrained(
        '../models', subfolder="arc2face", torch_dtype=torch.float16
    )

    pipeline = StableDiffusionPipeline.from_pretrained(
        base_model,
        text_encoder=encoder,
        unet=unet,
        torch_dtype=torch.float16,
        safety_checker=None
    )

    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline = pipeline.to('cuda')
    return pipeline


def get_face_detection_and_recognition_model():
    det_fr_model = FaceAnalysis(name='antelopev2', root='../', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    det_fr_model.prepare(ctx_id=0, det_size=(640, 640))
    return det_fr_model


def get_face_recognition_model():
    model = insightface.model_zoo.get_model("../models/antelopev2/arcface.onnx")
    model.prepare(ctx_id=0)
    return model


def get_random_float(min_max_list):
    if len(min_max_list) == 1:
        min_max_list.append(min_max_list[0])
    min_val, max_val = min_max_list
    factor = 100
    
    scaled_min = round(min_val * factor)
    scaled_max = round(max_val * factor)
    
    random_int = random.randint(scaled_min, scaled_max)
    random_float = random_int / factor
    return random_float


def load_embedding(embedd_path='', device='cuda:0'):
    if embedd_path.endswith('.pt'):
        embedd = torch.load(embedd_path).detach()
        embedd = torch.squeeze(embedd)
    elif embedd_path.endswith('.npy'):
        embedd = np.load(embedd_path)
        embedd = np.squeeze(embedd)
        embedd = torch.tensor(embedd, device=device)
        # print('embedd.device:', embedd.device)
    else:
        raise Exception(f'File format not supported: \'{embedd_path}\'')
    return embedd



def rotate_embedding_by_cosine_similarity(v1: torch.Tensor, cosine_similarity: float, random_vector=None) -> torch.Tensor:
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

    if random_vector is None:
        print('random_vector IS None')
        random_vector = torch.randn_like(v1)
    else:
        print('random_vector ISN\'T None')
        random_vector = torch.tensor(random_vector)
    
    projection_onto_u1 = torch.dot(random_vector, u1) * u1
    
    u2_raw = random_vector - projection_onto_u1
    
    u2_norm = torch.linalg.norm(u2_raw).to(torch.float32)
    
    if torch.isclose(u2_norm, torch.tensor(0.0).to(torch.float32)):
        raise RuntimeError("Failed to generate a non-collinear random vector. Try running again.")

    u2 = u2_raw / u2_norm
    u1_prime = (u1 * torch.cos(theta)) + (u2 * torch.sin(theta))
    v1_prime = u1_prime * v1_norm
    v1_prime = torch.unsqueeze(v1_prime, 0)
    return v1_prime, random_vector



def transfer_perturbation(e_u, e_i, f_u):
    e_u = e_u / e_u.norm()
    f_u = f_u / f_u.norm()
    e_i = e_i / e_i.norm()

    perturbation_ortho = e_i - torch.dot(e_i.flatten(), e_u.flatten()) * e_u
    f_i_unnorm = f_u + perturbation_ortho
    return f_i_unnorm



def transport_perturbation(e_u, e_i, f_u, v_random, cosine_similarity):
    """
    Transports perturbation while strictly preserving the geodesic distance 
    (angular norm) between the identity anchor and the sample.
    """
    # 1. Measure the exact angular distance (phi) of the original perturbation
    # Cosine similarity between sample and its mean identity
    cos_phi = torch.clamp(torch.dot(e_i.flatten(), e_u.flatten()), -1.0, 1.0)
    phi = torch.acos(cos_phi) # This is the "true" magnitude of change
    
    # If there's virtually no perturbation, return the target identity directly
    if phi < 1e-7:
        return f_u.clone()
        
    # 2. Isolate the direction of the original perturbation orthogonal to e_u
    p_dir = e_i - cos_phi * e_u
    p_dir = p_dir / p_dir.norm()
    
    # 3. Decompose p_dir relative to your rotation plane (e_u, v_random)
    dot_v = torch.dot(p_dir.flatten(), v_random.flatten())
    
    # Extract out-of-plane directional component
    p_dir_ortho = p_dir - (torch.dot(p_dir.flatten(), e_u.flatten()) * e_u) - (dot_v * v_random)
    
    # 4. Rotate the in-plane direction component by theta
    theta_rad = torch.acos(torch.tensor(cosine_similarity, device=e_u.device))
    v_rotated = -torch.sin(theta_rad) * e_u + torch.cos(theta_rad) * v_random
    
    # 5. Reconstruct the transported direction vector on the target identity f_u
    # (Note: dot_eu for p_dir is 0 by construction, so we only transport dot_v)
    p_dir_transported = (dot_v * v_rotated) + p_dir_ortho
    p_dir_transported = p_dir_transported / p_dir_transported.norm()
    
    # 6. Generate the final synthetic embedding using Spherical Linear Interpolation (Slerp)
    # This guarantees f_i is perfectly on the sphere *without* post-normalization shrinkage
    f_i = torch.cos(phi) * f_u + torch.sin(phi) * p_dir_transported
    
    return f_i



def parallel_transport_spherical_clean(e_u, e_i, f_u):
    """
    Transports the sample embedding e_i from the source identity e_u 
    to the target identity f_u along the unique geodesic path connecting them.
    
    Guarantees: cossim(f_i, f_u) == cossim(e_i, e_u)
    """
    # 1. Enforce strict unit norm (Hypersphere mapping)
    e_u = e_u / e_u.norm()
    f_u = f_u / f_u.norm()
    e_i = e_i / e_i.norm()
    
    # 2. Compute the cosine similarity and angle (phi) of the original sample
    cos_phi = torch.clamp(torch.dot(e_i.flatten(), e_u.flatten()), -1.0, 1.0)
    phi = torch.acos(cos_phi)
    
    if phi < 1e-7:
        return f_u.clone()  # No variation to transport
        
    # 3. Isolate the directional component of the perturbation orthogonal to e_u
    p_dir = e_i - cos_phi * e_u
    p_dir = p_dir / p_dir.norm()
    
    # 4. Construct the parallel transport matrix/mechanism from e_u to f_u
    # We find the part of p_dir that lies in the plane spanned by e_u and f_u
    cos_theta = torch.clamp(torch.dot(e_u.flatten(), f_u.flatten()), -1.0, 1.0)
    
    # If identities are identical or opposites, rotation plane collapses
    if abs(cos_theta) > 0.99999:
        # Just apply the original orthogonal direction directly onto f_u
        f_i = torch.cos(phi) * f_u + torch.sin(phi) * p_dir
        return f_i / f_i.norm()
        
    # Define the orthonormal basis for the identity-rotation plane
    u1 = e_u
    u2 = f_u - cos_theta * e_u
    u2 = u2 / u2.norm()
    
    # 5. Project the perturbation direction onto this identity-rotation plane
    proj_u1 = torch.dot(p_dir.flatten(), u1.flatten())
    proj_u2 = torch.dot(p_dir.flatten(), u2.flatten())
    
    # Isolate the component completely outside the identity-rotation plane
    p_dir_orthogonal = p_dir - (proj_u1 * u1) - (proj_u2 * u2)
    
    # 6. Transport the in-plane components to the new identity frame
    # u1 maps to f_u, u2 maps to the orthogonal vector to f_u within the same plane
    u2_transported = -torch.sqrt(1.0 - cos_theta**2) * e_u + cos_theta * u2
    u2_transported = u2_transported / u2_transported.norm()
    
    p_dir_transported = (proj_u1 * f_u) + (proj_u2 * u2_transported) + p_dir_orthogonal
    p_dir_transported = p_dir_transported / p_dir_transported.norm()
    
    # 7. Reconstruct on the target sphere surface using SLERP rules
    f_i = torch.cos(phi) * f_u + torch.sin(phi) * p_dir_transported
    
    # Final clamp normalization to defeat float16/float32 rounding errors
    return f_i / f_i.norm()




if __name__ == '__main__':

    args = parse_arguments()
    assert args.num_samples_by_id >= args.batch, f"Error, --num-samples-by-id must be greater or equal to --batch"
    assert args.num_samples_by_id % args.batch == 0, f"Error, --num-samples-by-id must be a multiple of --batch"
    assert os.path.isfile(args.subj_path) or os.path.isdir(args.subj_path), f"Error: no such file or dir \'{args.subj_path}\'"

    pipeline = get_arc2face_model()
    det_fr_model = get_face_detection_and_recognition_model()
    fr_model = get_face_recognition_model()

    if os.path.isfile(args.subj_path):
        if args.subj_path.endswith('.npy') or args.subj_path.endswith('.pt'):
            src_id_emb = load_embedding(args.subj_path)
        else:
            img = np.array(Image.open(args.subj_path))[:,:,::-1]
            if img.shape[0] == 112 and img.shape[1] == 112:   # face already aligned
                src_id_emb = fr_model.get_feat(img)
            else:
                faces = det_fr_model.get(img)   # detect face
                if len(faces) == 0:   # no face detected
                    raise Exception(f'No face detected in image: \'{args.subj_path}\'')
                faces = sorted(faces, key=lambda x:(x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]))[-1]  # select largest face (if more than one detected)
                src_id_emb = faces['embedding']

    elif os.path.isdir(args.subj_path):
        paths_files = get_all_files_in_path(args.subj_path, file_extension=['.jpg','.jpeg','.png', '.npy', '.pt'], ignore_pattern='_mean_embedding_')
        # for path in paths_files:
        #     print(path)
        # print('len(paths_files):', len(paths_files))
        # sys.exit(0)
        src_id_embedds = torch.zeros((len(paths_files), 512), dtype=torch.float16, device='cuda:0')
        for idx_path, path_file in enumerate(paths_files):
            if path_file.endswith('.npy') or path_file.endswith('.pt'):
                src_id_embedds[idx_path] = load_embedding(path_file)
            else:
                img = np.array(Image.open(path_file))[:,:,::-1]
                if img.shape[0] == 112 and img.shape[1] == 112:   # face already aligned
                    src_id_emb = fr_model.get_feat(img)
                    src_id_emb = torch.tensor(src_id_emb)
                else:
                    faces = fr_model.get(img)   # detect face
                    if len(faces) == 0:   # no face detected
                        raise Exception(f'No face detected in image: \'{path_file}\'')
                    faces = sorted(faces, key=lambda x:(x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]))[-1]  # select largest face (if more than one detected)
                    src_id_emb = faces['embedding']
                src_id_embedds[idx_path] = src_id_emb
        src_id_emb = src_id_embedds.mean(axis=0)

    src_id_emb = torch.tensor(src_id_emb, dtype=torch.float16)[None].cuda()

    similarity = get_random_float(args.similarity_range)
    print('similarity:', similarity)

    new_id_emb, random_vector = rotate_embedding_by_cosine_similarity(src_id_emb, similarity)


    # ------------------------------------------------------------

    # Generate images without transfer style (default Arc2Face):
    print(f'Generating {len(paths_files)} new images (default Arc2Face)...')
    new_id_emb_normalized = new_id_emb/torch.norm(new_id_emb, dim=1, keepdim=True)   # normalize embedding
    new_id_emb_proj = project_face_embs(pipeline, new_id_emb_normalized)    # pass through the encoder
    num_runs = int(len(paths_files) / args.batch)
    all_generated_images = []
    for idx_run in range(num_runs):
        print(f'    run {idx_run}/{num_runs}')
        images = pipeline(prompt_embeds=new_id_emb_proj, num_inference_steps=args.num_inference_steps, guidance_scale=3.0, num_images_per_prompt=args.batch).images
        all_generated_images.extend(images)

    output_folder = f"{os.path.join(args.output_path,args.subj_path.split('/')[-1])}_avgEmbedd_newId_sim={similarity}_defaultArc2Face"
    os.makedirs(output_folder, exist_ok=True)
    for i, img in enumerate(all_generated_images):
        output_img_name = os.path.splitext(os.path.basename(args.subj_path))[0]
        path_output_img = os.path.join(output_folder, f"{output_img_name}_newID_newSample_{i}.png")
        print(f"Saving output img: \'{path_output_img}\'", end='\r')
        img.save(path_output_img)
    print()


    # ------------------------------------------------------------

    # Transfer style face before generating face image
    all_generated_images = []
    print('\n---------------------------------')
    print(f'Generating {len(paths_files)} new images by transfering face style...')
    for idx_emb in range(len(paths_files)):
        # new_sample_emb = new_id_emb + src_id_embedds[idx_emb] - src_id_emb
        # new_sample_emb = transfer_perturbation(src_id_emb, src_id_embedds[idx_emb], new_id_emb)
        # new_sample_emb = transport_perturbation(src_id_emb, src_id_embedds[idx_emb], new_id_emb, random_vector, similarity)
        new_sample_emb = parallel_transport_spherical_clean(src_id_emb, src_id_embedds[idx_emb], new_id_emb)
        # new_sample_emb, _ = rotate_embedding_by_cosine_similarity(src_id_embedds[idx_emb], similarity, random_vector)
        # new_sample_emb = torch.unsqueeze(src_id_embedds[idx_emb], dim=0)

        new_sample_emb = new_sample_emb/torch.norm(new_sample_emb, dim=1, keepdim=True)   # normalize embedding

        print('torch.norm(src_id_embedds[idx_emb]-src_id_emb):', torch.norm(src_id_embedds[idx_emb]-src_id_emb))
        print('    sim:', torch.nn.functional.cosine_similarity(src_id_embedds[idx_emb], src_id_emb))
        print('torch.norm(new_sample_emb-new_id_emb)         :', torch.norm(new_sample_emb-new_id_emb))
        print('    sim:', torch.nn.functional.cosine_similarity(new_sample_emb, new_id_emb))
        

        # Generate images:
        print(f'{idx_emb}/{len(paths_files)} - Generating new image')
        print('    new_sample_emb.shape:', new_sample_emb.shape, '    new_id_emb_proj.shape:', new_id_emb_proj.shape)
        new_id_emb_proj = project_face_embs(pipeline, new_sample_emb)    # pass through the encoder
        # images = pipeline(prompt_embeds=new_id_emb_proj, num_inference_steps=25, guidance_scale=3.0, num_images_per_prompt=args.num_samples_by_id).images
    
        images = pipeline(prompt_embeds=new_id_emb_proj, num_inference_steps=args.num_inference_steps, guidance_scale=3.0, num_images_per_prompt=1).images
        all_generated_images.extend(images)

    output_folder = f"{os.path.join(args.output_path,args.subj_path.split('/')[-1])}_avgEmbedd_newId_sim={similarity}_transferStyle"
    os.makedirs(output_folder, exist_ok=True)
    for i, img in enumerate(all_generated_images):
        output_img_name = os.path.splitext(os.path.basename(args.subj_path))[0]
        path_output_img = os.path.join(output_folder, f"{output_img_name}_newID_newSample_{i}.png")
        print(f"Saving output img: \'{path_output_img}\'", end='\r')
        img.save(path_output_img)
    print()

    print('\nFinished!')