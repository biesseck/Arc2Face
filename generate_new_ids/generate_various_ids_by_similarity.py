'''
cd generate_new_ids
python generate_various_ids_by_similarity.py --path-dataset /hddevice/nobackup3/bjgbiesseck/datasets/face_recognition/CASIA-WebFace/imgs_crops_112x112_FACE_EMBEDDINGS --path-subj-list /hddevice/nobackup3/bjgbiesseck/datasets/face_recognition/CASIA-WebFace/merge_with_dataset_MS-Celeb-1M-ms1m-retinaface-t1-imgs_FACE_EMBEDDINGS_sim-range=[0.5,0.69]/dict_paths_new_subjs_base_subjs.json --similarity-range [0.5,0.69] --num-samples-by-id 50
'''


import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from insightface.app import FaceAnalysis
from PIL import Image
import numpy as np
import argparse
import random
import re
import json

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
    parser.add_argument("--path-dataset",      type=str, default="/hddevice/nobackup3/bjgbiesseck/datasets/face_recognition/CASIA-WebFace/imgs_crops_112x112_FACE_EMBEDDINGS")
    parser.add_argument("--path-subj-list",    type=str, default="/hddevice/nobackup3/bjgbiesseck/datasets/face_recognition/CASIA-WebFace/merge_with_dataset_MS-Celeb-1M-ms1m-retinaface-t1-imgs_FACE_EMBEDDINGS_sim-range=[0.5,0.69]/dict_paths_new_subjs_base_subjs.json")
    parser.add_argument("--similarity-range",  type=parse_list_arg, default=[0.5,0.69], required=True, help='A list of float values separated by commas, e.g., 0.5,0.69 or [0.5,0.69]')
    parser.add_argument("--num-samples-by-id", type=int, default=50)
    parser.add_argument("--path-output",       type=str, default="")
    args = parser.parse_args()
    return args


def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


def get_all_files_in_path(folder_path, file_extension=['.jpg','.jpeg','.png', '.npy', '.pt'], pattern=''):
    file_list = []
    for root, _, files in os.walk(folder_path):
        for filename in files:
            path_file = os.path.join(root, filename)
            for ext in file_extension:
                if pattern in path_file and path_file.lower().endswith(ext.lower()):
                    file_list.append(path_file)
                    # print(f'Found files: {len(file_list)}', end='\r')
    # print()
    file_list = natural_sort(file_list)
    return file_list


def load_json(path_file=''):
    assert os.path.isfile(path_file), f"Error, no such file: \'{path_file}\'"
    with open(path_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def save_json(dict_data={}, path_file=''):
    dir_name = os.path.dirname(path_file)
    if dir_name and not os.path.exists(dir_name):
        os.makedirs(dir_name)
    with open(path_file, 'w', encoding='utf-8') as f:
        json.dump(dict_data, f, indent=4, ensure_ascii=False)


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


def get_face_recognition_model():
    fr_model = FaceAnalysis(name='antelopev2', root='../', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    fr_model.prepare(ctx_id=0, det_size=(640, 640))
    return fr_model


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


def load_embedding(embedd_path=''):
    if embedd_path.endswith('.pt'):
        embedd = torch.load(embedd_path).detach()
        embedd = torch.squeeze(embedd)
    elif embedd_path.endswith('.npy'):
        embedd = np.load(embedd_path)
        embedd = np.squeeze(embedd)
    else:
        raise Exception(f'File format not supported: \'{embedd_path}\'')
    return embedd



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
    assert os.path.isdir(args.path_dataset), f"Error, no such dir: \'{args.path_dataset}\'"
    assert os.path.isfile(args.path_subj_list), f"Error, no such file: \'{args.path_subj_list}\'"

    if not args.path_output:
        args.path_output = f"{args.path_dataset}_newSynthIDs_Arc2Face_sim={args.similarity_range}".replace(' ','')
    else:
        args.path_output = os.path.join(args.path_output, f"{args.path_dataset.split('/')[-1]}_newSynthIDs_Arc2Face_sim={args.similarity_range}".replace(' ',''))

    pipeline = get_arc2face_model()
    fr_model = get_face_recognition_model()

    json_subjs_list = load_json(args.path_subj_list)
    subjs_orig_paths = [subj_path[0][0] for subj_path in json_subjs_list.values()]
    subjs_names = [subj_name.split('/')[-2] for subj_name in subjs_orig_paths]

    for idx_subj, subj_name in enumerate(subjs_names):
        path_dir_subj = os.path.join(args.path_dataset, subj_name)
        path_mean_embedding_subj = get_all_files_in_path(path_dir_subj, file_extension=['.jpg','.jpeg','.png', '.npy', '.pt'], pattern='_mean_embedding_')

        if len(path_mean_embedding_subj) > 0:
            src_id_emb = load_embedding(path_mean_embedding_subj[0])
        else:
            paths_files = get_all_files_in_path(path_dir_subj, file_extension=['.jpg','.jpeg','.png', '.npy', '.pt'])
            src_id_embedds = torch.zeros((len(paths_files), 512))
            for idx_path, path_file in enumerate(paths_files):
                if path_file.endswith('.npy') or path_file.endswith('.pt'):
                    src_id_embedds[idx_path] = load_embedding(path_file)
                else:
                    img = np.array(Image.open(path_file))[:,:,::-1]
                    if img.shape[0] == 112 and img.shape[1] == 112:   # face already aligned
                        # src_id_emb = 
                        pass
                    else:
                        faces = fr_model.get(img)   # detect face
                        if len(faces) == 0:   # no face detected
                            raise Exception(f'No face detected in image: \'{path_file}\'')
                        faces = sorted(faces, key=lambda x:(x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]))[-1]  # select largest face (if more than one detected)
                        src_id_embedds[idx_path] = faces['embedding']
            src_id_emb = src_id_embedds.mean(axis=0)
        
        src_id_emb = torch.tensor(src_id_emb, dtype=torch.float16)[None].cuda()

        similarity = get_random_float(args.similarity_range)
        print('similarity:', similarity)

        # Generate new identity embedding
        new_id_emb = rotate_embedding_by_cosine_similarity(src_id_emb, similarity)
        new_id_emb = new_id_emb/torch.norm(new_id_emb, dim=1, keepdim=True)   # normalize embedding

        # Generate images:
        print(f'{idx_subj}/{len(subjs_names)} Generating {args.num_samples_by_id} new images...')
        new_id_emb = project_face_embs(pipeline, new_id_emb)    # pass through the encoder
        images = pipeline(prompt_embeds=new_id_emb, num_inference_steps=25, guidance_scale=3.0, num_images_per_prompt=args.num_samples_by_id).images

        output_folder = f"{os.path.join(args.path_output,path_dir_subj.split('/')[-1])}_newId_sim={similarity}"
        os.makedirs(output_folder, exist_ok=True)
        for i, img in enumerate(images):
            output_img_name = os.path.splitext(os.path.basename(path_dir_subj))[0]
            path_output_img = os.path.join(output_folder, f"{output_img_name}_newID_newSample_{i}.png")
            print(f"    Saving output img: \'{path_output_img}\'", end='\r')
            img.save(path_output_img)
        print()
        print('------------')

    print('\nFinished!')
        


