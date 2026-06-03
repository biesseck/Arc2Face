# Load pipeline using diffusers:
from diffusers import (
    StableDiffusionPipeline,
    UNet2DConditionModel,
    DPMSolverMultistepScheduler,
    LCMScheduler
)

from arc2face import CLIPTextModelWrapper, project_face_embs

import os
import sys
import torch
import insightface
from insightface.app import FaceAnalysis
from PIL import Image
import numpy as np
import argparse
import re
import time


parser = argparse.ArgumentParser()
# parser.add_argument("--path-input", type=str, default="assets/examples/joacquin.png")
parser.add_argument("--path-dataset", type=str, default="/nobackup3/bjgbiesseck/Arc2Face_agedb-400")
parser.add_argument("--num-samples",  type=int, default=5)
parser.add_argument("--idx-start",    type=int, default=0)
parser.add_argument("--path-output",  type=str, default="")
parser.add_argument("--lcm-lora",     action='store_true')

args = parser.parse_args()
assert os.path.isdir(args.path_dataset), f"Error: dir not found \'{args.path_dataset}\'"





def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


def get_all_files_in_path(folder_path, file_extension=['.jpg','.jpeg','.png', '.npy', '.pt'], pattern=''):
    num_files_found = 0
    file_list = []
    for root, _, files in os.walk(folder_path):
        for filename in files:
            path_file = os.path.join(root, filename)
            for ext in file_extension:
                if pattern in path_file and path_file.lower().endswith(ext.lower()):
                    file_list.append(path_file)
                    num_files_found += 1
                    print(f'    Found {num_files_found}', end='\r')
                    # print(f'Found files: {len(file_list)}', end='\r')
    print()
    file_list = natural_sort(file_list)
    return file_list


def get_immediate_subdirs(parent_dir=''):
    subdirs = [os.path.join(parent_dir, name) for name in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, name))]
    subdirs = natural_sort(subdirs)
    return subdirs


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
    det_fr_model = FaceAnalysis(name='antelopev2', root='./', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    det_fr_model.prepare(ctx_id=0, det_size=(640, 640))
    return det_fr_model


def get_face_recognition_model():
    model = insightface.model_zoo.get_model("./models/antelopev2/arcface.onnx")
    model.prepare(ctx_id=0)
    return model





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

if args.lcm_lora:
    pipeline.load_lora_weights("latent-consistency/lcm-lora-sdv1-5")
    pipeline.scheduler = LCMScheduler.from_config(pipeline.scheduler.config)

pipeline = pipeline.to('cuda')



# Pick an image and extract the ID-embedding:
app = FaceAnalysis(name='antelopev2', root='./', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))


fr_model = get_face_recognition_model()


if not args.path_output:
    args.path_output = f"{args.path_dataset}_newSynthSamples_Arc2Face".replace(' ','')
else:
    args.path_output = os.path.join(args.path_output, f"{args.path_dataset.split('/')[-1]}_newSynthSamples_Arc2Face".replace(' ',''))
# os.makedirs(args.path_output, exist_ok=True)


print(f'Searching all files in path \'{args.path_dataset}\'')
all_imgs_paths = get_all_files_in_path(args.path_dataset)
# print('len(all_imgs_paths):', len(all_imgs_paths))
# sys.exit(0)
total_time = 0.0
for idx_img, path_img in enumerate(all_imgs_paths):
    start_time = time.time()
    if idx_img >= args.idx_start:
        print(f'{idx_img}/{len(all_imgs_paths)} - Loading \'{path_img}\'')
        # img = np.array(Image.open(args.path_input))[:,:,::-1]
        img = np.array(Image.open(path_img))
        if len(img.shape) == 2:  # monochromatic image
            img = np.dstack([img] * 3)
        img = img[:,:,::-1]

        if img.shape[0]==112 and img.shape[1]==112:   # face already aligned/cropped
            print('    Face already aligned/cropped!')
            id_emb = fr_model.get_feat(img)
            id_emb = np.squeeze(id_emb)        
        else:
            print('    Detecting face...')
            faces = app.get(img)   # detect face
            # print('faces:', faces)

            if len(faces) == 0:   # no face detected
                raise Exception(f'No face detected in image: \'{path_img}\'')
            else:
                faces = sorted(faces, key=lambda x:(x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]))[-1]  # select largest face (if more than one detected)
                id_emb = faces['embedding']

        id_emb = torch.tensor(id_emb, dtype=torch.float16)[None].cuda()
        id_emb = id_emb/torch.norm(id_emb, dim=1, keepdim=True)   # normalize embedding
        id_emb = project_face_embs(pipeline, id_emb)    # pass through the encoder

        # Generate images:
        print(f'    Generating {args.num_samples} new samples...')
        if args.lcm_lora:
            images = pipeline(prompt_embeds=id_emb, num_inference_steps=2, guidance_scale=1.0, num_images_per_prompt=args.num_samples).images
        else:
            images = pipeline(prompt_embeds=id_emb, num_inference_steps=25, guidance_scale=3.0, num_images_per_prompt=args.num_samples).images
        # print('images:', images)
        # output_folder = args.path_output
        output_sub_folder = os.path.join(args.path_output, os.path.splitext(os.path.basename(path_img))[0])
        os.makedirs(output_sub_folder, exist_ok=True)
        for i, img in enumerate(images):
            output_img_name = os.path.splitext(os.path.basename(path_img))[0]
            path_output_img = os.path.join(output_sub_folder, f"{output_img_name}_newSample_{i}.png")
            print(f"    Saving new sample img: \'{path_output_img}\'")
            img.save(path_output_img)

    else:
        print(f'{idx_img}/{len(all_imgs_paths)} - Skipping \'{path_img}\'')

    exec_time = time.time() - start_time
    total_time += exec_time
    remain_time = exec_time * (len(all_imgs_paths)-idx_img+1)
    print('      Exec time:          %.2fsec    %.2fmin    %.2fhour' % (exec_time, exec_time/60, exec_time/3600))
    print('      Total elapsed time: %.2fsec    %.2fmin    %.2fhour' % (total_time, total_time/60, total_time/3600))
    print('      Remaining time:     %.2fsec    %.2fmin    %.2fhour' % (remain_time, remain_time/60, remain_time/3600))
    print('------------')
    # sys.exit(0)
