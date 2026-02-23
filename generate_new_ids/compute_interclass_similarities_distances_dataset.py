# duo     (bjgbiesseck_MICA) export CUDA_VISIBLE_DEVICES=0; python compute_interclass_similarities_distances_dataset.py --input-path /datasets2/1st_frcsyn_wacv2024/datasets/real/1_CASIA-WebFace/imgs_crops_112x112_FACE_EMBEDDINGS --metric cosine_2d --file_ext .npy --mean_embedd_str mean_embedding
# diolkos (bjgbiesseck_MICA) export CUDA_VISIBLE_DEVICES=0; python compute_interclass_similarities_distances_dataset.py --input-path /nobackup/unico/datasets/face_recognition/synthetic/dcface_0.5m_oversample_xid/record/imgs_FACE_EMBEDDINGS --metric cosine_2d --file_ext .npy --mean_embedd_str mean_embedding


import os
import sys

import argparse
import random
import socket
import time
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
# from pytorch3d.io import load_obj
# from pytorch3d.loss import chamfer_distance
from mpl_toolkits.mplot3d import Axes3D

import glob
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# from pytorch3d.io import load_obj, load_ply
# from pytorch3d.loss import chamfer_distance


def get_parts_indices(sub_folders, divisions):
    begin_div = []
    end_div = []
    div_size = int(len(sub_folders) / divisions)
    remainder = int(len(sub_folders) % divisions)

    for i in range(0, divisions):
        begin_div.append(i*div_size)
        end_div.append(i*div_size + div_size)
    
    end_div[-1] += remainder

    # print('begin_div:', begin_div)
    # print('end_div:', end_div)
    # sys.exit(0)
    return begin_div, end_div


def load_sample(file_path):
    if file_path.endswith('.obj'):
        verts, _ = load_obj(file_path)
        vertices = verts.verts_packed()
    elif file_path.endswith('.ply'):
        data = load_ply(file_path)
        # vertices = data['vertices']
        vertices = data[0]
    elif file_path.endswith('.npy'):
        vertices = np.load(file_path)
        vertices = torch.from_numpy(vertices)
    else:
        raise ValueError("Unsupported file format. Only .obj and .ply files are supported.")
    return vertices
    

def compute_chamfer_distance(points1, points2):
    chamfer_dist = chamfer_distance(points1.unsqueeze(0), points2.unsqueeze(0))
    return chamfer_dist[0]


def compute_cosine_similarity(array1, array2, normalize=True):
    if array1.shape[0] == 1:
        array1 = array1[0]
    if array2.shape[0] == 1:
        array2 = array2[0]

    if isinstance(array1, np.ndarray):
         array1 = torch.from_numpy(array1)
    if isinstance(array2, np.ndarray):
         array2 = torch.from_numpy(array2)
    
    if normalize == True:
        array1 = torch.nn.functional.normalize(array1, dim=0)
        array2 = torch.nn.functional.normalize(array2, dim=0)
    cos_sim = nn.CosineSimilarity(dim=0, eps=1e-6)(array1, array2)
    return cos_sim


def compute_cosine_similarity_1_to_N(array1, array2, normalize=True):
    if isinstance(array1, np.ndarray):
         array1 = torch.from_numpy(array1)
    if isinstance(array2, np.ndarray):
         array2 = torch.from_numpy(array2)
        
    if len(array1.shape) == 1:
        array1 = torch.unsqueeze(array1, 0)
    if len(array2.shape) == 1:
        array2 = torch.unsqueeze(array2, 0)

    if normalize:
        array1 = F.normalize(array1, p=2, dim=1)
        array2 = F.normalize(array2, p=2, dim=1)
    similarity = torch.mm(array2, array1.T).squeeze(1)
    return similarity


def compute_euclidean_distance(array1, array2, normalize=True):
    # print('array1.shape:', array1.shape)
    if normalize == True:
        array1 = torch.nn.functional.normalize(array1, dim=0)
        array2 = torch.nn.functional.normalize(array2, dim=0)
    eucl_dist = torch.norm(array1 - array2)
    return eucl_dist


def find_files_by_extension(folder_path, target_file_substr, extension, ignore_file_with=''):
    matching_files = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            # Check if the file ends with the specified extension
            if file.endswith(extension):
                if target_file_substr in file and (ignore_file_with == '' or not ignore_file_with in file):
                    file_path = os.path.join(root, file)
                    matching_files.append(file_path)
    return sorted(matching_files)


def get_leaf_subdirs(base_path):
    base = Path(base_path)
    return [
        str(p) for p in base.rglob("*")
        if p.is_dir() and not any(child.is_dir() for child in p.iterdir())
    ]


def main(args):
    assert args.part < args.divs, f'Error, args.part ({args.part}) >= args.divs ({args.divs}), but should be args.part ({args.part}) < args.divs ({args.divs})'

    dataset_path = args.input_path.rstrip('/')
    output_path = f'{dataset_path}_INTERCLASS_SIMILARITIES_{args.metric}'
    os.makedirs(output_path, exist_ok=True)

    print('dataset_path:', dataset_path)
    print('Searching subject subfolders...')
    # subjects_paths = sorted([os.path.join(dataset_path,subj) for subj in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, subj))])
    subjects_paths = get_leaf_subdirs(dataset_path)
    # print('subjects_paths:', subjects_paths)
    print(f'Found {len(subjects_paths)} subjects!')

    begin_parts, end_parts = get_parts_indices(subjects_paths, args.divs)
    idx_subj_begin, idx_subj_end = begin_parts[args.part], end_parts[args.part]
    num_subjs_part = idx_subj_end - idx_subj_begin 
    print('\nbegin_parts:', begin_parts)
    print('end_parts:  ', end_parts)
    print(f'idx_subj_begin: {idx_subj_begin}    idx_subj_end: {idx_subj_end}')
    print('')
    # sub_folders = subjects_paths[begin_parts[args.part]:end_parts[args.part]]


    # Load 1 sample to get its size
    sample_path = find_files_by_extension(subjects_paths[0], args.mean_embedd_str, args.file_ext, ignore_file_with='')
    assert len(sample_path) > 0, f'Error, no such file with substr \'{args.mean_embedd_str}\' and ext \'{args.file_ext}\' in dir \'{subjects_paths[0]}\''
    assert len(sample_path) < 2, f'Error, more than 1 file with substr \'{args.mean_embedd_str}\' and ext \'{args.file_ext}\' in dir \'{subjects_paths[0]}\': {sample_path}'
    sample_path = sample_path[0]
    data = load_sample(sample_path)
    # print('data.shape:', data.shape, '    torch.squeeze(data).shape[0]:', torch.squeeze(data).shape[0])
    # sys.exit(0)



    print('Loading subjects mean embeddings...')
    subj_mean_embedds = torch.zeros((len(subjects_paths), torch.squeeze(data).shape[0]), dtype=data.dtype)
    # print('subj_mean_embedds.shape:', subj_mean_embedds.shape, '    subj_mean_embedds.dtype:', subj_mean_embedds.dtype)
    # sys.exit(0)
    for idx_subj, subj_path in enumerate(subjects_paths):
        print(f'{idx_subj}/{len(subjects_paths)} - Loading subject mean embedding in \'{subj_path}\'', end='\r')
        ignore_file_with = ''
        samples_paths = find_files_by_extension(subj_path, args.mean_embedd_str, args.file_ext, ignore_file_with)
        assert len(samples_paths) > 0, f'Error, no such file with substr \'{args.mean_embedd_str}\' and ext \'{args.file_ext}\' in dir \'{subj_path}\''
        assert len(samples_paths) < 2, f'Error, more than 1 file with substr \'{args.mean_embedd_str}\' and ext \'{args.file_ext}\' in dir \'{subj_path}\': {samples_paths}'
        # print('samples_paths:', samples_paths)
        # print('len(samples_paths):', len(samples_paths))
        # sys.exit(0)

        for idx_sf, sample_path in enumerate(samples_paths):
            # print(f'Loading samples - {idx_sf}/{len(samples_paths)}...', end='\r')
            data = load_sample(sample_path)
            # print('data.shape:', data.shape, '    type(data):', type(data), '    device:', {data.device})
            subj_mean_embedds[idx_subj] = data
        # print('')
        # print('subj_mean_embedds:', subj_mean_embedds)
        # print('len(loaded_samples):', len(loaded_samples))
        # sys.exit(0)
    print('')
    print(f'    subj_mean_embedds.shape: {subj_mean_embedds.shape}    dtype: {subj_mean_embedds.dtype}    device: {subj_mean_embedds.device}')
    # sys.exit(0)


    num_subjs_to_compare = -1       # ALL SIMILARITIES
    # num_subjs_to_compare = 10     # sampling

    if num_subjs_to_compare < 0:    # ALL SIMILARITIES
        num_total_similarities = int(((subj_mean_embedds.shape[0]-1 + 1)*(subj_mean_embedds.shape[0]-1)) / 2)
    else:
        num_total_similarities = int(subj_mean_embedds.shape[0] * num_subjs_to_compare)

    # print(f'\nAllocating similarities tensor of shape ({num_total_similarities},)')
    # all_interclass_similarities = torch.zeros((num_total_similarities,), dtype=subj_mean_embedds.dtype)
    # print('    all_interclass_similarities.shape:', all_interclass_similarities.shape, '    dtype:', all_interclass_similarities.dtype, '    device:', all_interclass_similarities.device)
    # sys.exit(0)


    print()
    # start_idx_all_sims = 0
    total_spent_time = 0.0
    num_computed_similarities = 0
    for idx_subj, subj_path in enumerate(subjects_paths[:-1]):
        subj_start_time = time.time()
        print('------------------')
        print('Computing interclass similarities/distances...')
        print(f'    idx_subj {idx_subj}/{subj_mean_embedds.shape[0]}')

        subj_mean_embedd = subj_mean_embedds[idx_subj]
        # print('    subj_mean_embedd.shape:', subj_mean_embedd.shape, '    dtype:', subj_mean_embedd.dtype, '    device:', subj_mean_embedd.device)

        if num_subjs_to_compare < 0:  # ALL SIMILARITIES
            other_subj_mean_embedd = subj_mean_embedds[idx_subj+1:]
        else:    # sampling
            raise Exception('Functionality not yet implemented!')

        # print('    other_subj_mean_embedd.shape:', other_subj_mean_embedd.shape, '    dtype:', other_subj_mean_embedd.dtype, '    device:', other_subj_mean_embedd.device)

        subj_similarities = compute_cosine_similarity_1_to_N(subj_mean_embedd, other_subj_mean_embedd)
        # print('    subj_similarities:', subj_similarities)
        print('    subj_similarities.shape:', subj_similarities.shape)
        num_computed_similarities += subj_similarities.shape[0]
        # print('    num_computed_similarities:', num_computed_similarities)

        # end_idx_all_sims = start_idx_all_sims + subj_similarities.shape[0]
        # all_interclass_similarities[start_idx_all_sims:end_idx_all_sims] = subj_similarities
        # start_idx_all_sims = end_idx_all_sims + 1

        subj_output_dir_path = subj_path.replace(dataset_path, output_path)
        # print('    subj_output_dir_path:', subj_output_dir_path)
        os.makedirs(subj_output_dir_path, exist_ok=True)
        subj_output_file_path = os.path.join(subj_output_dir_path, f'interclass_similarities_{args.metric}.pt')
        print(f'    Saving similarities: \'{subj_output_file_path}\'')
        torch.save(subj_similarities, subj_output_file_path)

        subj_elapsed_time = (time.time() - subj_start_time)
        avg_sim_elapsed_time = subj_elapsed_time / float(subj_similarities.shape[0])
        total_spent_time += subj_elapsed_time
        est_time_to_complete = avg_sim_elapsed_time * (num_total_similarities - num_computed_similarities)
        print('        subj_elapsed_time: %.2f sec' % (subj_elapsed_time))
        print('        avg_sim_elapsed_time: %f sec' % (avg_sim_elapsed_time))
        print(f'        num_computed_similarities: {num_computed_similarities}/{num_total_similarities} ({((num_computed_similarities/num_total_similarities)*100):.2f}%)')
        print('        Total spent time: %.2fsec, %.2fmin, %.2fhour' % (total_spent_time, total_spent_time/60.0, total_spent_time/3600.0))
        print('        Estimate time to complete: %.2fsec, %.2fmin, %.2fhour' % (est_time_to_complete, est_time_to_complete/60.0, est_time_to_complete/3600.0))

        # sys.exit(0)


    print('\nFinished!')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-path', type=str, default='/datasets2/1st_frcsyn_wacv2024/datasets/real/1_CASIA-WebFace/imgs_crops_112x112_FACE_EMBEDDINGS')

    # parser.add_argument('--str_begin', default='', type=str, help='Substring to find and start processing')
    # parser.add_argument('--str_end', default='', type=str, help='Substring to find and stop processing')
    # parser.add_argument('--str_pattern', default='', type=str, help='Substring to find and stop processing')

    parser.add_argument('--divs', default=1, type=int, help='How many parts to divide paths list (useful to paralelize process)')
    parser.add_argument('--part', default=0, type=int, help='Specific part to process (works only if -div > 1)')

    parser.add_argument('--metric', default='cosine_2d', type=str, help='Options: chamfer, cosine_3dmm, euclidean_3dmm, cosine_2d')
    parser.add_argument('--file_ext', default='.npy', type=str, help='.ply, .obj, .npy')
    parser.add_argument('--mean_embedd_str', default='mean_embedding', type=str, help='')

    # parser.add_argument('--dont_replace_existing_files', action='store_true', help='')

    args = parser.parse_args()

    main(args)
