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


def compute_euclidean_distance(array1, array2, normalize=True):
    # print('array1.shape:', array1.shape)
    if normalize == True:
        array1 = torch.nn.functional.normalize(array1, dim=0)
        array2 = torch.nn.functional.normalize(array2, dim=0)
    eucl_dist = torch.norm(array1 - array2)
    return eucl_dist


def find_files_by_extension(folder_path, extension, ignore_file_with=''):
    matching_files = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            # Check if the file ends with the specified extension
            if file.endswith(extension):
                file_path = os.path.join(root, file)
                if ignore_file_with == '' or not ignore_file_with in file_path:
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
    output_path = f'{dataset_path}_INTRACLASS_SIMILARITIES_{args.metric}'
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

    print('Computing similarities/distances...\n')
    for idx_subj, subj_path in enumerate(subjects_paths):
        if idx_subj >= idx_subj_begin and idx_subj < idx_subj_end:
            subj_start_time = time.time()

            subj = os.path.basename(subj_path)
            output_subj_path = os.path.join(output_path, subj)
            os.makedirs(output_subj_path, exist_ok=True)

            skip_distances_between_samples = False
            # distances_file_name = 'distances_'+args.metric+'.npy'
            distances_file_name = 'similarities_'+args.metric+'.npy'
            output_distances_path = os.path.join(output_subj_path, distances_file_name)

            skip_distances_to_mean_embedd = False
            # distances_to_mean_file_name = 'distances_'+args.metric+'_to_mean_class_embedd.pkl'
            distances_to_mean_file_name = 'similarities_'+args.metric+'_to_mean_class_embedd.pkl'
            output_distances_to_mean_path = os.path.join(output_subj_path, distances_to_mean_file_name)

            if args.dont_replace_existing_files:
                if os.path.isfile(output_distances_path):
                    skip_distances_between_samples = True
                if os.path.isfile(output_distances_to_mean_path):
                    skip_distances_to_mean_embedd = True
                if skip_distances_between_samples and skip_distances_to_mean_embedd:
                    print(f'Skipping subject {idx_subj-idx_subj_begin}/{num_subjs_part} - \'{subj}\', distances file already exists: \'{output_distances_path}\'')
                    continue

            # if args.dont_replace_existing_files:
            #     if os.path.isfile(output_distances_path) and os.path.isfile(output_distances_to_mean_path):
            #         print(f'Skipping subject {idx_subj-idx_subj_begin}/{num_subjs_part} - \'{subj}\', distances file already exists: \'{output_distances_path}\'')
            #         continue
            
            print(f'{idx_subj}/{len(subjects_paths)} - Searching subject samples in \'{subj_path}\'')
            ignore_file_with = 'mean_embedding'
            samples_paths = find_files_by_extension(subj_path, args.file_ext, ignore_file_with)
            # print('samples_paths:', samples_paths)
            # print('len(samples_paths):', len(samples_paths))
            # sys.exit(0)

            loaded_samples = [None] * len(samples_paths)
            for idx_sf, sample_path in enumerate(samples_paths):
                print(f'Loading samples - {idx_sf}/{len(samples_paths)}...', end='\r')
                data = load_sample(sample_path)
                loaded_samples[idx_sf] = data
            print('')
            # print('loaded_samples:', loaded_samples)
            # print('len(loaded_samples):', len(loaded_samples))
            # sys.exit(0)

            mean_embedd_file_pattern = os.path.join(subj_path, '*_mean_embedding_*.npy')
            mean_embedd_file_path = glob.glob(mean_embedd_file_pattern)
            if len(mean_embedd_file_path) > 0:
                mean_embedd_file_path = mean_embedd_file_path[0]
                mean_embedd = np.load(mean_embedd_file_path)
            else:
                embedds_subj = torch.zeros((len(samples_paths),1,loaded_samples[-1].shape[1]), dtype=torch.float32)
                for idx_embedd, embedd in enumerate(loaded_samples):
                    embedds_subj[idx_embedd] = embedd
                mean_embedd = embedds_subj.mean(axis=0)
                # print('embedds_subj:', embedds_subj)
                # print('embedds_subj.shape:', embedds_subj.shape)
                # print('mean_embedd.shape:', mean_embedd.shape)
                # sys.exit(0)                

            dist_to_mean_embedd = {}
            dist_samples_matrix = -np.ones((len(loaded_samples),len(loaded_samples)), dtype=np.float32)
            for i in range(len(loaded_samples)):
                sample1 = loaded_samples[i]

                if not skip_distances_to_mean_embedd:
                    if args.metric == 'chamfer':
                        dist_to_mean = compute_chamfer_distance(sample1, mean_embedd)
                    elif args.metric == 'cosine_3dmm' or args.metric == 'cosine_2d':
                        dist_to_mean = compute_cosine_similarity(sample1, mean_embedd)
                    elif args.metric == 'euclidean_3dmm':
                        dist_to_mean = compute_euclidean_distance(sample1, mean_embedd, normalize=False)
                    dist_to_mean_embedd[samples_paths[i]] = dist_to_mean

                if not skip_distances_between_samples:
                    for j in range(i+1, len(loaded_samples)):
                        print(f'    Computing intra-class \'{args.metric}\' distances - i: {i}/{len(loaded_samples)}  j: {j}/{len(loaded_samples)}', end='\r')
                        sample2 = loaded_samples[j]

                        if args.metric == 'chamfer':
                            dist = compute_chamfer_distance(sample1, sample2)
                        elif args.metric == 'cosine_3dmm' or args.metric == 'cosine_2d':
                            dist = compute_cosine_similarity(sample1, sample2)
                        elif args.metric == 'euclidean_3dmm':
                            dist = compute_euclidean_distance(sample1, sample2, normalize=False)

                        # chamfer_distances.append(chamfer_dist)
                        # print('dist:', dist)
                        dist_samples_matrix[i,j] = dist
            print('')

            if not skip_distances_to_mean_embedd:
                print(f'    Saving similarities/distances to mean class: \'{output_distances_to_mean_path}\'')
                # np.save(output_distances_to_mean_path, dist_to_mean_embedd)
                with open(output_distances_to_mean_path, 'wb') as fp:
                    pickle.dump(dist_to_mean_embedd, fp)

            if not skip_distances_between_samples:
                print(f'    Saving similarities/distances between samples: \'{output_distances_path}\'')
                np.save(output_distances_path, dist_samples_matrix)

            subj_elapsed_time = (time.time() - subj_start_time)
            print('    subj_elapsed_time: %.2f sec' % (subj_elapsed_time))
            print('---------------------')
            # sys.exit(0)

    print('\nFinished!')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-path', type=str, default='/datasets2/1st_frcsyn_wacv2024/datasets/3D_reconstruction_MICA/real/1_CASIA-WebFace/imgs_crops_112x112_ONLY-SAMPLED/output')
    
    # parser.add_argument('--str_begin', default='', type=str, help='Substring to find and start processing')
    # parser.add_argument('--str_end', default='', type=str, help='Substring to find and stop processing')
    # parser.add_argument('--str_pattern', default='', type=str, help='Substring to find and stop processing')

    parser.add_argument('--divs', default=1, type=int, help='How many parts to divide paths list (useful to paralelize process)')
    parser.add_argument('--part', default=0, type=int, help='Specific part to process (works only if -div > 1)')

    parser.add_argument('--metric', default='cosine_2d', type=str, help='Options: chamfer, cosine_3dmm, euclidean_3dmm, cosine_2d')
    parser.add_argument('--file_ext', default='.npy', type=str, help='.ply, .obj, .npy')

    parser.add_argument('--dont_replace_existing_files', action='store_true', help='')

    args = parser.parse_args()

    main(args)
