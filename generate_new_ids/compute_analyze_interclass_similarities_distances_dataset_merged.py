# duo     (bjgbiesseck_MICA) export CUDA_VISIBLE_DEVICES=0; python compute_interclass_similarities_distances_dataset.py --input-path /datasets2/1st_frcsyn_wacv2024/datasets/real/1_CASIA-WebFace/imgs_crops_112x112_FACE_EMBEDDINGS --metric cosine_2d --file_ext .npy --mean_embedd_str mean_embedding
# diolkos (bjgbiesseck_MICA) export CUDA_VISIBLE_DEVICES=0; python compute_interclass_similarities_distances_dataset.py --input-path /nobackup/unico/datasets/face_recognition/synthetic/dcface_0.5m_oversample_xid/record/imgs_FACE_EMBEDDINGS --metric cosine_2d --file_ext .npy --mean_embedd_str mean_embedding


import os
import sys

import argparse
import random
import socket
import time
import pickle
import json
import re

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


def save_json(data, path, indent=4):
    if not isinstance(data, dict):
        raise TypeError("The 'data' argument must be a dictionary.")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)

def load_json(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"The file '{path}' does not exist.")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def save_dict(data: dict, filename: str) -> None:
    serializable_data = {}
    for key, value in data.items():
        if isinstance(value, torch.Tensor):
            serializable_data[key] = value.detach().cpu().numpy()
        else:
            serializable_data[key] = value
    
    with open(filename, "wb") as f:
        pickle.dump(serializable_data, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_dict(filename: str) -> dict:
    with open(filename, "rb") as f:
        data = pickle.load(f)
    restored_data = {}
    for key, value in data.items():
        if isinstance(value, np.ndarray):
            try:
                restored_data[key] = torch.from_numpy(value)
            except Exception:
                restored_data[key] = value
        else:
            restored_data[key] = value
    return restored_data


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


def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


def get_leaf_subdirs(base_path):
    base = Path(base_path)
    subdir_paths = [
        str(p) for p in base.rglob("*")
        if p.is_dir() and not any(child.is_dir() for child in p.iterdir())
    ]
    subdir_paths = natural_sort(subdir_paths)
    return subdir_paths


def flat_array_remove_invalid_values(array, invalid_value=-1):
    if isinstance(array, dict):
        dict_data = [array[key] for key in array.keys()]
        array = np.array(dict_data)

    flat_array = array.flatten()
    valid_values = flat_array[flat_array != invalid_value]
    valid_values = np.clip(valid_values, 0.0, 1.0)
    return valid_values


def compute_metrics_distances_subject(dist_data):
    if dist_data.size > 0:
        metrics = {}
        metrics['all_distances'] = dist_data
        metrics['mean'] = np.mean(dist_data)
        metrics['std'] = np.std(dist_data)
        return metrics
    else:
        return None


def compute_histogram(metrics, nbins=20, lower=0.0, higher=1.0):
    # nbins = 20
    # lower, higher = 0.0, 1.0
    bins_edges = np.linspace(lower, higher, nbins+1)
    total_counts = np.zeros(nbins, dtype=np.int64)
    total_seen = 0

    bins_counts, _ = np.histogram(metrics['all_distances'], bins=bins_edges, range=(lower,higher))
    total_counts += bins_counts
    total_seen += metrics['all_distances'].size

    bins_widths = np.diff(bins_edges)
    n_in_range = total_counts.sum()
    density = total_counts / (n_in_range * bins_widths)
    pmf = total_counts / total_counts.sum()

    hist_computed_data = {}
    hist_computed_data['nbins']                               = nbins
    hist_computed_data['lower'], hist_computed_data['higher'] = lower, higher
    hist_computed_data['bins_edges']                          = bins_edges
    hist_computed_data['total_counts']                        = total_counts
    hist_computed_data['total_seen']                          = total_seen
    hist_computed_data['bins_widths']                         = bins_widths
    hist_computed_data['density']                             = density
    hist_computed_data['pmf']                                 = pmf
    return hist_computed_data


'''
def save_bar_plot_from_histogram(bins_edges, pmf, bins_widths, filename, title, color='blue'):
    plt.cla()
    plt.bar(bins_edges[:-1], pmf, width=bins_widths, align="edge", color=color, edgecolor='black', alpha=0.7, label='All similarities')
    
    # Add title, labels, and legend
    plt.title(title)
    plt.xlabel('Similarity')
    plt.ylabel('Frequency')
    plt.legend()

    plt.xlim([0, 1])
    # plt.ylim([0, 0.5])
    plt.ylim([0, 1.0])

    # Save the plot as PNG
    plt.savefig(filename)

    f_name, f_extension = os.path.splitext(filename)
    filename_svg = f_name + '.svg'
    plt.savefig(filename_svg)
'''

def save_bar_plot_from_histogram(bins_edges, pmf, bins_widths, total_counts, filename, title, color='blue'):
    plt.cla()
    plt.bar(bins_edges[:-1], pmf, width=bins_widths, align="edge", color=color, edgecolor='black', alpha=0.7, label='All similarities')
    
    # Add numbers on top of the bars
    # We calculate the center of each bar by adding half the width to the starting edge
    for edge, height, width, count in zip(bins_edges[:-1], pmf, bins_widths, total_counts):
        if height > 0: # Optional: skips labeling empty bins to avoid clutter
            bar_center = edge + (width / 2)
            # Adjust the vertical offset (height + 0.02) depending on your data scale
            plt.text(bar_center, height + 0.02, f'{count}', 
                     rotation=90, 
                     fontsize=8, 
                     ha='center', 
                     va='bottom')

    # Add title, labels, and legend
    plt.title(title)
    plt.xlabel('Similarity')
    plt.ylabel('Frequency')
    plt.legend()

    plt.xlim([0, 1])
    # plt.ylim([0, 0.5])
    plt.ylim([0, 1.0])

    # Save the plot as PNG
    plt.savefig(filename)

    f_name, f_extension = os.path.splitext(filename)
    filename_svg = f_name + '.svg'
    plt.savefig(filename_svg)



def main(args):
    assert args.part < args.divs, f'Error, args.part ({args.part}) >= args.divs ({args.divs}), but should be args.part ({args.part}) < args.divs ({args.divs})'
    assert os.path.isfile(args.other_dataset_subjs_list_path), f'Error, no such file \'{args.other_dataset_subjs_list_path}\''

    dataset_path = args.input_path.rstrip('/')
    output_path = f"{dataset_path}_INTERCLASS_SIMILARITIES_{args.metric}_{os.path.dirname(args.other_dataset_subjs_list_path).split('/')[-1]}"
    path_precomputed_data = os.path.join(output_path, f'precomputed_data_{os.path.basename(dataset_path)}.pkl')
    os.makedirs(output_path, exist_ok=True)


    print('dataset_path:', dataset_path)
    print('Searching subject subfolders...')
    # subjects_paths = sorted([os.path.join(dataset_path,subj) for subj in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, subj))])
    subjects_paths = get_leaf_subdirs(dataset_path)
    # print('subjects_paths:', subjects_paths)
    print(f'Found {len(subjects_paths)} subjects!')
    # sys.exit(0)


    print()
    print(f'Loading other dataset subjects list: \'{args.other_dataset_subjs_list_path}\'')
    other_dataset_subjs_list_dict = load_json(args.other_dataset_subjs_list_path)
    first_other_subj_path = list(other_dataset_subjs_list_dict.keys())[0]
    assert os.path.isfile(first_other_subj_path), f'Error, no such file \'{first_other_subj_path}\''
    other_dataset_subjs_paths = [k for k in list(other_dataset_subjs_list_dict.keys())]
    # print('other_dataset_subjs_list_dict.keys():', other_dataset_subjs_list_dict.keys())
    # print('len(other_dataset_subjs_list_dict):', len(other_dataset_subjs_list_dict))
    # print('other_dataset_subjs_paths:', other_dataset_subjs_paths)
    # sys.exit(0)


    if args.compute_from_scratch or not os.path.isfile(path_precomputed_data):
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

        print('Loading subjects mean embeddings of base dataset...')
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



        print()
        data = load_sample(other_dataset_subjs_paths[0])
        other_dataset_subj_mean_embedds = torch.zeros((len(other_dataset_subjs_paths), torch.squeeze(data).shape[0]), dtype=data.dtype)
        print('Loading other dataset mean embeddings')
        for idx_other_subj, other_subj_path in enumerate(other_dataset_subjs_paths):
            print(f'{idx_other_subj}/{len(other_dataset_subjs_paths)} - Loading subject mean embedding in \'{other_subj_path}\'', end='\r')
            data = load_sample(other_subj_path)
            # print('data.shape:', data.shape, '    type(data):', type(data), '    device:', {data.device})
            other_dataset_subj_mean_embedds[idx_other_subj] = data
        print('')
        print(f'    other_dataset_subj_mean_embedds.shape: {other_dataset_subj_mean_embedds.shape}    dtype: {other_dataset_subj_mean_embedds.dtype}    device: {other_dataset_subj_mean_embedds.device}')
        # sys.exit(0)



        print()
        all_subj_similarities_list = [None] * len(subjects_paths[:-1])
        print('Computing base dataset inner interclass similarities')
        for idx_subj, subj_path in enumerate(subjects_paths[:-1]):
            subj_mean_embedd = subj_mean_embedds[idx_subj]
            other_subj_mean_embedd = subj_mean_embedds[idx_subj+1:]
            subj_similarities = compute_cosine_similarity_1_to_N(subj_mean_embedd, other_subj_mean_embedd)
            print(f'    idx_subj {idx_subj}/{len(subjects_paths[:-1])} - subj_similarities.shape: {subj_similarities.shape}', end='\r')
            all_subj_similarities_list[idx_subj] = subj_similarities
        print()
        all_subj_similarities_concat = np.concatenate(all_subj_similarities_list)
        print(f'Flatting array and removing invalid values')
        all_subj_similarities_concat = flat_array_remove_invalid_values(all_subj_similarities_concat, invalid_value=-1)
        print(f'all_subj_similarities_concat.shape: {all_subj_similarities_concat.shape}\n')



        all_other_dataset_subj_similarities_list = [None] * len(other_dataset_subj_mean_embedds[:-1])
        print('Computing other dataset inner interclass similarities')
        for idx_subj, subj_mean_embedd in enumerate(other_dataset_subj_mean_embedds[:-1]):
            other_subj_mean_embedd = other_dataset_subj_mean_embedds[idx_subj+1:]
            subj_similarities = compute_cosine_similarity_1_to_N(subj_mean_embedd, other_subj_mean_embedd)
            print(f'    idx_subj {idx_subj}/{len(other_dataset_subj_mean_embedds[:-1])} - subj_similarities.shape: {subj_similarities.shape}', end='\r')
            all_other_dataset_subj_similarities_list[idx_subj] = subj_similarities
        print()
        all_other_dataset_subj_similarities_concat = np.concatenate(all_other_dataset_subj_similarities_list)
        print(f'Flatting array and removing invalid values')
        all_other_dataset_subj_similarities_concat = flat_array_remove_invalid_values(all_other_dataset_subj_similarities_concat, invalid_value=-1)
        print(f'all_other_dataset_subj_similarities_concat.shape: {all_other_dataset_subj_similarities_concat.shape}\n')



        all_merged_datasets_outer_similarities_list = [None] * len(subj_mean_embedds)
        print('Computing outer other dataset inner interclass similarities')
        for idx_base_subj, base_subj_mean_embedd in enumerate(subj_mean_embedds):
            similarities_base_subj_to_other_dataset = compute_cosine_similarity_1_to_N(base_subj_mean_embedd, other_dataset_subj_mean_embedds)
            print(f'    idx_base_subj {idx_base_subj}/{len(subj_mean_embedds)} - subj_similarities.shape: {similarities_base_subj_to_other_dataset.shape}', end='\r')
            all_merged_datasets_outer_similarities_list[idx_base_subj] = similarities_base_subj_to_other_dataset
        print()
        all_merged_datasets_outer_similarities_concat = np.concatenate(all_merged_datasets_outer_similarities_list)
        print(f'Flatting array and removing invalid values')
        all_merged_datasets_outer_similarities_concat = flat_array_remove_invalid_values(all_merged_datasets_outer_similarities_concat, invalid_value=-1)
        print(f'all_merged_datasets_outer_similarities_concat.shape: {all_merged_datasets_outer_similarities_concat.shape}\n')




        print('-------------------')
        print('Computing metrics of base dataset inner interclass similarities')
        metrics_base_dataset_inner_interclass_similarities    = compute_metrics_distances_subject(all_subj_similarities_concat)        
        print('Computing metrics of other dataset inner interclass similarities')
        metrics_other_dataset_inner_interclass_similarities   = compute_metrics_distances_subject(all_other_dataset_subj_similarities_concat)        
        print('Computing metrics of outer interclass similarities between base and other dataset')
        metrics_merged_datasets_outer_interclass_similarities = compute_metrics_distances_subject(all_merged_datasets_outer_similarities_concat)        


        precomputed_data = {}
        precomputed_data['metrics_base_dataset_inner_interclass_similarities']    = metrics_base_dataset_inner_interclass_similarities
        precomputed_data['metrics_other_dataset_inner_interclass_similarities']   = metrics_other_dataset_inner_interclass_similarities
        precomputed_data['metrics_merged_datasets_outer_interclass_similarities'] = metrics_merged_datasets_outer_interclass_similarities
        # precomputed_data['hist_base_dataset_inner_interclass_similarities']       = hist_base_dataset_inner_interclass_similarities
        # precomputed_data['hist_other_dataset_inner_interclass_similarities']      = hist_other_dataset_inner_interclass_similarities
        # precomputed_data['hist_merged_datasets_outer_interclass_similarities']    = hist_merged_datasets_outer_interclass_similarities
        print(f'\nSaving computed data: \'{path_precomputed_data}\'')
        save_dict(precomputed_data, path_precomputed_data)
        print('    Saved')


    else:
        print(f'\nLoading precomputed data: \'{path_precomputed_data}\'')
        precomputed_data = load_dict(path_precomputed_data)
        metrics_base_dataset_inner_interclass_similarities    = precomputed_data['metrics_base_dataset_inner_interclass_similarities']
        metrics_other_dataset_inner_interclass_similarities   = precomputed_data['metrics_other_dataset_inner_interclass_similarities']
        metrics_merged_datasets_outer_interclass_similarities = precomputed_data['metrics_merged_datasets_outer_interclass_similarities']
        # hist_base_dataset_inner_interclass_similarities       = precomputed_data['hist_base_dataset_inner_interclass_similarities']
        # hist_other_dataset_inner_interclass_similarities      = precomputed_data['hist_other_dataset_inner_interclass_similarities']
        # hist_merged_datasets_outer_interclass_similarities    = precomputed_data['hist_merged_datasets_outer_interclass_similarities']
        print('    Loaded')


    print()
    nbins = 20
    lower, higher = 0.0, 1.0
    print('Computing histogram of base dataset inner interclass similarities')
    hist_base_dataset_inner_interclass_similarities    = compute_histogram(metrics_base_dataset_inner_interclass_similarities, nbins, lower, higher)
    print("    hist_base_dataset_inner_interclass_similarities['bins_edges']:", hist_base_dataset_inner_interclass_similarities['bins_edges'])
    print("    hist_base_dataset_inner_interclass_similarities['total_counts']:", hist_base_dataset_inner_interclass_similarities['total_counts'])
    print('Computing histogram of other dataset inner interclass similarities')
    hist_other_dataset_inner_interclass_similarities   = compute_histogram(metrics_other_dataset_inner_interclass_similarities, nbins, lower, higher)
    print("    hist_other_dataset_inner_interclass_similarities['total_counts']:", hist_other_dataset_inner_interclass_similarities['total_counts'])
    print('Computing histogram of outer interclass similarities between base and other dataset')
    hist_merged_datasets_outer_interclass_similarities = compute_histogram(metrics_merged_datasets_outer_interclass_similarities, nbins, lower, higher)
    print("    hist_merged_datasets_outer_interclass_similarities['total_counts']:", hist_merged_datasets_outer_interclass_similarities['total_counts'])
    
        


    print()
    prefix_output_filename = 'INTERCLASS_SIMILARITIES'

    title = f"Base dataset - {len(subjects_paths)} subjects - {args.metric}"
    chart_file_name = f'0_{prefix_output_filename}_base_dataset_histogram_inner_interclass_similarities_' + args.metric + '.png'
    chart_file_path = os.path.join(output_path, chart_file_name)
    print(f'Saving histogram: \'{chart_file_path}\'')
    save_bar_plot_from_histogram(hist_base_dataset_inner_interclass_similarities['bins_edges'],
                                 hist_base_dataset_inner_interclass_similarities['pmf'],
                                 hist_base_dataset_inner_interclass_similarities['bins_widths'],
                                 hist_base_dataset_inner_interclass_similarities['total_counts'],
                                 chart_file_path, title, color='blue')

    title = f"Other dataset - {len(other_dataset_subjs_paths)} subjects - {args.metric}"
    chart_file_name = f'1_{prefix_output_filename}_other_dataset_histogram_inner_interclass_similarities_' + args.metric + '.png'
    chart_file_path = os.path.join(output_path, chart_file_name)
    print(f'Saving histogram: \'{chart_file_path}\'')
    save_bar_plot_from_histogram(hist_other_dataset_inner_interclass_similarities['bins_edges'],
                                 hist_other_dataset_inner_interclass_similarities['pmf'],
                                 hist_other_dataset_inner_interclass_similarities['bins_widths'],
                                 hist_other_dataset_inner_interclass_similarities['total_counts'],
                                 chart_file_path, title, color='orange')

    title = f"Merged datasets - {len(subjects_paths)} base subjects - {len(other_dataset_subjs_paths)} other subjects - {args.metric}"
    chart_file_name = f'2_{prefix_output_filename}_merged_datasets_histogram_outer_interclass_similarities_' + args.metric + '.png'
    chart_file_path = os.path.join(output_path, chart_file_name)
    print(f'Saving histogram: \'{chart_file_path}\'')
    save_bar_plot_from_histogram(hist_merged_datasets_outer_interclass_similarities['bins_edges'],
                                 hist_merged_datasets_outer_interclass_similarities['pmf'],
                                 hist_merged_datasets_outer_interclass_similarities['bins_widths'],
                                 hist_merged_datasets_outer_interclass_similarities['total_counts'],
                                 chart_file_path, title, color='green')

    print('\nFinished!')




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-path', type=str, default='/nobackup1/unico/datasets/face_recognition/CASIA-WebFace/imgs_crops_112x112_FACE_EMBEDDINGS')
    parser.add_argument('--other-dataset-subjs-list-path', type=str, default='/nobackup1/unico/datasets/face_recognition/CASIA-WebFace/merge_with_dataset_glint360k-glint360k-imgs_FACE_EMBEDDINGS_sim-range=[0.4,0.49]/dict_paths_new_subjs_base_subjs.json')

    # parser.add_argument('--str_begin', default='', type=str, help='Substring to find and start processing')
    # parser.add_argument('--str_end', default='', type=str, help='Substring to find and stop processing')
    # parser.add_argument('--str_pattern', default='', type=str, help='Substring to find and stop processing')

    parser.add_argument('--divs', default=1, type=int, help='How many parts to divide paths list (useful to paralelize process)')
    parser.add_argument('--part', default=0, type=int, help='Specific part to process (works only if -div > 1)')

    parser.add_argument('--metric', default='cosine_2d', type=str, help='Options: chamfer, cosine_3dmm, euclidean_3dmm, cosine_2d')
    parser.add_argument('--file-ext', default='.npy', type=str, help='.ply, .obj, .npy')
    parser.add_argument('--mean-embedd-str', default='mean_embedding', type=str, help='')

    parser.add_argument('--compute-from-scratch', action='store_true', help='')

    args = parser.parse_args()

    main(args)
