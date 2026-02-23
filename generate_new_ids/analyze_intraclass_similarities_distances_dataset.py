import os
import sys

import argparse
import random
import socket
import time
import glob
import pickle

import torch
import numpy as np
import matplotlib.pyplot as plt

# import torch
# import torch.nn as nn
# import matplotlib.pyplot as plt
# from pytorch3d.io import load_obj
# from pytorch3d.loss import chamfer_distance
# from mpl_toolkits.mplot3d import Axes3D

# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from pytorch3d.io import load_obj, load_ply
# from pytorch3d.loss import chamfer_distance


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


def load_distances(file_path):
    if file_path.endswith('.npy'):
        dists = np.load(file_path)
    elif file_path.endswith('.pkl'):
        with open(file_path, 'rb') as file:
            dists = pickle.load(file)
    else:
        raise ValueError(f"Unsupported file format \'{file_path}\'")
    return dists
    

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


def merge_metrics_dists(metrics_dist_subj):
    num_all_distances = 0
    for i, key in enumerate(metrics_dist_subj.keys()):
        num_all_distances += metrics_dist_subj[key]['all_distances'].shape[0]

    idx_begin_all_dist, idx_end_all_dist = 0, 0
    all_distances = np.zeros((num_all_distances,), dtype=np.float32)
    means = np.zeros((len(metrics_dist_subj),), dtype=np.float32)
    stds = np.zeros((len(metrics_dist_subj),), dtype=np.float32)
    for i, key in enumerate(metrics_dist_subj.keys()):
        idx_end_all_dist = idx_begin_all_dist + metrics_dist_subj[key]['all_distances'].shape[0]
        all_distances[idx_begin_all_dist:idx_end_all_dist] = metrics_dist_subj[key]['all_distances']
        idx_begin_all_dist = idx_end_all_dist
        
        means[i] = metrics_dist_subj[key]['mean']
        stds[i] = metrics_dist_subj[key]['std']
    return all_distances, means, stds


def save_histograms(all_distances, means, stds, filename, title):
    hist_all_dists, bin_all_dists_edges = np.histogram(all_distances, bins=20, density=True)
    bin_width = bin_all_dists_edges[1] - bin_all_dists_edges[0]
    plt.bar(bin_all_dists_edges[:-1], hist_all_dists/np.sum(hist_all_dists), width=bin_width, edgecolor='black', alpha=0.7, label='All dists')

    hist_means, bin_means_edges = np.histogram(means, bins=20, density=True)
    bin_width = bin_means_edges[1] - bin_means_edges[0]
    plt.bar(bin_means_edges[:-1], hist_means/np.sum(hist_means), width=bin_width, edgecolor='black', alpha=0.7, label='Means of dists')

    # Add title, labels, and legend
    plt.title(title)
    # plt.xlabel('Distance')
    plt.xlabel('Similarity')
    plt.ylabel('Frequency')
    plt.legend()

    plt.xlim([0, 1])
    # plt.xlim([0, 20])
    plt.ylim([0, 0.5])

    # Save the plot as PNG
    plt.savefig(filename)

    f_name, f_extension = os.path.splitext(filename)
    filename_svg = f_name + '.svg'
    plt.savefig(filename_svg)

    # Show the plot (optional)
    # plt.show()


def save_histograms_new(all_distances, means, stds, filename, title):
    nbins = 20
    lower, higher = 0.0, 1.0
    bins_edges = np.linspace(lower, higher, nbins+1)
    bins_widths = np.diff(bins_edges)
    # bins_counts, _ = np.histogram(metrics['all_distances'], bins=bins_edges, range=(lower,higher))

    # hist_all_dists, bin_all_dists_edges = np.histogram(all_distances, bins=20, density=True)
    hist_all_dists, bin_all_dists_edges = np.histogram(all_distances, bins=bins_edges, range=(lower,higher))
    bin_width = bin_all_dists_edges[1] - bin_all_dists_edges[0]
    plt.bar(bin_all_dists_edges[:-1], hist_all_dists/np.sum(hist_all_dists), width=bin_width, align="edge", edgecolor='black', alpha=0.7, label='All dists')

    # hist_means, bin_means_edges = np.histogram(means, bins=20, density=True)
    hist_means, bin_means_edges = np.histogram(means, bins=bins_edges, range=(lower,higher))
    bin_width = bin_means_edges[1] - bin_means_edges[0]
    plt.bar(bin_means_edges[:-1], hist_means/np.sum(hist_means), width=bin_width, align="edge", edgecolor='black', alpha=0.7, label='Means of dists')

    # Add title, labels, and legend
    plt.title(title)
    # plt.xlabel('Distance')
    plt.xlabel('Similarity')
    plt.ylabel('Frequency')
    plt.legend()

    plt.xlim([0, 1])
    # plt.xlim([0, 20])
    plt.ylim([0, 0.5])

    # Save the plot as PNG
    plt.savefig(filename)

    f_name, f_extension = os.path.splitext(filename)
    filename_svg = f_name + '.svg'
    plt.savefig(filename_svg)

    # Show the plot (optional)
    # plt.show()



def main(args):
    dataset_path = args.input_path.rstrip('/')
    assert os.path.isdir(dataset_path), f'Error, no such dir: \'{dataset_path}\''
    output_path = os.path.dirname(dataset_path)
    # os.makedirs(output_path, exist_ok=True)

    prefix_output_filename = 'INTRACLASS_SIMILARITIES'
    path_precomputed_histograms = os.path.join(output_path, f'{prefix_output_filename}_{os.path.basename(dataset_path)}.pkl')
    
    if args.compute_from_scratch or not os.path.isfile(path_precomputed_histograms):
        print('dataset_path:', dataset_path)
        print('Searching subject subfolders...')
        subjects_paths = sorted([os.path.join(dataset_path,subj) for subj in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, subj))])
        # print('subjects_paths:', subjects_paths)
        print(f'Found {len(subjects_paths)} subjects!')
        # sys.exit(0)

        metrics_dist_between_samples_subj = {}
        metrics_dist_to_mean_subj = {}
        print('Loading distances...\n')
        for idx_subj, subj_path in enumerate(subjects_paths):
            subj_start_time = time.time()
            
            subj_name = os.path.basename(subj_path)
            print(f'{idx_subj}/{len(subjects_paths)} - Loading subject \'{subj_name}\'', end='\r')

            # Distances between samples
            file_pattern_dist_between_samples = os.path.join(glob.escape(subj_path), '*' + args.file_ext)
            dist_between_samples_file_path = glob.glob(file_pattern_dist_between_samples)
            if len(dist_between_samples_file_path) > 0:
                # assert len(dist_between_samples_file_path) > 0, f'Error, file not found: \'{file_pattern}\''
                dist_between_samples_file_path = dist_between_samples_file_path[0]
                dist_between_samples_data = load_distances(dist_between_samples_file_path)
                # print('dist_between_samples_data.shape:', dist_between_samples_data.shape)
                # sys.exit(0)

                dist_between_samples_data = flat_array_remove_invalid_values(dist_between_samples_data, invalid_value=-1)
                # print('dist_between_samples_data.shape:', dist_between_samples_data.shape)
                # sys.exit(0)

                metrics = compute_metrics_distances_subject(dist_between_samples_data)
                if not metrics is None:
                    metrics_dist_between_samples_subj[subj_name] = metrics
                    # print('metrics_dist_between_samples_subj:', metrics_dist_between_samples_subj)
                    # sys.exit(0)

            # Distances to mean embedding
            file_pattern_dist_to_mean_subj = os.path.join(glob.escape(subj_path), '*.pkl')
            dist_to_mean_subj_file_path = glob.glob(file_pattern_dist_to_mean_subj)
            if len(dist_to_mean_subj_file_path) > 0:
                dist_to_mean_subj_file_path = dist_to_mean_subj_file_path[0]
                dist_to_mean_subj_data = load_distances(dist_to_mean_subj_file_path)
                # print('dist_to_mean_subj_data:', dist_to_mean_subj_data)
                # sys.exit(0)

                dist_to_mean_subj_data = flat_array_remove_invalid_values(dist_to_mean_subj_data, invalid_value=-1)
                # print('dist_to_mean_subj_data:', dist_to_mean_subj_data)
                # sys.exit(0)

                metrics_dist_to_mean_subj[subj_name] = compute_metrics_distances_subject(dist_to_mean_subj_data)
            
        total_num_subjs = idx_subj+1
        print('')
        
        print('Merging metrics...')
        all_dist_between_samples, all_means_dist_between_samples, all_stds_dist_between_samples = merge_metrics_dists(metrics_dist_between_samples_subj)
        all_dist_to_mean_embedd, all_means_dist_to_mean_embedd, all_stds_dist_to_mean_embedd = merge_metrics_dists(metrics_dist_to_mean_subj)
        # print('all_means_dist:', all_means_dist)
        # print('all_means_dist.shape:', all_means_dist.shape)

        hist_computed_data = {}
        hist_computed_data['total_num_subjs'] = total_num_subjs

        hist_computed_data['all_dist_between_samples']       = all_dist_between_samples
        hist_computed_data['all_means_dist_between_samples'] = all_means_dist_between_samples
        hist_computed_data['all_stds_dist_between_samples']  = all_stds_dist_between_samples

        hist_computed_data['all_dist_to_mean_embedd']       = all_dist_to_mean_embedd
        hist_computed_data['all_means_dist_to_mean_embedd'] = all_means_dist_to_mean_embedd
        hist_computed_data['all_stds_dist_to_mean_embedd']  = all_stds_dist_to_mean_embedd

        print(f'\nSaving computed data: \'{path_precomputed_histograms}\'')
        save_dict(hist_computed_data, path_precomputed_histograms)

    else:
        print(f'\nLoading computed data: \'{path_precomputed_histograms}\'')
        hist_computed_data = load_dict(path_precomputed_histograms)
        total_num_subjs = hist_computed_data['total_num_subjs']

        all_dist_between_samples       = hist_computed_data['all_dist_between_samples']
        all_means_dist_between_samples = hist_computed_data['all_means_dist_between_samples']
        all_stds_dist_between_samples  = hist_computed_data['all_stds_dist_between_samples']

        all_dist_to_mean_embedd        = hist_computed_data['all_dist_to_mean_embedd']
        all_means_dist_to_mean_embedd  = hist_computed_data['all_means_dist_to_mean_embedd']
        all_stds_dist_to_mean_embedd   = hist_computed_data['all_stds_dist_to_mean_embedd']



    title = f'dataset \'{args.dataset_name}\' - {total_num_subjs} subjects - {args.metric}'
    chart_file_name = f'{prefix_output_filename}_histograms_distances_between_samples_' + args.metric + '.png'
    chart_file_path = os.path.join(output_path, chart_file_name)
    print(f'Saving histograms: \'{chart_file_path}\'')
    # save_histograms(all_dist_between_samples, all_means_dist_between_samples, all_stds_dist_between_samples, chart_file_path, title)
    save_histograms_new(all_dist_between_samples, all_means_dist_between_samples, all_stds_dist_between_samples, chart_file_path, title)

    title = f'dataset \'{args.dataset_name}\' - {total_num_subjs} subjects - {args.metric}'
    chart_file_name = f'{prefix_output_filename}_histograms_distances_to_mean_embedd_' + args.metric + '.png'
    chart_file_path = os.path.join(output_path, chart_file_name)
    print(f'Saving histograms: \'{chart_file_path}\'')
    # save_histograms(all_dist_to_mean_embedd, all_means_dist_to_mean_embedd, all_stds_dist_to_mean_embedd, chart_file_path, title)
    save_histograms_new(all_dist_to_mean_embedd, all_means_dist_to_mean_embedd, all_stds_dist_to_mean_embedd, chart_file_path, title)

    print('\nFinished!')

        
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-path', type=str, default='/datasets2/1st_frcsyn_wacv2024/datasets/3D_reconstruction_MICA/real/1_CASIA-WebFace/imgs_crops_112x112/distances_cosine_3dmm')
    
    parser.add_argument('--metric', default='cosine_2d', type=str, help='Options: cosine_2d, euclidean_3dmm, cosine_3dmm, chamfer')
    parser.add_argument('--file_ext', default='.npy', type=str, help='.npy')
    parser.add_argument('--dataset_name', default='CASIA-WebFace', type=str, help='')

    parser.add_argument('--compute_from_scratch', action='store_true', help='')

    args = parser.parse_args()

    main(args)
