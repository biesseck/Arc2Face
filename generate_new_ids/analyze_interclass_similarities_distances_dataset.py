import os
import sys

import argparse
import random
import socket
import time
import glob
import pickle
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt



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
    if file_path.endswith('.pt'):
        dists = torch.load(file_path)
        dists = dists.numpy()
    elif file_path.endswith('.npy'):
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

    # hist_means, bin_means_edges = np.histogram(means, bins=20, density=True)
    # bin_width = bin_means_edges[1] - bin_means_edges[0]
    # plt.bar(bin_means_edges[:-1], hist_means/np.sum(hist_means), width=bin_width, edgecolor='black', alpha=0.7, label='Means of dists')

    # hist_stds, bin_stds_edges = np.histogram(stds, bins=20, density=True)
    # bin_width = bin_stds_edges[1] - bin_stds_edges[0]
    # plt.bar(bin_stds_edges[:-1], hist_stds/np.sum(hist_stds), width=bin_width, edgecolor='black', alpha=0.7, label='Stds')


    # Plot histograms
    # plt.hist(means, bins=20, density=False, stacked=True, alpha=0.5, label='Means')
    # plt.hist(stds, bins=20, density=True, stacked=True, alpha=0.5, label='Stds')

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


def save_bar_plot_from_histogram(bins_edges, pmf, bins_widths, filename, title):
    plt.bar(bins_edges[:-1], pmf, width=bins_widths, align="edge", edgecolor='black', alpha=0.7, label='All dists')
    
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


def get_leaf_subdirs(base_path):
    base = Path(base_path)
    return [
        str(p) for p in base.rglob("*")
        if p.is_dir() and not any(child.is_dir() for child in p.iterdir())
    ]


def main(args):
    dataset_path = args.input_path.rstrip('/')
    assert os.path.isdir(dataset_path), f'Error, no such dir: \'{dataset_path}\''
    output_path = os.path.dirname(dataset_path)
    # os.makedirs(output_path, exist_ok=True)

    prefix_output_filename = 'INTERCLASS_SIMILARITIES'
    path_precomputed_histograms = os.path.join(output_path, f'{prefix_output_filename}_{os.path.basename(dataset_path)}.pkl')
    
    if args.compute_from_scratch or not os.path.isfile(path_precomputed_histograms):
        print('dataset_path:', dataset_path)
        print('Searching subject subfolders...')
        # subjects_paths = sorted([os.path.join(dataset_path,subj) for subj in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, subj))])
        subjects_paths = get_leaf_subdirs(dataset_path)
        # print('subjects_paths:', subjects_paths)
        print(f'Found {len(subjects_paths)} subjects!')
        # sys.exit(0)

        nbins = 20
        lower, higher = 0.0, 1.0
        bins_edges = np.linspace(lower, higher, nbins+1)
        total_counts = np.zeros(nbins, dtype=np.int64)
        total_seen = 0
        
        print('\nLoading similarities...')
        for idx_subj, subj_path in enumerate(subjects_paths):
            subj_start_time = time.time()
            
            subj_name = os.path.basename(subj_path)
            print(f'{idx_subj}/{len(subjects_paths)} - Loading subject \'{subj_name}\'', end='\r')

            # Distances between subjects
            file_pattern_dist_between_samples = os.path.join(glob.escape(subj_path), '*' + args.file_ext)
            dist_between_samples_file_path = glob.glob(file_pattern_dist_between_samples)
            
            assert len(dist_between_samples_file_path) > 0, f'Error, file not found: \'{file_pattern}\''
            dist_between_samples_file_path = dist_between_samples_file_path[0]
            dist_between_samples_data = load_distances(dist_between_samples_file_path)
            # print('dist_between_samples_data:', dist_between_samples_data)
            # print('dist_between_samples_data.shape:', dist_between_samples_data.shape)
            # sys.exit(0)

            # if len(dist_between_samples_data.shape) > 1:
            dist_between_samples_data = flat_array_remove_invalid_values(dist_between_samples_data, invalid_value=-1)
            # print('dist_between_samples_data:', dist_between_samples_data)
            # print('\ndist_between_samples_data.shape:', dist_between_samples_data.shape)
            # sys.exit(0)

            metrics = compute_metrics_distances_subject(dist_between_samples_data)
            if not metrics is None:
                bins_counts, _ = np.histogram(metrics['all_distances'], bins=bins_edges, range=(lower,higher))
                total_counts += bins_counts
                total_seen += metrics['all_distances'].size

                # print('bins_counts:', bins_counts, '    bins_counts.sum():', bins_counts.sum())
                # print('bins_edges:', bins_edges)
                # print('total_counts:', total_counts)
                # print('total_seen:', total_seen)
                # print('-------------')
                
                # if idx_subj >= 9:
                #     break

        total_num_subjs = idx_subj+1
        print('')

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
        hist_computed_data['total_num_subjs']                     = total_num_subjs

        print(f'\nSaving computed data: \'{path_precomputed_histograms}\'')
        save_dict(hist_computed_data, path_precomputed_histograms)

    else:
        print(f'\nLoading computed data: \'{path_precomputed_histograms}\'')
        hist_computed_data = load_dict(path_precomputed_histograms)
        nbins           = hist_computed_data['nbins']
        lower, higher   = hist_computed_data['lower'], hist_computed_data['higher']
        bins_edges      = hist_computed_data['bins_edges']
        total_counts    = hist_computed_data['total_counts']
        total_seen      = hist_computed_data['total_seen']
        bins_widths     = hist_computed_data['bins_widths']
        density         = hist_computed_data['density']
        pmf             = hist_computed_data['pmf']
        total_num_subjs = hist_computed_data['total_num_subjs']

    # print('density:', density, '    density.sum():', density.sum())
    # print('pmf:', pmf, '    pmf.sum():', pmf.sum())
    # print('-------------')



    title = f"dataset \'{args.dataset_name}\' - {total_num_subjs} subjects - {args.metric}"
    chart_file_name = f'{prefix_output_filename}_histograms_distances_between_samples_' + args.metric + '.png'
    chart_file_path = os.path.join(output_path, chart_file_name)
    print(f'Saving histogram: \'{chart_file_path}\'')
    save_bar_plot_from_histogram(bins_edges, pmf, bins_widths, chart_file_path, title)

    print('\nFinished!')

        
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-path', type=str, default='/nobackup/unico/datasets/face_recognition/1_CASIA-WebFace/imgs_crops_112x112_FACE_EMBEDDINGS_INTERCLASS_SIMILARITIES_cosine_2d')
    
    parser.add_argument('--metric', default='cosine_2d', type=str, help='Options: cosine_2d, euclidean_3dmm, cosine_3dmm, chamfer')
    parser.add_argument('--file_ext', default='.pt', type=str, help='.pt, .npy, .pkl')
    parser.add_argument('--dataset_name', default='CASIA-WebFace', type=str, help='')

    parser.add_argument('--compute_from_scratch', action='store_true', help='')

    args = parser.parse_args()

    main(args)
