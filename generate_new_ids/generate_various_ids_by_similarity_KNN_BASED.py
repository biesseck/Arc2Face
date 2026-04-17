'''
cd generate_new_ids
python generate_various_ids_by_similarity.py --path-dataset /nobackup3/bjgbiesseck/CASIA-Webface/imgs_crops_112x112_FACE_EMBEDDINGS_R100_WebFace42M_ArcFace --path-subj-list /hddevice/nobackup3/bjgbiesseck/datasets/face_recognition/CASIA-WebFace/merge_with_dataset_MS-Celeb-1M-ms1m-retinaface-t1-imgs_FACE_EMBEDDINGS_sim-range=[0.5,0.69]/dict_paths_new_subjs_base_subjs.json --similarity-range [0.5,0.69] --num-samples-by-id 50
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
import time
import re
import json
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt


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
    parser.add_argument("--path-dataset",        type=str, default="/nobackup3/bjgbiesseck/CASIA-Webface/imgs_crops_112x112_FACE_EMBEDDINGS_R100_WebFace42M_ArcFace")
    parser.add_argument('--input-ext',           type=str, default='_embedding_r100_webface42m_arcface.npy')
    parser.add_argument("--similarity-range",    type=parse_list_arg, default=[0.5,0.69], required=True, help='A list of float values separated by commas, e.g., 0.5,0.69 or [0.5,0.69]')
    parser.add_argument("--num-new-ids",         type=int, default=-1)   # -1 == one new synthetic id for each real id
    parser.add_argument("--num-samples-by-id",   type=int, default=50)
    parser.add_argument("--batch",               type=int, default=25)
    parser.add_argument("--num-inference-steps", type=int, default=25)
    parser.add_argument("--k",                   type=int, default=5)
    parser.add_argument("--path-output",         type=str, default="")
    parser.add_argument('--dataset_name',        default='New Synthetic Subjects', type=str, help='')
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


def get_all_files_paths_with_dir_classes(folder_path, file_extension=['.jpg','.jpeg','.png'], pattern=''):
    file_list = []
    if isinstance(file_extension, str): file_extension = [file_extension]
    for root, _, files in os.walk(folder_path):
        for filename in files:
            path_file = os.path.join(root, filename)
            for ext in file_extension:
                if pattern in path_file and path_file.lower().endswith(ext.lower()):
                    file_list.append(path_file)
                    print(f'Found files: {len(file_list)}', end='\r')
    print('\nSorting paths...')
    file_list = natural_sort(file_list)
    classes_str_list = [f.split('/')[-2] for f in file_list]
    classes_str_unique_list = natural_sort(list(set(classes_str_list)))
    classes_str_unique_int  = list(range(0, len(classes_str_unique_list)))
    dict_classes_str_int = {class_str:class_int for (class_str,class_int) in zip(classes_str_unique_list,classes_str_unique_int)}
    # print('dict_classes_str_int:', dict_classes_str_int)
    # sys.exit(0)
    # for i, class_str in enumerate():
    classes_int_list = [dict_classes_str_int[class_str] for class_str in classes_str_list]

    return file_list, classes_str_list, classes_int_list


def get_immediate_subdirs(parent_dir=''):
    subdirs = [os.path.join(parent_dir, name) for name in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, name))]
    subdirs = natural_sort(subdirs)
    return subdirs


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


def save_dict_to_pickle(dictionary, filename):
    with open(filename, 'wb') as f:
        pickle.dump(dictionary, f, protocol=pickle.HIGHEST_PROTOCOL)
    

def load_dict_from_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


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
        embedd = embedd.cpu().detach().numpy()
    elif embedd_path.endswith('.npy'):
        embedd = np.load(embedd_path)
        embedd = np.squeeze(embedd)
    else:
        raise Exception(f'File format not supported: \'{embedd_path}\'')
    return embedd



def rotate_embedding_by_cosine_similarity(v1: torch.Tensor, void_vector: torch.Tensor, cosine_similarity: float) -> torch.Tensor:
    v1 = torch.tensor(v1)
    void_vector = torch.tensor(void_vector)

    v1 = torch.squeeze(v1)
    if not (0.0 <= cosine_similarity <= 1.0):
        raise ValueError("Cosine similarity must be between 0.0 and 1.0.")
    if v1.dim() != 1 or v1.size(0) != 512:
        raise ValueError("Input tensor must be 512-dimensional (1D tensor).")
    
    theta = torch.acos(cosine_similarity)
    
    if torch.isclose(theta, torch.tensor(0.0).to(torch.float32)):
        return v1.clone()
    
    v1_norm = torch.linalg.norm(v1).to(torch.float32)
    if torch.isclose(v1_norm, torch.tensor(0.0).to(torch.float32)):
        return v1.clone()
        
    u1 = v1 / v1_norm

    # random_vector = torch.randn_like(v1)
    
    # projection_onto_u1 = torch.dot(random_vector, u1) * u1
    projection_onto_u1 = torch.dot(void_vector, u1) * u1
    
    # u2_raw = random_vector - projection_onto_u1
    u2_raw = void_vector - projection_onto_u1
    
    u2_norm = torch.linalg.norm(u2_raw).to(torch.float32)
    
    if torch.isclose(u2_norm, torch.tensor(0.0).to(torch.float32)):
        raise RuntimeError("Failed to generate a non-collinear random vector. Try running again.")

    u2 = u2_raw / u2_norm
    u1_prime = (u1 * torch.cos(theta)) + (u2 * torch.sin(theta))
    v1_prime = u1_prime * v1_norm
    v1_prime = torch.unsqueeze(v1_prime, 0)
    return v1_prime


def compute_class_centroids(embedds_feats, embedds_labels_int, embedds_labels_str):
    feats = np.asarray(embedds_feats)
    labels_int = np.asarray(embedds_labels_int)
    
    unique_labels_int, inverse, counts = np.unique(labels_int, return_inverse=True, return_counts=True)
    unique_labels_str = list(dict.fromkeys(embedds_labels_str))
    num_classes = len(unique_labels_int)
    dim = feats.shape[1]
    
    centroids = np.zeros((num_classes, dim), dtype=feats.dtype)
    np.add.at(centroids, inverse, feats)
    
    centroids /= counts[:, np.newaxis]
    
    return centroids, unique_labels_int, unique_labels_str


def find_tangent_void_direction(target_centroid, neighbor_centroids):
    dots = np.dot(neighbor_centroids, target_centroid)
    projections = neighbor_centroids - np.outer(dots, target_centroid)
    crowd_direction = np.mean(projections, axis=0)
    void_vector = -crowd_direction
    
    norm = np.linalg.norm(void_vector)
    if norm < 1e-9:
        print("Warning: Neighbors surround target perfectly. Using random void.")
        void_vector = np.random.normal(0, 1, target_centroid.shape)
        void_vector -= np.dot(void_vector, target_centroid) * target_centroid
        void_vector /= np.linalg.norm(void_vector)
    else:
        void_vector /= norm
        
    return void_vector


def compute_similarities_histogram(all_similarities):
    nbins = 20
    lower, higher = 0.0, 1.0
    bins_edges = np.linspace(lower, higher, nbins+1)
    total_counts = np.zeros(nbins, dtype=np.int64)
    bins_counts, _ = np.histogram(all_similarities, bins=bins_edges, range=(lower,higher))
    print('bins_edges:', bins_edges)
    print('bins_counts:', bins_counts)
    total_counts += bins_counts
    bins_widths = np.diff(bins_edges)
    n_in_range = total_counts.sum()
    # print('(n_in_range * bins_widths):', (n_in_range * bins_widths))
    density = total_counts / (n_in_range * bins_widths)
    # print('total_counts.sum():', total_counts.sum())
    pmf = total_counts / total_counts.sum()
    return bins_edges, pmf, bins_widths


def save_bar_plot_from_histogram(bins_edges, pmf, bins_widths, filename, title):
    plt.bar(bins_edges[:-1], pmf, width=bins_widths, align="edge", edgecolor='black', alpha=0.7, label='All dists')
    
    # Add title, labels, and legend
    plt.title(title)
    plt.xlabel('Similarity')
    plt.ylabel('Frequency')
    plt.legend()

    plt.xlim([0, 1])
    
    # plt.ylim([0, 0.0001])
    plt.ylim([0, 1.0])

    # Save the plot as PNG
    plt.savefig(filename)

    f_name, f_extension = os.path.splitext(filename)
    filename_svg = f_name + '.svg'
    plt.savefig(filename_svg)

    plt.clf()



def flat_array_remove_invalid_values(array, invalid_value=-1, upper_triang=True):
    if upper_triang:
        iu = np.triu_indices(n=array.shape[0], k=1, m=array.shape[1])
        flat_array = array[iu]
    else:
        flat_array = array.flatten()
    valid_values = flat_array[flat_array != invalid_value]
    valid_values = np.clip(valid_values, 0.0, 1.0)
    return valid_values



if __name__ == '__main__':

    args = parse_arguments()
    assert args.num_samples_by_id >= args.batch, f"Error, --num-samples-by-id must be greater or equal to --batch"
    assert args.num_samples_by_id % args.batch == 0, f"Error, --num-samples-by-id must be a multiple of --batch"
    assert os.path.isdir(args.path_dataset), f"Error, no such dir: \'{args.path_dataset}\'"

    if not args.path_output:
        args.path_output = f"{args.path_dataset}_newSynthIDs_Arc2Face_sim={args.similarity_range}".replace(' ','')
    else:
        args.path_output = os.path.join(args.path_output, f"{args.path_dataset.split('/')[-1]}_newSynthIDs_Arc2Face_sim={args.similarity_range}".replace(' ',''))

    print('Loading Arc2Face model...')
    pipeline = get_arc2face_model()
    fr_model = get_face_recognition_model()

    subjs_orig_paths = get_immediate_subdirs(args.path_dataset)
    subjs_names = [subj_name.split('/')[-1] for subj_name in subjs_orig_paths]

    if args.num_new_ids == 0:
        print('\n--num-new-ids == 0, no new synthetic identities will be generated!')
        sys.exit(0)
    elif args.num_new_ids < 0:
        args.path_output += f'_{len(subjs_names)}ids'
    elif args.num_new_ids > 0:
        random.seed(440)
        random.shuffle(subjs_names)
        subjs_names = subjs_names[:args.num_new_ids]
        args.path_output += f'_{args.num_new_ids}ids'
    args.path_output += f'_KNN_BASED'
    print(f'Creating output folder: \'{args.path_output}\'')
    os.makedirs(args.path_output, exist_ok=True)




    dataset_all_embedds_file_name = f"{args.path_dataset.split('/')[-1]}.pkl"
    dataset_all_embedds_file_path = os.path.join(args.path_output, dataset_all_embedds_file_name)
    if not os.path.isfile(dataset_all_embedds_file_path):
        print(f"Searching embeddings in \'{args.path_dataset}\'")
        embedds_paths, embedds_classes_str, embedds_classes_int = get_all_files_paths_with_dir_classes(args.path_dataset, file_extension=args.input_ext)
        # print('embedds_classes_str:', embedds_classes_str)
        # print('embedds_classes_int:', embedds_classes_int)
        # sys.exit(0)

        dict_dataset_all_embedds = {}
        dict_dataset_all_embedds['embedds_paths']       = embedds_paths
        dict_dataset_all_embedds['embedds_classes_str'] = embedds_classes_str
        dict_dataset_all_embedds['embedds_classes_int'] = embedds_classes_int

        print(f'Saving all embeddings paths: \'{dataset_all_embedds_file_path}\'')
        save_dict_to_pickle(dict_dataset_all_embedds, dataset_all_embedds_file_path)
    else:
        print(f'Loading all embeddings paths: \'{dataset_all_embedds_file_path}\'')
        dict_dataset_all_embedds = load_dict_from_pickle(dataset_all_embedds_file_path)
        embedds_paths       = dict_dataset_all_embedds['embedds_paths']
        embedds_classes_str = dict_dataset_all_embedds['embedds_classes_str']
        embedds_classes_int = dict_dataset_all_embedds['embedds_classes_int']
    print('len(embedds_paths):', len(embedds_paths))
    print('type(embedds_classes_str):', type(embedds_classes_str))
    print('type(embedds_classes_int):', type(embedds_classes_int))
    print(f'------------------')



    if not 'embedds_feats' in dict_dataset_all_embedds:
        one_embedd = load_embedding(embedds_paths[0])
        print('one_embedd.shape:', one_embedd.shape)
        shape_embedds_feats = (len(embedds_paths), one_embedd.shape[0] if len(one_embedd.shape)==1 else one_embedd.shape[1])
        print('Allocating matrix:', shape_embedds_feats)
        embedds_feats = np.zeros(shape_embedds_feats, dtype=float)
        print('Done!')

        for idx_embedd, path_embedd in enumerate(embedds_paths):
            start_time = time.time()
            print(f'{idx_embedd}/{len(embedds_paths)} - Loading embedding \'{path_embedd}\'', end='\r')
            one_embedd = load_embedding(path_embedd)
            # one_embedd = one_embedd.cpu().detach().numpy()
            embedds_feats[idx_embedd] = one_embedd
        print()
        dict_dataset_all_embedds['embedds_feats'] = embedds_feats
        print(f'Saving all embeddings features: \'{dataset_all_embedds_file_path}\'')
        save_dict_to_pickle(dict_dataset_all_embedds, dataset_all_embedds_file_path)
    else:
        print(f'Loading all embeddings features: \'{dataset_all_embedds_file_path}\'')
        embedds_feats = dict_dataset_all_embedds['embedds_feats']
    print('embedds_feats.shape:', embedds_feats.shape)
    print(f'------------------')



    if not 'embedds_centroids' in dict_dataset_all_embedds:
        print('Computing centroids...')
        embedds_centroids, embedds_centroids_labels_int, embedds_centroids_labels_str = compute_class_centroids(embedds_feats, embedds_classes_int, embedds_classes_str)
        print('embedds_centroids.shape:', embedds_centroids.shape)
        print('embedds_centroids_labels_int.shape:', embedds_centroids_labels_int.shape)
        print('len(embedds_centroids_labels_str):', len(embedds_centroids_labels_str))
        centroids_norm = embedds_centroids / np.linalg.norm(embedds_centroids, axis=1, keepdims=True)
        # print(f'------------------')

        print('Computing cosine similarity matrix...')
        sim_matrix = cosine_similarity(centroids_norm)
        print('sim_matrix.shape:', sim_matrix.shape)
        # print(f'------------------')

        dict_dataset_all_embedds['embedds_centroids']            = embedds_centroids
        dict_dataset_all_embedds['embedds_centroids_labels_int'] = embedds_centroids_labels_int
        dict_dataset_all_embedds['embedds_centroids_labels_str'] = embedds_centroids_labels_str
        dict_dataset_all_embedds['embedds_centroids_norm']       = centroids_norm
        dict_dataset_all_embedds['sim_matrix']                   = sim_matrix
        print(f'Saving centroids: \'{dataset_all_embedds_file_path}\'')
        save_dict_to_pickle(dict_dataset_all_embedds, dataset_all_embedds_file_path)
    else:
        print(f'Loading centroids: \'{dataset_all_embedds_file_path}\'')
        embedds_centroids            = dict_dataset_all_embedds['embedds_centroids']
        embedds_centroids_labels_int = dict_dataset_all_embedds['embedds_centroids_labels_int']
        embedds_centroids_labels_str = dict_dataset_all_embedds['embedds_centroids_labels_str']
        centroids_norm               = dict_dataset_all_embedds['embedds_centroids_norm']
        sim_matrix                   = dict_dataset_all_embedds['sim_matrix']
    print(f'------------------')



    idx_end_subj = args.num_new_ids if args.num_new_ids > 0 else len(embedds_centroids)
    # idx_end_subj = len(embedds_centroids)



    path_output_knn = os.path.join(args.path_output, f'k={args.k}')
    print(f'Creating knn output folder: \'{path_output_knn}\'')
    os.makedirs(path_output_knn, exist_ok=True)



    print(f'------------------')
    print('Computing cosine similarity matrix between original real identities...')
    # print('    sim_matrix:', sim_matrix)
    print('    sim_matrix.shape:', sim_matrix.shape)
    unique_sim_matrix_original_dataset = flat_array_remove_invalid_values(sim_matrix, invalid_value=-1)
    bins_edges, pmf, bins_widths = compute_similarities_histogram(unique_sim_matrix_original_dataset)
    print('    pmf:', pmf)
    prefix_output_filename = 'INTERCLASS_SIMILARITIES_ORIGINAL_REAL_IDENTITIES'
    title = f"dataset \'{args.dataset_name}\' - {len(embedds_centroids)} subjects"
    chart_file_name = f'{prefix_output_filename}_histograms_distances_between_original_real_identities.png'
    chart_file_path = os.path.join(path_output_knn,chart_file_name)
    print(f'Saving histogram: \'{chart_file_path}\'')
    save_bar_plot_from_histogram(bins_edges, pmf, bins_widths, chart_file_path, title)
    print(f'------------------')



    print('Computing sums of similarities...')
    k = args.k
    k_sim_sums = np.sort(sim_matrix, axis=1)[:, -(k+1):-1].sum(axis=1)
    print('k_sim_sums', k_sim_sums)
    print('k_sim_sums.shape:', k_sim_sums.shape)
    # isolated_indices = np.argsort(k_sim_sums)
    isolated_indices = np.argsort(k_sim_sums)[:idx_end_subj]
    print('isolated_indices:', isolated_indices)
    print('isolated_indices.shape:', isolated_indices.shape)
    # sys.exit(0)
    print(f'------------------')



    # Compute new embeddings
    all_new_embedds = torch.zeros_like(torch.tensor(embedds_centroids), dtype=torch.float16)
    all_new_embedds_labels_int = []
    all_new_embedds_labels_str = []
    all_similarities = torch.empty((len(all_new_embedds,)), dtype=torch.float32)
    all_similarities_new_id_emb_to_k_neighbor_centroids = np.empty((idx_end_subj, k+1), dtype=np.float32)
    print()
    for idx_subj, target_idx in enumerate(isolated_indices):
        # target_idx = isolated_indices[idx_subj]
        print(f'{idx_subj}/{len(isolated_indices)} - subj \'{embedds_centroids_labels_str[target_idx]}\' - Computing void direction vectors', end='\r')
        # print(f'{idx_subj}/{len(embedds_centroids_labels_int)} - Computing void direction vectors',)
        all_new_embedds_labels_int.append(target_idx)
        all_new_embedds_labels_str.append(embedds_centroids_labels_str[target_idx])
        target_centroid = embedds_centroids[target_idx,:]
        # print('\ntarget_idx:', target_idx, f'    embedds_centroids_labels_str[{target_idx}]:', embedds_centroids_labels_str[target_idx])
        # print('target_centroid.shape:', target_centroid.shape)
        target_sims = sim_matrix[target_idx]
        sorted_indices = np.argsort(target_sims)[::-1]
        neighbor_indices = sorted_indices[1:k+1]
        neighbor_centroids = embedds_centroids[neighbor_indices]
        # void_vector = find_void_direction(target_centroid, neighbor_centroids)
        void_vector = find_tangent_void_direction(target_centroid, neighbor_centroids)

        all_similarities[idx_subj] = get_random_float(args.similarity_range)
        # print('similarity:', similarity)
        new_id_emb = rotate_embedding_by_cosine_similarity(target_centroid, void_vector, all_similarities[idx_subj])
        # print('new_id_emb.shape:', new_id_emb.shape)
        # print('new_id_emb.norm():', np.linalg.norm(new_id_emb))
        # new_id_emb = new_id_emb/torch.norm(new_id_emb, dim=1, keepdim=True)   # normalize embedding

        all_new_embedds[idx_subj] = new_id_emb
        # print('all_new_embedds:', all_new_embedds)
        # sys.exit(0)

        sim_matrix_new_id_emb_to_k_neighbor_centroids = cosine_similarity(new_id_emb, np.vstack((target_centroid,neighbor_centroids)))
        all_similarities_new_id_emb_to_k_neighbor_centroids[idx_subj] = sim_matrix_new_id_emb_to_k_neighbor_centroids
        # print('\nsim_matrix_new_id_emb_to_k_neighbor_centroids:', sim_matrix_new_id_emb_to_k_neighbor_centroids)
        # print('sim_matrix_new_id_emb_to_k_neighbor_centroids.shape:', sim_matrix_new_id_emb_to_k_neighbor_centroids.shape)
        # print('sim_matrix_new_id_emb_to_k_neighbor_centroids.dtype:', sim_matrix_new_id_emb_to_k_neighbor_centroids.dtype)
        # sys.exit(0)

        # print('----------')
    print('----------')
    print('Computing cosine similarity matrix between new synthetic embedds and their k neighbors...')
    # print('    all_similarities_new_id_emb_to_k_neighbor_centroids:', all_similarities_new_id_emb_to_k_neighbor_centroids)
    print('    all_similarities_new_id_emb_to_k_neighbor_centroids.shape:', all_similarities_new_id_emb_to_k_neighbor_centroids.shape)
    unique_sim_matrix_new_id_emb_to_k_neighbor_centroids = flat_array_remove_invalid_values(all_similarities_new_id_emb_to_k_neighbor_centroids, invalid_value=-1, upper_triang=False)
    bins_edges, pmf, bins_widths = compute_similarities_histogram(unique_sim_matrix_new_id_emb_to_k_neighbor_centroids)
    print('    pmf:', pmf)

    prefix_output_filename = 'INTERCLASS_SIMILARITIES_SYNTH_TO_THEIR_K-NEAREST-NEIGHBORS'
    title = f"dataset \'{args.dataset_name}\' - {len(embedds_centroids)} subjects"
    chart_file_name = f'{prefix_output_filename}_histograms_distances_between_samples_k={args.k}.png'
    chart_file_path = os.path.join(path_output_knn,chart_file_name)
    print(f'Saving histogram: \'{chart_file_path}\'')
    save_bar_plot_from_histogram(bins_edges, pmf, bins_widths, chart_file_path, title)
    print()




    print('Computing cosine similarity matrix between new synthetic embedds...')
    sim_matrix_all_new_embedds = cosine_similarity(all_new_embedds[:idx_end_subj])
    print('    sim_matrix_all_new_embedds.shape:', sim_matrix_all_new_embedds.shape)
    unique_sim_matrix_all_new_embedds = flat_array_remove_invalid_values(sim_matrix_all_new_embedds, invalid_value=-1)
    print('    unique_sim_matrix_all_new_embedds:', unique_sim_matrix_all_new_embedds)
    print('    unique_sim_matrix_all_new_embedds.shape:', unique_sim_matrix_all_new_embedds.shape)
    bins_edges, pmf, bins_widths = compute_similarities_histogram(unique_sim_matrix_all_new_embedds)
    print('    pmf:', pmf)

    prefix_output_filename = 'INTERCLASS_SIMILARITIES_SYNTH_TO_SYNTH_THEMSELVES'
    title = f"dataset \'{args.dataset_name}\' - {len(embedds_centroids)} subjects"
    chart_file_name = f'{prefix_output_filename}_histograms_distances_between_samples_k={args.k}.png'
    chart_file_path = os.path.join(path_output_knn,chart_file_name)
    print(f'Saving histogram: \'{chart_file_path}\'')
    save_bar_plot_from_histogram(bins_edges, pmf, bins_widths, chart_file_path, title)
    print()



    print('Computing cosine similarity matrix between new synthetic embedds and real embeddings...')
    sim_matrix_all_new_embedds_to_real_embedds = cosine_similarity(all_new_embedds[:idx_end_subj], centroids_norm)
    print('    sim_matrix_all_new_embedds_to_real_embedds.shape:', sim_matrix_all_new_embedds_to_real_embedds.shape)
    unique_sim_matrix_all_new_embedds_to_real_embedds = flat_array_remove_invalid_values(sim_matrix_all_new_embedds_to_real_embedds, invalid_value=-1, upper_triang=False)
    print('    unique_sim_matrix_all_new_embedds_to_real_embedds:', unique_sim_matrix_all_new_embedds_to_real_embedds)
    print('    unique_sim_matrix_all_new_embedds_to_real_embedds.shape:', unique_sim_matrix_all_new_embedds_to_real_embedds.shape)
    bins_edges, pmf, bins_widths = compute_similarities_histogram(unique_sim_matrix_all_new_embedds_to_real_embedds)
    print('    pmf:', pmf)

    prefix_output_filename = 'INTERCLASS_SIMILARITIES_SYNTH_TO_REAL'
    title = f"dataset \'{args.dataset_name}\' - {len(embedds_centroids)} subjects"
    chart_file_name = f'{prefix_output_filename}_histograms_distances_between_samples_k={args.k}.png'
    chart_file_path = os.path.join(path_output_knn,chart_file_name)
    print(f'Saving histogram: \'{chart_file_path}\'')
    save_bar_plot_from_histogram(bins_edges, pmf, bins_widths, chart_file_path, title)
    print()



    print('Computing cosine similarity matrix between ALL synthetic embedds and ALL real embeddings...')
    centroids_norm_with_new_embedds = np.vstack((centroids_norm, all_new_embedds[:idx_end_subj]))
    print('    centroids_norm_with_new_embedds.shape:', centroids_norm_with_new_embedds.shape)
    sim_matrix_centroids_norm_with_new_embedds = cosine_similarity(centroids_norm_with_new_embedds)
    print('    sim_matrix_centroids_norm_with_new_embedds.shape:', sim_matrix_centroids_norm_with_new_embedds.shape)
    unique_sim_matrix_centroids_norm_with_new_embedds = flat_array_remove_invalid_values(sim_matrix_centroids_norm_with_new_embedds, invalid_value=-1)
    print('    unique_sim_matrix_centroids_norm_with_new_embedds:', unique_sim_matrix_centroids_norm_with_new_embedds)
    print('    unique_sim_matrix_centroids_norm_with_new_embedds.shape:', unique_sim_matrix_centroids_norm_with_new_embedds.shape)
    bins_edges, pmf, bins_widths = compute_similarities_histogram(unique_sim_matrix_centroids_norm_with_new_embedds)
    print('    pmf:', pmf)

    prefix_output_filename = 'INTERCLASS_SIMILARITIES_ALL'
    title = f"dataset \'{args.dataset_name}\' - {len(embedds_centroids)} subjects"
    chart_file_name = f'{prefix_output_filename}_histograms_distances_between_samples_k={args.k}.png'
    chart_file_path = os.path.join(path_output_knn,chart_file_name)
    print(f'Saving histogram: \'{chart_file_path}\'')
    save_bar_plot_from_histogram(bins_edges, pmf, bins_widths, chart_file_path, title)

    print('----------')


    sys.exit(0)


    # Generate images from the computed embeddings
    # idx_end_subj = args.num_new_ids if args.num_new_ids > 0 else len(all_new_embedds_labels_str)
    all_new_embedds_labels_str = all_new_embedds_labels_str[:idx_end_subj]
    for idx_subj, subj_name in enumerate(all_new_embedds_labels_str):
        start_time = time.time()

        new_id_emb = all_new_embedds[idx_subj,:].to("cuda")
        new_id_emb = torch.unsqueeze(new_id_emb, 0)
        new_id_emb = new_id_emb/torch.norm(new_id_emb, dim=1, keepdim=True)   # normalize embedding
        # print('new_id_emb.shape:', new_id_emb.shape)
        # print('new_id_emb.norm():', torch.norm(new_id_emb))

        # Generate images:
        print(f'id {idx_subj}/{len(all_new_embedds_labels_str)} - subj \'{subj_name}\' - Generating {args.num_samples_by_id} new images...')
        new_id_emb_proj = project_face_embs(pipeline, new_id_emb)    # pass through the encoder
        # print('new_id_emb_proj.shape:', new_id_emb_proj.shape)
        # print('new_id_emb_proj.norm():', torch.norm(new_id_emb_proj))
        
        num_runs = int(args.num_samples_by_id / args.batch)
        all_generated_images = []
        for idx_run in range(num_runs):
            print(f'    run {idx_run}/{num_runs}')
            images = pipeline(prompt_embeds=new_id_emb_proj, num_inference_steps=args.num_inference_steps, guidance_scale=3.0, num_images_per_prompt=args.batch).images
            all_generated_images.extend(images)

        path_dir_subj = os.path.join(args.path_dataset, subj_name)
        output_folder = f"{os.path.join(path_output_knn,f'imgs_steps={args.num_inference_steps}',path_dir_subj.split('/')[-1])}_newId_sim={all_similarities[idx_subj]}"
        os.makedirs(output_folder, exist_ok=True)
        for i, img in enumerate(all_generated_images):
            output_img_name = os.path.splitext(os.path.basename(path_dir_subj))[0]
            path_output_img = os.path.join(output_folder, f"{output_img_name}_newID_newSample_{i}.png")
            print(f"    Saving output img: \'{path_output_img}\'", end='\r')
            img.save(path_output_img)
        print()

        exec_time = time.time() - start_time
        remain_time = exec_time * (len(all_new_embedds_labels_str)-idx_subj+1)
        print('    Exec time: %.2fsec    %.2fmin    %.2fhour' % (exec_time, exec_time/60, exec_time/3600))
        print('    Remaining time: %.2fsec    %.2fmin    %.2fhour' % (remain_time, remain_time/60, remain_time/3600))
        print('------------')

    print('\nFinished!')
        


