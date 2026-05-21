import os
import sys
import numpy as np
from sklearn.preprocessing import normalize
import argparse
import re
from glob import glob
import copy
import pickle
import json
from PIL import Image
import matplotlib.pyplot as plt



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_base_subj_embedds", type=str, default='/nobackup3/bjgbiesseck/CASIA-Webface/imgs_crops_112x112_FACE_EMBEDDINGS_R100_WebFace42M_ArcFace')
    parser.add_argument("--substring_file_base_subj", type=str, default='_mean_embedding_')

    parser.add_argument("--path_new_subjs_embedds", type=str, default='/nobackup1/bjgbiesseck/face_recognition/webface4m/imgs_FACE_EMBEDDINGS_R100_WebFace42M_ArcFace')
    parser.add_argument("--substring_file_new_subj", type=str,  default='_mean_embedding_')

    # parser.add_argument("--similarity_range", type=lambda s: [float(x) for x in s.strip('[]').split(',')], default='[0.0,0.3]')   # low similarity
    parser.add_argument("--similarity_range", type=lambda s: [float(x) for x in s.strip('[]').split(',')], default='[0.7,1.0]')     # high similarity

    args = parser.parse_args()

    return args


def save_pickle(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

def load_pickle(path):
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    return obj


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



def _numeric_sort_key(path):
    return [int(num) for num in re.findall(r'\d+', os.path.basename(path))]


def find_files_paths(path_embedds, exts=['.npy'], substring_file='', verbose=True):
    assert os.path.exists(path_embedds), f"Error: no such path or file '{path_embedds}'"
    matched_files = []
    count = 0
    for root, _, files in os.walk(path_embedds):
        for fname in files:
            # Check substring condition
            if substring_file not in fname:
                continue
            # Check extension condition (if exts is not empty)
            if exts and not any(fname.lower().endswith(ext.lower()) for ext in exts):
                continue

            full_path = os.path.join(root, fname)
            matched_files.append(full_path)
            count += 1
            if verbose:
                print(f"    Found files: {count}", end="\r")

    if verbose:
        print()
    matched_files.sort(key=_numeric_sort_key)
    return matched_files


def load_imgs_from_paths(paths_list, as_numpy=True):
    loaded_images = []
    for path in paths_list:
        img = Image.open(path).convert('RGB')
        if as_numpy:
            img = np.array(img)
        loaded_images.append(img)
    return loaded_images


def load_one_embedding(file_path):
    if '.npy' in os.path.splitext(file_path)[1]:
        embedding = np.load(file_path)
    return embedding


def load_embeddings(base_subj_embedds_paths):
    assert os.path.isfile(base_subj_embedds_paths[0]), f"Error, file not found \'{base_subj_embedds_paths[0]}\'"
    one_embedding = load_one_embedding(base_subj_embedds_paths[0])
    num_dims = one_embedding.shape[1] if len(one_embedding.shape) == 2 else one_embedding.shape[0]   # typically (1, 512)
    all_embeddings = np.zeros((len(base_subj_embedds_paths), num_dims), dtype=one_embedding.dtype)
    # print("    all_embeddings.shape:", all_embeddings.shape)
    # sys.exit(0)
    for idx_embedd_path, embedd_path in enumerate(base_subj_embedds_paths):
        print(f"    {idx_embedd_path}/{len(base_subj_embedds_paths)} - Loading \'{embedd_path}\'", end="\r")
        one_embedd = load_one_embedding(embedd_path)
        all_embeddings[idx_embedd_path,:] = one_embedd
        # print('all_embeddings:', all_embeddings)
    print()
    return all_embeddings


def make_indices_chunks(total_size=93431, chunk_size=1000):
    start_idx = 0
    curr_end_idx = min(total_size, chunk_size)
    chunk_indices = [[start_idx, curr_end_idx]]
    while curr_end_idx < total_size:
        start_idx = curr_end_idx
        curr_end_idx = min(total_size, start_idx+chunk_size)

        one_chunk = [start_idx, curr_end_idx]
        chunk_indices.append(one_chunk)
    # print('chunk_indices:', chunk_indices)
    # sys.exit(0)
    return chunk_indices


def load_select_new_subjs_embedds(base_subj_embedds, new_subj_embedds_paths, similarity_range):
    # base_subj_embedds = base_subj_embedds / np.linalg.norm(base_subj_embedds, axis=1, keepdims=True)
    base_subj_embedds = normalize(base_subj_embedds, axis=1)

    one_embedding = load_one_embedding(new_subj_embedds_paths[0])
    num_dims = one_embedding.shape[1] if len(one_embedding.shape) == 2 else one_embedding.shape[0]   # typically (1, 512)

    global_indices_selected_embedds = np.empty((0, 2), dtype=int)
    global_cossims_selected_embedds = np.empty((0, 1), dtype=float)
    
    # chunk_size = 1000
    # chunk_size = 5000
    chunk_size = 10000

    chunk_all_indices = make_indices_chunks(len(new_subj_embedds_paths), chunk_size)

    global_idx_embedd_path = 0
    for idx_chunk, chunk_idx in enumerate(chunk_all_indices):
        # one_embedding = load_one_embedding(new_subj_embedds_paths[0])
        # num_dims = one_embedding.shape[1] if len(one_embedding.shape) == 2 else one_embedding.shape[0]   # typically (1, 512)
        shape_chunk_embedd = (chunk_idx[1]-chunk_idx[0], num_dims)
        chunk_embeddings = np.zeros(shape_chunk_embedd, dtype=one_embedding.dtype)

        for idx_embedd_path, embedd_path in enumerate(new_subj_embedds_paths[chunk_idx[0]:chunk_idx[1]]):
            print(f"    chunk {idx_chunk}/{len(chunk_all_indices)} - chunk_idx: {chunk_idx} - chunk_embeddings.shape: {chunk_embeddings.shape} - global_idx_embedd: {global_idx_embedd_path}/{len(new_subj_embedds_paths)}", end='\r')
            # print(f"        chunk_embeddings.shape: {chunk_embeddings.shape} - global_idx_embedd: {global_idx_embedd_path}/{len(new_subj_embedds_paths)}", end='\r')
            one_embedd = load_one_embedding(embedd_path)
            chunk_embeddings[idx_embedd_path,:] = one_embedd

            global_idx_embedd_path += 1
        print()

        # chunk_embeddings = chunk_embeddings / np.linalg.norm(chunk_embeddings, axis=1, keepdims=True)
        chunk_embeddings = normalize(chunk_embeddings, axis=1)
        chunk_cossims = np.dot(chunk_embeddings, base_subj_embedds.T)
        # print('chunk_cossims:', chunk_cossims)
        # print('        chunk_cossims.shape:', chunk_cossims.shape)   # chunk_cossims.shape: (10000, 10572)

        # SELECT EMBEDDS INTO THE RANGE similarity_range
        mask = (chunk_cossims >= similarity_range[0]) & (chunk_cossims <= similarity_range[1])
        chunk_local_indices = np.argwhere(mask)
        chunk_cossims_selected_embedds = np.zeros((len(chunk_local_indices),1), dtype=float)
        for i in range(len(chunk_cossims_selected_embedds)):
            chunk_cossims_selected_embedds[i] = chunk_cossims[chunk_local_indices[i,0],chunk_local_indices[i,1]]
        # print('        chunk_local_indices:', chunk_local_indices)
        # print('        chunk_local_indices.shape:', chunk_local_indices.shape)

        chunk_global_indices = copy.deepcopy(chunk_local_indices)
        chunk_global_indices[:,0] += (idx_chunk * chunk_size)
        # print('        chunk_global_indices:', chunk_global_indices)
        print('        chunk_global_indices.shape:', chunk_global_indices.shape)
        # print('        chunk_global_indices.dtype:', chunk_global_indices.dtype)

        global_indices_selected_embedds = np.vstack((global_indices_selected_embedds, chunk_global_indices))
        global_cossims_selected_embedds = np.vstack((global_cossims_selected_embedds, chunk_cossims_selected_embedds))
        print('        global_indices_selected_embedds.shape:', global_indices_selected_embedds.shape)
        # print('        global_indices_selected_embedds.dtype:', global_indices_selected_embedds.dtype)

    return global_indices_selected_embedds, global_cossims_selected_embedds


def load_select_new_subjs_embedds_exclusive(base_subj_embedds, new_subj_embedds_paths, similarity_range):
    # base_subj_embedds = base_subj_embedds / np.linalg.norm(base_subj_embedds, axis=1, keepdims=True)
    base_subj_embedds = normalize(base_subj_embedds, axis=1)

    one_embedding = load_one_embedding(new_subj_embedds_paths[0])
    num_dims = one_embedding.shape[1] if len(one_embedding.shape) == 2 else one_embedding.shape[0]   # typically (1, 512)

    global_indices_selected_embedds = np.empty((0, 2), dtype=int)
    global_cossims_selected_embedds = np.empty((0, 1), dtype=float)
    
    # chunk_size = 1000
    # chunk_size = 5000
    chunk_size = 10000

    chunk_all_indices = make_indices_chunks(len(new_subj_embedds_paths), chunk_size)

    global_idx_embedd_path = 0
    for idx_chunk, chunk_idx in enumerate(chunk_all_indices):
        # one_embedding = load_one_embedding(new_subj_embedds_paths[0])
        # num_dims = one_embedding.shape[1] if len(one_embedding.shape) == 2 else one_embedding.shape[0]   # typically (1, 512)
        shape_chunk_embedd = (chunk_idx[1]-chunk_idx[0], num_dims)
        chunk_embeddings = np.zeros(shape_chunk_embedd, dtype=one_embedding.dtype)

        for idx_embedd_path, embedd_path in enumerate(new_subj_embedds_paths[chunk_idx[0]:chunk_idx[1]]):
            print(f"    chunk {idx_chunk}/{len(chunk_all_indices)} - chunk_idx: {chunk_idx} - chunk_embeddings.shape: {chunk_embeddings.shape} - global_idx_embedd: {global_idx_embedd_path}/{len(new_subj_embedds_paths)}", end='\r')
            # print(f"        chunk_embeddings.shape: {chunk_embeddings.shape} - global_idx_embedd: {global_idx_embedd_path}/{len(new_subj_embedds_paths)}", end='\r')
            one_embedd = load_one_embedding(embedd_path)
            chunk_embeddings[idx_embedd_path,:] = one_embedd

            global_idx_embedd_path += 1
        print()

        # chunk_embeddings = chunk_embeddings / np.linalg.norm(chunk_embeddings, axis=1, keepdims=True)
        chunk_embeddings = normalize(chunk_embeddings, axis=1)
        chunk_cossims = np.dot(chunk_embeddings, base_subj_embedds.T)
        # print('chunk_cossims:', chunk_cossims)
        # print('        chunk_cossims.shape:', chunk_cossims.shape)   # chunk_cossims.shape: (10000, 10572)

        # --- MODIFIED LOGIC HERE ---
        # 1. Find the highest overall similarity each new subject has to ANY base subject
        max_cossims_per_new_subj = np.max(chunk_cossims, axis=1, keepdims=True)
        
        # 2. Build the mask: 
        #    - Individual pairs must fall within the range
        #    - BUT the new subject's global maximum similarity cannot exceed your upper limit
        mask = (chunk_cossims >= similarity_range[0]) & (chunk_cossims <= similarity_range[1]) & (max_cossims_per_new_subj <= similarity_range[1])
        
        chunk_local_indices = np.argwhere(mask)
        
        # Optimized out the explicit for loop using advanced numpy indexing
        chunk_cossims_selected_embedds = chunk_cossims[chunk_local_indices[:, 0], chunk_local_indices[:, 1]].reshape(-1, 1)
        # ---------------------------

        # print('        chunk_local_indices:', chunk_local_indices)
        # print('        chunk_local_indices.shape:', chunk_local_indices.shape)

        chunk_global_indices = copy.deepcopy(chunk_local_indices)
        chunk_global_indices[:,0] += (idx_chunk * chunk_size)
        # print('        chunk_global_indices:', chunk_global_indices)
        print('        chunk_global_indices.shape:', chunk_global_indices.shape)
        # print('        chunk_global_indices.dtype:', chunk_global_indices.dtype)

        global_indices_selected_embedds = np.vstack((global_indices_selected_embedds, chunk_global_indices))
        global_cossims_selected_embedds = np.vstack((global_cossims_selected_embedds, chunk_cossims_selected_embedds))
        print('        global_indices_selected_embedds.shape:', global_indices_selected_embedds.shape)
        # print('        global_indices_selected_embedds.dtype:', global_indices_selected_embedds.dtype)

    return global_indices_selected_embedds, global_cossims_selected_embedds


def make_dict_new_subjs_base_subjs(new_subj_embedds_paths, base_subj_embedds_paths, indices_2d_selected_new_subjs, cossims_selected_new_subjs):
    dict_paths_new_subjs_base_subjs = {}
    for i, indices_2d in enumerate(indices_2d_selected_new_subjs):
        if not new_subj_embedds_paths[indices_2d[0]] in dict_paths_new_subjs_base_subjs:
            dict_paths_new_subjs_base_subjs[new_subj_embedds_paths[indices_2d[0]]] = []
        dict_paths_new_subjs_base_subjs[new_subj_embedds_paths[indices_2d[0]]].append((base_subj_embedds_paths[indices_2d[1]], cossims_selected_new_subjs[i][0]))
    # print('dict_paths_new_subjs_base_subjs:', dict_paths_new_subjs_base_subjs)
    # sys.exit(0)
    return dict_paths_new_subjs_base_subjs


def plot_two_sets_images(images_person1, images_person2, similarity, save_path='comparison.png', subj_labels=('Person 1', 'Person 2'), sup_title=''):
    ncols = max(len(images_person1), len(images_person2))
    nrows = 3

    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows))
    if ncols == 1:
        axes = axes.reshape(nrows, 1)

    # Plot first row
    for j in range(ncols):
        ax = axes[0, j]
        if j < len(images_person1):
            ax.imshow(images_person1[j])
        ax.axis('off')

    # Plot second row
    for j in range(ncols):
        ax = axes[1, j]
        if j < len(images_person2):
            ax.imshow(images_person2[j])
        ax.axis('off')

    # Plot third row
    for j in range(ncols):
        ax = axes[2, j]
        ax.axis('off')

    # Add row titles outside the grid
    fontsize = 30
    fig.text(0.1, 0.75, subj_labels[0],            fontsize=fontsize, va='center', ha='right')
    fig.text(0.1, 0.5, subj_labels[1],             fontsize=fontsize, va='center', ha='right')
    fig.text(0.1, 0.25, f"Sim = {similarity:.4f}", fontsize=fontsize, va='center', ha='right')

    fig.suptitle(sup_title, fontsize=50, x=0.05, ha='left')

    # plt.tight_layout(rect=[0.05, 0, 1, 1])
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)


def make_save_figures_selected_and_existing_identities(dict_paths_new_subjs_base_subjs, path_output_figures_ids_folder, split_str='_FACE_EMBEDDINGS_R100_WebFace42M_ArcFace/'):
    img_exts = ['.jpg','.jpeg','.png']
    for idx_key_new_subj, (key_new_subj, base_subj_list_of_list) in enumerate(dict_paths_new_subjs_base_subjs.items()):
        path_dir_new_subj  = os.path.join(key_new_subj.split(split_str)[0], key_new_subj.split('/')[-2])
        # print(f"{idx_key_new_subj}/{len(dict_paths_new_subjs_base_subjs)} - {path_dir_new_subj}")
        paths_imgs_new_subj = find_files_paths(path_dir_new_subj, exts=img_exts, substring_file='', verbose=False)
        assert len(paths_imgs_new_subj) > 0, f"Error, no files found with exts {img_exts} in path \'{path_dir_new_subj}\'"
        # print('paths_imgs_new_subj:', paths_imgs_new_subj)
        imgs_new_subj = load_imgs_from_paths(paths_imgs_new_subj)

        for idx_base_subj_list, base_subj_list in enumerate(base_subj_list_of_list):
            base_subj = base_subj_list[0]
            cossim    = base_subj_list[1]
            path_dir_base_subj = os.path.join(base_subj.split(split_str)[0], base_subj.split('/')[-2])
            # print(f"    {idx_base_subj_list}/{len(base_subj_list_of_list)} - {path_dir_base_subj}")
            paths_imgs_base_subj = []
            paths_imgs_base_subj = find_files_paths(path_dir_base_subj, exts=img_exts, substring_file='', verbose=False)
            assert len(paths_imgs_base_subj) > 0, f"Error, no files found with exts {img_exts} in path \'{path_dir_base_subj}\'"
            # print('paths_imgs_base_subj:', paths_imgs_base_subj)
            imgs_base_subj = load_imgs_from_paths(paths_imgs_base_subj)
        
            base_subj_label = f"{path_dir_base_subj.split('/')[-3]}_{path_dir_base_subj.split('/')[-1]}"
            new_subj_label  = f"{path_dir_new_subj.split('/')[-3]}_{path_dir_new_subj.split('/')[-1]}"
            title = f"Identity Cosine Similarity - Datasets \'{path_dir_base_subj.split('/')[-3]}\' and \'{path_dir_new_subj.split('/')[-3]}\'"
            figure_file_name = f"sim={cossim:.4f}_{base_subj_label}_and_{new_subj_label}.jpg"
            path_figure_file = os.path.join(path_output_figures_ids_folder, figure_file_name)
            print(f"    {idx_key_new_subj}/{len(dict_paths_new_subjs_base_subjs)} - {idx_base_subj_list}/{len(base_subj_list_of_list)} - {path_figure_file}")
            plot_two_sets_images(imgs_base_subj[:20], imgs_new_subj[:20], cossim, save_path=path_figure_file, subj_labels=(base_subj_label, new_subj_label), sup_title=title)
            # sys.exit(0)
        # print("-----------------")




def main(args):
    assert os.path.exists(args.path_base_subj_embedds), f'Error: no such path or file \'{args.path_base_subj_embedds}\''
    assert os.path.exists(args.path_new_subjs_embedds), f'Error: no such path or file \'{args.path_new_subjs_embedds}\''


    output_folder_name = f"merge_with_dataset_{'-'.join(args.path_new_subjs_embedds.split('/')[-3:])}_sim-range={args.similarity_range}_EXCLUSIVE".replace(' ','')
    path_output_folder = os.path.join(os.path.dirname(args.path_base_subj_embedds), output_folder_name)
    print(f"path_output_folder: \'{path_output_folder}\'")
    os.makedirs(path_output_folder, exist_ok=True)


    path_base_subj_data = os.path.join(path_output_folder, 'base_subj_data.pkl')
    if not os.path.isfile(path_base_subj_data):
        print('\nLoading base subjects embeddings...')
        print(f"    {args.path_base_subj_embedds}")
        base_subj_embedds_paths = find_files_paths(args.path_base_subj_embedds, exts=['.npy'], substring_file=args.substring_file_base_subj)
        base_subj_embedds       = load_embeddings(base_subj_embedds_paths)

        dict_base_subj_data = {'base_subj_embedds_paths': base_subj_embedds_paths,
                               'base_subj_embedds':       base_subj_embedds}
        print(f"    Saving pre-loaded data to disk: \'{path_base_subj_data}\'")
        save_pickle(dict_base_subj_data, path_base_subj_data)
    else:
        print(f"\nLoading pre-saved data from disk: \'{path_base_subj_data}\'")
        dict_base_subj_data = load_pickle(path_base_subj_data)
        base_subj_embedds_paths = dict_base_subj_data['base_subj_embedds_paths']
        base_subj_embedds       = dict_base_subj_data['base_subj_embedds']
    

    path_new_subj_data = os.path.join(path_output_folder, 'new_subj_data.pkl')
    if not os.path.isfile(path_new_subj_data):
        print('\nLoading new subjects embeddings...')
        print(f"    {args.path_new_subjs_embedds}")
        new_subj_embedds_paths        = find_files_paths(args.path_new_subjs_embedds, exts=['.npy'], substring_file=args.substring_file_new_subj)
        # indices_2d_selected_new_subjs, cossims_selected_new_subjs = load_select_new_subjs_embedds(base_subj_embedds, new_subj_embedds_paths, args.similarity_range)
        indices_2d_selected_new_subjs, cossims_selected_new_subjs = load_select_new_subjs_embedds_exclusive(base_subj_embedds, new_subj_embedds_paths, args.similarity_range)
    
        dict_new_subj_data = {'new_subj_embedds_paths':        new_subj_embedds_paths,
                              'indices_2d_selected_new_subjs': indices_2d_selected_new_subjs,
                              'cossims_selected_new_subjs':    cossims_selected_new_subjs}
        print(f"    Saving pre-loaded data to disk: \'{path_new_subj_data}\'")
        save_pickle(dict_new_subj_data, path_new_subj_data)
    else:
        print(f"\nLoading pre-saved data from disk: \'{path_new_subj_data}\'")
        dict_new_subj_data = load_pickle(path_new_subj_data)
        new_subj_embedds_paths        = dict_new_subj_data['new_subj_embedds_paths']
        indices_2d_selected_new_subjs = dict_new_subj_data['indices_2d_selected_new_subjs']
        cossims_selected_new_subjs    = dict_new_subj_data['cossims_selected_new_subjs']
        # print('cossims_selected_new_subjs:', cossims_selected_new_subjs)
        # print('cossims_selected_new_subjs.shape:', cossims_selected_new_subjs.shape)


    path_dict_paths_new_subjs_base_subjs = os.path.join(path_output_folder, 'dict_paths_new_subjs_base_subjs.json')
    if not os.path.isfile(path_dict_paths_new_subjs_base_subjs):
        print('\nMaking dict of subjs paths...')
        dict_paths_new_subjs_base_subjs = make_dict_new_subjs_base_subjs(new_subj_embedds_paths, base_subj_embedds_paths, indices_2d_selected_new_subjs, cossims_selected_new_subjs)
        print('\nlen(dict_paths_new_subjs_base_subjs):', len(dict_paths_new_subjs_base_subjs))
        print(f"    Saving dict_paths_new_subjs_base_subjs: \'{path_dict_paths_new_subjs_base_subjs}\'")
        save_json(dict_paths_new_subjs_base_subjs, path_dict_paths_new_subjs_base_subjs, indent=4)
    else:
        print(f"\nLoading dict_paths_new_subjs_base_subjs: \'{path_dict_paths_new_subjs_base_subjs}\'")
        dict_paths_new_subjs_base_subjs = load_json(path_dict_paths_new_subjs_base_subjs)


    # indices_1d_selected_new_subjs = indices_2d_selected_new_subjs[:,0]
    # selected_new_subj_embedds_paths = [new_subj_embedds_paths[idx] for idx in indices_1d_selected_new_subjs]
    # print('\nlen(selected_new_subj_embedds_paths):', len(selected_new_subj_embedds_paths))


    output_figures_ids_folder_name = "figures_ids"
    path_output_figures_ids_folder = os.path.join(path_output_folder, output_figures_ids_folder_name)
    print(f"\nSaving faces figures - path_output_figures_ids_folder: \'{path_output_figures_ids_folder}\'")
    os.makedirs(path_output_figures_ids_folder, exist_ok=True)
    make_save_figures_selected_and_existing_identities(dict_paths_new_subjs_base_subjs, path_output_figures_ids_folder, split_str='_FACE_EMBEDDINGS_R100_WebFace42M_ArcFace/')


    print('\nFinished!\n')




if __name__ == '__main__':
    args = parse_args()
    
    # chunk_idxs = make_indices_chunks(total_size=93431, chunk_size=10000)
    main(args)