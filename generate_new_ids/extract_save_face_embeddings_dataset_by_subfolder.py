import os, sys
import argparse

import cv2
from PIL import Image
import numpy as np
import torch
import re
import time

import insightface



def parse_args():
    parser = argparse.ArgumentParser(description='')
    # parser.add_argument('--network', type=str, default='r100', help='backbone network')
    parser.add_argument('--weights', type=str, default='../models/antelopev2/arcface.onnx')
    parser.add_argument('--imgs', type=str, default='/nobackup3/bjgbiesseck/datasets/face_recognition/CASIA-WebFace/imgs_crops_112x112')
    parser.add_argument('--output-path', type=str, default='')
    parser.add_argument('--start-idx', type=int, default=0)
    args = parser.parse_args()
    return args


def get_face_recognition_model(path_weights="../models/antelopev2/arcface.onnx"):
    model = insightface.model_zoo.get_model(path_weights)
    model.prepare(ctx_id=0)
    return model


def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


def get_all_files_in_path(folder_path, file_extension=['.jpg','.jpeg','.png'], pattern=''):
    file_list = []
    for root, _, files in os.walk(folder_path):
        for filename in files:
            path_file = os.path.join(root, filename)
            for ext in file_extension:
                if pattern in path_file and path_file.lower().endswith(ext.lower()):
                    file_list.append(path_file)
                    print(f'Found files: {len(file_list)}', end='\r')
    print()
    file_list = natural_sort(file_list)
    return file_list


def get_subdirs_folder(root_path):
    subfolders = [os.path.join(root_path, name) for name in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, name))]
    subfolders = natural_sort(subfolders)
    return subfolders


def load_img(path_img=''):
    img = np.array(Image.open(path_img))[:,:,::-1]
    return img


@torch.no_grad()
def get_face_embedd(model, img):
    # embedd = model.get_feat(img).cpu().numpy()
    embedd = model.get_feat(img)
    return embedd


def save_face_embedd(id_feat_img, output_path_id_feat):
    if output_path_id_feat.endswith('.pt'):
        torch.save(id_feat_img, output_path_id_feat)
    elif output_path_id_feat.endswith('.npy'):
        np.save(output_path_id_feat, id_feat_img)


def load_face_embedd(path_id_feat):
    if path_id_feat.endswith('.pt'):
        id_feat_img = torch.load(path_id_feat)
    elif path_id_feat.endswith('.npy'):
        id_feat_img = np.load(path_id_feat)
    return id_feat_img





if __name__ == '__main__':
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f'Loading trained model: {args.weights}')
    # model = load_trained_model(args.network, args.weight, device)
    model = get_face_recognition_model(args.weights)
    print()

    args.imgs = args.imgs.rstrip('/')
    if not args.output_path:
        args.output_path = args.imgs + '_FACE_EMBEDDINGS_R100_WebFace42M_ArcFace'
    os.makedirs(args.output_path, exist_ok=True)

    print(f'Searching images in \'{args.imgs}\'')
    # imgs_paths = get_all_files_in_path(args.imgs)
    subjs_folders_paths = get_subdirs_folder(args.imgs)
    print(f'Found {len(subjs_folders_paths)} subjects\n------------------\n')

    total_elapsed_time = 0.0
    for idx_subj_folder, path_subj_folder in enumerate(subjs_folders_paths):
        if idx_subj_folder >= args.start_idx:
            start_time = time.time()
            print(f'{idx_subj_folder}/{len(subjs_folders_paths)} - Computing face embedding')
            print(f'subj: {os.path.basename(path_subj_folder)}')

            subj_output_path = os.path.join(args.output_path, os.path.basename(path_subj_folder))
            # print('subj_output_path:', subj_output_path)
            os.makedirs(subj_output_path, exist_ok=True)

            imgs_paths = get_all_files_in_path(path_subj_folder)
            subj_embedds = np.zeros((len(imgs_paths),512), dtype=float)

            for idx_path, path_img in enumerate(imgs_paths):
                img_name, img_ext = os.path.splitext(os.path.basename(path_img))
                # output_path_id_feat = os.path.join(subj_output_path, img_name+'_embedding_r100_webface42m_arcface.pt')
                output_path_id_feat = os.path.join(subj_output_path, img_name+'_embedding_r100_webface42m_arcface.npy')

                if not os.path.isfile(output_path_id_feat):
                    img = load_img(path_img)
                    print(f"{idx_path}/{len(imgs_paths)} Computing and saving sample face embeddings", end='\r')
                    id_feat_img = get_face_embedd(model, img)

                    # output_path_dir = os.path.dirname(path_img.replace(args.imgs, args.output_path))
                    # print(f'output_path_dir: {output_path_dir}')
                    # os.makedirs(output_path_dir, exist_ok=True)

                    # print('output_path_id_feat:', output_path_id_feat)
                    if output_path_id_feat.endswith('.pt'):
                        torch.save(id_feat_img, output_path_id_feat)
                    elif output_path_id_feat.endswith('.npy'):
                        np.save(output_path_id_feat, id_feat_img)                

                else:
                    # print('Loading embedding already saved:', output_path_id_feat, end='\r')
                    print(f"{idx_path}/{len(imgs_paths)} Loading sample face embedding already saved", end='\r')
                    id_feat_img = load_face_embedd(output_path_id_feat)
                
                subj_embedds[idx_path] = id_feat_img
                # print('id_feat_img:', id_feat_img)
                # print('id_feat_img.shape:', id_feat_img.shape)
                # sys.exit(0)
            print()

            output_path_mean_embedd = os.path.join(subj_output_path, f'{os.path.basename(path_subj_folder)}_mean_embedding_r100_arcface.npy')
            if not os.path.isfile(output_path_mean_embedd):
                subj_embedd_mean = subj_embedds.mean(axis=0, keepdims=True)
                print(f'Saving mean subject embedding: \'{output_path_mean_embedd}\'')
                save_face_embedd(subj_embedd_mean, output_path_mean_embedd)
            else:
                print(f'Mean subject embedding already saved: \'{output_path_mean_embedd}\'')

            elapsed_time = time.time()-start_time
            total_elapsed_time += elapsed_time
            avg_sample_time = total_elapsed_time / ((idx_subj_folder-args.start_idx)+1)
            estimated_time = avg_sample_time * (len(subjs_folders_paths)-(idx_subj_folder+1))
            print("    Elapsed time: %.3fs" % elapsed_time)
            print("    Avg elapsed time: %.3fs" % avg_sample_time)
            print("    Total elapsed time: %.3fs,  %.3fm,  %.3fh" % (total_elapsed_time, total_elapsed_time/60, total_elapsed_time/3600))
            print("    Estimated Time to Completion (ETC): %.3fs,  %.3fm,  %.3fh" % (estimated_time, estimated_time/60, estimated_time/3600))
            print('--------------')
            # sys.exit(0)

        else:
            print(f'Skipping indices: {idx_subj_folder}/{len(subjs_folders_paths)}', end='\r')


    print('\nFinished!')