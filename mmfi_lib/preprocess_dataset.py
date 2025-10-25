import os
import numpy as np
import scipy.io as scio
from tqdm import tqdm
import glob

def preprocess_csi_file(input_path, output_path):
    data = scio.loadmat(input_path)['CSIamp']
    data[np.isinf(data)] = np.nan
    
    for i in range(10):
        temp_col = data[:, :, i]
        if np.isnan(temp_col).any():
            col_mean = np.nanmean(temp_col)
            temp_col[np.isnan(temp_col)] = col_mean
            data[:, :, i] = temp_col
    
    data = (data - np.min(data)) / (np.max(data) - np.min(data))
    np.save(output_path, data)

def preprocess_dataset(dataset_root):
    for scene in os.listdir(dataset_root):
        scene_path = os.path.join(dataset_root, scene)
        if not os.path.isdir(scene_path):
            continue
            
        for subject in os.listdir(scene_path):
            subject_path = os.path.join(scene_path, subject)
            if not os.path.isdir(subject_path):
                continue
                
            for action in os.listdir(subject_path):
                action_path = os.path.join(subject_path, action)
                csi_dir = os.path.join(action_path, 'wifi-csi')
                
                if not os.path.isdir(csi_dir):
                    continue
                
                for mat_file in tqdm(glob.glob(os.path.join(csi_dir, '*.mat')), 
                                    desc=f"{subject}/{action}"):
                    npy_file = mat_file.replace('.mat', '_processed.npy')
                    if not os.path.exists(npy_file):
                        preprocess_csi_file(mat_file, npy_file)

if __name__ == '__main__':
    dataset_root = r'D:\NCKH.2025-2026\VinWifi\MMFiDataset'
    preprocess_dataset(dataset_root)