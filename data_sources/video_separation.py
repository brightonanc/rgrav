# utilities to load video data from SUMMET:
# https://www.cs.colostate.edu/~vision/summet/
# can also use change detection dataset from:
# http://jacarini.dinf.usherbrooke.ca/dataset2012

# SUMMET instructions:
# 1. download the dataset from: https://www.cs.colostate.edu/~vision/summet/
# 2. put the dataset in the subfolder "Summet_Data"
#   a. .mat file should be Summet_Data/smaller_action_labels_2345.mat relative to this file
# 3. run "python video_separation.py" to make sure everything works


import os
import glob
import torch
import numpy as np
import scipy.io
import cv2
import h5py


this_dir = os.path.dirname(os.path.abspath(__file__))
def find_video_dir(video_name):
    dataset_dirs = ['dataset2012/dataset', 'dataset2014/dataset']
    for dataset_dir in dataset_dirs:
        subdirs = glob.glob(os.path.join(this_dir, dataset_dir, '*'))
        for subdir in subdirs:
            video_dir = os.path.join(subdir, video_name)
            if os.path.exists(os.path.join(video_dir, 'input')) and os.path.exists(os.path.join(video_dir, 'groundtruth')):
                return video_dir
    return None

class Loader_CDW():
    def __init__(self, video_name):
        video_dir = find_video_dir(video_name)
        if video_dir is None:
            raise ValueError(f'video {video_name} not found')
        self.data_dir = os.path.join(video_dir, 'input')
        self.gt_dir = os.path.join(video_dir, 'groundtruth')

        self.fnames = sorted(glob.glob(os.path.join(self.data_dir, '*.jpg')))
        self.fnames_gt = sorted(glob.glob(os.path.join(self.gt_dir, '*.png')))
        assert len(self.fnames) == len(self.fnames_gt), \
            '{} dataset has different number of frames and groundtruths'.format(video_dir)
        self.n_samples = len(self.fnames)

    def load_data(self, frame_idx):
        assert 0 <= frame_idx < self.n_samples, 'frame index out of range'
        img = cv2.imread(self.fnames[frame_idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img)
        img = img / 255.0
        return img

    def load_gt(self, frame_idx):
        assert 0 <= frame_idx < self.n_samples, 'frame index out of range'
        img = cv2.imread(self.fnames_gt[frame_idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img)
        img = img / 255.0
        return img

class SUMMET_Loader():
    def __init__(self):
        mat_loc = os.path.join(this_dir, 'Summet_Data/smaller_action_labels_2345.mat')
        assert os.path.exists(mat_loc), 'summet dataset not found'
        action_labels = scipy.io.loadmat(mat_loc)['smaller_action_labels'][0, 0]
        labels = action_labels[2]
        # just take the first label for each tracklet
        self.labels = labels[:, 0]
        self.labels = [label[0] for label in self.labels]

        mat_loc = os.path.join(this_dir, 'Summet_Data/DARPA_tracklets_2345_09_05.mat')
        assert os.path.exists(mat_loc), 'summet dataset not found'
        with h5py.File(mat_loc, 'r') as f:
            tracklet_refs = [f['tracklets'][0, i] for i in range(len(f['tracklets'][0]))]
            self.tracklets = [np.array(f[ref]['data']) for ref in tracklet_refs]
        self.tracklets = np.array(self.tracklets)
        self.tracklets = torch.from_numpy(self.tracklets)
        print('tracklets shape: ', self.tracklets.shape)

    def load_data(self, frame_idx):
        assert 0 <= frame_idx < len(self.tracklets), 'frame index out of range'
        return self.tracklets[frame_idx]


if __name__ == '__main__':
    summet_loader = SUMMET_Loader()
    ind = 1000
    track = summet_loader.load_data(ind)
    label = summet_loader.labels[ind]
    print(track.shape)
    import matplotlib.pyplot as plt
    plt.figure()
    plt.suptitle(f'tracklet {label}')
    for i in range(len(track)):
        plt.imshow(track[i].T, cmap='gray')
        plt.pause(0.2)
    exit()
    loader = Loader_CDW('highway')
    print('n_samples: ', loader.n_samples)
    print('data: ', loader.fnames[:3])
    print('ground truth: ', loader.fnames_gt[:3])

    # load an image and show it
    import matplotlib.pyplot as plt
    plt.figure(figsize=(15, 10))
    for m, idx in enumerate([0, 500, 1000]):
        img = loader.load_data(idx)
        img_gt = loader.load_gt(idx)

        plt.subplot(2, 3, m + 1)
        plt.imshow(img)
        plt.title(f'frame {idx}')

        plt.subplot(2, 3, m + 4)
        plt.imshow(img_gt)
        plt.title(f'GT {idx}')
    plt.show()
