# utilities to load video data from SUMMET:
# https://www.cs.colostate.edu/~vision/summet/
# can also use highway dataset from: 
# http://jacarini.dinf.usherbrooke.ca/dataset2012

# Alex also has on his OneDrive:
# https://gtvault-my.sharepoint.com/:f:/g/personal/asf3_gatech_edu/El8WVy1iy5RHsvCRw0lZgkoB6MB9-NhQ6XHDAz4ixKSePQ?e=Yxm6nB

import os
import glob
import numpy as np
import cv2


this_dir = os.path.dirname(os.path.abspath(__file__))


class Highway_Loader():
    def __init__(self):
        highway_dir = os.path.join(this_dir, 'highway')
        self.data_dir = os.path.join(highway_dir, 'input')
        self.gt_dir = os.path.join(highway_dir, 'groundtruth')
        assert os.path.exists(self.data_dir) and os.path.exists(self.gt_dir), \
            'highway dataset not found'

        self.fnames = sorted(glob.glob(os.path.join(self.data_dir, '*.jpg')))
        self.fnames_gt = sorted(glob.glob(os.path.join(self.gt_dir, '*.png')))
        assert len(self.fnames) == len(self.fnames_gt), \
            'highway dataset has different number of frames and groundtruths'
        self.n_samples = len(self.fnames)

    def load_data(self, frame_idx):
        assert 0 <= frame_idx < self.n_samples, 'frame index out of range'
        img = cv2.imread(self.fnames[frame_idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def load_gt(self, frame_idx):
        assert 0 <= frame_idx < self.n_samples, 'frame index out of range'
        img = cv2.imread(self.fnames_gt[frame_idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

class SUMMET_Loader():
    def __init__(self):
        self.data_dir = os.path.join(this_dir, 'summet')
        assert os.path.exists(self.data_dir), 'summet dataset not found'

    def load_data(self):
        pass


if __name__ == '__main__':
    loader = Highway_Loader()
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
