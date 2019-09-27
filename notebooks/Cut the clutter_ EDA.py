import pandas as pd
import numpy as np
import cv2
from PIL import Image
from matplotlib import pyplot as plt
from pathlib import Path

# Load the SDK
from lyft_dataset_sdk.lyftdataset import LyftDataset, LyftDatasetExplorer
from lyft_dataset_sdk.utils.data_classes import LidarPointCloud


train = pd.read_csv('../data/train.csv')
train.head()


level5data = LyftDataset(data_path='../data/', json_path='../data/train_data', verbose=True)


# In[5]:


token0 = train.iloc[0]['Id']
token0


my_sample = level5data.get('sample', token0)
my_sample

my_scene = level5data.get('scene', my_sample['scene_token'])
my_scene

out = Path('out.avi')
level5data.render_scene(my_scene['token'], out_path=out)

