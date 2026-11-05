import os
import sys
import pandas as pd

project_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(project_dir, 'derm7pt'))

from derm7pt.dataset import Derm7PtDatasetGroupInfrequent

dir_release = os.path.join(project_dir, 'release_v0')

# Dataset after grouping infrequent labels.
derm_data = Derm7PtDatasetGroupInfrequent(
    dir_images=os.path.join(dir_release, 'images'), 
    metadata_df=pd.read_csv(os.path.join(dir_release, 'meta/meta.csv')), 
    train_indexes=list(pd.read_csv(os.path.join(dir_release, 'meta/train_indexes.csv'))['indexes']), 
    valid_indexes=list(pd.read_csv(os.path.join(dir_release, 'meta/valid_indexes.csv'))['indexes']), 
    test_indexes=list(pd.read_csv(os.path.join(dir_release, 'meta/test_indexes.csv'))['indexes']))

# Outputs to screen the preprocessed dataset in a Pandas format.
derm_data.df