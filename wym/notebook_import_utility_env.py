"""
%load_ext autoreload
%autoreload 2
import os
prefix = ''
if os.path.expanduser('~') == '/home/baraldian': # UNI env
    prefix = '/home/baraldian'
else:
    from google.colab import drive
    drive.mount('/content/drive')
softlab_path = os.path.join(prefix + '/content/drive/Shareddrives/SoftLab/')
project_path = os.path.join(softlab_path, 'Projects', 'Fairness','scalable-fairlearn')
import sys
sys.path.append(os.path.join(project_path))
sys.path = list(set(sys.path))

exec(open(os.path.join(project_path,'notebook_import_utility_env.py')).read())
"""

import ast
import copy
import gc
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import re
import requests
import socket
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils import io
from multiprocessing import Pool
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from torch.utils.data import Dataset, DataLoader
from tqdm.autonotebook import tqdm
from warnings import simplefilter
from datetime import datetime
import jsonlines

# simplefilter(action='ignore', category=FutureWarning)
# simplefilter(action='ignore')

host_name = socket.gethostname()
# print(f'{host_name = }')
print(f'\nCurrently executing on {host_name}')

prefix = ''
if '/home/' in os.path.expanduser('~'):  # UNI env
    prefix = os.path.expanduser('~')
else:
    # install here for colab env
    # !pip install lime
    # !pip install -q spacy
    # !pip install -q pytorch-lightning
    # !pip install -q transformers
    # !pip install -q -U sentence-transformers
    # !pip install -U nltk
    # !pip install pyyaml==5.4.1
    prefix = '.'


def check_pathc_exist(path):
    if not os.path.exists(path):
        print(f"Missing path. Expected path: {path}")
        sys.exit(1)


softlab_path = os.path.join(prefix + '/content/drive/Shareddrives/SoftLab/')
project_path = os.path.join(softlab_path, 'Projects', 'WYM')
dataset_path = os.path.join(softlab_path, 'Dataset', 'Entity Matching')
model_files_path = os.path.join(project_path, 'dataset_files')
base_files_path = os.path.join(project_path, 'dataset_files')

tabelle_paper_path = os.path.join(project_path, 'RISULTATI')
os.makedirs(tabelle_paper_path, exist_ok=True)

landmark_path = os.path.join(softlab_path, 'Projects', 'Landmark Explanation EM')
sys.path.append(landmark_path)
github_code_path = os.path.join(landmark_path, 'Landmark_github')

sys.path.append(os.path.join(softlab_path, 'Projects/external_github/ditto'))
sys.path.append(os.path.join(softlab_path, 'Projects/external_github'))
sys.path.append(github_code_path)

bert_src_path = os.path.join(project_path, 'wym_github')

for path in [softlab_path, project_path, dataset_path, model_files_path, base_files_path, github_code_path,
             bert_src_path]:
    check_pathc_exist(path)

sys.path.append(os.path.join(project_path, 'common_functions'))
sys.path.append(os.path.join(project_path, 'notebooks' if 'baraldian' in prefix else 'notebooks giacomo'))
sys.path.append(os.path.join(project_path, 'src'))
sys.path.append(os.path.join(project_path, 'src', 'BERT'))

sys.path = list(set(sys.path))

# pandas and numpy setup
pd.options.display.float_format = '{:.4f}'.format
np.set_printoptions(formatter={'float_kind': '{:.4f}'.format})
pd.options.display.max_rows = 150
pd.options.display.max_columns = 150
pd.options.display.max_colwidth = 100
pd.options.display.precision = 15
pd.options.display.max_info_columns = 150
InteractiveShell.ast_node_interactivity = "all"  # Display all statements

excluded_cols = ['id', 'left_id', 'right_id']

import seaborn as sns

sns.set()  # for plot styling
sns.set(rc={"figure.dpi": 200, 'savefig.dpi': 400})
sns.set_context('notebook')
sns.set_style('whitegrid')
# plt.rcParams.update({'font.size': 14})
# plt.tight_layout()

print("Environment set up correctly.")

# plt.rcParams['legend.fontsize'] = 14
# plt.rcParams["figure.figsize"] = (14,5)
# plt.rcParams['axes.titlepad'] = 2
# fig, axes = plt.subplots(2,6, sharex=True, sharey=True)
# ax_flat = axes.flatten()
# fig.text(0, 0.5, 'F1', va='center', rotation='vertical')
# labels = to_plot.columns.str.replace('del_','').map(
# {'useful_impact': 'MoRF', 'useless_impact':'LeRF', 'random':'random'})
# fig.legend(axes,     # The line objects
#            labels=labels,   # The labels for each line
#            loc="lower center",   # Position of legend
#            borderaxespad=-0.4,    # Small spacing around legend box
#           #  title="Legend Title"  # Title for the legend
#            ncol=5,
#            )
#
# plt.tight_layout()
# fig.savefig(os.path.join(tabelle_paper_path, '...')) # , bbox_inches='tight'
