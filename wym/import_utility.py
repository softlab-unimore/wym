def in_notebook():
    try:
        from IPython import get_ipython
        if 'IPKernelApp' not in get_ipython().config:  # pragma: no cover
            return False
    except ImportError:
        return False
    except AttributeError:
        return False
    return True

def check_pathc_exist(path):
    if not os.path.exists(path):
        print(f"Missing path. Expected path: {path}")
        sys.exit(1)

import os
import sys

import ast
import copy
import gc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import re
import requests
import socket
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
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

if in_notebook():
    print("Jupyter Notebook detected.")
    from wym.notebook_import_utility_env import *
else:
    from wym.import_utility_env import *