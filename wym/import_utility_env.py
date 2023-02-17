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

import os
import sys
import pandas as pd
import numpy as np
from IPython.core.interactiveshell import InteractiveShell
from .import_utility import check_pathc_exist, prefix

softlab_path = os.path.join(prefix, 'content/drive/Shareddrives/SoftLab')
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
