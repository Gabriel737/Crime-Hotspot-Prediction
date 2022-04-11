## Vancouver crime data config file

VAN_DATA_RAW = '../data/raw'
VAN_DATA_PRCD = '../data/processed'
VAN_DATA_SHP = '../data/shape_files'
MODEL_SAVE_PATH = '../data/model_states'

# Vancouver UTM zone number and letter
UTM_ZONE_NO = 10
UTM_ZONE_LTR = 'U'

# Relevant columns 
REL_COLS = ['TYPE','YEAR','MONTH','DAY','NEIGHBOURHOOD','X','Y']

# Self-defined crime categories
CRIME_CATS = ['Break and Enter', 'Homicide', 'Mischief', 'Assualt', 'Theft', 'Vehicle Collision']

# Crime types present in Vancouver crime data
CRIME_TYPES = ['Break and Enter Commercial','Break and Enter Residential/Other',
               'Homicide','Mischief','Offence Against a Person','Other Theft',
               'Theft from Vehicle','Theft of Bicycle','Theft of Vehicle',
               'Vehicle Collision or Pedestrian Struck (with Fatality)',
               'Vehicle Collision or Pedestrian Struck (with Injury)']

# Crime type to crime category mapping
TYPE2CAT = {'Break and Enter Commercial':'Break and Enter',
            'Break and Enter Residential/Other':'Break and Enter',
            'Homicide':'Homicide', 'Mischief':'Mischief',
            'Offence Against a Person':'Assualt',
            'Other Theft': 'Theft',
            'Theft from Vehicle':'Theft',
            'Theft of Bicycle':'Theft',
            'Theft of Vehicle':'Theft',
            'Vehicle Collision or Pedestrian Struck (with Fatality)':'Vehicle Collision',
            'Vehicle Collision or Pedestrian Struck (with Injury)':'Vehicle Collision'}

# Vertices coordinates for bounding box
BB_VERTICES = {'BB_NE': {'lat': 49.30112, 'long': -123.02245}, 'BB_SE': {'lat': 49.18444, 'long': -123.02245},
             'BB_SW': {'lat': 49.18444, 'long': -123.20071}, 'BB_NW': {'lat': 49.30112, 'long': -123.20071}}

# Bounding box edge length (in kms)
BB_DIST = 13

# Bounding box cell length (in kms)
BB_CELL_LEN = 0.5

# Number of cells along an axis 
CELL_COUNT = 26

# Number of cells in each dimension
NUM_CELLS = BB_DIST * BB_CELL_LEN

# Size of input batch to constitute a single temporal training sample
SEQ_LEN = 16

# Torch device
DEVICE = 'cuda'

# Train batch size
TRAIN_BATCH_SIZE = 32

# Number of filters in ConvLSTM
HIDDEN_DIM = 64

# Kernel size in ConvLSTM - CNN
KERNEL_SIZE = 3

# Dropout probability
DROP_P = 0.5

# Learning rate
LR = 3e-5

# Save Model
SAVE = True

# No. of epochs
N_EPOCHS = 75

# Classification threshold
CLASS_THRESH = 0.45

# Random seed
RANDOM_SEED = 42

# Optimiser 
OPTIM_NAME = 'Adam'

# Evaluate thresholds 
EVAL_THRESHOLDS = [0.3, 0.4, 0.5 ,0.6, 0.7, 0.8]

# No. of secondary features
N_SEC_FEATS = 6

# Cross entropy loss weights
CROSS_ENT_LOSS_WEIGHTS = [1,3]



