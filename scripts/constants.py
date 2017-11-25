# Work space directory
HOME_DIR = '/home/mhasek/saliency-salgan-2017/'

# Path to SALICON raw data
pathToImages = '/home/mhasek/Documents/CIS680/FinalProject/Datasets/SALICON/images/'
pathToMaps = '/home/mhasek/Documents/CIS680/FinalProject/Datasets/SALICON/maps/'
pathToFixationMaps = '/home/mhasek/Documents/CIS680/FinalProject/Datasets/SALICON/fixations/'

# Path to processed data
pathOutputImages = '/home/mhasek/Documents/CIS680/FinalProject/Datasets/SALICON/images320x240/'
pathOutputMaps = '/home/mhasek/Documents/CIS680/FinalProject/Datasets/SALICON/saliency320x240/'
pathToPickle = '/home/mhasek/Documents/CIS680/FinalProject/Datasets/SALICON/320x240/'

# Path to pickles which contains processed data
TRAIN_DATA_DIR = '/home/mhasek/Documents/CIS680/FinalProject/Datasets/SALICON/320x240/fix_trainData.pickle'
VAL_DATA_DIR = '/home/mhasek/Documents/CIS680/FinalProject/Datasets/SALICON/320x240/fix_validationData.pickle'
TEST_DATA_DIR = '/home/mhasek/Documents/CIS680/FinalProject/Datasets/SALICON/256x192/testData.pickle'

# Path to vgg16 pre-trained weights
PATH_TO_VGG16_WEIGHTS = '/home/mhasek/saliency-salgan-2017/vgg16.pkl'

# Input image and saliency map size
INPUT_SIZE = (256, 192)

# Directory to keep snapshots
DIR_TO_SAVE = 'test'
