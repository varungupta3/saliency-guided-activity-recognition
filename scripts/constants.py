# # # Work space directory
HOME_DIR = '/media/autel/5C4C-7166/Anand/CIS680/FinalProject/Saliency-GAN/'

# Path to SALICON raw data
pathToImages = '/home/autel/Downloads/CIS680Final/Datasets/images/'
pathToMaps = '/home/autel/Downloads/CIS680Final/Datasets/maps/'
pathToFixationMaps = '/home/autel/Downloads/CIS680Final/Datasets/fixations/'

# Path to processed data
pathOutputImages = '/home/autel/Downloads/CIS680Final/Datasets/images320x240/'
pathOutputMaps = '/home/autel/Downloads/CIS680Final/Datasets/saliency320x240/'
pathToPickle = '/home/autel/Downloads/CIS680Final/Datasets/320x240/'

# Path to pickles which contains processed data
TRAIN_DATA_DIR = '/home/autel/Downloads/CIS680Final/Datasets/320x240/trainData.pickle'
VAL_DATA_DIR = '/home/autel/Downloads/CIS680Final/Datasets/320x240/validationData.pickle'
TEST_DATA_DIR = '/home/autel/Downloads/CIS680Final/Datasets/320x240/testData.pickle'

# Path to vgg16 pre-trained weights
PATH_TO_VGG16_WEIGHTS = '/home/autel/Downloads/saliency-salgan-2017/vgg16.pkl'

# Input image and saliency map size
ORIG_SIZE = (1280, 960)
INPUT_SIZE = (256, 192)

# Directory to keep snapshots
DIR_TO_SAVE = 'test'

	# Pretrained train pytorch paths

# trainImagesPath = '../../cookingtrainimages.npy'
# trainMasksPath = '../../cookingtrainmask.npy'
# trainSaliencyImagesPath = '../../cookingtrainsaliencyimages.npy'

testImagesPath = '../../testimages.npy'
testMasksPath = '../../testmask.npy'

genWeightsPath = "../../gen_modelWeights0090.npz"
discWeightsPath = "../../discrim_modelWeights0090.npz"

gazeDataDir = '../../gaze/'
imageDataDir = '../../'
saveDataPath = "../../cooking/"
labelsDataDir = '../../labels_cleaned/'

		# LSTM paths

trainImagesPath = '../../cooking/images.npy'
trainMasksPath = '../../cooking/saliency_maps.npy'
trainActionsPath = '../../cooking/action1.npy'
trainObjectsPath = '../../cooking/action2.npy'
trainActionsListPath = '../../cooking/ordered_action1.npy'
trainObjectsListPath = '../../cooking/ordered_action2.npy'



#----------------------------------------------------------------------------------------------------------------
# Yasin's Workspace
#----------------------------------------------------------------------------------------------------------------

# # Work space directory
# HOME_DIR = '/home/mhasek/saliency-salgan-2017/'

# # Path to SALICON raw data
# pathToImages = '/home/mhasek/Documents/CIS680/FinalProject/Datasets/SALICON/images/'
# pathToMaps = '/home/mhasek/Documents/CIS680/FinalProject/Datasets/SALICON/maps/'
# pathToFixationMaps = '/home/mhasek/Documents/CIS680/FinalProject/Datasets/SALICON/fixations/'

# # Path to processed data
# pathOutputImages = '/home/mhasek/Documents/CIS680/FinalProject/Datasets/SALICON/images320x240/'
# pathOutputMaps = '/home/mhasek/Documents/CIS680/FinalProject/Datasets/SALICON/saliency320x240/'
# pathToPickle = '/home/mhasek/Documents/CIS680/FinalProject/Datasets/SALICON/320x240/'

# # Path to pickles which contains processed data
# TRAIN_DATA_DIR = '/home/mhasek/Documents/CIS680/FinalProject/Datasets/SALICON/320x240/fix_trainData.pickle'
# VAL_DATA_DIR = '/home/mhasek/Documents/CIS680/FinalProject/Datasets/SALICON/320x240/fix_validationData.pickle'
# TEST_DATA_DIR = '/home/mhasek/Documents/CIS680/FinalProject/Datasets/SALICON/256x192/testData.pickle'

# # Path to vgg16 pre-trained weights
# PATH_TO_VGG16_WEIGHTS = '/home/mhasek/saliency-salgan-2017/vgg16.pkl'

# # Input image and saliency map size
# INPUT_SIZE = (256, 192)

# # Directory to keep snapshots
# DIR_TO_SAVE = 'test'

#----------------------------------------------------------------------------------------------------------------
# Varun's Workspace
#----------------------------------------------------------------------------------------------------------------

# # Work space directory
# HOME_DIR = '/home/varun/Courses/CIS680/project/datasets/SALICON/saliency-salgan-2017/'

# # Path to SALICON raw data
# pathToImages = '/home/varun/Courses/CIS680/project/datasets/SALICON/images/'
# pathToMaps = '/home/varun/Courses/CIS680/project/datasets/SALICON/maps/'
# pathToFixationMaps = '/home/varun/Courses/CIS680/project/datasets/SALICON/fixations/'

# # Path to processed data
# pathOutputImages = '/home/varun/Courses/CIS680/project/datasets/SALICON/images320x240/'
# pathOutputMaps = '/home/varun/Courses/CIS680/project/datasets/SALICON/saliency320x240/'
# pathToPickle = '/home/varun/Courses/CIS680/project/datasets/SALICON/320x240/'

# # Path to pickles which contains processed data
# TRAIN_DATA_DIR = '/home/varun/Courses/CIS680/project/datasets/SALICON/320x240/trainData.pickle'
# VAL_DATA_DIR = '/home/varun/Courses/CIS680/project/datasets/SALICON/320x240/validationData.pickle'
# TEST_DATA_DIR = '/home/varun/Courses/CIS680/project/datasets/SALICON/256x192/testData.pickle'

# # Path to vgg16 pre-trained weights
# PATH_TO_VGG16_WEIGHTS = '/home/varun/Courses/CIS680/saliency-salgan-2017/vgg16.pkl'

# # Input image and saliency map size
# ORIG_SIZE = (1280, 960)
# INPUT_SIZE = (256, 192)

# # Directory to keep snapshots
# DIR_TO_SAVE = 'test'

# trainImagesPath = '../../../datasets/data/resizedcookingtrainimages.npy'
# trainMasksPath = '../../../datasets/data/resizedcookingtrainmask.npy'
# trainSaliencyImagesPath = '../../../datasets/data/resizedcookingtrainsaliencyimages.npy'

# testImagesPath = '../../../datasets/data/testimages.npy'
# testMasksPath = '../../../datasets/data/testmask.npy'

# genWeightsPath = "../../../datasets/gen_modelWeights0090.npz"
# discWeightsPath = "../../../datasets/discrim_modelWeights0090.npz"

# saveDataPath = "../../../datasets/GTEA/cooking/"
# labelsDataDir = '../../../datasets/GTEA/labels_cleaned/'
# gazeDataDir = '../../../datasets/GTEA/gaze/'
# imageDataDir = '../../../datasets/GTEA/'
