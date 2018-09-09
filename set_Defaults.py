from update_conmat import *
import operator

class Init_param(object):
    def __init__(self):
        """
         set initial Parameters
        """
        # The number of computation threads to use
        self.comp_threads = 4

        """
        Plotting and Saving Directories
        """
        self.PLOT_RESULTS = 0
        # If >0 then results will be plotted every PLOT_RESULTS many epochs in the various forms
        self.SAVE_RESULTS = 5
        # If >0 then error function evaluation will be printed out
        self.DISPLAY_ERRORS = 1
        # Where to save the results
        self.fullsavepath = './Results/fruit_100_100/Run_1/'
        # Path to a folder containing only image files
        self.fulldatapath = './Datasets/Images/fruit_100_100/'
        # Set this to the last folder
        self.tag = 'fruit_100_100'
        # Used for Reconstruction: this is a path to a previously trained model used for reconstructing an image
        self.fullmodelpath = './Results/fruit_100_100/Run_1/epoch5_layer2.mat'

        """
        Preprocessing
        """
        # Parameters for the image creation/preprocessing.
        self.ZERO_MEAN = 1    # binary (subtract mean or not)
        self.CONTRAST_NORMALIZE = 1  # binary (constrast normalize images or not)
        self.COLOR_IMAGES = "gray"  # string: 'gray', 'rgb', 'ycbcr', 'hsv'
        self.SQUARE_IMAGES = 1   # binary (square images or not)

        """
        Training Adjustments
        """
        # Filters updates after all samples update z (doesn't work well)
        self.UPDATE_FILTERS_IN_BATCH = 0
        # Batch size used in training (leave at 1 as batching doesn't work well)
        self.batch_size = 1
        # If this is set to 0, then they are picked in order (1 is best)
        self.RANDOM_IMAGE_ORDER = 1
        # Number of epochs per layer through the training set (5 is usually sufficient)
        self.maxepochs = [1, 1, 5, 5]
        # Number of steps of conjugate gradient used when updating filters
        # and feature maps at each iteration (2 is best)
        self.min_iterations = 2
        # Threshold for the gradients
        self.grad_threshold = 0.01
        # For Yann's inference scheme, if you want to train the 1st layer initially
        self.LAYER1_FIRST = 0

        """
        model structure
        """
        # Number of layers total in the model you want to train Note: below you will
        # see many variables defined for a 4 layer model
        self.num_layers = 2
        # Size of each filters in each layer (assumes square). (7 is good)
        self.filter_size = [7, 7, 7, 7]
        # Number of feature maps in each layer. (this is the defualt)
        self.num_feature_maps = [3, 9, 100, 250]
        # Number of input maps in the first layer (do not modify this)
        if(operator.eq(self.COLOR_IMAGES, 'ycbcr') or operator.eq(self.COLOR_IMAGES, 'rgb') or operator.eq(self.COLOR_IMAGES, 'hsv')):
            num_input_maps = 3
        else:
            num_input_maps = 1
        # Number of input maps in all layers (do not modify this)
        self.num_input_maps = [num_input_maps, self.num_feature_maps[0], self.num_feature_maps[1],
                               self.num_feature_maps[2]]
        # The default types for the connectivity matrix (from cvpr2010)
        self.conmat_types = ['Full', 'SAD', 'Random Doubles', 'Random Doubles']

       # self.conmat_value = update_conmat(self.num_layers, self.conmat_types, self.num_input_maps, self.num_feature_maps)

        """
        Learning parameters
        """
        # Reconstruction term lambda*||sum(conv(z,f)) - y||^2 weighting for each layer (1 works well)
        self.lamb = [1, 1, 1, 1]
        # The alpha-norm on the auxilary variable kappa*|x|^alpha for each layer
        self.alphavalue = [0.8, 0.8, 1, 1]
        # A coefficient on the filter L2 regularization
        self.kappa = [1, 1, 1, 1]

        # The regeme over the continutaion variable, beta
        self.Binitial = 1   # start value
        self.Bmultiplier = 10  # beta = beta*Bmultiplier each iteration
        self.betaT = 4       # number of iterations

        """
        z0 map parameters, still fairly experimental
        """
        # Make the z0 filter size the same as the first layer's by default.
        self.z0_filter_size = self.filter_size[0]
        # The coefficient on the gradient(z0) term when training z0
        self.psi = 1
        # Check to determine if you even want to train the z0 map (while training z and f)
        self.TRAIN_Z0 = 0

        """
        Experimental Features (do not modify)
        """
        # Noise, leave this as 'none' for no noise.
        self.noisetype = 'None (Reconstruct)'
        # If you want to update the y variable when reconstructing
        self.UPDATE_INPUT = 0
        # Lambda for the reconstruction error of the updated and input images
        self.lambda_input = 1

        # Normalization type for each layer and the input image (first one)
        self.norm_types = ['Max', 'Max', 'None', 'None', 'None']
        # The size of the pooling regions.
        self.norm_sizes = [[2, 2], [2, 2], [2, 2], [2, 2], [2, 2]]
        # If you want to loop and unloop the pooling operation at each iteration
        self.SUBSAMPLING_UPDATES = 1

        """
        GUI Specific fields (not included so do not modify)
        """
        # The type of experiment you want to run (used by the gui)
        self.exptype = 'Train Layer By Layer'
        # The actual file to run the experiment at. Leave a space before first char
        self.expfile = 'train'
        # The location where the job was run from. Leave a space before first char.
        self.machine = 'laptop'
        # The dataset directory (where to load images from for training/reconstruction)
        self.datadirectory = './Datasets/Images/fruit_100_100/'
        # The model directory (used when loading a previous model)
        self.modeldirectory = './Results/train/city_100_100/color_filters_9_45/Run_0/epoch25_layer1'
        # Where to save the results
        self.savedirectory = './Results/+'

        self.pix_noise_rec_error = np.zeros((10, 10))
        self.pix_update_rec_error = np.zeros((10, 10))


