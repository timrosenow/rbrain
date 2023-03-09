"""
Neural Net segmentations including training and applying on novel data.
"""
import configparser
import keras
import keras.utils
import nibabel as nib
import numpy as np
import os.path
import pathlib
import rbrain.utils as rbutils
from scipy import ndimage


# Helper functions
def simple_iou(mask1, mask2):
    """
    Simple helper function to determine the IOU of two masks.

    Args:
        mask1 (np.ndarray): The first mask.
        mask2 (np.ndarray): The second mask.

    Returns:
        float: The IOU (between 0 and 1).
    """
    m1_area = np.count_nonzero(mask1)
    m2_area = np.count_nonzero(mask2)
    intersection = np.count_nonzero(
        np.logical_and(mask1, mask2)
    )
    return intersection / (m1_area + m2_area - intersection)


# Classes
class TFDataSequence(keras.utils.Sequence):
    """
    Helper class used to provide batches of tensorflow images to the neural network.

    Mostly boilerplate code, overrides the required virtual methods for keras/TF data sequences.

    Args:
        batch_size (int): Batch size for training / analysis.
        img_size ((int, int)): Image size tuple in pixels e.g. (192, 192).
        input_scans (np.ndarray): Image data: np.ndarray of size [192, 192, n] where n is number of images.
        input_masks (np.ndarray): Mask data: binary np.ndarray of size [192, 192, n] where n is number of images.

    Attributes:
        batch_size (int): Batch size for training / analysis.
        img_size ((int, int)): Image size tuple in pixels e.g. (192, 192).
        input_scans (np.ndarray): Image data: np.ndarray of size [192, 192, n] where n is number of images.
        input_masks (np.ndarray): Mask data: binary np.ndarray of size [192, 192, n] where n is number of images.
    """
    def __init__(self, batch_size, img_size, input_scans, input_masks):
        self.batch_size = batch_size
        self.img_size = img_size
        self.input_scans = input_scans
        self.input_masks = input_masks

    def __len__(self):
        return len(self.input_scans) // self.batch_size

    def __getitem__(self, idx):
        i = idx * self. batch_size
        return self.input_scans[i : i + self.batch_size], self.input_masks[i : i + self.batch_size]


class NNModelCreator:
    """
    A class to define and generate a neural network model for use in deep learning.

    This class is easily extensible to new neural net models: just create a new method create_[model_name] with
    your desired model structure that returns a tensorflow.keras.Model.
    """
    def __init__(self):
        pass

    # The following methods are just helper methods to make the actual definition method easier to read
    @staticmethod
    def _convolution_operation(entered_input, filters=64):
        conv1 = keras.layers.Conv2D(filters, kernel_size=(3, 3), padding="same")(entered_input)
        batch_norm1 = keras.layers.BatchNormalization()(conv1)
        act1 = keras.layers.ReLU()(batch_norm1)
        # Taking first input and implementing the second conv block
        conv2 = keras.layers.Conv2D(filters, kernel_size=(3, 3), padding="same")(act1)
        batch_norm2 = keras.layers.BatchNormalization()(conv2)
        act2 = keras.layers.ReLU()(batch_norm2)
        return act2

    @staticmethod
    def _encoder(entered_input, filters=64):
        # Collect the start and end of each sub-block for normal pass and skip connections
        enc1 = NNModelCreator._convolution_operation(entered_input, filters)
        MaxPool1 = keras.layers.MaxPooling2D(strides=(2, 2))(enc1)
        return enc1, MaxPool1

    @staticmethod
    def _decoder(entered_input, skip, filters=64):
        # Upsampling and concatenating the essential features
        Upsample = keras.layers.Conv2DTranspose(filters, (2, 2), strides=2, padding="same")(entered_input)
        Connect_Skip = keras.layers.Concatenate()([Upsample, skip])
        out = NNModelCreator._convolution_operation(Connect_Skip, filters)
        return out

    @staticmethod
    def create_unet(img_size):
        """
        Creates a tf/keras UNet model for image training/analysis.

        Args:
            img_size ((int, int)): Tuple containing the size of the images in pixels (e.g. 192, 192).

        Returns:
             keras.models.Model: The UNet keras model.
        """
        # Take the image size and shape
        input1 = keras.layers.Input((img_size[0], img_size[1], 1))

        # Construct the encoder blocks
        skip1, encoder_1 = NNModelCreator._encoder(input1, 64)
        skip2, encoder_2 = NNModelCreator._encoder(encoder_1, 64 * 2)
        skip3, encoder_3 = NNModelCreator._encoder(encoder_2, 64 * 4)
        skip4, encoder_4 = NNModelCreator._encoder(encoder_3, 64 * 8)

        # Preparing the next block
        conv_block = NNModelCreator._convolution_operation(encoder_4, 64 * 16)

        # Construct the decoder blocks
        decoder_1 = NNModelCreator._decoder(conv_block, skip4, 64 * 8)
        decoder_2 = NNModelCreator._decoder(decoder_1, skip3, 64 * 4)
        decoder_3 = NNModelCreator._decoder(decoder_2, skip2, 64 * 2)
        decoder_4 = NNModelCreator._decoder(decoder_3, skip1, 64)

        out = keras.layers.Conv2D(2, 1, padding="same", activation="sigmoid")(decoder_4)

        model = keras.models.Model(input1, out)
        model.compile(optimizer="rmsprop", loss="sparse_categorical_crossentropy")
        return model


class NNManager:
    """
    Base class to manage Neural Networks for image rbrain-segmenter. Sets up all the base file and neural network methods,
    while leaving the actual work to subclasses.

    Args:
        scan_dir (str): Directory in which your scan images reside.
        mask_dir (str): Directory in which your mask images reside, or will reside.
        model_dir (str): Directory where your model is saved. Created on demand if needed.
        config_file (str): Location of the config file, if overriding defaults. Can be None.

    Attributes:
        scan_dir (str): Directory in which your scan images reside.
        mask_dir (str): Directory in which your mask images reside, or will reside.
        model_dir (str): Directory where your model is saved. Created on demand if needed.
        model (keras.Models.model): The keras Model, defined and created at runtime.
        img_size ((int, int)): Image size fed into the neural network.
        random_seed (int): Random seed (for repeatability). Can be overwritten at runtime if desired.
    """
    def __init__(self, scan_dir, mask_dir, model_dir, config_file=None):
        self.scan_dir = scan_dir
        self.mask_dir = mask_dir
        self.model_dir = model_dir
        self._config = None
        self.model = None

        # Some parameters from the config file should be explicitly stored
        self.img_size = None
        self.random_seed = None

        # First read in the default config file, for sensible defaults
        self.load_config(rbutils.DEF_CONFIG_FILE)
        # Now read in the config file, if it exists
        if config_file is not None:
            self.load_config(config_file)

    def load_config(self, config_file):
        """
        Loads in a configuration file and sets class variables accordingly.

        Args:
            config_file (str): Configuration file to load.

        """
        if not os.path.exists(config_file):
            raise rbutils.ConfigFileNotFoundException()

        # If no config exists yet, read in the whole file and reference the Segmentation section
        if self._config is None:
            config_reader = configparser.ConfigParser()
            config_reader.read(config_file)
            self._config = config_reader['Segmentation']
        # Otherwise, update the existing config file
        else:
            self._config.parser.read(config_file)

        # Some parameters need explicit storage
        self.img_size = (self._config.getint("ImgSizeX"), self._config.getint("ImgSizeY"))
        self.random_seed = self._config.getint("RandomSeed")

    @staticmethod
    def get_nifti_files(img_path):
        """
        Returns a list of all nifti files in a directory, sorted by name. This is useful when loading images
        with their accompanying masks. If a single nifti-image is provided, instead of a directory, this returns a list
        with one element (the image provided).

        Args:
            img_path (str): The directory to be listed.

        Returns:
            [str]: A list of nifti files in this directory.
        """

        if rbutils.is_nifti(img_path):
            return [img_path]

        img_files = sorted(
            [
                os.path.join(img_path, filename) for filename in os.listdir(img_path) if rbutils.is_nifti(filename)
            ]
        )
        return img_files

    def load_images(self, img_path, img_type="float32", is_mask=False, transpose=True, is_dir=True):
        """
        Loads images and masks, and converts to a numpy.ndarray in a format ready for training - i.e.
        a large array with shape (img_size_i x img_size_j x N),
        where N is the number of slices per image x number of images.

        Args:
            img_path (str): Directory containing the images to be loaded.
            img_type (str): Image type, string from np.dtype. Use "uint16" for masks (as this is readable by nifti).
            is_mask (bool): If these are mask files. If so, this affects interpolation.
            transpose (bool): Include all three X/Y/Z views separately in the image slices, e.g. for 3D scans.
            is_dir (bool): If img_path is a directory, process all files. If it's a single file, process that only.
                Defaults to True (i.e. assumes a directory).

        Returns:
            np.ndarray: the image array.
        """
        # Set up the array to save images
        img_slices = np.ndarray((self.img_size[0], self.img_size[1], 0), dtype=img_type)

        # Use higher order interpolation for images, and nearest-neighbour if it is a mask
        if is_mask:
            interp_order = 0
        else:
            interp_order = 2

        # List all nifti files, or just get a single file if is_file is true
        img_files = []
        if is_dir:
            img_files = self.get_nifti_files(img_path)
        else:
            img_files = [img_path]

        if len(img_files) == 0:
            raise rbutils.InvalidDatasetException()

        for img_file in img_files:
            img_data = rbutils.read_nifti_file(img_file)
            # Resize the image first - zoom factor tells us the ratio to get the desired resolution
            zoom_factor = [self.img_size[0] / img_data.shape[0], self.img_size[1] / img_data.shape[1], 1]
            img_slices = np.append(img_slices,
                                   ndimage.zoom(img_data, zoom_factor, order=interp_order).astype(img_type), 2
                                   )

            # If we wish to use all three planes, also add the transposed data
            if transpose:
                img_data_y = img_data.transpose([0, 2, 1])
                zoom_factor = [self.img_size[0] / img_data_y.shape[0], self.img_size[1] / img_data_y.shape[1], 1]
                img_slices = np.append(img_slices,
                                       ndimage.zoom(img_data_y, zoom_factor, order=interp_order).astype(img_type), 2
                                       )
                img_data_z = img_data.transpose([2, 1, 0])
                zoom_factor = [self.img_size[0] / img_data_z.shape[0], self.img_size[1] / img_data_z.shape[1], 1]
                img_slices = np.append(img_slices,
                                       ndimage.zoom(img_data_z, zoom_factor, order=interp_order).astype(img_type), 2
                                       )

            # Finally, we need to transpose the images from image-notation (XYZ) to array notation (ZYX) for use in TF
        img_slices = np.transpose(img_slices, [2, 1, 0])
        return img_slices


class NNTrainer(NNManager):
    """
    Class to train neural networks from annotated image masks.

    Provide this class with two directories: the scan dir, and the annotation dir, which must have files in the same
    order (e.g. img1, img2, img3; and mask1, mask2, mask3 corresponding to three pairs of images and masks).

    Attributes:
        training_scan_data (np.ndarray): Training image numeric data, including augmentation, used for training the neural network.
        training_mask_data (np.ndarray): Training mask data, including augmentation, used for training the neural network.
        validation_scan_data (np.ndarray): Validation image data.
        validation_mask_data (np.ndarray): Validation mask data.
    """
    def __init__(self, scan_dir, mask_dir, model_dir, config_file=None):
        super().__init__(scan_dir, mask_dir, model_dir, config_file)
        self.training_scan_data = None
        self.training_mask_data = None
        self.validation_scan_data = None
        self.validation_mask_data = None

    @staticmethod
    def img_mask_random_rotation(img, mask, max_amt):
        """
        Given a 2D image and image mask, return both after a rotation by a random amount up to max_amt.

        Args:
            img (np.ndarray): The image to be rotated.
            mask (np.ndarray): The image mask to be rotated.
            max_amt (float): The maximum allowable rotation amount

        Returns:
            (np.ndarray, np.ndarray): The image and mask, after rotation
        """
        rot_amt = (np.random.random() - 0.5) * max_amt
        rot_img = ndimage.rotate(img, angle=rot_amt, reshape=False, order=2)
        rot_mask = ndimage.rotate(mask, angle=rot_amt, reshape=False, order=0)
        return rot_img, rot_mask

    @staticmethod
    def img_mask_random_zoom(img, mask, max_amt):
        """
        Given a 2D image and image mask, return both after zooming in or out by a random factor up to max_amt
        Args:
            img (np.ndarray): The image to be zoomed.
            mask (np.ndarray): The mask to be zoomed.
            max_amt (float): Maximum factor by which an image is zoomed.

        Returns:
            (np.ndarray, np.ndarray): The image and mask, after zooming.
        """
        zoom_amt = np.random.random() * max_amt
        img_newsize = (int(img.shape[0] * (1 - zoom_amt)), int(img.shape[1] * (1 - zoom_amt)))
        img_skipsize = ((img.shape[0] - img_newsize[0]) // 2, (img.shape[1] - img_newsize[1]) // 2)

        img_zoom = img[img_skipsize[0]:img_skipsize[0] + img_newsize[0],
                        img_skipsize[1]:img_skipsize[1] + img_newsize[1]]
        mask_zoom = mask[img_skipsize[0]:img_skipsize[0] + img_newsize[0],
                        img_skipsize[1]:img_skipsize[1] + img_newsize[1]]
        zoom_factor = [img.shape[0] / img_zoom.shape[0], img.shape[1] / img_zoom.shape[1]]

        img_zoom = ndimage.zoom(img_zoom, zoom_factor, order=2)
        mask_zoom = ndimage.zoom(mask_zoom, zoom_factor, order=0)

        return img_zoom, mask_zoom

    def setup_training_data(self):
        """
        Prepares the images and masks: converts from nifti, reshapes, normalises, splits into training
        and validation sets, then adds data augmentation to the training images.
        """
        scan_data = self.load_images(self.scan_dir,
                                     img_type="float32",
                                     is_mask=False,
                                     transpose=self._config.getboolean("Transpose3D")
                                     )

        mask_data = self.load_images(self.mask_dir,
                                     img_type="uint16",
                                     is_mask=True,
                                     transpose=self._config.getboolean("Transpose3D")
                                     )

        # Raise an invalid dataset exception if the scans and masks are not of equal length
        if scan_data.shape[0] != mask_data.shape[0]:
            raise rbutils.InvalidDatasetException()

        # Normalise the scan data, for training purposes.
        scan_data = (scan_data - scan_data.mean()) / scan_data.std()

        # Split into training and validation datasets
        num_validation = int(len(scan_data) * self._config.getfloat("ValSplit"))
        np.random.seed(self.random_seed)
        np.random.shuffle(scan_data)
        np.random.seed(self.random_seed)
        np.random.shuffle(mask_data)
        self.training_scan_data = scan_data[:-num_validation]
        self.validation_scan_data = scan_data[-num_validation:]
        self.training_mask_data = mask_data[:-num_validation]
        self.validation_mask_data = mask_data[-num_validation:]

        # Perform data augmentation on the training set, if desired.
        # This section is a bit spaghetti-like, but basically, loop over the whole dataset and
        # for each slice, append an augmented version of it to the original dataset, then repeat for every
        # augmentation loop (e.g. if NumAugs is 3, do it 3 times).
        # Note if num_augs is zero, then nothing will happen.
        num_scans = self.training_scan_data.shape[0]
        num_augs = self._config.getint("NumAugs")

        for i in range(num_augs):
            scan_aug = np.zeros(
                (num_scans, self.training_scan_data.shape[1], self.training_scan_data.shape[2]),
                dtype=self.training_scan_data.dtype)
            mask_aug = np.zeros(
                (num_scans, self.training_mask_data.shape[1], self.training_mask_data.shape[2]),
                dtype=self.training_mask_data.dtype)
            # Rotate by a random amount, then zoom by a random amount
            for img_num in range(num_scans):
                scan, mask = self.img_mask_random_rotation(
                    self.training_scan_data[img_num],
                    self.training_mask_data[img_num],
                    self._config.getfloat("AugMaxAngle")
                )
                scan, mask = self.img_mask_random_zoom(scan, mask, self._config.getfloat("AugMaxZoom"))
                scan_aug[img_num] = scan
                mask_aug[img_num] = mask
            self.training_scan_data = np.append(self.training_scan_data, scan_aug, axis=0)
            self.training_mask_data = np.append(self.training_mask_data, mask_aug, axis=0)

    def train_network(self):
        """
        Trains a neural network from the (already input and prepared) training and validation data.

        Note that this assumes that everything is already set up and ready to go - data is loaded and prepared, etc.
        Minimal error checking is performed, so please make sure everything is set up before running this. The model
        and checkpoints will be saved in the output directory provided, but no error checking is performed (i.e. it
        will overwrite existing file, etc).
        """

        # Raise an exception if the data has not been set up yet.
        if self.training_scan_data is None or self.training_mask_data is None:
            raise rbutils.DatasetNotLoadedException()

        # Create the model directory, if it doesn't exist
        pathlib.Path(self.model_dir).mkdir(parents=True, exist_ok=True)

        # Instantiate data sequence classes for training and validation data, for passing into the NN
        train_gen = TFDataSequence(
            self._config.getint("BatchSize"),
            self.img_size,
            self.training_scan_data,
            self.training_mask_data
        )
        val_gen = TFDataSequence(
            self._config.getint("BatchSize"),
            self.img_size,
            self.validation_scan_data,
            self.validation_mask_data
        )

        # Create the model and start training
        self.model = NNModelCreator.create_unet(self.img_size)
        callbacks = [
            keras.callbacks.ModelCheckpoint(f"{self.model_dir}/checkpoints/train_checkpoint.h5", save_best_only=True)
            ]
        self.model.fit(train_gen, epochs=self._config.getint("NumEpochs"), validation_data=val_gen, callbacks=callbacks)

        # Finally re-analyse the validation data and print the IOU
        val_preds = self.model.predict(val_gen)
        print("Basic IOU for validation set:",
              simple_iou(np.argmax(val_preds, axis=-1), val_gen.input_masks[0:len(val_preds), :, :])
              )

    def save_model(self):
        """
        Saves the model to a file (within self.model_dir) for later use.
        """
        if self.model is None:
            raise rbutils.NNModelNotFoundException()

        self.model.save(self.model_dir)


class NNAnalyser(NNManager):
    """
    Class to use a trained neural network to analyse a new set of images to product segmentation masks.

    Provide this class with two directories: the scan dir, which has your images for analysis, and the mask dir, which
    is where the masks will be stored (with the same filenames as the scans). Note that this class does some
    post-processing on the scans (dilates the mask by 1 voxel in 3D, and then selects the largest contiguous volume).
    """
    def __init__(self, scan_dir, mask_dir, model_dir, config_file=None):
        super().__init__(scan_dir, mask_dir, model_dir, config_file)
        self.load_model(model_dir)

    def load_model(self, model_dir):
        """
        Sets the model directory and loads the keras model.

        Args:
            model_dir (str): The new model directory.
        """
        self.model = keras.models.load_model(model_dir)
        self.model_dir = model_dir

    def analyse_scans(self):
        """
        Analyses all the scans in scan_dir and then saves the output masks in mask_dir. Includes post-processing,
        etc.
        """

        # Analyse each scan separately
        for scan_file in self.get_nifti_files(self.scan_dir):
            # We need the original scan size for later processing
            scan_nifti = nib.load(scan_file)
            scan_size = scan_nifti.shape
            scan_data = self.load_images(scan_file, "float32", is_mask=False, transpose=False, is_dir=False)
            # Normalise the image
            scan_data = (scan_data - scan_data.mean()) / scan_data.std()
            # Create some dummy mask data to feed to the generator - not sure if necessary or not.
            mask_data = np.zeros(scan_data.shape, dtype="uint16")

            # Create an image generator and feed it to the NN
            analysis_gen = TFDataSequence(
                1,  # Batch size of 1, otherwise the residual slices will be excluded
                self.img_size,
                scan_data,
                mask_data
            )
            predictions = self.model.predict(analysis_gen)
            mask_data = np.argmax(predictions, axis=-1).astype("uint16")

            # We need to rearrange the axes back to their original shape (XYZ) and un-resize the image
            mask_data = np.transpose(mask_data, [2, 1, 0])
            zoom_factor = np.divide(scan_size, mask_data.shape)
            mask_data = ndimage.zoom(mask_data, zoom_factor, order=0)

            # Dilate by 1, then exclude all but the largest contiguous volume
            mask_data = ndimage.binary_dilation(mask_data).astype("uint16")
            mask_objects, num_objects = ndimage.label(mask_data)
            largest_object = 1 + np.argmax(
                [np.count_nonzero(mask_objects[mask_objects == x+1]) for x in range(num_objects)]
            )
            mask_data = (mask_objects == largest_object).astype("uint16")

            # Save the resulting mask, in nifti format. Use the original scan as a template e.g. for headers etc.
            mask_nifti = nib.Nifti1Image(mask_data, scan_nifti.affine, scan_nifti.header)
            mask_nifti.header.set_data_dtype("uint16")
            nib.save(mask_nifti, os.path.join(self.mask_dir, os.path.basename(scan_file)))


