# -*- coding: utf-8 -*-
import os, random, pdb
import matplotlib.pyplot as plt

from cntk import constant, use_default_device, cross_entropy_with_softmax, classification_error
from cntk import load_model, Trainer, UnitType
from cntk.io import MinibatchSource, ImageDeserializer, StreamDefs, StreamDef
import cntk.io.transforms as xforms
from cntk.layers import placeholder, GlobalAveragePooling, Dropout, Dense
from cntk.learners import momentum_sgd, learning_rate_schedule, momentum_schedule
from cntk.logging import log_number_of_parameters, ProgressPrinter, graph
from cntk.logging.graph import find_by_name
from cntk.ops import input_variable, combine
from cntk.ops.functions import CloneMethod

from helpers import *

random.seed(0)

#NOTE: the functionality in this file is adapted and extended from CNTK's transfer learning tutorial:
#  https://github.com/Microsoft/CNTK/wiki/Build-your-own-image-classifier-using-Transfer-Learning


################################
# helper functions - cntk
################################
def printDeviceType(boGpuRequired = False):
    if use_default_device().type() != 0:
        print("Using GPU for CNTK training/scoring.")
    else:
        print("WARNING: using CPU for CNTK training/scoring.")
        if boGpuRequired:
            raise Exception("Cannot find GPU or GPU is already locked.")


# Creates a minibatch source for training or testing
def create_mb_source(map_file, image_width, image_height, num_channels, num_classes, boTrain):
    transforms = []
    if boTrain:
        # Scale to square-sized image. without this the cropping transform would chop the larger dimension of an
        # image to make it squared, and then take 0.9 crops from within the squared image.
        transforms += [xforms.scale(width=2*image_width, height=2*image_height, channels=num_channels,
                                    interpolations='linear', scale_mode='pad', pad_value=114)]
        transforms += [xforms.crop(crop_type='randomside', side_ratio=0.9, jitter_type='uniratio')]     # Randomly crop square area
    transforms += [xforms.scale(width=image_width, height=image_height, channels=num_channels,          # Scale down and pad
                                interpolations='linear', scale_mode='pad', pad_value=114)]
    if boTrain:
        transforms += [xforms.color(brightness_radius=0.2, contrast_radius=0.2, saturation_radius=0.2)]

    return MinibatchSource(ImageDeserializer(map_file, StreamDefs(
            features  = StreamDef(field='image', transforms=transforms),
            labels    = StreamDef(field='label', shape=num_classes))),
            randomize = boTrain,
            multithreaded_deserializer=True)


# Creates the network model for transfer learning
def create_model(base_model_file, input_features, num_classes,  dropout_rate = 0.5, freeze_weights = False):
    # Load the pretrained classification net and find nodes
    base_model   = load_model(base_model_file)
    feature_node = find_by_name(base_model, 'features')
    beforePooling_node = find_by_name(base_model, "z.x.x.r")
    #graph.plot(base_model, filename="base_model.pdf") # Write graph visualization

    # Clone model until right before the pooling layer, ie. until including z.x.x.r
    modelCloned = combine([beforePooling_node.owner]).clone(
        CloneMethod.freeze if freeze_weights else CloneMethod.clone,
        {feature_node: placeholder(name='features')})

    # Center the input around zero and set model input.
    # Do this early, to avoid CNTK bug with wrongly estimated layer shapes
    feat_norm = input_features - constant(114)
    model = modelCloned(feat_norm)

    # Pool over all spatial dimensions and add dropout layer
    avgPool = GlobalAveragePooling(name = "poolingLayer")(model)
    if dropout_rate > 0:
        avgPoolDrop = Dropout(dropout_rate)(avgPool)
    else:
        avgPoolDrop = avgPool

    # Add new dense layer for class prediction
    finalModel = Dense(num_classes, activation=None, name="prediction") (avgPoolDrop)
    return finalModel


# Trains a transfer learning model
def train_model(base_model_file, train_map_file, test_map_file, input_resolution,
                num_epochs, mb_size, max_train_images, lr_per_mb, momentum_per_mb, l2_reg_weight,
                dropout_rate, freeze_weights, num_channels = 3):

    #init
    image_width  = input_resolution
    image_height = input_resolution
    epoch_size_test  = len(readTable(test_map_file))
    epoch_size_train = len(readTable(train_map_file))
    epoch_size_train = min(epoch_size_train, max_train_images)
    num_classes = max(ToIntegers(getColumn(readTable(train_map_file), 1))) + 1

    # Create the minibatch source
    minibatch_source_train = create_mb_source(train_map_file, image_width, image_height, num_channels, num_classes, True)
    minibatch_source_test  = create_mb_source(test_map_file,  image_width, image_height, num_channels, num_classes, False)

    # Define mapping from reader streams to network inputs
    label_input = input_variable(num_classes)
    image_input = input_variable((num_channels, image_height, image_width), name = "input")
    input_map = {
        image_input: minibatch_source_train['features'],
        label_input: minibatch_source_train['labels']
    }

    # Instantiate the transfer learning model and loss function
    cntkModel = create_model(base_model_file, image_input, num_classes, dropout_rate, freeze_weights)
    ce = cross_entropy_with_softmax(cntkModel, label_input)
    pe = classification_error(cntkModel, label_input)

    # Instantiate the trainer object
    lr_schedule = learning_rate_schedule(lr_per_mb, unit=UnitType.minibatch)
    mm_schedule = momentum_schedule(momentum_per_mb)
    learner = momentum_sgd(cntkModel.parameters, lr_schedule, mm_schedule, l2_regularization_weight=l2_reg_weight)
    progress_writers = [ProgressPrinter(tag='Training', num_epochs=num_epochs)]
    trainer = Trainer(cntkModel, (ce, pe), learner, progress_writers)

    # Run training epochs
    print("Training transfer learning model for {0} epochs (epoch_size_train = {1}).".format(num_epochs, epoch_size_train))
    errsTest  = []
    errsTrain = []
    log_number_of_parameters(cntkModel)

    for epoch in range(num_epochs):
        # Train model
        err_numer = 0
        sample_counts = 0
        while sample_counts < epoch_size_train:  # Loop over minibatches in the epoch
            sample_count = min(mb_size, epoch_size_train - sample_counts)
            data = minibatch_source_train.next_minibatch(sample_count, input_map = input_map)
            trainer.train_minibatch(data)        # Update model with it
            sample_counts += sample_count        # Count samples processed so far
            err_numer += trainer.previous_minibatch_evaluation_average * sample_count

            if sample_counts % (100 * mb_size) == 0:
                print ("Training: processed {0} samples".format(sample_counts))

            # Visualize training images
            # img_data = data[image_input].asarray()
            # for i in range(len(img_data)):
            #     debugImg = img_data[i].squeeze().swapaxes(0, 1).swapaxes(1, 2) / 255.0
            #     imshow(debugImg)

        # Compute accuracy on training and test sets
        errsTrain.append(err_numer / float(sample_counts))
        trainer.summarize_training_progress()
        errsTest.append(cntkComputeTestError(trainer, minibatch_source_test, mb_size, epoch_size_test, input_map))
        trainer.summarize_test_progress()

        # Plot training progress
        plt.plot(errsTrain, 'b-', errsTest, 'g-')
        plt.xlabel('Epoch number')
        plt.ylabel('Error')
        plt.title('Training error (blue), test error (green)')
        plt.draw()
    return cntkModel


# Evaluate model accuracy
def cntkComputeTestError(trainer, minibatch_source_test, mb_size, epoch_size, input_map):
    acc_numer = 0
    sample_counts = 0
    while sample_counts < epoch_size:  # Loop over minibatches in the epoch
        sample_count = min(mb_size, epoch_size - sample_counts)
        data = minibatch_source_test.next_minibatch(sample_count, input_map = input_map)
        acc_numer     += trainer.test_minibatch(data) * sample_count
        sample_counts += sample_count
    return acc_numer / float(sample_counts)


def runCntkModel(model, map_file, node_name = [], mb_size = 1):
    # Get minibatch source
    num_classes = model.shape[0]
    (image_width, image_height) = find_by_name(model, "input").shape[1:]
    minibatch_source = create_mb_source(map_file, image_width, image_height, 3, num_classes, False)
    features_si = minibatch_source['features']

    # Set output node
    if node_name == []:
        output_node = model
    else:
        node_in_graph = model.find_by_name(node_name)
        output_node   = combine([node_in_graph.owner])

    # Evaluate DNN for all images
    data = []
    sample_counts = 0
    imgPaths = getColumn(readTable(map_file), 0)
    while sample_counts < len(imgPaths):
        sample_count = min(mb_size, len(imgPaths) - sample_counts)
        mb = minibatch_source.next_minibatch(sample_count)
        output = output_node.eval(mb[features_si])
        data += [o.flatten() for o in output]
        sample_counts += sample_count
        if sample_counts % 100 < mb_size:
            print("Evaluating DNN (output dimension = {}) for image {} of {}: {}".format(len(data[-1]), sample_counts,
                                                                                         len(imgPaths),
                                                                                         imgPaths[sample_counts - 1]))
    data = [[imgPath, feat] for imgPath, feat in zip(imgPaths, data)]
    return data


def featurizeImages(model, imgFilenamesPath, imgDir, map_file, node_name = [], mb_size = 1):
    # Get image paths
    imgFilenames = loadFromPickle(imgFilenamesPath)
    imgLabelMap  = getImgLabelMap(imgFilenames, imgDir)
    imgLabelMap  = zip(getColumn(imgLabelMap,0), [0] * len(imgLabelMap))  # Set labels to all 0's since not used anyway

    # Run CNTK model for each image
    # Note: CNTK's MinibatchSource/ImageReader currently does not support in-memory
    #       calls, hence need to save input map to disk.
    writeTable(map_file, imgLabelMap)
    cntkOutput = runCntkModel(model, map_file, node_name)

    # Store all features in a dictionary
    features = dict()
    for imgPath, feat in cntkOutput:
        imgFilename = os.path.basename(imgPath)
        imgSubdir   = os.path.split(os.path.split(imgPath)[0])[1]
        key = imgSubdir + "/" + imgFilename
        features[key] = feat
    return features



