# Malaria detection using deep learning

This Tensorflow code trains and tests a Malaria parasite (Plasmodioum) detector using Convolutional Neural Networks.
You should be able to get accuracy of up to 98% with a few hours of training.


# Download and Process images
1 - Get the dataset from intestinalparasites-images.zip and extract
2 - Run convertimage.py to extract images of malaria and normal classes and resize them

# Train on Malaria dataset
3 - Run vgg16_trainable.py to train. Set the trainging and validation paths in the file before doing so.
4 - Note the numpy file saved by the previous script

# Test Malaria parasite detector
5 - Run test_fcn32.py
6 - The image with segmentation will be outputted 
