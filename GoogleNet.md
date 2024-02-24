# Train a Deep Learning Model using GoogleNet

Script for creating and training a deep learning network with the following properties:

```matlab:Code(Display)
Number of layers: 144
Number of connections: 170
Training setup file: C:\Users\melahi\Documents\MATLAB\Examples\R2021b\nnet\TransferLearningUsingGoogLeNetExample\params_2022_04_24__22_42_52.mat
```

Run this script to create the network layers, import training and validation data, and train the network. The network layers are stored in the workspace variable `lgraph`. The trained network is stored in the workspace variable `net`.

# Load Initial Parameters

Load parameters for network initialization. For transfer learning, the network initialization parameters are the parameters of the initial pretrained network.

```matlab:Code
trainingSetup = load("C:\Users\melahi\Documents\MATLAB\Examples\R2021b\nnet" + ...
    "\TransferLearningUsingGoogLeNetExample\CSV\params_2022_04_24__22_42_52.mat");
```

# Import Data

Import training and validation data.

```matlab:Code
imdsTrain = imageDatastore("C:\Users\melahi\Documents\MATLAB\Examples\R2021b\nnet" + ...
    "\TransferLearningUsingGoogLeNetExample\CSV","IncludeSubfolders",true,"LabelSource","foldernames");
[imdsTrain, imdsValidation] = splitEachLabel(imdsTrain,0.7);

% Resize the images to match the network input layer.
augimdsTrain = augmentedImageDatastore([224 224 3],imdsTrain);
augimdsValidation = augmentedImageDatastore([224 224 3],imdsValidation);

figure
numImages = length(imdsTrain.Files);
perm = randperm(numImages,32);
for i = 1:32
    subplot(8,4,i);
    imshow(imdsTrain.Files{perm(i)});
    drawnow
end
```

![figure_0.png](GoogleNet_images/figure_0.png)

# Set Training Options

Specify options to use when training.

```matlab:Code
opts = trainingOptions("sgdm",...
    "ExecutionEnvironment","auto",...
    "InitialLearnRate",0.0001,...
    "MaxEpochs",8,...
    "MiniBatchSize",4,...
    "Shuffle","every-epoch",...
    "ValidationFrequency",5,...
    "Plots","training-progress",...
    "ValidationData",augimdsValidation);
```

# Create Layer Graph

Create the layer graph variable to contain the network layers.

```matlab:Code
lgraph = layerGraph();
```

# Add Layer Branches

Add the branches of the network to the layer graph. Each branch is a linear array of layers.

```matlab:Code
tempLayers = [
    imageInputLayer([224 224 3],"Name","data","Mean",trainingSetup.data.Mean)
    convolution2dLayer([7 7],64,"Name","conv1-7x7_s2","BiasLearnRateFactor",2,"Padding",[3 3 3 3],"Stride",[2 2],"Bias",trainingSetup.conv1_7x7_s2.Bias,"Weights",trainingSetup.conv1_7x7_s2.Weights)
    reluLayer("Name","conv1-relu_7x7")
    maxPooling2dLayer([3 3],"Name","pool1-3x3_s2","Padding",[0 1 0 1],"Stride",[2 2])
    crossChannelNormalizationLayer(5,"Name","pool1-norm1","K",1)
    convolution2dLayer([1 1],64,"Name","conv2-3x3_reduce","BiasLearnRateFactor",2,"Bias",trainingSetup.conv2_3x3_reduce.Bias,"Weights",trainingSetup.conv2_3x3_reduce.Weights)
    reluLayer("Name","conv2-relu_3x3_reduce")
    convolution2dLayer([3 3],192,"Name","conv2-3x3","BiasLearnRateFactor",2,"Padding",[1 1 1 1],"Bias",trainingSetup.conv2_3x3.Bias,"Weights",trainingSetup.conv2_3x3.Weights)
    reluLayer("Name","conv2-relu_3x3")
    crossChannelNormalizationLayer(5,"Name","conv2-norm2","K",1)
    maxPooling2dLayer([3 3],"Name","pool2-3x3_s2","Padding",[0 1 0 1],"Stride",[2 2])];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],16,"Name","inception_3a-5x5_reduce","BiasLearnRateFactor",2,"Bias",trainingSetup.inception_3a_5x5_reduce.Bias,"Weights",trainingSetup.inception_3a_5x5_reduce.Weights)
    reluLayer("Name","inception_3a-relu_5x5_reduce")
    convolution2dLayer([5 5],32,"Name","inception_3a-5x5","BiasLearnRateFactor",2,"Padding",[2 2 2 2],"Bias",trainingSetup.inception_3a_5x5.Bias,"Weights",trainingSetup.inception_3a_5x5.Weights)
    reluLayer("Name","inception_3a-relu_5x5")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Name","inception_3a-pool","Padding",[1 1 1 1])
    convolution2dLayer([1 1],32,"Name","inception_3a-pool_proj","BiasLearnRateFactor",2,"Bias",trainingSetup.inception_3a_pool_proj.Bias,"Weights",trainingSetup.inception_3a_pool_proj.Weights)
    reluLayer("Name","inception_3a-relu_pool_proj")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],64,"Name","inception_3a-1x1","BiasLearnRateFactor",2,"Bias",trainingSetup.inception_3a_1x1.Bias,"Weights",trainingSetup.inception_3a_1x1.Weights)
    reluLayer("Name","inception_3a-relu_1x1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],96,"Name","inception_3a-3x3_reduce","BiasLearnRateFactor",2,"Bias",trainingSetup.inception_3a_3x3_reduce.Bias,"Weights",trainingSetup.inception_3a_3x3_reduce.Weights)
    reluLayer("Name","inception_3a-relu_3x3_reduce")
    convolution2dLayer([3 3],128,"Name","inception_3a-3x3","BiasLearnRateFactor",2,"Padding",[1 1 1 1],"Bias",trainingSetup.inception_3a_3x3.Bias,"Weights",trainingSetup.inception_3a_3x3.Weights)
    reluLayer("Name","inception_3a-relu_3x3")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = depthConcatenationLayer(4,"Name","inception_3a-output");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Name","inception_3b-pool","Padding",[1 1 1 1])
    convolution2dLayer([1 1],64,"Name","inception_3b-pool_proj","BiasLearnRateFactor",2,"Bias",trainingSetup.inception_3b_pool_proj.Bias,"Weights",trainingSetup.inception_3b_pool_proj.Weights)
    reluLayer("Name","inception_3b-relu_pool_proj")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],128,"Name","inception_3b-3x3_reduce","BiasLearnRateFactor",2,"Bias",trainingSetup.inception_3b_3x3_reduce.Bias,"Weights",trainingSetup.inception_3b_3x3_reduce.Weights)
    reluLayer("Name","inception_3b-relu_3x3_reduce")
    convolution2dLayer([3 3],192,"Name","inception_3b-3x3","BiasLearnRateFactor",2,"Padding",[1 1 1 1],"Bias",trainingSetup.inception_3b_3x3.Bias,"Weights",trainingSetup.inception_3b_3x3.Weights)
    reluLayer("Name","inception_3b-relu_3x3")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],128,"Name","inception_3b-1x1","BiasLearnRateFactor",2,"Bias",trainingSetup.inception_3b_1x1.Bias,"Weights",trainingSetup.inception_3b_1x1.Weights)
    reluLayer("Name","inception_3b-relu_1x1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],32,"Name","inception_3b-5x5_reduce","BiasLearnRateFactor",2,"Bias",trainingSetup.inception_3b_5x5_reduce.Bias,"Weights",trainingSetup.inception_3b_5x5_reduce.Weights)
    reluLayer("Name","inception_3b-relu_5x5_reduce")
    convolution2dLayer([5 5],96,"Name","inception_3b-5x5","BiasLearnRateFactor",2,"Padding",[2 2 2 2],"Bias",trainingSetup.inception_3b_5x5.Bias,"Weights",trainingSetup.inception_3b_5x5.Weights)
    reluLayer("Name","inception_3b-relu_5x5")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    depthConcatenationLayer(4,"Name","inception_3b-output")
    maxPooling2dLayer([3 3],"Name","pool3-3x3_s2","Padding",[0 1 0 1],"Stride",[2 2])];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Name","inception_4a-pool","Padding",[1 1 1 1])
    convolution2dLayer([1 1],64,"Name","inception_4a-pool_proj","BiasLearnRateFactor",2,"Bias",trainingSetup.inception_4a_pool_proj.Bias,"Weights",trainingSetup.inception_4a_pool_proj.Weights)
    reluLayer("Name","inception_4a-relu_pool_proj")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],16,"Name","inception_4a-5x5_reduce","BiasLearnRateFactor",2,"Bias",trainingSetup.inception_4a_5x5_reduce.Bias,"Weights",trainingSetup.inception_4a_5x5_reduce.Weights)
    reluLayer("Name","inception_4a-relu_5x5_reduce")
    convolution2dLayer([5 5],48,"Name","inception_4a-5x5","BiasLearnRateFactor",2,"Padding",[2 2 2 2],"Bias",trainingSetup.inception_4a_5x5.Bias,"Weights",trainingSetup.inception_4a_5x5.Weights)
    reluLayer("Name","inception_4a-relu_5x5")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],96,"Name","inception_4a-3x3_reduce","BiasLearnRateFactor",2,"Bias",trainingSetup.inception_4a_3x3_reduce.Bias,"Weights",trainingSetup.inception_4a_3x3_reduce.Weights)
    reluLayer("Name","inception_4a-relu_3x3_reduce")
    convolution2dLayer([3 3],208,"Name","inception_4a-3x3","BiasLearnRateFactor",2,"Padding",[1 1 1 1],"Bias",trainingSetup.inception_4a_3x3.Bias,"Weights",trainingSetup.inception_4a_3x3.Weights)
    reluLayer("Name","inception_4a-relu_3x3")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],192,"Name","inception_4a-1x1","BiasLearnRateFactor",2,"Bias",trainingSetup.inception_4a_1x1.Bias,"Weights",trainingSetup.inception_4a_1x1.Weights)
    reluLayer("Name","inception_4a-relu_1x1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = depthConcatenationLayer(4,"Name","inception_4a-output");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],160,"Name","inception_4b-1x1","BiasLearnRateFactor",2,"Bias",trainingSetup.inception_4b_1x1.Bias,"Weights",trainingSetup.inception_4b_1x1.Weights)
    reluLayer("Name","inception_4b-relu_1x1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],112,"Name","inception_4b-3x3_reduce","BiasLearnRateFactor",2,"Bias",trainingSetup.inception_4b_3x3_reduce.Bias,"Weights",trainingSetup.inception_4b_3x3_reduce.Weights)
    reluLayer("Name","inception_4b-relu_3x3_reduce")
    convolution2dLayer([3 3],224,"Name","inception_4b-3x3","BiasLearnRateFactor",2,"Padding",[1 1 1 1],"Bias",trainingSetup.inception_4b_3x3.Bias,"Weights",trainingSetup.inception_4b_3x3.Weights)
    reluLayer("Name","inception_4b-relu_3x3")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],24,"Name","inception_4b-5x5_reduce","BiasLearnRateFactor",2,"Bias",trainingSetup.inception_4b_5x5_reduce.Bias,"Weights",trainingSetup.inception_4b_5x5_reduce.Weights)
    reluLayer("Name","inception_4b-relu_5x5_reduce")
    convolution2dLayer([5 5],64,"Name","inception_4b-5x5","BiasLearnRateFactor",2,"Padding",[2 2 2 2],"Bias",trainingSetup.inception_4b_5x5.Bias,"Weights",trainingSetup.inception_4b_5x5.Weights)
    reluLayer("Name","inception_4b-relu_5x5")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Name","inception_4b-pool","Padding",[1 1 1 1])
    convolution2dLayer([1 1],64,"Name","inception_4b-pool_proj","BiasLearnRateFactor",2,"Bias",trainingSetup.inception_4b_pool_proj.Bias,"Weights",trainingSetup.inception_4b_pool_proj.Weights)
    reluLayer("Name","inception_4b-relu_pool_proj")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = depthConcatenationLayer(4,"Name","inception_4b-output");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],128,"Name","inception_4c-3x3_reduce","BiasLearnRateFactor",2,"Bias",trainingSetup.inception_4c_3x3_reduce.Bias,"Weights",trainingSetup.inception_4c_3x3_reduce.Weights)
    reluLayer("Name","inception_4c-relu_3x3_reduce")
    convolution2dLayer([3 3],256,"Name","inception_4c-3x3","BiasLearnRateFactor",2,"Padding",[1 1 1 1],"Bias",trainingSetup.inception_4c_3x3.Bias,"Weights",trainingSetup.inception_4c_3x3.Weights)
    reluLayer("Name","inception_4c-relu_3x3")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Name","inception_4c-pool","Padding",[1 1 1 1])
    convolution2dLayer([1 1],64,"Name","inception_4c-pool_proj","BiasLearnRateFactor",2,"Bias",trainingSetup.inception_4c_pool_proj.Bias,"Weights",trainingSetup.inception_4c_pool_proj.Weights)
    reluLayer("Name","inception_4c-relu_pool_proj")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],128,"Name","inception_4c-1x1","BiasLearnRateFactor",2,"Bias",trainingSetup.inception_4c_1x1.Bias,"Weights",trainingSetup.inception_4c_1x1.Weights)
    reluLayer("Name","inception_4c-relu_1x1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],24,"Name","inception_4c-5x5_reduce","BiasLearnRateFactor",2,"Bias",trainingSetup.inception_4c_5x5_reduce.Bias,"Weights",trainingSetup.inception_4c_5x5_reduce.Weights)
    reluLayer("Name","inception_4c-relu_5x5_reduce")
    convolution2dLayer([5 5],64,"Name","inception_4c-5x5","BiasLearnRateFactor",2,"Padding",[2 2 2 2],"Bias",trainingSetup.inception_4c_5x5.Bias,"Weights",trainingSetup.inception_4c_5x5.Weights)
    reluLayer("Name","inception_4c-relu_5x5")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = depthConcatenationLayer(4,"Name","inception_4c-output");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],144,"Name","inception_4d-3x3_reduce","BiasLearnRateFactor",2,"Bias",trainingSetup.inception_4d_3x3_reduce.Bias,"Weights",trainingSetup.inception_4d_3x3_reduce.Weights)
    reluLayer("Name","inception_4d-relu_3x3_reduce")
    convolution2dLayer([3 3],288,"Name","inception_4d-3x3","BiasLearnRateFactor",2,"Padding",[1 1 1 1],"Bias",trainingSetup.inception_4d_3x3.Bias,"Weights",trainingSetup.inception_4d_3x3.Weights)
    reluLayer("Name","inception_4d-relu_3x3")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],112,"Name","inception_4d-1x1","BiasLearnRateFactor",2,"Bias",trainingSetup.inception_4d_1x1.Bias,"Weights",trainingSetup.inception_4d_1x1.Weights)
    reluLayer("Name","inception_4d-relu_1x1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],32,"Name","inception_4d-5x5_reduce","BiasLearnRateFactor",2,"Bias",trainingSetup.inception_4d_5x5_reduce.Bias,"Weights",trainingSetup.inception_4d_5x5_reduce.Weights)
    reluLayer("Name","inception_4d-relu_5x5_reduce")
    convolution2dLayer([5 5],64,"Name","inception_4d-5x5","BiasLearnRateFactor",2,"Padding",[2 2 2 2],"Bias",trainingSetup.inception_4d_5x5.Bias,"Weights",trainingSetup.inception_4d_5x5.Weights)
    reluLayer("Name","inception_4d-relu_5x5")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Name","inception_4d-pool","Padding",[1 1 1 1])
    convolution2dLayer([1 1],64,"Name","inception_4d-pool_proj","BiasLearnRateFactor",2,"Bias",trainingSetup.inception_4d_pool_proj.Bias,"Weights",trainingSetup.inception_4d_pool_proj.Weights)
    reluLayer("Name","inception_4d-relu_pool_proj")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = depthConcatenationLayer(4,"Name","inception_4d-output");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],160,"Name","inception_4e-3x3_reduce","BiasLearnRateFactor",2,"Bias",trainingSetup.inception_4e_3x3_reduce.Bias,"Weights",trainingSetup.inception_4e_3x3_reduce.Weights)
    reluLayer("Name","inception_4e-relu_3x3_reduce")
    convolution2dLayer([3 3],320,"Name","inception_4e-3x3","BiasLearnRateFactor",2,"Padding",[1 1 1 1],"Bias",trainingSetup.inception_4e_3x3.Bias,"Weights",trainingSetup.inception_4e_3x3.Weights)
    reluLayer("Name","inception_4e-relu_3x3")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],32,"Name","inception_4e-5x5_reduce","BiasLearnRateFactor",2,"Bias",trainingSetup.inception_4e_5x5_reduce.Bias,"Weights",trainingSetup.inception_4e_5x5_reduce.Weights)
    reluLayer("Name","inception_4e-relu_5x5_reduce")
    convolution2dLayer([5 5],128,"Name","inception_4e-5x5","BiasLearnRateFactor",2,"Padding",[2 2 2 2],"Bias",trainingSetup.inception_4e_5x5.Bias,"Weights",trainingSetup.inception_4e_5x5.Weights)
    reluLayer("Name","inception_4e-relu_5x5")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],256,"Name","inception_4e-1x1","BiasLearnRateFactor",2,"Bias",trainingSetup.inception_4e_1x1.Bias,"Weights",trainingSetup.inception_4e_1x1.Weights)
    reluLayer("Name","inception_4e-relu_1x1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Name","inception_4e-pool","Padding",[1 1 1 1])
    convolution2dLayer([1 1],128,"Name","inception_4e-pool_proj","BiasLearnRateFactor",2,"Bias",trainingSetup.inception_4e_pool_proj.Bias,"Weights",trainingSetup.inception_4e_pool_proj.Weights)
    reluLayer("Name","inception_4e-relu_pool_proj")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    depthConcatenationLayer(4,"Name","inception_4e-output")
    maxPooling2dLayer([3 3],"Name","pool4-3x3_s2","Padding",[0 1 0 1],"Stride",[2 2])];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],160,"Name","inception_5a-3x3_reduce","BiasLearnRateFactor",2,"Bias",trainingSetup.inception_5a_3x3_reduce.Bias,"Weights",trainingSetup.inception_5a_3x3_reduce.Weights)
    reluLayer("Name","inception_5a-relu_3x3_reduce")
    convolution2dLayer([3 3],320,"Name","inception_5a-3x3","BiasLearnRateFactor",2,"Padding",[1 1 1 1],"Bias",trainingSetup.inception_5a_3x3.Bias,"Weights",trainingSetup.inception_5a_3x3.Weights)
    reluLayer("Name","inception_5a-relu_3x3")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Name","inception_5a-pool","Padding",[1 1 1 1])
    convolution2dLayer([1 1],128,"Name","inception_5a-pool_proj","BiasLearnRateFactor",2,"Bias",trainingSetup.inception_5a_pool_proj.Bias,"Weights",trainingSetup.inception_5a_pool_proj.Weights)
    reluLayer("Name","inception_5a-relu_pool_proj")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],32,"Name","inception_5a-5x5_reduce","BiasLearnRateFactor",2,"Bias",trainingSetup.inception_5a_5x5_reduce.Bias,"Weights",trainingSetup.inception_5a_5x5_reduce.Weights)
    reluLayer("Name","inception_5a-relu_5x5_reduce")
    convolution2dLayer([5 5],128,"Name","inception_5a-5x5","BiasLearnRateFactor",2,"Padding",[2 2 2 2],"Bias",trainingSetup.inception_5a_5x5.Bias,"Weights",trainingSetup.inception_5a_5x5.Weights)
    reluLayer("Name","inception_5a-relu_5x5")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],256,"Name","inception_5a-1x1","BiasLearnRateFactor",2,"Bias",trainingSetup.inception_5a_1x1.Bias,"Weights",trainingSetup.inception_5a_1x1.Weights)
    reluLayer("Name","inception_5a-relu_1x1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = depthConcatenationLayer(4,"Name","inception_5a-output");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],192,"Name","inception_5b-3x3_reduce","BiasLearnRateFactor",2,"Bias",trainingSetup.inception_5b_3x3_reduce.Bias,"Weights",trainingSetup.inception_5b_3x3_reduce.Weights)
    reluLayer("Name","inception_5b-relu_3x3_reduce")
    convolution2dLayer([3 3],384,"Name","inception_5b-3x3","BiasLearnRateFactor",2,"Padding",[1 1 1 1],"Bias",trainingSetup.inception_5b_3x3.Bias,"Weights",trainingSetup.inception_5b_3x3.Weights)
    reluLayer("Name","inception_5b-relu_3x3")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],384,"Name","inception_5b-1x1","BiasLearnRateFactor",2,"Bias",trainingSetup.inception_5b_1x1.Bias,"Weights",trainingSetup.inception_5b_1x1.Weights)
    reluLayer("Name","inception_5b-relu_1x1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Name","inception_5b-pool","Padding",[1 1 1 1])
    convolution2dLayer([1 1],128,"Name","inception_5b-pool_proj","BiasLearnRateFactor",2,"Bias",trainingSetup.inception_5b_pool_proj.Bias,"Weights",trainingSetup.inception_5b_pool_proj.Weights)
    reluLayer("Name","inception_5b-relu_pool_proj")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],48,"Name","inception_5b-5x5_reduce","BiasLearnRateFactor",2,"Bias",trainingSetup.inception_5b_5x5_reduce.Bias,"Weights",trainingSetup.inception_5b_5x5_reduce.Weights)
    reluLayer("Name","inception_5b-relu_5x5_reduce")
    convolution2dLayer([5 5],128,"Name","inception_5b-5x5","BiasLearnRateFactor",2,"Padding",[2 2 2 2],"Bias",trainingSetup.inception_5b_5x5.Bias,"Weights",trainingSetup.inception_5b_5x5.Weights)
    reluLayer("Name","inception_5b-relu_5x5")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    depthConcatenationLayer(4,"Name","inception_5b-output")
    globalAveragePooling2dLayer("Name","pool5-7x7_s1")
    dropoutLayer(0.4,"Name","pool5-drop_7x7_s1")
    fullyConnectedLayer(6,"Name","fc","BiasLearnRateFactor",10,"WeightLearnRateFactor",10)
    softmaxLayer("Name","prob")
    classificationLayer("Name","classoutput")];
lgraph = addLayers(lgraph,tempLayers);

% clean up helper variable
clear tempLayers;
```

# Connect Layer Branches

Connect all the branches of the network to create the network graph.

```matlab:Code
lgraph = connectLayers(lgraph,"pool2-3x3_s2","inception_3a-5x5_reduce");
lgraph = connectLayers(lgraph,"pool2-3x3_s2","inception_3a-pool");
lgraph = connectLayers(lgraph,"pool2-3x3_s2","inception_3a-1x1");
lgraph = connectLayers(lgraph,"pool2-3x3_s2","inception_3a-3x3_reduce");
lgraph = connectLayers(lgraph,"inception_3a-relu_pool_proj","inception_3a-output/in4");
lgraph = connectLayers(lgraph,"inception_3a-relu_1x1","inception_3a-output/in1");
lgraph = connectLayers(lgraph,"inception_3a-relu_3x3","inception_3a-output/in2");
lgraph = connectLayers(lgraph,"inception_3a-relu_5x5","inception_3a-output/in3");
lgraph = connectLayers(lgraph,"inception_3a-output","inception_3b-pool");
lgraph = connectLayers(lgraph,"inception_3a-output","inception_3b-3x3_reduce");
lgraph = connectLayers(lgraph,"inception_3a-output","inception_3b-1x1");
lgraph = connectLayers(lgraph,"inception_3a-output","inception_3b-5x5_reduce");
lgraph = connectLayers(lgraph,"inception_3b-relu_pool_proj","inception_3b-output/in4");
lgraph = connectLayers(lgraph,"inception_3b-relu_3x3","inception_3b-output/in2");
lgraph = connectLayers(lgraph,"inception_3b-relu_1x1","inception_3b-output/in1");
lgraph = connectLayers(lgraph,"inception_3b-relu_5x5","inception_3b-output/in3");
lgraph = connectLayers(lgraph,"pool3-3x3_s2","inception_4a-pool");
lgraph = connectLayers(lgraph,"pool3-3x3_s2","inception_4a-5x5_reduce");
lgraph = connectLayers(lgraph,"pool3-3x3_s2","inception_4a-3x3_reduce");
lgraph = connectLayers(lgraph,"pool3-3x3_s2","inception_4a-1x1");
lgraph = connectLayers(lgraph,"inception_4a-relu_pool_proj","inception_4a-output/in4");
lgraph = connectLayers(lgraph,"inception_4a-relu_3x3","inception_4a-output/in2");
lgraph = connectLayers(lgraph,"inception_4a-relu_1x1","inception_4a-output/in1");
lgraph = connectLayers(lgraph,"inception_4a-relu_5x5","inception_4a-output/in3");
lgraph = connectLayers(lgraph,"inception_4a-output","inception_4b-1x1");
lgraph = connectLayers(lgraph,"inception_4a-output","inception_4b-3x3_reduce");
lgraph = connectLayers(lgraph,"inception_4a-output","inception_4b-5x5_reduce");
lgraph = connectLayers(lgraph,"inception_4a-output","inception_4b-pool");
lgraph = connectLayers(lgraph,"inception_4b-relu_pool_proj","inception_4b-output/in4");
lgraph = connectLayers(lgraph,"inception_4b-relu_1x1","inception_4b-output/in1");
lgraph = connectLayers(lgraph,"inception_4b-relu_5x5","inception_4b-output/in3");
lgraph = connectLayers(lgraph,"inception_4b-relu_3x3","inception_4b-output/in2");
lgraph = connectLayers(lgraph,"inception_4b-output","inception_4c-3x3_reduce");
lgraph = connectLayers(lgraph,"inception_4b-output","inception_4c-pool");
lgraph = connectLayers(lgraph,"inception_4b-output","inception_4c-1x1");
lgraph = connectLayers(lgraph,"inception_4b-output","inception_4c-5x5_reduce");
lgraph = connectLayers(lgraph,"inception_4c-relu_1x1","inception_4c-output/in1");
lgraph = connectLayers(lgraph,"inception_4c-relu_5x5","inception_4c-output/in3");
lgraph = connectLayers(lgraph,"inception_4c-relu_pool_proj","inception_4c-output/in4");
lgraph = connectLayers(lgraph,"inception_4c-relu_3x3","inception_4c-output/in2");
lgraph = connectLayers(lgraph,"inception_4c-output","inception_4d-3x3_reduce");
lgraph = connectLayers(lgraph,"inception_4c-output","inception_4d-1x1");
lgraph = connectLayers(lgraph,"inception_4c-output","inception_4d-5x5_reduce");
lgraph = connectLayers(lgraph,"inception_4c-output","inception_4d-pool");
lgraph = connectLayers(lgraph,"inception_4d-relu_1x1","inception_4d-output/in1");
lgraph = connectLayers(lgraph,"inception_4d-relu_pool_proj","inception_4d-output/in4");
lgraph = connectLayers(lgraph,"inception_4d-relu_5x5","inception_4d-output/in3");
lgraph = connectLayers(lgraph,"inception_4d-relu_3x3","inception_4d-output/in2");
lgraph = connectLayers(lgraph,"inception_4d-output","inception_4e-3x3_reduce");
lgraph = connectLayers(lgraph,"inception_4d-output","inception_4e-5x5_reduce");
lgraph = connectLayers(lgraph,"inception_4d-output","inception_4e-1x1");
lgraph = connectLayers(lgraph,"inception_4d-output","inception_4e-pool");
lgraph = connectLayers(lgraph,"inception_4e-relu_pool_proj","inception_4e-output/in4");
lgraph = connectLayers(lgraph,"inception_4e-relu_1x1","inception_4e-output/in1");
lgraph = connectLayers(lgraph,"inception_4e-relu_3x3","inception_4e-output/in2");
lgraph = connectLayers(lgraph,"inception_4e-relu_5x5","inception_4e-output/in3");
lgraph = connectLayers(lgraph,"pool4-3x3_s2","inception_5a-3x3_reduce");
lgraph = connectLayers(lgraph,"pool4-3x3_s2","inception_5a-pool");
lgraph = connectLayers(lgraph,"pool4-3x3_s2","inception_5a-5x5_reduce");
lgraph = connectLayers(lgraph,"pool4-3x3_s2","inception_5a-1x1");
lgraph = connectLayers(lgraph,"inception_5a-relu_3x3","inception_5a-output/in2");
lgraph = connectLayers(lgraph,"inception_5a-relu_1x1","inception_5a-output/in1");
lgraph = connectLayers(lgraph,"inception_5a-relu_pool_proj","inception_5a-output/in4");
lgraph = connectLayers(lgraph,"inception_5a-relu_5x5","inception_5a-output/in3");
lgraph = connectLayers(lgraph,"inception_5a-output","inception_5b-3x3_reduce");
lgraph = connectLayers(lgraph,"inception_5a-output","inception_5b-1x1");
lgraph = connectLayers(lgraph,"inception_5a-output","inception_5b-pool");
lgraph = connectLayers(lgraph,"inception_5a-output","inception_5b-5x5_reduce");
lgraph = connectLayers(lgraph,"inception_5b-relu_3x3","inception_5b-output/in2");
lgraph = connectLayers(lgraph,"inception_5b-relu_1x1","inception_5b-output/in1");
lgraph = connectLayers(lgraph,"inception_5b-relu_pool_proj","inception_5b-output/in4");
lgraph = connectLayers(lgraph,"inception_5b-relu_5x5","inception_5b-output/in3");
```

# Train Network

Train the network using the specified options and training data.

```matlab:Code
analyzeNetwork(net)
[net, traininfo] = trainNetwork(augimdsTrain,lgraph,opts);
```

```text:Output
Training on single CPU.
Initializing input data normalization.
|======================================================================================================================|
|  Epoch  |  Iteration  |  Time Elapsed  |  Mini-batch  |  Validation  |  Mini-batch  |  Validation  |  Base Learning  |
|         |             |   (hh:mm:ss)   |   Accuracy   |   Accuracy   |     Loss     |     Loss     |      Rate       |
|======================================================================================================================|
|       1 |           1 |       00:00:03 |       25.00% |       20.00% |       3.2166 |       2.8498 |      1.0000e-04 |
|       1 |           5 |       00:00:05 |       75.00% |       13.33% |       1.4265 |       2.4104 |      1.0000e-04 |
|       2 |          10 |       00:00:08 |       25.00% |       33.33% |       1.6979 |       1.6221 |      1.0000e-04 |
|       2 |          15 |       00:00:10 |       50.00% |       33.33% |       1.3097 |       2.0775 |      1.0000e-04 |
|       3 |          20 |       00:00:13 |       25.00% |       40.00% |       1.9315 |       1.2386 |      1.0000e-04 |
|       3 |          25 |       00:00:16 |       50.00% |       46.67% |       1.6579 |       1.5231 |      1.0000e-04 |
|       4 |          30 |       00:00:18 |       75.00% |       60.00% |       0.8088 |       1.0055 |      1.0000e-04 |
|       4 |          35 |       00:00:21 |      100.00% |       73.33% |       0.4539 |       0.8974 |      1.0000e-04 |
|       5 |          40 |       00:00:24 |      100.00% |       66.67% |       0.4828 |       0.8838 |      1.0000e-04 |
|       5 |          45 |       00:00:26 |      100.00% |       66.67% |       0.3857 |       0.7889 |      1.0000e-04 |
|       6 |          50 |       00:00:29 |      100.00% |       73.33% |       0.3089 |       0.6608 |      1.0000e-04 |
|       7 |          55 |       00:00:32 |      100.00% |       86.67% |       0.2535 |       0.5544 |      1.0000e-04 |
|       7 |          60 |       00:00:34 |      100.00% |       80.00% |       0.2824 |       0.5272 |      1.0000e-04 |
|       8 |          65 |       00:00:37 |       75.00% |       93.33% |       0.3928 |       0.4979 |      1.0000e-04 |
|       8 |          70 |       00:00:40 |      100.00% |       93.33% |       0.3023 |       0.4629 |      1.0000e-04 |
|       8 |          72 |       00:00:41 |      100.00% |       86.67% |       0.1395 |       0.4461 |      1.0000e-04 |
|======================================================================================================================|
Training finished: Max epochs completed.
```

![figure_1.png](GoogleNet_images/figure_1.png)

# **Confusion Chart**

```matlab:Code
[YPred] = classify(net,augimdsValidation)
plotconfusion(imdsValidation.Labels,YPred)
```

# **Classify Test Images**

```matlab:Code
imagesGoogleNet = imageDatastore(['C:\Users\melahi\Documents\' ...
    'MATLAB\Examples\R2021b\nnet\TransferLearningUsing' ...
    'GoogLeNetExample\Test\']);
for K = 1 : 30

    subplot(10,3,K);
    I_org = imread(imagesGoogleNet.Files{K});
    I = imresize(I_org,[224,224]);
    [YPred,probs] = classify(net,I);
    imshow(I_org)
    label = YPred;
    title(string(label) + ", " + num2str(100*max(probs),3) + "%");
end
%save image
 userinput = input('Enter file name: ' , 's') 
```

```text:Output
userinput = 'GoogleNet classification'
```

```matlab:Code
 filename = sprintf('%s', userinput);
 saveas(gcf, filename, 'png')
```

![figure_2.png](GoogleNet_images/figure_2.png)

***
*Generated from GoogleNet.mlx with [Live Script to Markdown Converter](https://github.com/roslovets/Live-Script-to-Markdown-Converter)*
