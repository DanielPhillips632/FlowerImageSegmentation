close all;
clear;
clc;

imDir = "images_256";
pxDir = "labels_256";

% Configure datastores, with class info
imds = imageDatastore(imDir);
classNames = ["flower" "background"];
pixelLabelID = [1 3];
pxds = pixelLabelDatastore(pxDir,classNames,pixelLabelID);
% Retain only flower and (adjusted) background classes
pxds.ReadFcn = @removeOtherClasses;

% Get base names of files in the datastores
[~, imNames, ~] = cellfun(@(x) fileparts(x), imds.Files, 'UniformOutput', false);
[~, pxNames, ~] = cellfun(@(x) fileparts(x), pxds.Files, 'UniformOutput', false);

% Find indices of images that have corresponding labels
[~, labelIndices] = intersect(imNames, pxNames, 'stable');

% Retain only the images with corresponding labels
imds = subset(imds, labelIndices);

% Shuffle the indices
rng(0);
numFiles = numel(imds.Files);
shuffledIndices = randperm(numFiles);

% Create dataset splitting indicies
splitIndexTrain = round(0.8 * numFiles);
splitIndexVal = round(0.9 * numFiles);

% Create subset indices
trainingIndices = shuffledIndices(1:splitIndexTrain);
validationIndices = shuffledIndices(splitIndexTrain+1:splitIndexVal);
testingIndices = shuffledIndices(splitIndexVal+1:end);

% Partition images and annotations into training, validation and test sets
trainingImds = subset(imds, trainingIndices);
trainingPxds = subset(pxds, trainingIndices);
validationImds = subset(imds, validationIndices);
validationPxds = subset(pxds, validationIndices);
testingImds = subset(imds, testingIndices);
testingPxds = subset(pxds, testingIndices);

% Combine images with ground truths
trainingData = combine(trainingImds, trainingPxds);
validationData = pixelLabelImageDatastore(validationImds, validationPxds);
testingData = pixelLabelImageDatastore(testingImds, testingPxds);

% Augment, enlarge and shuffle the training data
augmentedTrainingData = transform(trainingData, @augment);
combinedData = combine(trainingData, augmentedTrainingData);
trainingData = shuffle(combinedData);

% Configure and load DeepLabv3+ network with ResNet-18 backbone
imageSize = [256 256 3];
numClasses = 2;
existingNet = deeplabv3plusLayers(imageSize,numClasses,"resnet18","DownsamplingFactor",16);

%%

% Set training parameters
opts = trainingOptions('adam', ...
    'InitialLearnRate',1e-3, ...
    'MaxEpochs',10, ... 
    'MiniBatchSize',64, ...
    'Plots','training-progress', ...
    'ValidationData',validationData, ... 
    'ValidationFrequency',10, ...
    'OutputNetwork','best-validation-loss');

% Train and save model
net = trainNetwork(trainingData,existingNet,opts);
%save('segmentexistnet','net');

%%
% Load and apply model to test set
load('segmentexistnet.mat')
testResults = semanticseg(testingImds,net,"WriteLocation","out");

%%
% Show example segmentation of test image
overlayOut = labeloverlay(readimage(testingImds,26),readimage(testResults,26)); %overlay
figure
imshow(overlayOut);
title('overlayOut')

%%
% Evaluate segmentation performance
metrics = evaluateSemanticSegmentation(testResults, testingPxds);

% Display confusion matrix
cm = confusionchart(metrics.ConfusionMatrix.Variables, classNames);
cm.Title = 'Normalised Confusion Matrix (%)';
cm.Normalization = 'row-normalized';
ax = gca;
ax.FontSize = 20;

% Display mean IOU
meanIoU = metrics.ImageMetrics.MeanIoU

% Display histogram of IOU values
figure
histogram(meanIoU)
title('Image Mean IoU')

% Display class metrics
metrics.ClassMetrics

%%

% Assign every pixel that is not assigned to the flower class to the
% background class
function data = removeOtherClasses(pxds)
    data = imread(pxds);
    data(data ~= 1) = 3;
end

% Augment an image with random colour and affine adjustments, adapted from
% https://uk.mathworks.com/help/deeplearning/ug/augment-pixel-labels-for-semantic-segmentation.html
function out = augment(data)
im = data{1};
px = data{2};

im = jitterColorHSV(im,"Hue",0.1,"Saturation",0.3,"Brightness",0.3,"Contrast",0.3);
tform = randomAffine2d("Scale",[0.8 1.2],"XReflection",true,'Rotation',[-25 25], ...
    "XShear",[-10 10],'YShear',[-10 10],'XTranslation',[-10 10],'YTranslation',[-10 10]);
rout = affineOutputView(size(im),tform);

augmentedIm = imwarp(im,tform,"OutputView",rout);
augmentedPx = imwarp(px,tform,"OutputView",rout,'Interp','nearest');

out = {augmentedIm,augmentedPx};
end

