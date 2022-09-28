function outputIDNew = FaceRecognitionNew(trainImgSet, trainPersonID, testPath)
%%   A face reconition method based on Viola-Jones face detection algorithm, HOG feature and KNN classifier
%    trainImgSet: contains all the given training face images
%    trainPersonID: indicate the corresponding ID of all the training images
%    testImgSet: constains all the test face images
%    outputID - predicted ID for all tested images 

%% Extract HOG features from the training images. 
trainTmpSet = [];
for i=1:size(trainImgSet,4)    
    tmpI = FaceDetection(trainImgSet(:,:,:,i)); % Extract face from the image
    hog = extractHOGFeatures(tmpI); % Extract HOG feature
    trainTmpSet = [trainTmpSet; hog];
end

%% Train a KNN model
Mdl = fitcknn(trainTmpSet,trainPersonID,'NumNeighbors',1,'Standardize',1,'Distance','correlation');

%% Face recognition for all the test images
outputIDNew=[];
testImgNames=dir([testPath,'*.jpg']);

for i=1:size(testImgNames,1)
    testImg=imread([testPath, testImgNames(i,:).name]); %load one of the test images
    tmpI = FaceDetection(testImg); % Extract face from the image
    tmpI = extractHOGFeatures(tmpI); % Extract HOG feature
    [predictIndex,~] = predict(Mdl,tmpI); % Predict the class using KNN model
    outputIDNew=[outputIDNew; predictIndex];
end

