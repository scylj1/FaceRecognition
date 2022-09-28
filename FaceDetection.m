function face = FaceDetection(image)
%%   A face detection method based on Viola-Jones algorithm 
%    image: A single training image
%    face - Extract the face from the input image 

%% Apply Viola-Jones algorithm using built-in functions
image = rgb2gray(image);
faceDetector = vision.CascadeObjectDetector;
faces_auto = step(faceDetector, image);

% Return the whole image if no face is detected
if isempty(faces_auto)
    face = image;

% Return the largest image if multiple faces are detected
elseif size(faces_auto, 1) > 1
    faces = faces_auto(:, 3)';
    [~ , index] = max(faces);
    face = image(faces_auto(index , 2):(faces_auto(index , 2)+faces_auto(index , 4)), faces_auto(index , 1):(faces_auto(index , 1)+faces_auto(index , 3)));

% Return the detected face
else
    face = image(faces_auto(2):(faces_auto(2)+faces_auto(4)), faces_auto(1):(faces_auto(1)+faces_auto(3)));
end

% resize the image
face = imresize(face, [50 50]);
face = imresize(face, [100 100]);
    
