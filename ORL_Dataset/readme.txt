There are two variables in ORL dataset.
alls is a d x n dimensional matrix whose column vector represents a sample.
gnd is a 1 x n dimensional vector whose element indicates the label of the corresponding sample (the sample collection and label collection are following the same sequence).

matlab commands (show the image):
load('ORL_32_32.mat')
i=3; % choose the face sample you wanna see.
imshow(reshape(uint8(alls(:, i)), 32,32 ))
