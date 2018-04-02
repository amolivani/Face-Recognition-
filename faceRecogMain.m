close all; %closes all the previous open windows
clc %clears command window

TrainingDatabasePath = '/Users/amolivani/Documents/MATLAB/Training/';%give path of the folder containing training dataset images
TestDatabasePath = '/Users/amolivani/Documents/MATLAB/Test/'; %give path of the folder containing test dataset images

% Training 
% Considering the 8 training data images provided

trainI1 = strcat(TrainingDatabasePath,'1','.jpg');
TI1 = imresize(double(imread(trainI1)),[120,80]);
R1= reshape(TI1,120*80,1);

trainI2 = strcat(TrainingDatabasePath,'2','.jpg');
TI2 = imresize(double(imread(trainI2)),[120,80]);
R2= reshape(TI2,120*80,1);

trainI3 = strcat(TrainingDatabasePath,'3','.jpg');
TI3 = imresize(double(imread(trainI3)),[120,80]);
R3= reshape(TI3,120*80,1);

trainI4 = strcat(TrainingDatabasePath,'4','.jpg');
TI4 = imresize(double(imread(trainI4)),[120,80]);
R4= reshape(TI4,120*80,1);

trainI5 = strcat(TrainingDatabasePath,'5','.jpg');
TI5 = imresize(double(imread(trainI5)),[120,80]);
R5= reshape(TI5,120*80,1);

trainI6 = strcat(TrainingDatabasePath,'6','.jpg');
TI6 = imresize(double(imread(trainI6)),[120,80]);
R6= reshape(TI6,120*80,1);

trainI7 = strcat(TrainingDatabasePath,'7','.jpg');
TI7 = imresize(double(imread(trainI7)),[120,80]);
R7= reshape(TI7,120*80,1);

trainI8 = strcat(TrainingDatabasePath,'8','.jpg');
TI8 = imresize(double(imread(trainI8)),[120,80]);
R8= reshape(TI8,120*80,1);

% calculating mean face 
m = (R1+R2+R3+R4+R5+R6+R7+R8)/8;
%R_m = reshape(m,120,80);

% subtracting mean face from all the training images
R1_bar = R1-m;
R2_bar = R2-m;
R3_bar = R3-m;
R4_bar = R4-m;
R5_bar = R5-m;
R6_bar = R6-m;
R7_bar = R7-m;
R8_bar = R8-m;

% putting all training faces into a single matrix A
A = [reshape(R1_bar,1,120*80)
    reshape(R2_bar,1,120*80)
    reshape(R3_bar,1,120*80)
    reshape(R4_bar,1,120*80)
    reshape(R5_bar,1,120*80)
    reshape(R6_bar,1,120*80)
    reshape(R7_bar,1,120*80)
    reshape(R8_bar,1,120*80)];

% calculating the covariance matrix
C = (A)*(A');

% finding eigenvalues and putting eigenvectors into a single matrix V
[V,lamda] = eigs(C,8,'largestabs');

%calculating the face space u 
u = V*A;
U = u';

% displaying each training face onto face space
figure('Name','Training images after subtracting mean face')
subplot(3,3,1), F1=reshape(U(:,1),120,80); pcolor(flipud(F1)), shading interp, colormap(gray), set(gca, 'Xtick', [], 'Ytick', []); title('1.jpg');
subplot(3,3,2), F2=reshape(U(:,2),120,80); pcolor(flipud(F2)), shading interp, colormap(gray), set(gca, 'Xtick', [], 'Ytick', []); title('2.jpg');
subplot(3,3,3), F3=reshape(U(:,3),120,80); pcolor(flipud(F3)), shading interp, colormap(gray), set(gca, 'Xtick', [], 'Ytick', []); title('3.jpg');
subplot(3,3,4), F4=reshape(U(:,4),120,80); pcolor(flipud(F4)), shading interp, colormap(gray), set(gca, 'Xtick', [], 'Ytick', []); title('4.jpg');
subplot(3,3,5), F5=reshape(U(:,5),120,80); pcolor(flipud(F5)), shading interp, colormap(gray), set(gca, 'Xtick', [], 'Ytick', []); title('5.jpg');
subplot(3,3,6), F6=reshape(U(:,6),120,80); pcolor(flipud(F6)), shading interp, colormap(gray), set(gca, 'Xtick', [], 'Ytick', []); title('6.jpg');
subplot(3,3,7), F7=reshape(U(:,7),120,80); pcolor(flipud(F7)), shading interp, colormap(gray), set(gca, 'Xtick', [], 'Ytick', []); title('7.jpg');
subplot(3,3,8), F8=reshape(U(:,8),120,80); pcolor(flipud(F8)), shading interp, colormap(gray), set(gca, 'Xtick', [], 'Ytick', []); title('8.jpg');

% calculating the PCA coefficient omega
omega = horzcat(u*R1_bar,u*R2_bar,u*R3_bar,u*R4_bar,u*R5_bar,u*R6_bar,u*R7_bar,u*R8_bar);

% Recognition
% Reading the image to be tested 
test_image = strcat(TestDatabasePath,'1','.jpg');%give the test image name
I = imresize( double( imread(test_image)), [120,80]);
I = reshape(I,120*80,1);

% subtracting mean face m from input face I
I_bar = I-m;

% computing its projection onto face space
projected_omega = u*I_bar;

% reconstructing input face image from eigenfaces
IR_bar = U*projected_omega;
IR_bar_reshape = reshape(IR_bar,120,80);

%displaying the reconstructed image
figure('Name','Reconstructed Image'), imshow(IR_bar_reshape);

%manually chosen threshold 
t0 = 9.0000e+10;

% calculating distance between input face image and its reconstruction
d0 = norm( IR_bar - I );

% calculating distance between input face and training images in face space
edist = [];
for i = 1 : 8
    q = omega(:,i);
    d = ( norm( projected_omega - q ) );
    edist = [edist d];
end

% finding the minimum distance and recognizing its index for classification
[edist_min , Recognized_index] = min(edist);
outputI = strcat(int2str(Recognized_index),'.jpg');
img_select = strcat(TrainingDatabasePath,outputI);
img_select = imread(img_select);

% displaying the output and its classification result
figure('Name','Output Image');
I = imresize(double(imread(test_image)),[231,195]);
subplot(1,2,1), pcolor(flipud(I)), shading interp, colormap(gray), set(gca, 'Xtick', [], 'Ytick', []), title('Known face detected');
subplot(1,2,2), pcolor(flipud(img_select)), shading interp, colormap(gray), set(gca, 'Xtick', [], 'Ytick', []), title('Classification Result');
str = strcat('Test image classified to the following training image: ',outputI);
disp(str)
