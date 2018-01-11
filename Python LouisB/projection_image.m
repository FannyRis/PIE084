clear all

% test de projection sur 03_00000178.jpg
% taille image 2448 x 2048 pixels
% equivalent 8.44560000000000066223 x 7.06560000000000076881mm
image = imread('data/gourd_c1818/Images/03/03_00000260.jpg');
[m,n,p] = size(image);
image_size_x = 8.44560000000000066223e-3;
image_size_y = 7.06560000000000076881e-3;

%%% point a projeter %%%
% centre de l'image situe a 44.791975, 1.393504
% equivalent UTM31 372926.070216281, 4961097.17070787, 157
% autre point UTM31 372907.93 4961136.45 154.49

%pylone 206
X = [372904.4705323878;  4961525.41389998;   176.55100000000002];
%pylone 207
%X = [372827.01153238775;  4961516.86289998;  174.191];


%coord camera dans l'espace
C = [372780.215804; 4961458.993612; 305.361986]; 

% orientation de la camera
omega = deg2rad(24.468344);
phi = deg2rad(-19.920028);
kappa = deg2rad(44.589921);

% cf "New Calibration and Computing Method for Direct Georeferencing of Image"
R_kappa = [cos(kappa) sin(kappa) 0;...
           -sin(kappa) cos(kappa) 0;...
           0 0 1];
R_omega = [1 0 0;...
           0 cos(omega) sin(omega);...
           0 -sin(omega) cos(omega)];
R_phi = [cos(phi) 0 -sin(phi);...
         0 1 0;...
         sin(phi) 0 cos(phi)];
     
R = R_kappa*R_phi*R_omega; %marche mieux
% R = R_kappa*R_omega*R_phi; %selon le bouquin


% caracteristiques camera
% cf "Multiple View Geometry in Computer Vision (Second Edition)"
f = 11.97211421866878922060e-3;
px = 4.15244279911911640113e-3; %point principal
py = 3.51430396435733793936e-3;
px_pixel = px*n/image_size_x;
py_pixel = py*m/image_size_y;
K = [f 0 px; 0 f py; 0 0 1]; %(6.4)


%cgt de repere UTM31 -> camera
Xcam =  R*(X-C); % (6.6)

%projection sur image
x = K*Xcam; % (6.5)

x_proj = x(1)/x(3); % coord du point sur l'image
y_proj = x(2)/x(3);
x_proj_pixel = x_proj*n/image_size_x;
y_proj_pixel = y_proj*m/image_size_y;


%%%correction distorsion%%%
% Symmetrical Lens Distortion Coeffs
K1= -0.39617281077986588045;
K2= 0.11834006879098386833;
K3= -0.01605847433237215724;
% Tangential Lens Distortion Coeffs
T1= 0.00025977502617845189;
T2= -0.00021034257794291539;

% p.191 %radial distorsion
% r = sqrt((4*(x_proj_pixel-px_pixel)/(n+m))^2+(4*(y_proj_pixel-py_pixel)/(n+m))^2);
% x_proj_corr_pixel = px_pixel + (1+K1*r+K2*r^2+K3*r^3)*(x_proj_pixel - px_pixel);
% y_proj_corr_pixel = py_pixel + (1+K1*r+K2*r^2+K3*r^3)*(y_proj_pixel - py_pixel);

% r = norm([x_proj-px; y_proj-py]);
r = sqrt(((x_proj-px)/f)^2+((y_proj-py)/f)^2);
x_proj_corr_pixel = (px + (1+K1*r+K2*r^2+K3*r^3)*(x_proj - px))*n/image_size_x;
y_proj_corr_pixel = (py + (1+K1*r+K2*r^2+K3*r^3)*(y_proj - py))*m/image_size_y;

%tangential distorsion
%x_proj_corr_pixel = x_proj_corr_pixel + T1*(r^2+2*(x_proj_pixel-px_pixel)^2) + 2*T2*(x_proj_pixel-px_pixel)*(y_proj_pixel-py_pixel)
%y_proj_corr_pixel = y_proj_corr_pixel + T2*(r^2+2*(y_proj_pixel-py_pixel)^2) + 2*T1*(x_proj_pixel-px_pixel)*(y_proj_pixel-py_pixel)


% affichage
figure
imshow(image);
hold on
%avant correction
plot(n-x_proj_pixel,y_proj_pixel, 'linestyle','none','marker','x','markersize',25,'linewidth',4)

%apres correction
plot(n-x_proj_corr_pixel,y_proj_corr_pixel, 'linestyle','none','marker','x','markersize',25,'linewidth',4)
