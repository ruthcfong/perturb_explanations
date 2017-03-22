img_path = '/home/ruthfong/packages/caffe/examples/images/cat.jpg';
N = 25;
sigma = 10;

ind = -floor(N/2):floor(N/2);
[X,Y] = meshgrid(ind, ind);
h = exp(-(X.^2 + Y.^2) / (2*sigma^2));
h = h / sum(h(:));

h_col = h(:);

I = imread(img_path);
I = im2double(rgb2gray(I));
I_pad = padarray(I, [floor(N/2), floor(N/2)]);
C = im2col(I_pad, [N, N], 'sliding');
C_filter = sum(bsxfun(@times, C, h_col), 1);
out = col2im(C_filter, [N, N], size(I_pad), 'sliding');

% figure;
% subplot(1,2,1);
% imshow(I);
% subplot(1,2,2);
% imshow(out);

weights = h;
%weights = repmat(h, [1,1,3,1]);
%weights = single(weights / sum(weights(:)));
%weights = 1/(5*5*3)*rand(5,5,3,1, 'single')
net.layers{1} = struct(...
    'name',  'conv1', ...
    'type', 'conv', ...
    'weights', {{weights, zeros(1,1,'single')}}, ...
    'pad', floor(N/2), ...
    'stride', 1, ...
    'dilate', 1, ...
    'opts', {{}});

%img = single(imread('/home/ruthfong/packages/caffe/examples/images/cat.jpg'));
img = rand(224,224,'single');
res = vl_simplenn(net, img);
gradient = ones(size(res(end).x), 'like', res(end).x);
res = vl_simplenn(net, img, gradient);

figure;
subplot(2,2,1);
imshow(uint8(img));
subplot(2,2,2);
imshow(uint8(res(end).x));
subplot(2,2,3);
imshow(res(1).dzdx);