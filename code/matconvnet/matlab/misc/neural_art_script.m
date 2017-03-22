% alexnet
net = load('/home/ruthfong/packages/matconvnet/data/models/imagenet-caffe-alex.mat');
 
% vgg16
%net = load('/home/ruthfong/packages/matconvnet/data/models/imagenet-matconvnet-vgg-verydeep-16.mat');

img_content = imread('/home/ruthfong/neural_coding/sample_images/tubingen.jpg');
img_style = imread('/home/ruthfong/neural_coding/sample_images/starry_night_google.jpg');

x_content = cnn_normalize(net.meta.normalization, img_content, 1);
x_style = cnn_normalize(net.meta.normalization, img_style, 1);

opts = struct();

% alexnet
opts.content_layer = 11;
opts.style_layers = [1,5,9,11,13];

% vgg16
%opts.content_layer = 20; % conv4_2
%opts.style_layers = [1,6,11,18,25]; % conv(1-5)_1

opts.style_weights = 1/length(opts.style_layers)*ones(size(opts.style_layers));
opts.debug = true;
opts.learning_rate = 1e1;
opts.plot_step = 10;

x_art = create_neural_art(net, x_content, x_style, opts);

