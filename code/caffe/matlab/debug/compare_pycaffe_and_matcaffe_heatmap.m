heatmap_py = importdata('/home/ruthfong/neural_coding/saliency_heatmap_python.txt');
bb_py_orig = [96 50 434 341];

gpu = 0;
caffe_model_dir = '/home/ruthfong/packages/caffe/models';
model_dir = fullfile(caffe_model_dir, 'bvlc_googlenet');
net_model = fullfile(model_dir, 'deploy_force_backward.prototxt');
net_weights = fullfile(model_dir, 'bvlc_googlenet.caffemodel');
net = caffe.Net(net_model, net_weights, 'test');

imdb_paths = load('/data/ruthfong/ILSVRC2012/val_imdb_paths.mat');
path = imdb_paths.images.paths{1};
label = imdb_paths.images.labels(1);

img = caffe.io.load_image(path);
mean_img = caffe.io.read_mean('/home/ruthfong/packages/caffe/data/ilsvrc12/imagenet_mean.binaryproto');
mean_c = squeeze(mean(mean(mean_img)));
resize = [224 224 3];
im1 = normalize_img(img, mean_img, resize);
%im2 = permute(bsxfun(@minus, permute(imresize(img, resize(1:2), 'bilinear'), [3 1 2]), mean_c), [2 3 1]);
im2 = permute(bsxfun(@minus, permute(imresize(img, resize(1:2), 'bilinear'), [3 1 2]), mean_c), [2 3 1]);
opts = struct();
opts.gpu = gpu;

heatmap_mat1 = convert_im_order(compute_heatmap(net, im1, label, 'saliency', opts)); 
heatmap_mat2 = convert_im_order(compute_heatmap(net, im2, label, 'saliency', opts));

caffe.reset_all();

img = imread(path);
img_size = size(img);
[bb_mat1, heatmap_mat1_thres] = getbb_from_heatmap(heatmap_mat1, 5.0, img_size(1:2));
[bb_mat2, heatmap_mat2_thres] = getbb_from_heatmap(heatmap_mat2, 5.0, img_size(1:2));
[bb_py_new, heatmap_py_thres] = getbb_from_heatmap(heatmap_py, 5.0, img_size(1:2));

figure;
subplot(4,3,1);
imshow(img);

subplot(4,3,4);
imagesc(heatmap_mat1);
colorbar;
axis square;
title('heatmap mat1');

subplot(4,3,5);
imagesc(heatmap_mat2);
colorbar;
axis square;
title('heatmap mat2');

subplot(4,3,6);
imagesc(heatmap_py);
colorbar;
axis square;
title('heatmap py');

subplot(4,3,7);
imagesc(heatmap_mat1_thres);
colorbar;
axis square;
hold on;
rectangle('Position', [bb_mat1(1) bb_mat1(2) (bb_mat1(3) - bb_mat1(1)) (bb_mat1(4) - bb_mat1(2))], ...
    'EdgeColor', 'r');
title('mat1 thres');

subplot(4,3,8);
imagesc(heatmap_mat2_thres);
colorbar;
axis square;
hold on;
rectangle('Position', [bb_mat2(1) bb_mat2(2) (bb_mat2(3) - bb_mat2(1)) (bb_mat2(4) - bb_mat2(2))], ...
    'EdgeColor', 'r');
title('mat2 thres');

subplot(4,3,9);
imagesc(heatmap_py_thres);
colorbar;
axis square;
hold on;
rectangle('Position', [bb_py_new(1) bb_py_new(2) (bb_py_new(3) - bb_py_new(1)) (bb_py_new(4) - bb_py_new(2))], ...
    'EdgeColor', 'r');
rectangle('Position', [bb_py_orig(1) bb_py_orig(2) (bb_py_orig(3) - bb_py_orig(1)) (bb_py_orig(4) - bb_py_orig(2))], ...
    'EdgeColor', 'b');
title('py thres');

subplot(4,3,10);
imagesc(heatmap_mat1 - heatmap_py);
colorbar;
axis square;
title('mat1 - py');

subplot(4,3,11);
imagesc(heatmap_mat2 - heatmap_py);
colorbar;
axis square;
title('mat2 - py');

subplot(4,3,12);
imagesc(heatmap_mat1 - heatmap_mat2);
colorbar;
axis square;
title('mat1 - mat2');

disp(bb_mat1);
disp(bb_mat2);
disp(bb_py_orig);
disp(bb_py_new);