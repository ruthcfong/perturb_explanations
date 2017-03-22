net = load('/home/ruthfong/packages/matconvnet/data/models/imagenet-caffe-alex.mat');
imdb_paths = load('/data/ruthfong/ILSVRC2012/val_imdb_paths.mat');
img_idx = 1;
img_idx = [1,2,5,8,3,6,7,20,57,12,14,18,21,27,37,41,61,70,76,91];

opts = struct();

opts.num_transforms = 100;
opts.num_top = 10;
opts.size = 25;
opts.flip = false;
opts.gpu = 0;
opts.null_img = [];
opts.save_fig_path = '';
opts.save_res_path = '';

for i=1:length(img_idx)
    curr_opts = opts;
    img_i = img_idx(i);
    img = imread(imdb_paths.images.paths{img_i});
    img = cnn_normalize(net.meta.normalization, img, 1);
%     res = vl_simplenn(net, img);
%     [~,target_class] = max(res(end).x);
    target_class = imdb_paths.images.labels(img_i);
    curr_opts.save_fig_path = sprintf(['/data/ruthfong/neural_coding/figures11/', ...
        'random_masks/target_class_no_shearing_size_%d_num_%d_flip_%d/%d.jpg'], ...
        opts.size, opts.num_transforms, opts.flip, img_i);
    curr_opts.save_res_path = sprintf(['/data/ruthfong/neural_coding/results11/', ...
        'random_masks/target_class_no_shearing_size_%d_num_%d_flip_%d/%d.mat'], ...
        opts.size, opts.num_transforms, opts.flip, img_i);

    res = generate_random_masks(net,img,target_class,curr_opts);
    
    close all;
end
