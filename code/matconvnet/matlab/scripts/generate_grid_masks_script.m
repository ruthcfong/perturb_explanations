net_type = 'googlenet';
switch net_type
    case 'alexnet'
        net = load('/home/ruthfong/packages/matconvnet/data/models/imagenet-caffe-alex.mat');
    case 'vgg16'
        net = load('/home/ruthfong/packages/matconvnet/data/models/imagenet-vgg-verydeep-16.mat');
    case 'googlenet'
        net = load('/home/ruthfong/packages/matconvnet/data/models/imagenet-googlenet-dag.mat');
    otherwise
        assert(false);
end

use_train = false;
use_heldout = true;

if use_train
    imdb_paths = load('/data/ruthfong/ILSVRC2012/annotated_train_imdb_paths.mat');
    img_idx = load('/data/ruthfong/ILSVRC2012/annotated_train_heldout_idx.mat');
    img_idx = img_idx.heldout_idx;
    dataset_description = 'annotated_train_heldout';
else
    imdb_paths = load('/data/ruthfong/ILSVRC2012/val_imdb_paths.mat');
    if use_heldout
        img_idx = [1,2,5,8,3,6,7,20,57,12,14,18,21,27,37,41,61,70,76,91];
        dataset_description = 'val_heldout';
    else
        img_idx = 1:50000;
        dataset_description = 'val';
    end
end

num_top = 1;

opts = struct();
opts.flip = false;
opts.sigma = 500;
opts.num_centers = 484;
opts.gpu = 0;
opts.save_res_path = '';
opts.save_fig_path = '';

isDag = isfield(net, 'params') || isprop(net, 'params');

if isDag
    net = dagnn.DagNN.loadobj(net);
    net.mode = 'test';
    order = net.getLayerExecutionOrder();
    input_i = net.layers(order(1)).inputIndexes;
    output_i = net.layers(order(end)).outputIndexes;
    assert(length(input_i) == 1);
    assert(length(output_i) == 1);
    input_name = net.vars(input_i).name;
    output_name = net.vars(output_i).name;
    net.vars(output_i).precious = 1;
end

for i=1:length(img_idx)
    curr_opts = opts;
    
    img_i = img_idx(i);
    img = cnn_normalize(net.meta.normalization, imread(imdb_paths.images.paths{img_i}), 1);

    if isDag
        net.move('cpu');
        inputs = {input_name, img};
        net.eval(inputs);
        gradient = zeros(size(net.vars(output_i).value), 'like', net.vars(output_i).value);
        output_val = net.vars(output_i).value;
    else
        res = vl_simplenn(net, img);
        gradient = zeros(size(res(end).x), 'like', res(end).x);
        output_val = res(end).x;
    end
    
    %curr_opts.null_img = imgaussfilt(img, 10);
    
    [~,top_idx] = sort(squeeze(output_val), 'descend');
    gradient(top_idx(1:num_top)) = 1;
    
%     target_class = imdb_paths.images.labels(img_i);
%     gradient(target_class) = 1;

    curr_opts.save_fig_path = sprintf('/data/ruthfong/neural_coding/figures12/grid_masks/%s_%s/num_top_%d_flip_%d_sigma_%d_num_centers_%d/%d.jpeg', ...
        net_type, dataset_description, num_top, curr_opts.flip, curr_opts.sigma, curr_opts.num_centers, img_i);
    curr_opts.save_res_path = sprintf('/data/ruthfong/neural_coding/results12/grid_masks/%s_%s/num_top_%d_flip_%d_sigma_%d_num_centers_%d/%d.mat', ...
        net_type, dataset_description, num_top, curr_opts.flip, curr_opts.sigma, curr_opts.num_centers, img_i);
    res = generate_grid_masks(net, img, gradient, curr_opts);
    
    disp(i);
    
    close all;
end