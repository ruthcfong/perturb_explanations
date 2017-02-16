net_type = 'alexnet';
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
        %img_idx = [3,6,7,20,57,12,14,18,21,27,37,41,61,70,76,91];
        dataset_description = 'val_heldout';
    else
        img_idx = 1:50000;
        dataset_description = 'val';
    end
end

opts = struct();

opts.adam.beta1 = 0.9;
opts.adam.beta2 = 0.999;
opts.adam.epsilon = 1e-8;

opts.num_iters = 500;
opts.plot_step = 500;

opts.l1_ideal = 1;

opts.learning_rate = 1e0;
opts.lambda = 5e-8; %2.5e-8; %1e-10;
opts.tv_lambda = 5e-6; %2.5e-6; %1e-8; 
opts.beta = 1.5; % 1.2;

opts.null_img_type = 'provided';
opts.null_img_imdb_paths = imdb_paths;

opts.noise.use = true;
opts.noise.mean = 0;
opts.noise.std = 1e-3;
opts.jitter = 8;
opts.mask_params.type = 'direct';
opts.update_func = 'adam';

opts.gpu = 1;

% random sample
% opts.learning_rate = 1e0;
% opts.lambda = 1e-4;
% opts.tv_lambda = 1e-2;
% opts.beta = 3;

% opts.learning_rate = 1e0;
% opts.lambda = 1e-5; 
% opts.tv_lambda = 5e-3;  
% opts.beta = 3;

% opts.null_img_type = 'index_sample';
% opts.null_img_imdb_paths = imdb_paths;
% 
% opts.noise.use = true;
% opts.noise.mean = 0;
% opts.noise.std = 1e-3;
% opts.jitter = 10;
% opts.mask_params.type = 'direct';
% opts.update_func = 'adam';

%max_diff_layer = create_max_diff_layer();
%net.layers{end} = max_diff_layer;

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
end

% net.layers{end} = struct(...
%     'type','mseloss',...
%     'class',zeros([1 1 1000], 'single'));

%lambdas = mtimes([1 2.5 5 7.5]', (10.^(-8:1:-1)));
%lambdas = reshape(lambdas, [1 numel(lambdas)]);
lambdas = 10.^(-8:-1);
%betas = 1:0.5:3;
betas = 1:3;
jitters = [0 2 4 8];

for lambda=lambdas
    for tv_lambda=lambdas
        for tv_beta=betas
            for jitter=jitters
                opts.lambda = lambda;
                opts.tv_lambda = tv_lambda;
                opts.beta = tv_beta;
                opts.jitter = jitter;
                for i=1:length(img_idx)
                    curr_opts = opts;
                    img_i = img_idx(i);
                    img = imread(imdb_paths.images.paths{img_i});
                    img_ = cnn_normalize(net.meta.normalization, img, true);
                    target_class = imdb_paths.images.labels(img_i);

                    if isDag
                        net.move('cpu');
                        inputs = {input_name, img_};
                        net.vars(output_i).precious = 1;
                        net.eval(inputs);
                        gradient = zeros(size(net.vars(output_i).value), 'like', net.vars(output_i).value);
                        output_val = net.vars(output_i).value;
                    else
                        res = vl_simplenn(net, img_);
                        gradient = zeros(size(res(end).x), 'single');
                        output_val = res(end).x;
                    end

                    %gradient(target_class) = 1;

                    [~,top_idx] = sort(squeeze(output_val), 'descend');
                    gradient(top_idx(1:5)) = 1;

                    %gradient = 1;

                    curr_opts.null_img = imgaussfilt(img, 10);

                    if strcmp(curr_opts.null_img_type, 'index_sample')
                        sigmas = 10:-0.1:0;
                        curr_opts.null_img = zeros([size(img) length(sigmas)], 'single');
                        curr_opts.null_img(:,:,:,end) = img;
                        for j=1:(length(sigmas)-1)
                            curr_opts.null_img(:,:,:,j) = imgaussfilt(img, sigmas(j));
                        end
                    end

                    curr_opts.save_fig_path = fullfile(sprintf(strcat('/data/ruthfong/neural_coding/figures10/', ...
                        'imagenet/%s_%s/L0/min_classlabel_5_%s_%s/lr_%f_reg_lambda_%f_tv_norm_%f_beta_%f_num_iters_%d_noise_%d_jitter_%d_%s/', ...
                        '%d_mask_dim_2.jpg'), ...
                     net_type, dataset_description, curr_opts.mask_params.type, curr_opts.null_img_type, curr_opts.learning_rate, log10(curr_opts.lambda), log10(curr_opts.tv_lambda), ...
                     curr_opts.beta, curr_opts.num_iters, curr_opts.noise.use, curr_opts.jitter, curr_opts.update_func), num2str(img_i));

                    curr_opts.save_res_path = sprintf(strcat('/data/ruthfong/neural_coding/results10/', ...
                        'imagenet/%s_%s/L0/min_classlabel_5_%s_%s/lr_%f_reg_lambda_%f_tv_norm_%f_beta_%f_num_iters_%d_noise_%d_jitter_%d_%s/', ...
                        '%d.mat'), ...
                     net_type, dataset_description, curr_opts.mask_params.type, curr_opts.null_img_type, curr_opts.learning_rate, log10(curr_opts.lambda), log10(curr_opts.tv_lambda), ...
                     curr_opts.beta, curr_opts.num_iters, curr_opts.noise.use, curr_opts.jitter, curr_opts.update_func, img_i);

                    if exist(curr_opts.save_res_path, 'file')
                        fprintf('%s already exists; so skipping\n', curr_opts.save_res_path);
                        continue;
                    end

                    mask_res = optimize_mask(net, img, gradient, curr_opts);

                    close all;
                end
            end
        end
    end
end
        
