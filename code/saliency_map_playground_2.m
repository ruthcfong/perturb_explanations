%% set parameters
load_network = true;
load_data = true;
run_on_batch = false;
show_figures = false;
is_local = false;
dataset = 'imagenet';


switch dataset
    case 'mnist' 
        end_l = 7;
        start_l = 1;
        batch_range = 1:1000;
        class_offset = 0;
    case 'places' % conv5 in places alexnet
        end_l = 20;
        start_l = 13;
        batch_range = 1:10:20500;
        class_offset = 1; % needed for places alexnet
        if exist('places_net')
            net = places_net;
        end
        if exist('places_imdb')
            imdb = places_imdb;
        end
        if exist('places_res')
            res = places_res;
        end
    case 'imagenet' % relu5 in imagenet alexnet
        batch_range = 1:10000; 
        start_l = 14;
        end_l = 20;
%         if exist('val_imdb_small')
%             imdb = val_imdb_small;
%         end
        class_offset = 0;
    otherwise
        assert(false);
end

%% load variables

% load network
if load_network
    switch dataset
        case 'mnist'
            if is_local
                load('/Users/brian/matconvnet-1.0-beta21/data/trained_nets/mnist-net.mat');
            else
                load('/home/ruthfong/packages/matconvnet/data/mnist-baseline-simplenn/net-final.mat');
            end
        case 'places'
            if is_local
                net = load('/Users/brian/neural_coding/models/places-caffe-ref-upgraded-tidy-with-classes.mat');
            else
                net = load('/home/ruthfong/neural_coding/models/places-caffe-ref-upgraded-tidy-with-classes.mat');
            end
        case 'imagenet'
            if is_local
                net = load('//Users/brian/matconvnet-1.0-beta21/data/models/imagenet-caffe-alex.mat');
            else
                net = load('/home/ruthfong/packages/matconvnet/data/models/imagenet-caffe-alex.mat');
            end
    end
end

% load images
if load_data
    switch dataset
        case 'mnist'
            if is_local
                imdb = load('/Users/brian/matconvnet-1.0-beta21/data/mnist/imdb.mat');
            else
                imdb = load('/home/ruthfong/packages/matconvnet/data/mnist-baseline-simplenn/imdb.mat');
            end
        case 'places'
            if is_local
                imdb = load('/Users/brian/neural_coding/data/places205_imdb_val.mat');
            else
                imdb = load('/data/datasets/places205/imdb_val_resized_227.mat');
            end
        case 'imagenet'
            if is_local
                imdb_paths = load('~/neural_coding/data/ILSVRC2012/val_imdb_paths.mat');
            else
                imdb_paths = load('/data/ruthfong/ILSVRC2012/val_imdb_paths.mat');
            end
    end
end

%% add a softmaxloss layer if needed
if ~isequal(net.layers{end}.type, 'softmaxloss')
    net.layers{end+1} = struct('type', 'softmaxloss') ;
end

%% forward and backward pass
% if run_on_batch
%     net.layers{end}.class = imdb.images.labels(batch_range) + class_offset;
%     res = vl_simplenn(net, imdb.images.data(:,:,:,batch_range),1);
% end


%% load variables
%null_img = mean(imdb.images.data(:,:,:,batch_range),4);
layer = 0;

%img_idx =1:100;
%img_idx = [4,7];
%img_idx = [1,2,5,8,3,6,7,20,57,12,14,18,21,27,37,41,61,70,76,91];
img_idx = 1:500;
opts = struct();
%opts.batch_range = batch_range;
%opts.class_offset = class_offset;
%opts.null_img = null_img;
opts.num_iters = 300;
opts.plot_step = 50;
opts.debug = false;
opts.save_fig_path = '';
opts.save_res_path = '';
opts.loss = 'min_classlabel';
opts.mask_init = 'rand';
%opts.error_stopping_threshold = 100;%-100; 
opts.num_masks = 1;
opts.denom_reg = false; % obsolete
opts.mask_transform_function = 'linear';
%opts.mask_transform_function = 'generalized_sigmoid';
% opts.num_average = 5;
%opts.gen_sig_a = 50;
opts.num_class_labels = 5;
%opts.premask = 'superpixels';
% opts.sim_layernums = [4];
% opts.sim_layercoeffs = [1e-8];

for i=1:length(img_idx),
    curr_opts = opts;
    img_i = img_idx(i);
    if is_local
        [~, filename, ext] = fileparts(imdb_paths.images.paths{img_i})
        img_path = fullfile('~/neural_coding/data/ILSVRC2012/images/val/', strcat(filename, ext));
        img = cnn_normalize(net.meta.normalization, imread(img_path), true);
    else
        img = cnn_normalize(net.meta.normalization, imread(imdb_paths.images.paths{img_i}), true);
    end
    target_class = imdb_paths.images.labels(img_i) + class_offset;
    for m=2,
        curr_opts.mask_dims = m;

        switch curr_opts.mask_dims
            case 1
                curr_opts.learning_rate = 2e1;
                curr_opts.lambda = 5e-4;
                curr_opts.tv_lambda = 0;
                curr_opts.beta = 0;
            case 2
                % L15
%                 curr_opts.learning_rate = 1e1; %2e1;
%                 curr_opts.lambda =2.75e-3; %2.2e-3; %1e-3; %5e-5;
%                 curr_opts.tv_lambda = 1e-3;
                %L8 (pool2) -- not too tuned
%                 curr_opts.learning_rate = 1e1; %2e1;
%                 curr_opts.lambda = 5e-3; %1e-3; %5e-5;
%                 curr_opts.tv_lambda = 2e-3;
%                 curr_opts.beta = 2;
%L3
%                 curr_opts.learning_rate = 1e1; %2e1;
%                 curr_opts.lambda = 2.2e-3; %1e-3; %5e-5;
%                 curr_opts.tv_lambda = 1e-3;
%                 curr_opts.beta = 2;
                %L2
%                 curr_opts.learning_rate = 1e1; %2e1;
%                 curr_opts.lambda = 2.2e-3; %1e-3; %5e-5;
%                 curr_opts.tv_lambda = 1e-3;
%                 curr_opts.beta = 2;
% L0 -min_classlabelsoftmax for top5/max_softmax
                curr_opts.learning_rate = 1e2; %2e1;
                curr_opts.lambda = 5e-7; %1e-3; %5e-5;
                curr_opts.tv_lambda = 1e-3;
                curr_opts.beta = 3;
% L0 = min_preclasslabel for top5
%                 curr_opts.learning_rate = 1e2; %2e1;
%                 curr_opts.lambda = 1e-5; %1e-3; %5e-5;
%                 curr_opts.tv_lambda = 1e-3;
%                 curr_opts.beta = 3;
% L0 - mask ensemble
%                 curr_opts.learning_rate = 1e1; %2e1;
%                 curr_opts.lambda = 5e-7; %1e-3; %5e-5;
%                 curr_opts.tv_lambda = 1e-3;
%                 curr_opts.beta = 3;
% L0 - superpixels - max_classlabel
%                 curr_opts.learning_rate = 1e1;
%                 curr_opts.lambda = 1e-5;
%                 curr_opts.tv_lambda = 1e-5;
%                 curr_opts.beta = 3;
% L0 - superpixels - min_classlabel
%                 curr_opts.learning_rate = 1e1;
%                 curr_opts.lambda = 1e-6;
%                 curr_opts.tv_lambda = 1e-4;
%                 curr_opts.beta = 3;
                % L0 - closest
%                 curr_opts.learning_rate = 4e1; %2e1;
%                 curr_opts.lambda = 1e-5; %2.2e-3; %1e-3; %5e-5;
%                 curr_opts.tv_lambda = 3e-3;
%                 curr_opts.beta = 3.1;
                % L0 - min classlabel for top5
%                 curr_opts.learning_rate = 1e3; 
%                 curr_opts.lambda = 7.25e-8; 
%                 curr_opts.tv_lambda = 1e-4;
%                 curr_opts.beta = 2;
                % L0 - min classlabel for top5 using sigmoid
%                 curr_opts.learning_rate = 1e1; 
%                 curr_opts.lambda = 1e-4; 
%                 curr_opts.tv_lambda = 1e-4;
%                 curr_opts.beta = 2;
% L0 - preserve_class_vector- pretty good but long
%                 curr_opts.learning_rate = 1e2;
%                 curr_opts.lambda = 5e-6;
%                 curr_opts.tv_lambda = 1e-4;
%                 curr_opts.beta = 2;
% L0 - preserve_class_vector - good for 43
%                 curr_opts.learning_rate = 5e2;
%                 curr_opts.lambda = 5e-6;
%                 curr_opts.tv_lambda = 1e-4;
%                 curr_opts.beta = 2;
% L0 - preserve_class_vector - WIP for 3
%                 curr_opts.learning_rate = 1e2;
%                 curr_opts.lambda = 1e-5;
%                 curr_opts.tv_lambda = 2e-4;
%                 curr_opts.beta = 2;
% L0 - min_max_classlabel
%                 curr_opts.learning_rate = 1e2;
%                 curr_opts.lambda = 1e-5;
%                 curr_opts.tv_lambda = 2e-4;
%                 curr_opts.beta = 2;
            case 3
                curr_opts.learning_rate = 2e1;
                curr_opts.lambda = 1e-6;
                curr_opts.tv_lambda = 0;
                curr_opts.beta = 0;
        end
        if is_local 
            curr_opts.save_fig_path = fullfile('/Users/brian/neural_coding/figures/local/week8',...
                'softmaxloss_L15_mask_dim_2_init_0.5_no_truncate_mask', ...
                sprintf('%d_%s.jpg', img_i, get_short_class_name(net, target_class,false)));
        else 
            curr_opts.save_fig_path = fullfile(sprintf('/home/ruthfong/neural_coding/figures8/%s/L%d/%s_%d/lr_%f_reg_lambda_%f_tv_norm_%f_beta_%f_num_iters_%d_%s_trans_%s_adam/', ...
             dataset, layer, curr_opts.loss, curr_opts.num_class_labels, curr_opts.learning_rate, curr_opts.lambda, curr_opts.tv_lambda, ...
             curr_opts.beta, curr_opts.num_iters, curr_opts.mask_init,  ...
             curr_opts.mask_transform_function), ...
             strcat(num2str(img_i), sprintf('_mask_dim_%d.jpg', curr_opts.mask_dims)));
            curr_opts.save_res_path = fullfile(sprintf('/home/ruthfong/neural_coding/results8/%s/L%d/%s_%d/lr_%f_reg_lambda_%f_tv_norm_%f_beta_%f_num_iters_%d_%s_trans_%s_adam/', ...
             dataset, layer, curr_opts.loss, curr_opts.num_class_labels, curr_opts.learning_rate, curr_opts.lambda, curr_opts.tv_lambda, ...
             curr_opts.beta, curr_opts.num_iters, curr_opts.mask_init, ...
             curr_opts.mask_transform_function),...
             strcat(num2str(img_i), sprintf('_mask_dim_%d.mat', curr_opts.mask_dims)));
         
%             curr_opts.save_fig_path = fullfile(sprintf('/home/ruthfong/neural_coding/figures7/%s/L%d/%s_%d_ensemble_%d/lr_%f_reg_lambda_%f_tv_norm_%f_beta_%f_num_iters_%d_%s_stop_%f_trans_%s_a_%f_adam/', ...
%              dataset, layer, curr_opts.loss, curr_opts.num_class_labels, curr_opts.num_masks, curr_opts.learning_rate, curr_opts.lambda, curr_opts.tv_lambda, ...
%              curr_opts.beta, curr_opts.num_iters, curr_opts.mask_init, curr_opts.error_stopping_threshold, ...
%              curr_opts.mask_transform_function, curr_opts.gen_sig_a), ...
%              strcat(num2str(img_i), sprintf('_mask_dim_%d.jpg', curr_opts.mask_dims)));
%             curr_opts.save_res_path = fullfile(sprintf('/home/ruthfong/neural_coding/results7/%s/L%d/%s_%d_ensemble_%d/lr_%f_reg_lambda_%f_tv_norm_%f_beta_%f_num_iters_%d_%s_stop_%f_trans_%s_a_%f_adam/', ...
%              dataset, layer, curr_opts.loss, curr_opts.num_class_labels, curr_opts.num_masks, curr_opts.learning_rate, curr_opts.lambda, curr_opts.tv_lambda, ...
%              curr_opts.beta, curr_opts.num_iters, curr_opts.mask_init, curr_opts.error_stopping_threshold, ...
%              curr_opts.mask_transform_function, curr_opts.gen_sig_a),...
%              strcat(num2str(img_i), sprintf('_mask_dim_%d.mat', curr_opts.mask_dims)));


%             curr_opts.save_fig_path = fullfile(sprintf('/home/ruthfong/neural_coding/figures6/%s/L%d/%s/reg_lambda_%f_tv_norm_%f_beta_%f_%s_stop_%f_trans_%s_a_%f_style_%s_w_%s_adam/', ...
%              dataset, layer, curr_opts.loss, curr_opts.lambda, curr_opts.tv_lambda, ...
%              curr_opts.beta, curr_opts.mask_init, curr_opts.error_stopping_threshold, ...
%              curr_opts.mask_transform_function, curr_opts.gen_sig_a, ...
%              strrep(num2str(opts.sim_layernums),  '  ', '_'),...
%              strrep(num2str(opts.sim_layercoeffs), '  ', '_')), ...
%              strcat(num2str(img_i), sprintf('_mask_dim_%d.jpg', curr_opts.mask_dims)));
%             curr_opts.save_res_path = fullfile(sprintf('/home/ruthfong/neural_coding/results6/%s/L%d/%s/reg_lambda_%f_tv_norm_%f_beta_%f_%s_stop_%f_trans_%s_a_%f_style_%s_w_%s_adam/', ...
%              dataset, layer, curr_opts.loss, curr_opts.lambda, curr_opts.tv_lambda, ...
%              curr_opts.beta, curr_opts.mask_init, curr_opts.error_stopping_threshold, ...
%              curr_opts.mask_transform_function, curr_opts.gen_sig_a, ...
%              strrep(num2str(opts.sim_layernums),  '  ', '_'),...
%              strrep(num2str(opts.sim_layercoeffs), '  ', '_')),...
%              strcat(num2str(img_i), sprintf('_mask_dim_%d.mat', curr_opts.mask_dims)));
%          
            if exist(curr_opts.save_fig_path, 'file') && exist(curr_opts.save_res_path, 'file')
                fprintf('%s already exists; so skipping\n', curr_opts.save_fig_path);
                continue;
            end
         end
        
        if isempty(strfind(curr_opts.loss, 'min_max'))
%             res_mask = optimize_layer_feats(net, img, target_class, layer, curr_opts);
            res_mask = optimize_layer_feats(net, img, target_class, layer, curr_opts);
        else
            res_mask = optimize_layer_feats_multitask(net, img, target_class, layer, curr_opts);
        end
           close all;
    end
end