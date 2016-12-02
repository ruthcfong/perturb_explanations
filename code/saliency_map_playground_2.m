%% set parameters
load_network = false;
load_data = false;
run_on_batch = false;
show_figures = false;
is_local = true;
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
                imdb_paths = load('/data/ruthfong/ILSVRC2012/val_imdb_paths_small.mat');
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
layer = 15;

img_idx = 1:10;
%img_idx = [6];

opts = struct();
%opts.batch_range = batch_range;
%opts.class_offset = class_offset;
%opts.null_img = null_img;
opts.num_iters = 70;
opts.plot_step = 10;
opts.debug = true;
opts.save_fig_path = '';
opts.save_res_path = '';
opts.loss = 'softmaxloss';
opts.denom_reg = false;

for i=1:length(img_idx),
    curr_opts = opts;
    img_i = img_idx(i);
    if is_local
        [~, filename, ext] = fileparts(imdb_paths.images.paths{img_i});
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
                curr_opts.learning_rate = 2e1;
                curr_opts.lambda =2.2e-3; %1e-3; %5e-5;
                curr_opts.tv_lambda = 1e-3;
                curr_opts.beta = 1;
            case 3
                curr_opts.learning_rate = 2e1;
                curr_opts.lambda = 1e-6;
                curr_opts.tv_lambda = 0;
                curr_opts.beta = 0;
        end
        
        curr_opts.save_fig_path = fullfile('/Users/brian/neural_coding/figures/local/week8',...
            'softmaxloss_L15_mask_dim_2_init_0.5_no_truncate_mask', ...
            sprintf('%d_%s.jpg', img_i, get_short_class_name(net, target_class,false)));
%         curr_opts.save_fig_path = fullfile(sprintf('/home/ruthfong/neural_coding/figures5/%s/L%d/%s/reg_lambda_%f_tv_norm_%f_beta_%f/', ...
%          dataset, layer, curr_opts.loss, curr_opts.lambda, curr_opts.tv_lambda, curr_opts.beta), ...
%          strcat(num2str(img_i), sprintf('_mask_dim_%d.jpg', curr_opts.mask_dims)));
%         curr_opts.save_res_path = fullfile(sprintf('/home/ruthfong/neural_coding/results5/%s/L%d/%s/reg_lambda_%f_tv_norm_%f_beta_%f/', ...
%          dataset, layer, curr_opts.loss, curr_opts.lambda, curr_opts.tv_lambda, curr_opts.beta), ...
%          strcat(num2str(img_i), sprintf('_mask_dim_%d.mat', curr_opts.mask_dims)));

        res_mask = optimize_layer_feats(net, img, target_class, layer, curr_opts);
        
%         close all;
    end
end


%%
% figure;
% for i=1:9
%     subplot(3,3,i);
%     mask = mask_t(:,:,:,i);
%     x = actual_feats .* mask + null_feats .* (1 - mask);
%     curr_feat_map = x(:,:,max_feature_idx);
%     curr_diff_map = actual_feat_map - curr_feat_map;
%     curr_saliency_map = get_saliency_map_from_difference_map(curr_diff_map, layer, rf_info, img_size);
%     curr_saliency_map_rep = repmat(normalize(curr_saliency_map),[1 1 3]);
%     imshow(normalize((display_img+imdb.images.data_mean).*curr_saliency_map_rep));
% end
% %% plot
% figure;
% subplot(1,3,1);
% imshow(normalize(img+imdb.images.data_mean));
% title('Orig Img')
% subplot(1,3,2);
% imshow(normalize(saliency_map));
% title('Sal Map')
% subplot(1,3,3);
% saliency_map_rep = repmat(normalize(saliency_map),[1 1 3]);
% mask = saliency_map_rep > 0.5;
% anti_mask = saliency_map_rep <= 0.5;
% display_img = zeros(img_size,'single');
% display_img(mask) = img(mask);
% display_img(anti_mask) = null_img(anti_mask);
% imshow(normalize(display_img+imdb.images.data_mean));
% title('Dot Product');
