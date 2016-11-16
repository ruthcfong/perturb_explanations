%% set parameters
load_network = false;
load_data = false;
run_on_batch = false;
show_figures = false;
is_local = false;
dataset = 'places';


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
    otherwise
        assert(false);
end

%% load variables

% load network
if load_network
    if strcmp(dataset, 'mnist')
        if is_local
            load('/Users/brian/matconvnet-1.0-beta21/data/trained_nets/mnist-net.mat');
        else
            load('/home/ruthfong/packages/matconvnet/data/mnist-baseline-simplenn/net-final.mat');
        end
    else
        net = load('/home/ruthfong/neural_coding/models/places-caffe-ref-upgraded-tidy-with-classes.mat');
    end
end

% load images
if load_data
    if strcmp(dataset, 'mnist')
        if is_local
            imdb = load('/Users/brian/matconvnet-1.0-beta21/data/mnist/imdb.mat');
        else
            imdb = load('/home/ruthfong/packages/matconvnet/data/mnist-baseline-simplenn/imdb.mat');
        end
    else
        imdb = load('/data/datasets/places205/imdb_val_resized_227.mat');
    end
end

%% forward and backward pass

if run_on_batch
    % for places alexnet, add a loss layer
    if strcmp(dataset, 'places') && ~isequal(net.layers{end}.type, 'softmaxloss')
        net.layers{end+1} = struct('type', 'softmaxloss') ;
    end

    net.layers{end}.class = imdb.images.labels(batch_range) + class_offset;
    res = vl_simplenn(net, imdb.images.data(:,:,:,batch_range),1);
end
%% load variables
null_img = mean(imdb.images.data(:,:,:,batch_range),4);
layer = 14;

img_idx = [1 12 21 32 42 52 61 79 83];
%img_idx = [32];

opts = struct();
opts.batch_range = batch_range;
opts.class_offset = class_offset;
opts.null_img = null_img;
opts.num_iters = 500;
opts.mask_dims = 3;
switch opts.mask_dims
    case 1
        opts.learning_rate = 2e1;
        opts.lambda = 5e-4;
    case 2
        opts.learning_rate = 2e1;
        opts.lambda = 5e-5;
    case 3
        opts.learning_rate = 2e1;
        opts.lambda = 1e-6;
end
opts.plot_step = 50;
opts.debug = true;
opts.save_fig_path = '';
opts.save_res_path = '';
for i=1:length(img_idx),
    curr_opts = opts;
    img_i = img_idx(i);
    curr_opts.img_i = img_i;
     curr_opts.save_fig_path = fullfile('/home/ruthfong/neural_coding/figures3/places/L14/softmaxloss/figures/', ...
         strcat(num2str(img_i), '_mask_dim_3.jpg'));
     curr_opts.save_res_path = fullfile('/home/ruthfong/neural_coding/figures3/places/L14/softmaxloss/results/', ...
         strcat(num2str(img_i), '_mask_dim_3.mat'));
    res_mask = optimize_layer_feats(net, imdb, res, layer, curr_opts);
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
