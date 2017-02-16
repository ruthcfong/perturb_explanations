net_type = 'alexnet';
use_train = false;
use_heldout = true;
mask_dir = '/data/ruthfong/neural_coding/results10/imagenet/alexnet_val_heldout/L0/min_classlabel_5_direct_blur/lr_1.000000_reg_lambda_-7.301030_tv_norm_-5.301030_beta_1.500000_num_iters_500_noise_1_jitter_10_adam';

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

if use_train
    imdb_paths = load('/data/ruthfong/ILSVRC2012/annotated_train_imdb_paths.mat');
    img_idx = load('/data/ruthfong/ILSVRC2012/annotated_train_heldout_idx.mat');
    img_idx = img_idx.heldout_idx;
    dataset_description = 'annotated_train_heldout';
else
    imdb_paths = load('/data/ruthfong/ILSVRC2012/val_imdb_paths.mat');
    if use_heldout
        img_idx = [1,2,5,8,3];
        %img_idx = [1,2,5,8,3,6,7,20,57,12,14,18,21,27,37,41,61,70,76,91];
        dataset_description = 'val_heldout';
    else
        img_idx = 1:50000;
        dataset_description = 'val';
    end
end

heatmap_types = {'mask', 'saliency', 'deconvnet', 'guided_backprop', 'lrp_epsilon', 'lrp_alpha_beta'};
norm_deg = Inf;
flip_mask = true;

num_imgs = length(img_idx);
num_heatmaps = length(heatmap_types);

%figure;
for i=1:num_imgs
    img_i = img_idx(i);
    img = cnn_normalize(net.meta.normalization, imread(imdb_paths.images.paths{img_i}), 1);
    target_class = imdb_paths.images.labels(img_i);
    %subplot(num_imgs, num_heatmaps + 1, (i-1)*(num_heatmaps+1) + 1);
    figure;
    subplot(1, num_heatmaps+1, 1);
    imshow(uint8(cnn_denormalize(net.meta.normalization, img)));
    title('Orig');
    for j=1:num_heatmaps
        heatmap_type = heatmap_types{j};
        if strcmp(heatmap_type, 'mask')
            mask_res = load(fullfile(mask_dir, [num2str(img_i) '.mat']));
            mask = mask_res.new_res.mask;
            if flip_mask, mask = 1-mask; end
            heatmap = mask;
        else
            heatmap = compute_heatmap(net, img, target_class, heatmap_type, norm_deg);
        end
        %subplot(num_imgs, num_heatmaps + 1, (i-1)*(num_heatmaps+1) + j + 1);
        subplot(1, num_heatmaps+1, j + 1);
        imshow(normalize(bsxfun(@times, cnn_denormalize(net.meta.normalization, img), heatmap)));
        title(heatmap_type);
    end
end
