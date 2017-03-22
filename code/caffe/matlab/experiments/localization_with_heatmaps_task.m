net_type = 'googlenet';
is_training = false;
use_heldout = false;
gpu = 0;
heatmap_types = {'saliency','guided_backprop','excitation_backprop'};
alphas = [5.0, 4.5, 1.5];
batch_size = 50;

caffe_model_dir = '/home/ruthfong/packages/caffe/models';
switch net_type
    case 'alexnet'
        model_dir = fullfile(caffe_model_dir, 'bvlc_reference_caffenet');
        net_model = fullfile(model_dir,'deploy_force_backward.prototxt');
        net_weights = fullfile(model_dir,'bvlc_reference_caffenet.caffemodel');
    case 'vgg16'
        model_dir = fullfile(caffe_model_dir, 'vgg16');
        net_model = fullfile(model_dir, 'VGG_ILSVRC_16_layers_deploy_force_backward.prototxt');
        net_weights = fullfile(model_dir, 'VGG_ILSVRC_16_layers.caffemodel');
    case 'googlenet'
        model_dir = fullfile(caffe_model_dir, 'bvlc_googlenet');
        net_model = fullfile(model_dir, 'deploy_force_backward.prototxt');
        net_weights = fullfile(model_dir, 'bvlc_googlenet.caffemodel');
    otherwise
        error('%s net type is not supported', net_type);
end
net = caffe.Net(net_model, net_weights, 'test');

if is_training
    imdb_paths = load('/data/ruthfong/ILSVRC2012/annotated_train_imdb_paths.mat');
    all_img_idx = load('/vdata/ruthfong/ILSVRC2012/annotated_train_heldout_idx.mat');
    all_img_idx = all_img_idx.heldout_idx;
else
    imdb_paths = load('/data/ruthfong/ILSVRC2012/val_imdb_paths.mat');
    if use_heldout
        all_img_idx = [1,2,3,5,6,7,8,12,14,18,20,21,27,37,41,57,61,70,76,91];
    else
        all_img_idx = 1:50000;
    end
end

paths = imdb_paths.images.paths(all_img_idx);
labels = imdb_paths.images.labels(all_img_idx);

opts = struct();
softmax_i = find(cellfun(@(l) strcmp(l, 'prob'), net.layer_names));
if ~isempty(softmax_i)
    opts.start_layer = net.layer_names{softmax_i-1};
else
    opts.start_layer = net.layer_names{end};
end
opts.end_layer = net.layer_names{1};

opts.indexing = importdata('/home/ruthfong/packages/caffe/data/ilsvrc12/ascii_order_to_synset_order.txt');
opts.batch_size = batch_size;
opts.default_img_size = get_net_default_img_size(net_type);
opts.mean_img = caffe.io.read_mean('/home/ruthfong/packages/caffe/data/ilsvrc12/imagenet_mean.binaryproto');
opts.gpu = gpu;
opts.norm_deg = Inf;

if use_heldout
    for j=1:length(heatmap_types)
        heatmap_type = heatmap_types{j};
        alphas = 0:0.5:10;
        curr_opts = opts;
        if strcmp(heatmap_type, 'excitation_backprop')
            curr_opts.norm_deg = -1;
            switch net_type
                case 'alexnet'
                    curr_opts.end_layer = 'conv5'; % setting for CNN-S, an improved version of AlexNet
                case 'vgg16'
                    curr_opts.end_layer = 'pool4';
                case 'googlenet'
                    curr_opts.end_layer = 'pool2/3x3_s2';
            end
        end
        for i=1:length(alphas)
            alpha = alphas(i);
            out_file = sprintf('/data/ruthfong/neural_coding/loc_preds/%s_annotated_train_heldout_gt_caffe/%s/alpha_%.1f%s_norm_deg_%d.txt', ...
                net_type, heatmap_type, alpha, curr_opts.end_layer, curr_opts.norm_deg);
            if exist(out_file, 'file')
                fprintf('skipping %s because it already exists\n', out_file);
                continue;
            end
            localization_with_heatmaps(net, paths, labels, alpha, heatmap_type, out_file, curr_opts);
        end
    end
else
    for j=1:length(heatmap_types)
        heatmap_type = heatmap_types{j};
        alpha = alphas(j);
        curr_opts = opts;
        if strcmp(heatmap_type, 'excitation_backprop')
            curr_opts.norm_deg = -1;
            switch net_type
                case 'alexnet'
                    curr_opts.end_layer = 'conv5'; % setting for CNN-S, an improved version of AlexNet
                case 'vgg16'
                    curr_opts.end_layer = 'pool4';
                case 'googlenet'
                    curr_opts.end_layer = 'pool2/3x3_s2';
            end
        end
        out_file = sprintf('/data/ruthfong/neural_coding/loc_preds/%s_val_gt_caffe/%s/alpha_%.1f%s_norm_deg_%d.txt', ...
            net_type, heatmap_type, alpha, curr_opts.end_layer, curr_opts.norm_deg);
        if exist(out_file, 'file')
            fprintf('skipping %s because it already exists\n', out_file);
            continue;
        else
            localization_with_heatmaps(net, paths, labels, alpha, heatmap_type, out_file, curr_opts);
        end
    end
end

caffe.reset_all();