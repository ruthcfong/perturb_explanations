net = load('/home/ruthfong/packages/matconvnet/data/models/imagenet-caffe-alex.mat');

use_training_heldout = true;

if use_training_heldout
    imdb_paths = load('/data/ruthfong/ILSVRC2012/train_imdb_paths.mat');
    all_img_idx = load('/data/ruthfong/ILSVRC2012/training_5000_heldout_idx.mat');
    all_img_idx = all_img_idx.heldout_idx;
else
    imdb_paths = load('/data/ruthfong/ILSVRC2012/val_imdb_paths.mat');
    all_img_idx = 1:50000;
end

heatmap_type = 'saliency';

if use_training_heldout
    for alpha=0:0.5:10
        out_file = sprintf('/data/ruthfong/ILSVRC2012/loc_preds/%s_alpha_%.1f.txt', ...
            heatmap_type, alpha);
        localization_with_heatmaps(net, imdb_paths, all_img_idx, alpha, heatmap_type, out_file);
    end
else
    alpha = 5;
    out_file = sprintf('/data/ruthfong/ILSVRC2012/loc_preds/%s_alpha_%.1f.txt', ...
        heatmap_type, alpha);
    localization_with_heatmaps(net, imdb_paths, all_img_idx, alpha, heatmap_type, out_file);
end