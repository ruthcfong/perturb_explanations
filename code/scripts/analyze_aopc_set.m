net = load('/home/ruthfong/packages/matconvnet/data/models/imagenet-caffe-alex.mat');
save_res_dir = '/home/ruthfong/neural_coding/results8/imagenet/L0/min_classlabel_5/aopc_iters_30_perturbs_1_with_pert_imgs_lr_10.000000_reg_lambda_0.000000_tv_norm_0.001000_beta_3.000000_num_iters_350_rand_trans_linear_adam/';
filenames = dir(save_res_dir);
filenames = filenames(3:end);

load(fullfile(save_res_dir, filenames(1).name));
num_iters = length(res.mask.diff_scores);
num_examples = length(filenames);
num_heatmaps = 6;

diff_scores = zeros([num_iters num_examples num_heatmaps], 'single');
for i=1:num_examples
   load(fullfile(save_res_dir, filenames(i).name));
   
   diff_scores(:,i,1) = res.mask.diff_scores;
   diff_scores(:,i,2) = res.saliency.diff_scores;
   diff_scores(:,i,3) = res.deconvnet.diff_scores;
   diff_scores(:,i,4) = res.guided.diff_scores;
   diff_scores(:,i,5) = res.lrp_epsilon.diff_scores;
   diff_scores(:,i,6) = res.random.diff_scores;
end

aopcs = bsxfun(@rdivide, squeeze(mean(cumsum(diff_scores, 1),2))',(1+(1:num_iters)))';

divisor = repmat((1+(1:num_iters))',[1 num_examples num_heatmaps]);
ind_aopcs = cumsum(diff_scores, 1)./divisor;

figure; 
plot(bsxfun(@minus, aopcs(:,1:end), aopcs(:,end)));
legend({'mask','sal','deconvnet','guided','lrp','rand'});

[~,sorted_idx] = sort(ind_aopcs(end,:,2) - ind_aopcs(end,:,1), 'descend');

for i=1:5
    ind = sorted_idx(i);
    load(fullfile(save_res_dir, filenames(ind).name));
    
    heatmap_mask = res.mask.heatmap;
    heatmap_sal = res.saliency.heatmap;
    img = imread(res.image.path);
    img_ = cnn_normalize(net.meta.normalization, img, 1);
    figure;
    subplot(3,3,1);
    imshow(normalize(img_));
    subplot(3,3,2);
    imshow(normalize(bsxfun(@times, img_, normalize(heatmap_mask))));
    subplot(3,3,3);
    imshow(normalize(bsxfun(@times, img_, normalize(heatmap_sal))));
    subplot(3,3,4);
    plot(bsxfun(@minus, squeeze(ind_aopcs(:,ind,:)), squeeze(ind_aopcs(:,ind,end))));
    legend({'mask','sal','deconvnet','guided','lrp','rand'});
    axis square;
    subplot(3,3,5);
    imshow(normalize(img_)*0.5 + map2jpg(im2double(normalize(heatmap_mask)))*0.5);
    subplot(3,3,6);
    imshow(normalize(img_)*0.5 + map2jpg(im2double(normalize(heatmap_sal)))*0.5);
    subplot(3,3,7);
    if isfield(res.random, 'pert_img')
        imshow(res.random.pert_img);
        title('rand');
    end
    subplot(3,3,8);
    if isfield(res.mask, 'pert_img')
        imshow(res.mask.pert_img);
        title('mask');
    end
    subplot(3,3,9);
    if isfield(res.saliency, 'pert_img')
        imshow(res.saliency.pert_img);
        title('sal');
    end
end