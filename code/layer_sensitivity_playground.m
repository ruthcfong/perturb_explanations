load_network = true;
is_local = false;
if load_network
    if is_local
        net = load('~/neural_coding/models/places-caffe-ref-upgraded-tidy-with-classes.mat');
    else
        net = load('/home/ruthfong/neural_coding/models/places-caffe-ref-upgraded-tidy-with-classes.mat');
    end
end

if is_local
    orig_img = imread('/Users/brian/neural_coding/data/gsun_fff7749c15474a28ebea4b4ad0c97cb6.jpg');
else
    orig_img = imread('/data/datasets/places205/images256/a/alley/gsun_fff7749c15474a28ebea4b4ad0c97cb6.jpg');
end
norm_img = cnn_normalize(net.meta.normalization, orig_img, true);

%% color experiment

%size_orig_img = size(orig_img);

gray_orig_img = repmat(rgb2gray(orig_img), [1 1 3]);
gray_norm_img = cnn_normalize(net.meta.normalization, gray_orig_img, true);

%% show figure for difference between pre- and post-norm grayscale
% figure; 
% subplot(1,3,1); 
% imshow(gray_orig_img); 
% title('Pre-Norm rgb2gray GS'); 
% 
% subplot(1,3,2); 
% imshow(normalize(repmat(mean(norm_img,3),[1 1 3]))); 
% title('Post-Norm mean GS'); 
% 
% subplot(1,3,3); 
% imagesc(gray_norm_img-repmat(mean(norm_img,3),[1 1 3])); 
% axis square;
% title('Diff btwn Pre- and Post- Norm');

%% forward pass
rgb_res = vl_simplenn(net, norm_img); 
gray_res = vl_simplenn(net, gray_norm_img);

%% 
%layer = 14;
layers = [2, 6, 10, 12, 14];
figure;
subplot(3,3,1);
imshow(normalize(norm_img));
title('RGB Img');

subplot(3,3,2);
imshow(normalize(gray_norm_img));
title('Gray Img');

for i=1:length(layers)
    layer = layers(i);
    layer_name = net.layers{layer}.name;
    
    size_feats = size(rgb_res(layer+1).x);

    diff_feats = reshape(rgb_res(layer+1).x - gray_res(layer+1).x, ...
        [prod(size_feats(1:2)), size_feats(3)]);

    abs_mean_act_diffs = abs(mean(diff_feats, 1));
    std_act_diffs = std(diff_feats, 1);
    [~, sorted_idx] = sort(abs_mean_act_diffs);
    
    mean_std = mean(std_act_diffs);
    std_std = std(std_act_diffs);
    
    %[h,p] = jbtest(mean_act_diffs); % reject null hypothesis at all layers
    
    subplot(3,3,i+2);
    bar(1:size_feats(end), abs_mean_act_diffs(sorted_idx));
    hold on;
    errorbar(1:size_feats(end), abs_mean_act_diffs(sorted_idx), ...
        std_act_diffs(sorted_idx)/sqrt(prod(size_feats(1:2))), '.');
    hold off;
    title(sprintf('%s (Mean Std=%.2f, Std Std= %.2f)', layer_name, mean_std, std_std));
    ylabel('Abs Mean Act Diff');
    xlabel(sprintf('Sorted %s Filters', layer_name));
end

