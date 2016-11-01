net_path = '../models/places-caffe-ref-upgraded-tidy.mat';

net = load(net_path);

%im = imread('../data/gsun_fff7749c15474a28ebea4b4ad0c97cb6.jpg');
im_ = single(im);
im_ = imresize(im_, net.meta.normalization.imageSize(1:2));
im_ = im_ - net.meta.normalization.averageImage;

target_y = zeros([1 1 205]);
target_y(3) = single(1);
res = vl_simplenn(net, im_);

[error, dzdy] = loss(res(end).x, target_y);

res = vl_simplenn(net, im_,dzdy);

% show the classification result
scores = squeeze(gather(res(end).x)) ;
[bestScore, best] = max(scores) ;
%figure(1) ; clf ; imagesc(im) ; axis image ;
% title(sprintf('%s (%d), score %.3f',...
% net.meta.classes.description{best}, best, bestScore));
%title(sprintf('score %.3f',bestScore));

%vl_simplenn_display(net);
layernum = 14; % conv5

x = res(layernum).x;
x_gap = mean(x,3);
x_gap_norm = (x_gap - min(min(x_gap)))/(max(max(x_gap)) - min(min(x_gap)));

dzdx = res(layernum).dzdx;
dzdx_gap = mean(dzdx,3);
dzdx_gap_norm = (dzdx_gap - min(min(dzdx_gap)))/(max(max(dzdx_gap)) - min(min(dzdx_gap))); 

weighted_x = x .* dzdx;
weighted_x_gap = mean(weighted_x, 3);
weighted_x_norm = (weighted_x_gap - min(min(weighted_x_gap)))/(max(max(weighted_x_gap))-min(min(weighted_x_gap)));


figure; 
subplot(1,3,1); 
imshow(x_gap_norm);
subplot(1,3,2);
imshow(dzdx_gap_norm);
subplot(1,3,3);
imshow(weighted_x_norm);

gradient = weighted_x;
gradient_gap = mean(gradient, 3);
gradient_norm = (gradient_gap - min(min(gradient_gap)))/(max(max(gradient_gap)) - min(min(gradient_gap)));

snet = create_scaling_net(net_path);
snet_res = pass_through_scaling_net(snet, im_, layernum - 1, gradient);

dzdx_s = snet_res(1).dzdx;
dzdx_s_norm = (dzdx_s - min(min(min(dzdx_s))))/(max(max(max(dzdx_s))) - min(min(min(dzdx_s))));

dzdx_size = size(dzdx);
gradient_gap_rep = repmat(gradient_gap, [1 1 dzdx_size(3)]);
snet_res_2 = pass_through_scaling_net(snet, im_, layernum - 1, gradient_gap_rep);

dzdx_s_gap = snet_res_2(1).dzdx;
dzdx_s_gap_norm = (dzdx_s_gap - min(min(min(dzdx_s_gap))))/(max(max(max(dzdx_s_gap))) - min(min(min(dzdx_s_gap))));

figure;
subplot(2,2,1);
imshow(im_);
subplot(2,2,2);
imshow(gradient_norm);
subplot(2,2,3);
imshow(dzdx_s_norm);
subplot(2,2,4);
imshow(dzdx_s_gap_norm);

figure;
subplot(1,3,1);
dzdx_s_norm_mean = mean(dzdx_s_norm,3);

function [error, dzdy] = loss(pred, target)
    error = sum((target - pred).^2);
    dzdy = -2*(target-pred);
end