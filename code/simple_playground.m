net = load('/home/ruthfong/neural_coding/models/places-caffe-ref-upgraded.mat');

im = imread('/data/datasets/places205/images256/a/alley/gsun_fff7749c15474a28ebea4b4ad0c97cb6.jpg');
im_ = single(im);
im_ = imresize(im_, net.meta.normalization.imageSize(1:2));
im_ = im_ - net.meta.normalization.averageImage;

vl_simplenn(net, im_);

% show the classification result
scores = squeeze(gather(res(end).x)) ;
[bestScore, best] = max(scores) ;
figure(1) ; clf ; imagesc(im) ; axis image ;
title(sprintf('%s (%d), score %.3f',...
net.classes.description{best}, best, bestScore)) ;