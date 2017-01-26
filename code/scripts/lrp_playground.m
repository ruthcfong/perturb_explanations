net = load('/home/ruthfong/packages/matconvnet/data/models/imagenet-caffe-alex.mat');
imdb_paths = load('/data/ruthfong/ILSVRC2012/val_imdb_paths.mat');

img_i = 3; % 3, 6, and 13 look off
img_path = imdb_paths.images.paths{img_i};
target_class = imdb_paths.images.labels(img_i);

img = imread(img_path);
img_ = cnn_normalize(net.meta.normalization, img, 1);

res = vl_simplenn(net, img_, 1);

opts = {};
opts.lrp_epsilon = 100;
heatmap = compute_heatmap(net, img_, target_class, 'lrp_epsilon', 0, opts);
%figure; imshow(normalize(bsxfun(@times, img_, heatmap)));
%figure; imshow(normalize(bsxfun(@times, single(img), imresize(heatmap, [size(img,1), size(img,2)]))));
figure; imshow(vl_imsc_am(heatmap));
%%
% res(l).x is H_in x W_in x K_in x N
% res(l+1).x is H_out x W_out x K_out
% net.layers{l}
% weights{1} is H_w x W_w x K_in x K_out
% weights{2} is K_out x 1

% l = 1;
% 
% assert(strcmp(net.layers{l}.type, 'conv'));
% W = net.layers{l}.weights{1};
% b = net.layers{l}.weights{2};
% [H_in,W_in,K_in,~] = size(res(l).x);
% [H_out,W_out,K_out,~] = size(res(l+1).x);
% 
% hstride = net.layers{l}.stride(1);
% wstride = net.layers{l}.stride(2);
% [hf, wf, df, nf] = size(W);
% 
% relevance = zeros(size(res(l).x), 'single');
% 
% next_relevance = res(l+1).dzdx;
% 
% epsilon = 0.01; % parameterize
% 
% for h=1:H_out
%     for w=1:W_out
%         x = res(l).x((h-1)*hstride+1:(h-1)*hstride+hf,(w-1)*wstride+1:(w-1)*wstride+wf,:); % [hf, wf, d_in]
%         x = repmat(x, [1 1 1 nf]); % [hf, wf, d_in, nf]
%         Z = W .* x; % [hf, wf, df, nf]
%         
%         Zs = sum(sum(sum(Z,1),2),3); % [1 1 1 nf] (convolution summing here)
%         Zs = Zs + reshape(b, size(Zs));
%         Zs = Zs + epsilon*sign(Zs);
%         Zs = repmat(Zs, [hf, wf, df, 1]);
%         
%         zz = Z ./ Zs;
%         
%         rr = repmat(reshape(next_relevance(h,w,:), [1 1 1 nf]), [hf, wf, df, 1]); % [hf, wf, df, nf]
%         %%
%         rx = relevance((h-1)*hstride+1:(h-1)*hstride+hf,(w-1)*wstride+1:(w-1)*wstride+wf,:);
%         relevance((h-1)*hstride+1:(h-1)*hstride+hf,(w-1)*wstride+1:(w-1)*wstride+wf,:) = ...
%             rx + sum(zz .* rr, 4);
%     end
% end


% relevance = zeros(size(res(l).x), 'single');
% weighted_activations = zeros([size(res(l+1).x) K_in], 'single');
% for i=1:K_in
%     for j=1:K_out
%         conv_layer = net.layers{l};
%         conv_layer.weights = {W(:,:,i,j) b(j,:)};
%         nnet = {};
%         nnet.layers{1} = conv_layer;
%         nres = vl_simplenn(nnet, res(l).x(:,:,i));
%         weighted_activations(:,:,j,i) = nres(end).x;
%     end
% end
% 
% %%
% relevance = zeros([H_out, W_out, K_in], 'single');
% next_relevance = res(l+1).dzdx;
% summed_weighted_activations = sum(weighted_activations, 4);
% epsilon = 0.01;
% for i=1:K_in
%     temp = zeros([H_out, W_out], 'single');
%     for j=1:K_out
%         temp = temp + ...
%             (weighted_activations(:,:,j,i) ...
%             ./ (summed_weighted_activations(:,:,j) ...
%             + epsilon*sign(summed_weighted_activations(:,:,j)))) ...
%             .* next_relevance(:,:,j);
%     end
%     relevance(:,:,i) = temp;
% end
