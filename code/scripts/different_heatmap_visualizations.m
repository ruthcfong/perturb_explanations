net = load('/home/ruthfong/packages/matconvnet/data/models/imagenet-caffe-alex.mat');
imdb_paths = load('/data/ruthfong/ILSVRC2012/val_imdb_paths.mat');

if ~isequal(net.layers{end}.type, 'softmaxloss')
    net.layers{end+1} = struct('type', 'softmaxloss') ;
end

img_i = 1;

target_class = imdb_paths.images.labels(img_i);
net.layers{end}.class = target_class;
img = cnn_normalize(net.meta.normalization, imread(imdb_paths.images.paths{img_i}), true);
display_img = cnn_denormalize(net.meta.normalization, img);

res = vl_simplenn(net, img, 1);

figure; 
subplot(2,3,1);
imshow(normalize(display_img));
title('Orig Img');

%% Simonyan et al., 2014
norm_p = @(x, p) sum(abs(x).^p,3).^(1/p); % q-norm across color channels of image

hm_2 = norm_p(res(1).dzdx, q);

subplot(2,3,2);
imshow(normalize(bsxfun(@times, display_img, hm_2)));
title('Simonyan L2-norm');

% doesn't do anything with Inf
% hm_inf = norm_p(res(1).dzdx, Inf);
% subplot(2,3,3);
% imshow(normalize(bsxfun(@times, display_img, hm_inf)));
% title('Simonyan Inf-norm');

%% LRP
rf_info = get_rf_info(net);

%%
test_layernum = 13; %4;
curr_layer = res(test_layernum).x;
next_layer = res(test_layernum+1).x;
test_spec = net.layers{test_layernum};
disp(test_spec.type);

figure;
subplot(2,2,1);
imagesc(curr_layer(:,:,1));
colorbar;
axis square;

subplot(2,2,2);
imagesc(next_layer(:,:,1));
colorbar;
axis square;


tnet = truncate_net(net, test_layernum, test_layernum);
res_test1 = vl_simplenn(tnet, curr_layer, ones(size(next_layer), 'single'));

subplot(2,2,3);
imagesc(res_test1(1).dzdx(:,:,1));
colorbar;
axis square;

res_testp = vl_simplenn(tnet, curr_layer, next_layer);
subplot(2,2,4);
imagesc(res_testp(1).dzdx(:,:,1));
colorbar;
axis square;

%%
debug = true;
num_layers = length(net.layers);
res(num_layers+1).r = 1; 

for l=num_layers:-1:1
    switch net.layers{l}
        case 'conv'
            switch lrp_rule
                case 'epsilon'
                    
                case 'alpha_beta'
                    
            end
        case 'pool'
            tnet = truncate_net(net, l, l);
            res_pool = vl_simplenn(tnet, res(l).x, res(l+1).r);
            res(l).r = res_pool(1).dzdx;
        otherwise
            % relu, lrn/norm, softmax -- do nothing; just propagate directly back
            if isequal(size(res(l).x), size(res(l+1).x))
                res(l).r = res(l+1).r;
            % loss layer, distribute equally over curr layer's shape
            else
                assert(numel(res(l+1).x) == 1);
                res(l).r = ones(size(res(l).x, 'single'));
                res(l).r = res(l+1).x / numel(res(l).x) * res(l).r;
            end
    end
    
    % maintain conservation principle
    assert(sum(res(l).r(:)) == sum(res(l+1).r(:)));
    
    if debug
        disp(net.layers{l});
    end
    assert(isequal(size(res(l).r), size(res(l).x)));
end

%%
figure; 
subplot(1,3,1); 
imagesc(net.layers{13}.weights{1}(:,:,1,1)); 
colorbar;
axis square;
subplot(1,3,2); 
imagesc(res(13).x(:,:,1));
colorbar;
axis square;
subplot(1,3,3); 
imshow(res(14).x(:,:,1));
colorbar;
axis square;
res(l).x.*