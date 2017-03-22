load_network = true;
load_dataset = true;
is_local = false;
save_figure = false;
delete_window = false;

softmaxscore = @(target_class, feats) exp(feats(target_class))/sum(exp(feats));
softmaxloss = @(target_class, feats) -log(softmaxscore(target_class, feats));

if load_network
    if is_local
        % TODO
        assert(false);
    else
        net = load('/home/ruthfong/packages/matconvnet/data/models/imagenet-caffe-alex.mat');
    end
end

if load_dataset
    if is_local
        assert(false); % TODO
    else
        imdb_paths = load('/data/ruthfong/ILSVRC2012/val_imdb_paths.mat');
    end
end

% class_i = 818; % sports car
% image_idx_for_class = find(imdb_paths.images.labels == class_i);
% img_i = image_idx_for_class(27);

%img_i = 1;

for img_i=1:1000,
orig_img = imread(imdb_paths.images.paths{img_i});
disp(imdb_paths.images.paths{img_i});
norm_img = cnn_normalize(net.meta.normalization, orig_img, true);

img_size = size(norm_img);

% figure; imshow(orig_img);

% add a softmaxloss layer if needed
if ~isequal(net.layers{end}.type, 'softmaxloss')
    net.layers{end+1} = struct('type', 'softmaxloss') ;
end

target_class = imdb_paths.images.labels(img_i);
net.layers{end}.class = target_class;
res = vl_simplenn(net, norm_img);

orig_output = res(end-1).x;

class_split = strsplit(net.meta.classes.description{target_class},',');
target_class_name = strrep(class_split{1}, ' ', '_');
f = figure;
subplot(2,3,1);
imshow(normalize(norm_img));
title(sprintf('%d: %s (%.4f)', img_i, strrep(target_class_name, '_', '\_'), ...
    softmaxscore(target_class, res(end-1).x)));

layers = [2,6, 10, 12, 14]; % relu 3-5 layers for alexnet
occ_size = [5, 3, 1, 1, 1];
occ_stride = [5, 3, 1, 1, 1];

for i=1:length(layers)
    layer = layers(i);
    
    % prepare spatially occluded feats
    size_feats = size(res(layer+1).x);
    num_occs_in_side = size_feats(1)/occ_stride(i);
    if delete_window
        occ_feats = repmat(res(layer+1).x, [1, 1, 1, num_occs_in_side^2]);
    else
        occ_feats = repmat(zeros(size_feats, 'single'), [1, 1, 1, num_occs_in_side^2]);
    end
    for r=1:num_occs_in_side
        for c=1:num_occs_in_side
            start_r = (r-1)*occ_stride(i)+1;
            end_r = start_r + occ_size(i) - 1;
            start_c = (c-1)*occ_stride(i)+1;
            end_c = start_c + occ_size(i) - 1;
            if delete_window
                occ_feats(start_r:end_r,start_c:end_c,:,(r-1)*num_occs_in_side+c) = 0;
            else
                occ_feats(start_r:end_r,start_c:end_c,:,(r-1)*num_occs_in_side+c) = ...
                    res(layer+1).x(start_r:end_r,start_c:end_c,:);
            end
        end
    end

    tnet = truncate_net(net, layer+1, length(net.layers));
    tnet.layers{end}.class = repmat(target_class,[1 num_occs_in_side^2]);
    tres = vl_simplenn(tnet, occ_feats, 1);

    occ_output = tres(end-1).x;

    orig_score = softmaxscore(target_class, orig_output);
    occ_map = zeros([num_occs_in_side num_occs_in_side]);
    for r=1:num_occs_in_side
        for c=1:num_occs_in_side
            id = (r-1)*num_occs_in_side+c;
            occ_map(r,c) = orig_score - softmaxscore(target_class, occ_output(:,:,:,id));
        end
    end

    occ_heatmap = map2jpg(im2double(imresize(occ_map, img_size(1:2), 'nearest')));
    
    subplot(2,3,i+1);
    imshow(normalize(norm_img)*0.5 + 0.5*occ_heatmap);
    title(net.layers{layer}.name);
    
end

drawnow;

if save_figure
    fig_path = fullfile('/home/ruthfong/neural_coding/figures5/imagenet/bruteforce_occlusion_exclude', ...
        sprintf('%s_%d.jpg', target_class_name, img_i));
    print(f, fig_path, '-djpeg');
end

% [~,sorted_idx] = sort(res(end-1).x,'descend');
% for i=1:5
%    disp(sprintf('%d - %s %.4f', i, net.meta.classes.description{sorted_idx(i)}, ...
%        softmaxscore(sorted_idx(i), res(end-1).x)));
% end

end