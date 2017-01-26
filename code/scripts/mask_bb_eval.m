function mask_bb_eval()
net = load('/home/ruthfong/packages/matconvnet/data/models/imagenet-caffe-alex.mat');
imdb_paths = load('/data/ruthfong/ILSVRC2012/val_imdb_paths.mat');
bb_dir = '/data/ruthfong/ILSVRC2012/val';
mask_res_dir = '/home/ruthfong/neural_coding/results7/imagenet/L0/min_classlabel_5/reg_lambda_0.000000_tv_norm_0.001000_beta_3.000000_num_iters_500_rand_stop_-100.000000_trans_linear_a_50.000000_adam/';
save_dir = strrep(strrep(mask_res_dir, 'results','figures'), 'lr', 'analysis_lr');
mask_dims = 2;

if isempty(strfind(mask_res_dir, 'max_softmaxloss')) ...
        && isempty(strfind(mask_res_dir, 'min_classlabel'))
    flip_mask = false;
else
    flip_mask = true;
end

img_idx = [1,2,5,8,3,6,7,20,57,12,14,18,21,27,37,41,61,70,76,91];
%for img_i=1:(length(dir(mask_res_dir))-3)
for img_i=img_idx
    img_path = imdb_paths.images.paths{img_i};
    target_class = imdb_paths.images.labels(img_i);
    img = imread(img_path);

    [~,filename,~] = fileparts(img_path);

    bb_path = fullfile(bb_dir, strcat(filename, '.xml')); 
    rec = VOCreadxml(bb_path);
    bb = rec.annotation.object.bndbox;
    bb_disp = [str2double(bb.xmin)+1, str2double(bb.ymin)+1, str2double(bb.xmax)-str2double(bb.xmin), ...
        str2double(bb.ymax)-str2double(bb.ymin)];

    mask_res = load(fullfile(mask_res_dir, sprintf('%d_mask_dim_%d.mat', img_i, mask_dims)));
    mask_res = mask_res.new_res;

    save_path = fullfile(save_dir, sprintf('%d_mask_dim_%d.jpg', img_i, mask_dims));
    generate_fig(net, img, target_class, bb_disp, mask_res.mask, flip_mask, save_path);
end

end

function generate_fig(net, img, target_class, bb, mask, flip_mask, save_path)
    % flip if necessary
    if ~flip_mask
        mask_resized = imresize(mask, [size(img,1), size(img,2)]);
    else
        mask_resized = imresize(1 - mask, [size(img,1), size(img,2)]);
    end
    mask_resized = clip_map(mask_resized);
    [aoi_mask, aou_mask] = calculate_aoi_aou(mask_resized, bb);

    %% Simonyan et al, 2014 (Gradient-based Saliency)
    if ~isequal(net.layers{end}.type, 'softmaxloss')
        net.layers{end+1} = struct('type', 'softmaxloss') ;
    end
    net.layers{end}.class = target_class;

    img_ = cnn_normalize(net.meta.normalization, img, 1);
    res_sal = vl_simplenn(net, img_, 1);

    saliency_resized = imresize(max(abs(res_sal(1).dzdx),[],3), [size(img,1), size(img,2)]);
    saliency_resized = clip_map(saliency_resized);
    [aoi_sal, aou_sal] = calculate_aoi_aou(saliency_resized, bb);

    %% Zeiler & Fergus, 2014 (Deconvolutional network)
    % create a deconvnet
    deconvnet = net;
    for i=1:length(deconvnet.layers)
        switch deconvnet.layers{i}.type
            case 'relu'
                deconvnet.layers{i}.type = 'relu_deconvnet';
            otherwise
                continue;
        end
    end

    res_deconv = vl_simplenn(deconvnet, img_, 1);
    deconv_resized = imresize(max(abs(res_deconv(1).dzdx),[],3), [size(img,1), size(img,2)]);
    deconv_resized = clip_map(deconv_resized);
    [aoi_deconv, aou_deconv] = calculate_aoi_aou(deconv_resized, bb);
 
    %% Springenberg et al., 2015, Mahendran and Vedaldi, 2015 (DeSalNet/Guided Backprop)
    
    % create guided backprop net
    guided_net = net;
    for i=1:length(deconvnet.layers)
        switch guided_net.layers{i}.type
            case 'relu'
                guided_net.layers{i}.type = 'relu_eccv16';
            otherwise
                continue;
        end
    end

    res_guided = vl_simplenn(guided_net, img_, 1);
    guided_resized = imresize(max(abs(res_guided(1).dzdx),[],3), [size(img,1), size(img,2)]);
    guided_resized = clip_map(guided_resized);
    [aoi_guided, aou_guided] = calculate_aoi_aou(guided_resized, bb);
 
    f = figure;
    subplot(2,3,1);
    imshow(img);
    hold on;
    rectangle('Position', bb, 'EdgeColor','r');
    title('Orig Img + BB');

    %fprintf('%f %f %f\n', aoi_bb, aou_bb, aoi_bb/aou_bb);
    subplot(2,3,2);
    imshow(normalize(bsxfun(@times, single(img), mask_resized)));
    hold on;
    rectangle('Position', bb, 'EdgeColor', 'r');
    title(sprintf('Opt Mask (%f)', aoi_mask/aou_mask));

    %fprintf('%f %f %f\n', aoi_sal, aou_sal, aoi_sal/aou_sal);
    subplot(2,3,3);
    imshow(normalize(bsxfun(@times, single(img), saliency_resized)));
    hold on;
    rectangle('Position', bb, 'EdgeColor', 'r');
    title(sprintf('Saliency (%f)', aoi_sal/aou_sal));

    subplot(2,3,4);
    imshow(normalize(bsxfun(@times, single(img), deconv_resized)));
    hold on;
    rectangle('Position', bb, 'EdgeColor', 'r');
    title(sprintf('Deconvnet (%f)', aoi_deconv/aou_deconv));
    
    subplot(2,3,5);
    imshow(normalize(bsxfun(@times, single(img), guided_resized)));
    hold on;
    rectangle('Position', bb, 'EdgeColor', 'r');
    title(sprintf('Guided Backprop (%f)', aoi_guided/aou_guided));
    
    prep_path(save_path);
    print(f, save_path, '-djpeg');

    close(f);
end

function map = clip_map(map)
    map(map > 1) = 1;
    map(map < 0) = 0;
end

function [aoi, aou] = calculate_aoi_aou(map, bb)
    map_bb = map(bb(2):bb(2)+bb(4),bb(1):bb(1)+bb(3));
    aoi = sum(map_bb(:));
    aou = sum(map(:)) + bb(3)*bb(4) - aoi;
end

