function run_population_encoding_imagenet_experiment(net, layer, target_class, K, ...
    varargin)
    opts.use_norm = true;
    opts.save_fig_dir = '';
    
    opts = vl_argparse(opts, varargin);
    
    class_imdb_paths = load(sprintf('/data/ruthfong/ILSVRC2012/class_train_imdb_paths/%d_train_imdb_paths.mat', ...
        target_class));
%     % average image isn't the right format for vgg-very-deep-16 (1x3 single)
%     normalization = net.meta.normalization;
%     averageImage = imresize(net_alexnet.meta.normalization.averageImage, ...
%         normalization.imageSize(1:2));
%     normalization.averageImage = averageImage;
%     class_imdb = build_imacgenet_class_imdb(class_imdb_paths, normalization);
    class_imdb = build_imagenet_class_imdb(class_imdb_paths, net.meta.normalization);
    disp('class imdb created');

    class_res_path = sprintf('/data/ruthfong/ILSVRC2012/class_res/train/%d_res.mat', target_class);
    if exist(class_res_path, 'file')
        cres = load();
        cres = cres.res;
        fprintf('loaded existing class res from %s', class_res_path);
    else
%         if ~isequal(net.layers{end}.type, 'softmaxloss')
%             net.layers{end+1} = struct('type', 'softmaxloss') ;
%         end
% 
%         net.layers{end}.class = class_imdb.images.labels;
        net = truncate_net(net, 1, layer);
        start_time = cputime;
        cres = vl_simplenn(net, class_imdb.images.data);
        fprintf('finished forward run for %d images in %.2f seconds\n', ...
            length(class_imdb.images.labels), cputime - start_time);
%         res = cres;
%         save(class_res_path, 'res', '-v7.3');
%         disp(sprintf('saved class res at %s', class_res_path));
    end
    %show_images(cres(1).x(:,:,:,1:49));


    %% reshape layer features
    disp(net.layers{end}.name);

    size_feats_vol = size(cres(end).x);
    feats_per_img = reshape(cres(end).x, [prod(size_feats_vol(1:2)), ...
        size_feats_vol(3), size_feats_vol(4)]);
    [num_acts, num_feats, num_imgs] = size(feats_per_img);

    % L2 normalization per population response
    if opts.use_norm
        norms = arrayfun(@(x) norm(feats_per_img(:,:,x)), 1:num_imgs);
        for i=1:num_imgs
            feats_per_img(:,:,i) = feats_per_img(:,:,i) / norms(i);
        end
    end
    feats = zeros([num_acts*num_imgs, num_feats], 'single');

    for i=1:num_acts
        for j=1:num_imgs
            feats((j-1)*num_acts + i, :) = feats_per_img(i,:,j);
        end
    end

    %% run kmeans++
    [idx,C,sumd,D] = kmeans(feats, K);
    start_time = cputime;
    fprintf('finished running kmeans++ with K=%d in %.2f seconds\n', K, ...
        cputime - start_time);

    f = figure; % reuse the same figure
    for c_i=1:K
        [~,sorted_idx] = sort(D(:,c_i), 'ascend');
        c_idx = sorted_idx(1:49); % show nearest 49 patches to cluster center
        img_idx = floor((c_idx-1)./num_acts) + 1;
        img_pos_idx = mod(c_idx-1, num_acts) + 1;
        r_in = floor(img_pos_idx./size_feats_vol(1));
        c_in = mod(img_pos_idx, size_feats_vol(1));

        rf_info = get_rf_info(net);

        [r_start, r_end, c_start, c_end] = get_patch_coordinates(rf_info, layer, ...
            net.meta.normalization.imageSize(1:3), r_in, c_in);

        num_patches = length(c_idx);
        side_length = ceil(sqrt(num_patches));
        for i=1:num_patches
            subplot(side_length, side_length, i);
            imshow(normalize(cres(1).x(r_start(i):r_end(i),c_start(i):c_end(i),:,img_idx(i))));
        end

        if ~isempty(opts.save_fig_dir)
            fig_path = fullfile(opts.save_fig_dir, sprintf('cluster_%d.jpg', c_i));
            prep_path(fig_path);
            print(f, fig_path, '-djpeg');
        end
    end

end