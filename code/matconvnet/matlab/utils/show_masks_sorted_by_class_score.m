function show_masks_sorted_by_class_score(net, img_idx, imdb_paths, mask_path_format)
    num_images = length(img_idx);
    target_scores = zeros([1 num_images], 'single');
    target_labels = zeros([1 num_images], 'single');
    top_scores = zeros([1 num_images], 'single');
    top_labels = zeros([1 num_images], 'single');
    side_length = ceil(sqrt(num_images));
    
    for i=1:num_images
        img_i = img_idx(i);
        img = cnn_normalize(net.meta.normalization, imread(imdb_paths.images.paths{img_i}), 1);
        target_labels(i) = imdb_paths.images.labels(img_i);
        res = vl_simplenn(net, img);
        target_scores(i) = res(end).x(:,:,target_labels(i));
        [top_scores(i), top_labels(i)] = max(squeeze(res(end).x));
    end
    
    [~, sorted_idx] = sort(target_scores);
    figure('units','normalized','outerposition',[0 0 1 1]); % open a maxed out figure
    for i=1:num_images
        img_i = sorted_idx(i);
        img = cnn_normalize(net.meta.normalization, ...
            imread(imdb_paths.images.paths{img_idx(img_i)}), 1);
        mask_res = load(sprintf(mask_path_format, img_idx(img_i)));
        mask_res = mask_res.new_res;        
        mask_heatmap = map2jpg(im2double(mask_res.mask));
        subplot(side_length, side_length, i);
        imshow(normalize(img)*0.7 + mask_heatmap*0.3);
%         title(sprintf('%.2f: %s (%.2f)', top_scores(img_i), ...
%             get_short_class_name(net, top_labels(img_i), true), target_scores(img_i)));
        title(sprintf('%d: %.2f', img_idx(img_i), target_scores(img_i)));
    end
end