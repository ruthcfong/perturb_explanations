function run_color_sensitivity_experiment(net, imdb_paths, class_i, layers)
    image_idx_for_class = find(imdb_paths.images.labels == class_i);

    %% get dataset
    [rgb_norm_imgs, gray_norm_imgs] = get_rgb_and_gray_datasets(...
        imdb_paths.images.paths(image_idx_for_class), ...
        net.meta.normalization);

    %% forward pass
    rgb_res = vl_simplenn(net, rgb_norm_imgs); 
    gray_res = vl_simplenn(net, gray_norm_imgs);

    %% 
    rf_info = get_rf_info(net);

    f = figure('units','normalized','outerposition',[0 0 1 1]); % open a maxed out figure
    subplot(2,3,6);
    imshow(normalize(rgb_norm_imgs(:,:,:,1)));
    title(sprintf('Example Img of %d %s', class_i, net.meta.classes.description{class_i}));

    for i=1:length(layers)
        layer = layers(i);
        layer_name = net.layers{layer}.name;
        disp(layer_name);
        size_feats = size(rgb_res(layer+1).x);

        diff_vol = rgb_res(layer+1).x - gray_res(layer+1).x;
        diff_feats = reshape(diff_vol, ...
            [prod(size_feats(1:2)), size_feats(3), size_feats(4)]);

        abs_mean_act_diffs_per_img = abs(mean(diff_feats, 1));
        std_act_diffs = std(diff_feats, 1);
        mean_abs_mean = mean(abs_mean_act_diffs_per_img,3);
        std_abs_mean = std(transpose(squeeze(abs_mean_act_diffs_per_img)));
        [~, sorted_idx] = sort(mean_abs_mean);

        mean_std = mean(std_act_diffs(:));
        std_std = std(std_act_diffs(:));

        opts = struct();
        opts.rf_info = rf_info;
        opts.activation_layer = layer;

        opts.space_type = 'top';
        opts.fig_path = fullfile('/home/ruthfong/neural_coding/figures5/imagenet/color_sensitivity/',...
            sprintf('%d_%s/color_mean_%s_most_sensitive_unit_sorted_activation.jpg', ...
            class_i, get_short_class_name(net, class_i, false), net.layers{layer}.name));
        show_images_sorted_by_activation(rgb_norm_imgs, ...
            squeeze(mean(diff_feats(:,sorted_idx(end),:),1)), opts);

        opts.num_images = 100;

        opts.space_type = 'top';
        opts.fig_path = fullfile('/home/ruthfong/neural_coding/figures5/imagenet/color_sensitivity/',...
            sprintf('%d_%s/color_%s_%s_most_sensitive_unit_sorted_patch_activation.jpg', ...
            class_i, get_short_class_name(net, class_i, false), opts.space_type, ...
            net.layers{layer}.name));
        show_images_sorted_by_activation(rgb_norm_imgs, ...
            squeeze(diff_vol(:,:,sorted_idx(end),:)), opts);

        opts.space_type = 'bottom';
        opts.fig_path = fullfile('/home/ruthfong/neural_coding/figures5/imagenet/color_sensitivity/',...
            sprintf('%d_%s/color_%s_%s_most_sensitive_unit_sorted_%s_patch_activation.jpg', ...
            class_i, get_short_class_name(net, class_i, false), opts.space_type, ...
            net.layers{layer}.name));
        show_images_sorted_by_activation(rgb_norm_imgs, ...
            squeeze(diff_vol(:,:,sorted_idx(end),:)), opts);

        opts.space_type = 'spaced_out';
        opts.fig_path = fullfile('/home/ruthfong/neural_coding/figures5/imagenet/color_sensitivity/',...
            sprintf('%d_%s/color_%s_%s_most_sensitive_unit_sorted_%s_patch_activation.jpg', ...
            class_i, get_short_class_name(net, class_i, false), opts.space_type, ...
            net.layers{layer}.name));
        show_images_sorted_by_activation(rgb_norm_imgs, ...
            squeeze(diff_vol(:,:,sorted_idx(end),:)), opts);


        %[h,p] = jbtest(mean_act_diffs); % reject null hypothesis at all layers
        figure(f);
        subplot(2,3,i);
        bar(1:size_feats(end-1), mean_abs_mean(sorted_idx));
        hold on;
        errorbar(1:size_feats(end-1), mean_abs_mean(sorted_idx), ...
            std_abs_mean(sorted_idx)/sqrt(size_feats(end)), '.');
        hold off;
        title(sprintf('%s (Mean Std=%.2f, Std Std= %.2f)', layer_name, mean_std, std_std));
        ylabel('Abs Mean Act Diff');
        xlabel(sprintf('Sorted %s Filters', layer_name));
    end

    fig_path = fullfile('/home/ruthfong/neural_coding/figures5/imagenet/color_sensitivity/',...
            sprintf('%d_%s.jpg', class_i, get_short_class_name(net, class_i, false)));
    prep_path(fig_path);
    print(f, fig_path, '-djpeg');
end