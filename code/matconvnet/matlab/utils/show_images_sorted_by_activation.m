function show_images_sorted_by_activation(data, activations, varargin)
    data_size = size(data);
    activation_size = size(activations);
    opts.batch_range = 1:data_size(end);
    opts.space_type = 'spaced_out';
    opts.fig_path = '';
    opts.num_images = min(100, data_size(end));
    opts.all_step = 1;
    opts.activation_layer = -1;
    opts.rf_info = {};
    opts.show_unique_images = false;
    num_figures = 1;
    opts = vl_argparse(opts, varargin);
    
    activations_1d = reshape(activations, [1 numel(activations)]);
    [~,sorted_idx] = sort(activations_1d, 'descend');
    
    switch opts.space_type
        case 'spaced_out'
            space_range = ceil(linspace(1,length(sorted_idx),opts.num_images));
        case 'top'
            space_range = 1:opts.num_images;
        case 'bottom'
            space_range = length(sorted_idx):-1:length(sorted_idx)-opts.num_images;
        case 'all'
            space_range = 1:opts.all_step:length(sorted_idx);
            num_figures = ceil(length(sorted_idx)/opts.num_images);
        otherwise
            assert(false);
    end
    num_on_side = ceil(sqrt(opts.num_images));
    
    for n=1:num_figures,
        f = figure('units','normalized','outerposition',[0 0 1 1]); % open a maxed out figure
        for i=1:opts.num_images
            subplot(num_on_side, num_on_side, i); 
            idx = sorted_idx(space_range(i+opts.num_images*(n-1)));
            if length(activation_size) < 3
                imshow(normalize(data(:,:,:,opts.batch_range(idx))));
                title([num2str(idx), ': ', num2str(activations(idx), '%.2f')]);
            else
                feature_map_size = prod(activation_size(1:end-1));
                img_i = ceil(idx/feature_map_size);
                orig_img = normalize(data(:,:,:,opts.batch_range(img_i)));
                [y_i, x_i, ~, ~] = ind2sub(activation_size, idx);
                img_size = size(orig_img);
                bb_x_start = max(1,opts.rf_info.stride(opts.activation_layer) ...
                    * x_i + opts.rf_info.offset(opts.activation_layer) ...
                    - (opts.rf_info.size(opts.activation_layer) - 1)/2);
                bb_y_start = max(1,opts.rf_info.stride(opts.activation_layer) ...
                    * y_i + opts.rf_info.offset(opts.activation_layer) ...
                    - (opts.rf_info.size(opts.activation_layer) - 1)/2);
                bb_x_end = min(img_size(2), bb_x_start + opts.rf_info.size(opts.activation_layer) - 1);
                bb_y_end = min(img_size(1), bb_y_start + opts.rf_info.size(opts.activation_layer) - 1);
                if ismatrix(orig_img)
                    imshow(orig_img(bb_y_start:bb_y_end,bb_x_start:bb_x_end));
                else
                    imshow(orig_img(bb_y_start:bb_y_end,bb_x_start:bb_x_end,:));
                end
                title([num2str(img_i), ': ', num2str(activations(idx), '%.2f')]);
            end
        end
        if ~isempty(opts.fig_path)
            if num_figures == 1
                fig_path = opts.fig_path;
                prep_path(fig_path);
                print(f, fig_path, '-djpeg');
            else
                [folder, name, ext] = fileparts(opts.fig_path);
                fig_path = fullfile(folder, sprintf('%s_%d%s', name, n, ext));
                print(f, fig_path, '-djpeg');
            end
        end
    end
end