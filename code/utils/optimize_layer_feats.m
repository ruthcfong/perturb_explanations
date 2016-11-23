function new_res = optimize_layer_feats(net, img, target_class, layer, varargin)
    %opts.null_img = zeros(size(imdb.images.data_mean));
    opts.num_iters = 500;
    opts.learning_rate = 0.95;
    opts.lambda = 1e-6;
    opts.save_fig_path = '';
    opts.save_res_path = '';
    opts.plot_step = floor(opts.num_iters/20);
    opts.debug = false;
    opts.mask_dims = 2;
    opts.loss = 'softmaxloss';
    
    opts = vl_argparse(opts, varargin);
    
    type = 'double';
    type_fh = @double;
    
    net = convert_net_value_type(net, type_fh);
    
    img_size = size(net.meta.normalization.averageImage);
    rf_info = get_rf_info(net);
    
    if isfield(net.meta.classes,'description')
        classes = net.meta.classes.description;
    else
        classes = net.meta.classes;
    end
    
    % prepare truncated network
    net.layers{end}.class = type_fh(target_class);
    % res_null = vl_simplenn(net, opts.null_img, 1);
    
    tnet = truncate_net(net, layer+1, length(net.layers));

    res = vl_simplenn(net, img,1);
    actual_feats = type_fh(res(layer+1).x);
    size_feats = size(actual_feats);
    % null_feats = res_null(layer+1).x;
    
    % get maximum feature map (similar to Fergus and Zeiler, 2014)
    [~, max_feature_idx] = max(sum(sum(res(layer+1).x,1),2));


    switch opts.mask_dims
        case 1
            mask = rand([1 1 size_feats(3)], type);
            mask_t = zeros([size_feats(3) opts.num_iters], type);
        case 2
            mask = rand([size_feats(1:2) 1],type);
            mask_t = zeros([size_feats(1:2) opts.num_iters],type);
        case 3
            mask = rand(size_feats,type);
            mask_t = zeros([size_feats opts.num_iters],type);
        otherwise
            assert(false);
    end
    E = zeros([3 opts.num_iters]);

    tnet_cam = tnet;
    tnet_cam.layers = tnet_cam.layers(1:end-1); % exclude softmax loss layer
    gradient = zeros(size(res(end-1).x), type);
    gradient(target_class) = 1;
    res_orig_cam = vl_simplenn(tnet_cam, actual_feats, gradient);
    
    cam_weights_orig = sum(sum(res_orig_cam(1).dzdx,1),2) / prod(size_feats(1:2));
    cam_map_orig = bsxfun(@max, sum(bsxfun(@times, res_orig_cam(1).x, cam_weights_orig),3), 0);
    large_heatmap_orig = map2jpg(im2double(imresize(cam_map_orig, img_size(1:2))));

    display_im = normalize(img+net.meta.normalization.averageImage);

    [sorted_orig_scores, sorted_orig_class_idx] = sort(res(end-1).x, 'descend');
    num_top_scores = 5;
    interested_scores = zeros([num_top_scores+1 opts.num_iters]);
    switch opts.loss
        case 'softmaxloss'
            orig_err = -log(exp(res(end-1).x(:,:,target_class))/(...
                sum(exp(res(end-1).x))));
        case 'min_classlabel'
            orig_err = res_orig_cam(end).x(:,:,target_class);
            %orig_err = mean((res_orig_cam(end).x - gradient).^2);
        case 'max_classlabel'
            orig_err = res_orig_cam(end).x(:,:,target_class);
            neg_gradient = zeros(size(res(end-1).x), type);
            neg_gradient(target_class) = -1;

        otherwise
            assert(false);
    end
        
    fig = figure;
    for t=1:opts.num_iters,
        switch opts.mask_dims
            case 1
                mask_t(:,t) = squeeze(mask);
                x = bsxfun(@times, actual_feats, mask);
            case 2
                mask_t(:,:,t) = mask;
                %x = actual_feats .* mask + null_feats .* (1 - mask);
                x = bsxfun(@times, actual_feats, mask);
            case 3
                mask_t(:,:,:,t) = mask;
                x = actual_feats .* mask;
        end
        
        E(2,t) = opts.lambda * sum(abs(mask(:)));

        switch opts.loss
            case 'softmaxloss'
                tres = vl_simplenn(tnet, x, 1);
                E(1,t) = tres(end).x;
            case 'min_classlabel'
                tres = vl_simplenn(tnet_cam, x, gradient);
                E(1,t) = tres(end).x(:,:,target_class);
                E(2,t) = opts.lambda * sum(abs(1-mask(:)));
            case 'max_classlabel'
                tres = vl_simplenn(tnet_cam, x, neg_gradient);
                E(1,t) = tres(end).x(:,:,target_class);
            otherwise
                assert(false);
        end
        E(3,t) = E(1,t) + E(2,t);
                 
        interested_scores(1:num_top_scores,t) = tres(end-1).x(sorted_orig_class_idx(1:num_top_scores));
        interested_scores(end,t) = tres(end-1).x(target_class);
        
        switch opts.mask_dims
            case 1
                softmax_der = sum(sum(tres(1).dzdx.*actual_feats,1),2);
            case 2
                softmax_der = sum(tres(1).dzdx.*actual_feats,3);
                %reg_der = sum(sign(mask),3);
            case 3
                softmax_der = tres(1).dzdx.*actual_feats;
                %reg_der = sign(mask);
        end
        
        if strcmp(opts.loss, 'min_classlabel')      
            reg_der = sign(1-mask);
        else
            reg_der = sign(mask);
        end

        mask = mask - opts.learning_rate*(softmax_der+opts.lambda*reg_der);
        mask(mask > 1) = 1;
        mask(mask < 0) = 0;

%         if mod(t-1,10) == 0
%             ex = rand(size(mask), type);
%             eta = 0.0001;
%             xp = bsxfun(@times, actual_feats, mask + eta * ex);
%             tresp = vl_simplenn(tnet, xp, 1);
%             dzdx_emp = 1 * (tresp(end).x - tres(end).x) / eta;
%             dzdx_comp = sum(sr_der(:) .* ex(:));
%             fprintf('der: emp: %f, comp: %f, error %.2f %%\n', ...
%                 dzdx_emp, dzdx_comp, abs(1 - dzdx_emp/dzdx_comp)*100);
%         end

        % plotting
        if t == opts.num_iters || (opts.debug && mod(t-1,opts.plot_step) == 0)
            res_new_cam = vl_simplenn(tnet_cam, x, gradient);

            switch opts.mask_dims
                case 1
                    subplot(3,3,1);
                    actual_max_feat_map = res(layer+1).x(:,:,max_feature_idx);
                    curr_saliency_map = get_saliency_map_from_difference_map(...
                        actual_max_feat_map - x(:,:,max_feature_idx), layer, rf_info, img_size);
                    curr_saliency_map_rep = repmat(normalize(curr_saliency_map),[1 1 3]);
                    imshow(display_im.*curr_saliency_map_rep);
                    title('Diff Max Feat Saliency');

                    subplot(3,3,2);
                    curr_saliency_map = get_saliency_map_from_difference_map(mean(res(layer+1).x ...
                        - x,3), layer, rf_info, img_size);
                    curr_saliency_map_rep = repmat(normalize(curr_saliency_map),[1 1 3]);
                    imshow(display_im.*curr_saliency_map_rep);
                    title('Diff Avg Feats Saliency');

                    res_new_cam = vl_simplenn(tnet_cam, x, gradient);
                    cam_weights_new = sum(sum(res_new_cam(1).dzdx,1),2) / prod(size_feats(1:2));
                    cam_map_new = bsxfun(@max, sum(bsxfun(@times, res_new_cam(1).x, cam_weights_new),3), 0);
                    large_heatmap_new = map2jpg(im2double(imresize(cam_map_new, img_size(1:2))));

%                     hm_new = map2jpg(im2double(imresize(mean(x .* tres(1).dzdx,3), img_size(1:2))), [], 'jet');
%                     hm_old = map2jpg(im2double(imresize(mean(actual_feats .* ...
%                         res(layer+1).dzdx(:,:,:,opts.img_i),3), img_size(1:2) )), [], 'jet');
                                        
                    [best_opt_score, best_opt_class_i] = max(tres(end-1).x);

                    subplot(3,3,3);
                    imshow(display_im*0.3 + large_heatmap_new*0.7);
                    title(sprintf('CAM Opt (%.3f: %.8s)', best_opt_score, classes{best_opt_class_i}));
                    subplot(3,3,5);
                    imshow(display_im*0.3 + large_heatmap_orig*0.7);
                    title(sprintf('CAM Orig (%.3f: %.8s)', sorted_orig_scores(1), classes{sorted_orig_class_idx(1)}));
                        [sorted_orig_scores, sorted_orig_class_idx] = sort(res(end-1).x, 'descend');

%                     subplot(2,3,3);
%                     curr_saliency_map = get_saliency_map_from_difference_map(mask, layer, rf_info, img_size);
%                     curr_saliency_map_rep = repmat(normalize(curr_saliency_map),[1 1 3]);
%                     imshow(normalize((img+imdb.images.data_mean).*curr_saliency_map_rep));
%                     title('Mask Saliency');

                    subplot(3,3,4);
                    imagesc(reshape(mask,[ceil(sqrt(length(mask))), ceil(sqrt(length(mask)))]));
                    colorbar;
                    axis square;
                    title('feature mask');

                    subplot(3,3,6);
                    plot(transpose(E(1,1:t)));
                    hold on;
                    plot(transpose(E(3,1:t)));
                    plot(repmat(orig_err, [1 t]));
                    hold off;
                    axis square;
                    title(sprintf('log(lr) = %.2f, log(lambda) = %.2f', log10(opts.learning_rate), log10(opts.lambda)));
                    %legend('Softmax Loss','Tot Loss','Orig SM Loss');

                    subplot(3,3,7);
                    imshow(display_im);
                    title('orig im');
                    
                    subplot(3,3,8);
                    plot(transpose(interested_scores(:,1:t)));
                    axis square;
                    %legend([classes(sorted_orig_class_idx(1:num_top_scores)) classes(target_class)]);
                    title(sprintf('top %d scores', num_top_scores));

                    drawnow;
                case 2
                    subplot(3,3,1);
                    actual_max_feat_map = res(layer+1).x(:,:,max_feature_idx);
                    curr_saliency_map = get_saliency_map_from_difference_map(...
                        actual_max_feat_map - x(:,:,max_feature_idx), layer, rf_info, img_size);
                    curr_saliency_map_rep = repmat(normalize(curr_saliency_map),[1 1 3]);
                    imshow(display_im.*curr_saliency_map_rep);
                    title('Diff Max Feat Saliency');

                    subplot(3,3,2);
                    curr_saliency_map = get_saliency_map_from_difference_map(mean(res(layer+1).x ...
                        - x,3), layer, rf_info, img_size);
                    curr_saliency_map_rep = repmat(normalize(curr_saliency_map),[1 1 3]);
                    imshow(display_im.*curr_saliency_map_rep);
                    title('Diff Avg Feats Saliency');

                    subplot(3,3,3);
                    curr_saliency_map = get_saliency_map_from_difference_map(mask, layer, rf_info, img_size);
                    curr_saliency_map_rep = repmat(normalize(curr_saliency_map),[1 1 3]);
                    imshow(display_im.*curr_saliency_map_rep);
                    title('Mask Saliency');

%                     hm_new = map2jpg(im2double(imresize(bsxfun(@max,sum(...
%                         bsxfun(@times, x, mean(mean(tres(1).dzdx,1),2)),3),0), img_size(1:2))), [], 'jet');
%                     hm_old = map2jpg(im2double(imresize(bsxfun(@max,sum(...
%                         bsxfun(@times, actual_feats,...
%                         mean(mean(res(layer+1).dzdx(:,:,:,opts.img_i),1),2)),3),0), img_size(1:2))), [], 'jet');
%                     display_im = normalize(img+imdb.images.data_mean);
                    
                    hm_mask = map2jpg(im2double(imresize(mask,img_size(1:2))));

                    res_new_cam = vl_simplenn(tnet_cam, x, gradient);
                    cam_weights_new = sum(sum(res_new_cam(1).dzdx,1),2) / prod(size_feats(1:2));
                    cam_map_new = bsxfun(@max, sum(bsxfun(@times, res_new_cam(1).x, cam_weights_new),3), 0);
                    large_heatmap_new = map2jpg(im2double(imresize(cam_map_new, img_size(1:2))));
                    
                    [best_opt_score, best_opt_class_i] = max(tres(end-1).x);

                    subplot(3,3,4);
                    imshow(display_im*0.3 + large_heatmap_new*0.7);
                    title(sprintf('CAM Opt (%.3f: %.8s)', best_opt_score, classes{best_opt_class_i}));
                    subplot(3,3,5);
                    imshow(display_im*0.3 + large_heatmap_orig*0.7);
                    title(sprintf('CAM Orig (%.3f: %.8s)', sorted_orig_scores(1), classes{sorted_orig_class_idx(1)}));
                        [sorted_orig_scores, sorted_orig_class_idx] = sort(res(end-1).x, 'descend');

                    subplot(3,3,6);
                    plot(transpose(interested_scores(:,1:t)));
                    axis square;
                    %legend([classes(sorted_orig_class_idx(1:num_top_scores)) classes(target_class)]);
                    title(sprintf('top %d scores', num_top_scores));
                    
                    subplot(3,3,7);
                    imagesc(mask);
                    colorbar;
                    axis square;
                    title('spatial mask');

                    subplot(3,3,8);
                    imshow(display_im*0.3 + hm_mask*0.7);
                    title('mask overlay');
                    
                    subplot(3,3,9);
                    plot(transpose(E(1,1:t)));
                    hold on;
                    plot(transpose(E(3,1:t)));
                    plot(repmat(orig_err, [1 t]));
                    hold off;
                    axis square;
                    title(sprintf('log(lr) = %.2f, log(lambda) = %.2f', log10(opts.learning_rate), log10(opts.lambda)));

                    %legend('Softmax Loss','Tot Loss','Orig SM Loss');

                    drawnow;
                case 3
                    subplot(3,3,1);
                    actual_max_feat_map = res(layer+1).x(:,:,max_feature_idx);
                    curr_saliency_map = get_saliency_map_from_difference_map(...
                        actual_max_feat_map - x(:,:,max_feature_idx), layer, rf_info, img_size);
                    curr_saliency_map_rep = repmat(normalize(curr_saliency_map),[1 1 3]);
                    imshow(display_im.*curr_saliency_map_rep);
                    title('Diff Max Feat Saliency');

                    subplot(3,3,2);
                    curr_saliency_map = get_saliency_map_from_difference_map(mean(res(layer+1).x ...
                        - x,3), layer, rf_info, img_size);
                    curr_saliency_map_rep = repmat(normalize(curr_saliency_map),[1 1 3]);
                    imshow(display_im.*curr_saliency_map_rep);
                    title('Diff Avg Feats Saliency');

                    subplot(3,3,3);
                    curr_saliency_map = get_saliency_map_from_difference_map(mean(mask,3), layer, rf_info, img_size);
                    curr_saliency_map_rep = repmat(normalize(curr_saliency_map),[1 1 3]);
                    imshow(display_im.*curr_saliency_map_rep);
                    title('Mean Mask Saliency');

                    subplot(3,3,4);
                    curr_saliency_map = get_saliency_map_from_difference_map(mask(:,:,max_feature_idx), layer, rf_info, img_size);
                    curr_saliency_map_rep = repmat(normalize(curr_saliency_map),[1 1 3]);
                    imshow(display_im.*curr_saliency_map_rep);
                    title('Max Mask Saliency');

                    subplot(3,3,5);
                    imagesc(mask(:,:,max_feature_idx));
                    colorbar;
                    axis square;
                    title('Max Mask');

%                     max_mask = map2jpg(im2double(imresize(max(mask,[],3),img_size(1:2))));
%                     imshow(display_im*0.3 + max_mask*0.7);
%                     title('Max Mask Overlay');

                    subplot(3,3,6);
                    plot(transpose(E(1,1:t)));
                    hold on;
                    plot(transpose(E(3,1:t)));
                    plot(repmat(orig_err, [1 t]));
                    hold off;
                    axis square;
                    title(sprintf('log(lr) = %.2f, log(lambda) = %.2f', log10(opts.learning_rate), log10(opts.lambda)));
                    %legend('Softmax Loss','Tot Loss','Orig SM Loss');

                    hm_mask = map2jpg(im2double(imresize(mean(mask,3),img_size(1:2))));

                    res_new_cam = vl_simplenn(tnet_cam, x, gradient);
                    cam_weights_new = sum(sum(res_new_cam(1).dzdx,1),2) / prod(size_feats(1:2));
                    cam_map_new = bsxfun(@max, sum(bsxfun(@times, res_new_cam(1).x, cam_weights_new),3), 0);
                    large_heatmap_new = map2jpg(im2double(imresize(cam_map_new, img_size(1:2))));
                    
                    [best_opt_score, best_opt_class_i] = max(tres(end-1).x);

                    subplot(3,3,7);
                    imshow(display_im*0.3 + large_heatmap_new*0.7);
                    title(sprintf('CAM Opt (%.3f: %.8s)', best_opt_score, classes{best_opt_class_i}));
                    subplot(3,3,8);
                    imshow(display_im*0.3 + large_heatmap_orig*0.7);
                    title(sprintf('CAM Orig (%.3f: %.8s)', sorted_orig_scores(1), classes{sorted_orig_class_idx(1)}));
                        [sorted_orig_scores, sorted_orig_class_idx] = sort(res(end-1).x, 'descend');

                    
                    subplot(3,3,9);
                    imshow(display_im*0.3 + hm_mask*0.7);
                    title('Mean Mask Overlay');
                    
                    drawnow;
            end
            fprintf(strcat('loss at epoch %d : orig: %f, softmax: %f, reg: %f, tot: %f\n', ...
                'derivs at epoch %d: softmax: %f, reg (unnorm): %f, reg (norm): %f\n'), ...
                t, orig_err, E(1,t), E(2,t), E(3,t), t, mean(softmax_der(:)), ...
                mean(reg_der(:)), opts.lambda * mean(reg_der(:)));
            
        end
    end
    
%     act_opts = struct();
%     act_opts.batch_range = opts.batch_range;
%     act_opts.space_type = 'top';
%     act_opts.rf_info = rf_info;
%     act_opts.activation_layer = layer;
%     [~,sorted_idx] = sort(sum(sum(x,1),2));
%     num_top = 1;
%     for i=1:num_top
%         show_images_sorted_by_activation(imdb, res(layer+1).x(:,:,sorted_idx(end-i+1),:), act_opts);
%     end

    if ~strcmp(opts.save_fig_path, ''),
        [folder, ~, ~] = fileparts(opts.save_fig_path);
        if ~exist(folder, 'dir')
            mkdir(folder);
        end
        print(fig, opts.save_fig_path, '-djpeg');
    end
    
    new_res = struct();
    
    new_res.mask = mask_t;
    new_res.error = E;
    new_res.optimized_feats = x;
    new_res.actual_feats = actual_feats;
    new_res.tnet = tnet;
    new_res.tnet_cam = tnet_cam;
    new_res.target_class = target_class;
    new_res.img = img;
    new_res.opts = opts;
    
    
    if ~strcmp(opts.save_res_path, ''),
        [folder, ~, ~] = fileparts(opts.save_res_path);
        if ~exist(folder, 'dir')
            mkdir(folder);
        end

        save(opts.save_res_path, 'new_res');
    end
end