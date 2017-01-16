function new_res = optimize_layer_feats_multitask(net, img, target_class, layer, varargin)
    %opts.null_img = zeros(size(imdb.images.data_mean));
    opts.num_iters = 500;
    opts.learning_rate = 0.95;
    opts.lambda = 1e-6;
    opts.tv_lambda = 0;
    opts.beta = 1;
    opts.save_fig_path = '';
    opts.save_res_path = '';
    opts.plot_step = floor(opts.num_iters/20);
    opts.debug = false;
    opts.mask_dims = 2;
    opts.loss = 'min_max_classlabel';
    opts.denom_reg = false; % obsolete
    opts.mask_init = 'rand';
    opts.error_stopping_threshold = 100;
    opts.num_average = 5;
    opts.sim_layernums = [];
    opts.sim_layercoeffs = [];
    opts.num_class_labels = 0;
    opts.num_masks = 1;
    opts.adam.beta1 = 0.999;
    opts.adam.beta2 = 0.999;
    opts.adam.epsilon = 1e-8;
    opts.premask = '';
    opts.num_superpixels = 500;

    opts.mask_transform_function = 'linear';
    opts.gen_sig_c = 0.5;
    opts.gen_sig_a = 10;
    
    opts = vl_argparse(opts, varargin);

    type = 'single';
    type_fh = @single;

    gen_sig_func = @(x) (1./(1+exp(-opts.gen_sig_a*(x-opts.gen_sig_c))));
    d_gen_sig_func = @(x) (opts.gen_sig_a*gen_sig_func(x).*(1-gen_sig_func(x)));
    linear_func = @(x) (x);
    d_linear_func = @(x) (ones(size(x),type));

    switch opts.mask_transform_function
        case 'linear'
            mask_transform = linear_func;
            d_mask_transform = d_linear_func;
        case 'generalized_sigmoid'
            mask_transform = gen_sig_func;
            d_mask_transform = d_gen_sig_func;
    end
        
    stop_early = false;
        
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
    tnet = truncate_net(net, layer+1, length(net.layers));

    res = vl_simplenn(net, img,1);
    actual_feats = type_fh(res(layer+1).x);
    size_feats = size(actual_feats);
    % null_feats = res_null(layer+1).x;
    
    % get maximum feature map (similar to Fergus and Zeiler, 2014)
    [~, max_feature_idx] = max(sum(sum(res(layer+1).x,1),2));

    % set mask initialization function
    switch opts.mask_init
        case 'rand'
            mask_init_func = @rand;
        case 'ones'
            mask_init_func = @ones;
        case 'half'
            mask_init_func = @(s, t) (0.5 * ones(s, t));
    end
    
    assert(opts.mask_dims == 2);
    
    switch opts.premask
        case 'superpixels'
            assert(opts.num_masks == 1);
            [superpixel_labels, num_superpixels] = superpixels(img, opts.num_superpixels);
            premask = mask_init_func([1 num_superpixels],type);
            mask = zeros([size_feats(1:2) 1], type);
            for i=1:num_superpixels
                mask(superpixel_labels == i) = premask(i);
            end
            d_superpixel_mask = @(d) arrayfun(@(i) sum(d(superpixel_labels == i)), 1:num_superpixels);
            d_mask_func = d_superpixel_mask;
            differentiable_mask = premask;
        otherwise
            mask = mask_init_func([size_feats(1:2) opts.num_masks],type);
            d_average_mask = @(d) d;
            d_mask_func = d_average_mask;
            differentiable_mask = mask;
    end

    mask_t = zeros([size_feats(1:2) opts.num_iters],type);
    E = zeros([6 opts.num_iters]); % min, max, sim, L2norm, tv, total

    % for adam update
    m_t = zeros(size(differentiable_mask), type);
    v_t = zeros(size(differentiable_mask), type);
    
    tnet_cam = tnet;
    tnet_cam.layers = tnet_cam.layers(1:end-1); % exclude softmax loss layer
    
    % prepare gradients for min and max class label tasks
    min_gradient = zeros(size(res(end-1).x), type);
    min_gradient(target_class) = 1;
    max_gradient = zeros(size(res(end-1).x), type);
    max_gradient(target_class) = -1;
        
    res_orig_cam = vl_simplenn(tnet_cam, actual_feats, min_gradient);
        
    cam_weights_orig = sum(sum(res_orig_cam(1).dzdx,1),2) / prod(size_feats(1:2));
    cam_map_orig = bsxfun(@max, sum(bsxfun(@times, res_orig_cam(1).x, cam_weights_orig),3), 0);
    large_heatmap_orig = map2jpg(im2double(imresize(cam_map_orig, img_size(1:2))));

    display_im = normalize(img+net.meta.normalization.averageImage);

    [sorted_orig_scores, sorted_orig_class_idx] = sort(res(end-1).x, 'descend');
    num_top_scores = 5;
    interested_scores = zeros([num_top_scores+1 opts.num_iters]);
    
    if opts.num_class_labels > 0
        min_gradient = zeros(size(res(end-1).x), type);
        max_gradient = zeros(size(res(end-1).x), type);
        for i=1:opts.num_class_labels
            min_gradient(sorted_orig_class_idx(i)) = 1;
            max_gradient(sorted_orig_class_idx(i)) = -1;
        end
    end

    % min_gradient(sorted_orig_class_idx(1:num_top_scores)) = 1; % testing
    % orig_denom = sum(exp(res(end-1).x));
    assert(strcmp(opts.loss, 'min_max_classlabel'));
    orig_err = res_orig_cam(end).x(:,:,target_class);
    
    % null_feats = net.meta.normalization.averageImage; % only for pixel
    % space
    
    % prepare mask and masked input feats for first iteration
    mask_t(:,:,1) = mean(mask, 3);
    x_min = bsxfun(@times, actual_feats, mask_transform(1-mean(mask, 3)));
    x_max = bsxfun(@times, actual_feats, mask_transform(mean(mask, 3)));

    tres_max = vl_simplenn(tnet_cam, x_max, max_gradient);

    fig = figure('units','normalized','outerposition',[0 0 1 1]); % open a maxed out figure
    for t=1:opts.num_iters,
        sim_der = zeros(size(x_max), type);
        if ~isempty(opts.sim_layernums)
            for i=1:length(opts.sim_layernums)
                l = opts.sim_layernums(i);
                w = opts.sim_layercoeffs(i);
                sim_net = truncate_net(net, layer+1, l);
                sim_net.layers{end+1} = struct('type','gramsimloss');
                sim_net.layers{end}.class = res_orig_cam(l+1).x;
                sim_res = vl_simplenn(sim_net, x_max, 1);
                E(3,t) = E(3,t) + w*sim_res(end).x;
                sim_der = sim_der + w*sim_res(1).dzdx;
            end           
        end
        
        % calculate L1 regularization error and deriv
        E(4,t) = opts.lambda * sum(abs(mask(:)))/opts.num_masks;
        reg_der = sign(mask)/opts.num_masks;
        reg_der = d_mask_func(reg_der);
        
        % calculate tv-norm error and deriv
        if opts.tv_lambda ~= 0 && opts.beta ~= 0
            tv_err = 0;
            tv_der = zeros(size(mask), type);
            for i=1:opts.num_masks
                [tv_err_m, tv_der_m] = tv(mask(:,:,i), opts.beta);
                tv_err = tv_err + tv_err_m;
                tv_der(:,:,i) = tv_der_m;
            end
            tv_err = tv_err/opts.num_masks;
        else
            tv_err = 0;
            tv_der = zeros(size(mask), type);
        end
        tv_der = d_mask_func(tv_der);
        
        E(5,t) = opts.tv_lambda* tv_err/opts.num_masks;
        
        % calculate output and errors for min and max problems
        tres_min = vl_simplenn(tnet_cam, x_min, min_gradient);
        E(1,t) = tres_min(end).x(:,:,target_class);
        % already calculated
        E(2,t) = tres_max(end).x(:,:,target_class);
        
        E(end,t) = sum(E(1:end-1,t));
                
        % keep track of original top5 scores and target class score
        % (using max_classlabel input feats)
        interested_scores(1:num_top_scores,t) = tres_max(end).x(sorted_orig_class_idx(1:num_top_scores));
        interested_scores(end,t) = tres_max(end).x(target_class);
        
        maxscore_der = bsxfun(@times, d_mask_transform(mask),sum(tres_max(1).dzdx.*actual_feats,3));
        minscore_der = bsxfun(@times, d_mask_transform(mask),sum(tres_min(1).dzdx.*(1-actual_feats),3));
        sim_der = bsxfun(@times, d_mask_transform(mask), sum(sim_der.*actual_feats,3));
        
        maxscore_der = d_mask_func(maxscore_der / opts.num_masks);
        minscore_der = d_mask_func(minscore_der / opts.num_masks);
        sim_der = d_mask_func(sim_der / opts.num_masks);
        
        gradient = maxscore_der + minscore_der ...
            + sim_der...
            + opts.lambda*reg_der...
            + opts.tv_lambda*tv_der;

        % adam update
        if isfield(opts, 'adam')
            m_t = opts.adam.beta1*m_t + (1-opts.adam.beta1)*gradient;
            v_t = opts.adam.beta2*v_t + (1-opts.adam.beta2)*(gradient.^2);
            m_hat = m_t/(1-opts.adam.beta1^t);
            v_hat = v_t/(1-opts.adam.beta2^t);

            differentiable_mask = differentiable_mask - opts.learning_rate./(sqrt(v_hat)+opts.adam.epsilon).*m_hat;
        % vanilla gradient descent
        else
            differentiable_mask = differentiable_mask - opts.learning_rate*gradient;
        end
        
        % clip mask
        differentiable_mask(differentiable_mask > 1) = 1;
        differentiable_mask(differentiable_mask < 0) = 0;
        
        % prepare mask and masked input feats for next iteration
        switch opts.premask
            case 'superpixels'
                premask = differentiable_mask;
                mask = zeros([size_feats(1:2) 1], type);
                for i=1:num_superpixels
                    mask(superpixel_labels == i) = premask(i);
                end
            otherwise
                mask = differentiable_mask;
        end
        mask_t(:,:,t) = mean(mask, 3);
        x_min = bsxfun(@times, actual_feats, mask_transform(1-mean(mask,3)));
        x_max = bsxfun(@times, actual_feats, mask_transform(mean(mask,3)));

        tres_max = vl_simplenn(tnet_cam, x_max, max_gradient);

        % early stopping check
        if t > 50 && E(2,t) - tres_max(end).x(:,:,target_class) ...
                > opts.error_stopping_threshold
            stop_early = true;
            
            % reset for plotting
            mask = mean(mask_t(:,:,t-opts.num_average:t),3);
            x_max = bsxfun(@times, actual_feats, mean(mask, 3));
        end

        % plotting
        if t == opts.num_iters || opts.debug || mod(t-1,opts.plot_step) == 0 ...
                || stop_early
            %set(fig, 'Visible', 'off');
            
            subplot(3,4,1);
            imshow(normalize(img));
            title('Orig Img');

            subplot(3,4,2);
            if layer == 0
                imshow(normalize(x_max));
                if ~stop_early
                    title('Masked Img');
                else
                    title('Avg Masked Img');
                end
            else
                curr_saliency_map = get_saliency_map_from_difference_map(mean(mask, 3), layer, rf_info, img_size);
                curr_saliency_map_rep = repmat(normalize(curr_saliency_map),[1 1 3]);
                imshow(display_im.*curr_saliency_map_rep);
                title('Mask Saliency');
            end

            subplot(3,4,3);
            curr_saliency_map = get_saliency_map_from_difference_map(mean(res(layer+1).x ...
                - x_max,3), layer, rf_info, img_size);
            hm_avg_diff_feats = map2jpg(im2double(imresize(...
                normalize(curr_saliency_map),img_size(1:2))));
            imshow(display_im*0.5 + hm_avg_diff_feats*0.5);
            title('Diff Avg Feats Saliency');

            hm_mask = map2jpg(im2double(imresize(mean(mask, 3),img_size(1:2))));

            res_new_cam = vl_simplenn(tnet_cam, x_max, max_gradient);
            
            cam_weights_new = sum(sum(res_new_cam(1).dzdx,1),2) / prod(size_feats(1:2));
            cam_map_new = bsxfun(@max, sum(bsxfun(@times, res_new_cam(1).x, cam_weights_new),3), 0);
            large_heatmap_new = map2jpg(im2double(imresize(cam_map_new, img_size(1:2))));

            [best_opt_score, best_opt_class_i] = max(res_new_cam(end).x);

            subplot(3,4,5);
            imshow(display_im*0.5 + large_heatmap_new*0.5);
            title(sprintf('CAM Opt (%.3f: %s)', best_opt_score, ...
                get_short_class_name(net, best_opt_class_i, true)));
            
            subplot(3,4,6);
            imshow(display_im*0.5 + large_heatmap_orig*0.5);
            title(sprintf('CAM Orig (%.3f: %s)', sorted_orig_scores(1), ...
                get_short_class_name(net, sorted_orig_class_idx(1), true)));
                [sorted_orig_scores, sorted_orig_class_idx] = sort(res(end-1).x, 'descend');

            subplot(3,4,7);
            plot(transpose(interested_scores(:,1:t)));
            axis square;
            legend([get_short_class_name(net, [squeeze(sorted_orig_class_idx(1:num_top_scores)); target_class], true)]);
            title(sprintf('top %d scores', num_top_scores));

            subplot(3,4,8);
            plot(transpose(E(1,1:t)));
            hold on;
            plot(transpose(E(2,1:t)));
            plot(transpose(E(end,1:t)));
            plot(repmat(orig_err, [1 t]));
            hold off;
            axis square;
            title(sprintf('log(lr) = %.2f, log(lambda) = %.2f, log(tv) = %.2f, beta = %.2f', ...
                log10(opts.learning_rate), log10(opts.lambda), ...
                log10(opts.tv_lambda), opts.beta));

            legend('min','max', 'loss + reg','orig err');


%                     subplot(3,3,8);
%                     imagesc(tv_der);
%                     colorbar;
%                     axis square;
%                     title('tv der');

            subplot(3,4,9);
            imagesc(mean(minscore_der, 3));
            colorbar;
            axis square;
            title('minscore deriv');

            subplot(3,4,10);
            imagesc(mean(maxscore_der, 3));
            colorbar;
            axis square;
            title('maxscore deriv');
            
            subplot(3,4,4);
            imagesc(mean(mask, 3));
            colorbar;
            axis square;
            if ~stop_early
                title('spatial mask');
            else
                title('avg spatial mask');
            end

            subplot(3,4,11);
            imagesc(mask_transform(mean(mask, 3)));
            colorbar;
            axis square;
            if ~stop_early
                title('transf spatial mask');
            else
                title('avg trans spatial mask');
            end

            subplot(3,4,12);
            imshow(display_im*0.5 + hm_mask*0.5);
            if ~stop_early
                title('mask overlay');
            else
                title('avg mask overlay');
            end

            drawnow;
            
            fprintf(strcat('loss at epoch %d: orig: %f, min_score: %f, max_score: %f, sim: %f, reg: %f, tv: %f, tot: %f\n', ...
                'mean derivs at epoch %d: min_score: %f, max_score: %f, sim: %f, reg (unnorm): %f, reg (norm): %f, tv (unnorm): %f, tv (norm): %f\n'), ...
                t, orig_err, E(1,t), E(2,t), E(3,t), E(4,t), E(5,t), E(6,t), t, mean(minscore_der(:)), ...
                mean(maxscore_der(:)), mean(sim_der(:)), mean(reg_der(:)), opts.lambda * mean(reg_der(:)), ...
                mean(abs(tv_der(:))), opts.tv_lambda * mean(abs(tv_der(:))));
            
        end
        
        if stop_early,
            break;
        end
        
    end
    
    %if ~strcmp(opts.save_fig_path, ''),
    prep_path(opts.save_fig_path);
    print(fig, opts.save_fig_path, '-djpeg');
    %end
    
    new_res = struct();
    
%     new_res.mask_t = mask_t(:,:,1:t);
    new_res.error = E(:,1:t);
%     new_res.optimized_feats = x;
%     new_res.actual_feats = actual_feats;
%     new_res.tnet = tnet;
%     new_res.tnet_cam = tnet_cam;
    new_res.target_class = target_class;
%     new_res.img = img;
%     new_res.opts = opts;
    new_res.mask = mean(mask_t(:,:,t-opts.num_average:t),3);
    new_res.actual_feats = actual_feats;
    new_res.layer = layer;
    new_res.opts = opts;
    
    if ~strcmp(opts.save_res_path, ''),
        [folder, ~, ~] = fileparts(opts.save_res_path);
        if ~exist(folder, 'dir')
            mkdir(folder);
        end

        save(opts.save_res_path, 'new_res');
    end
end


% --------------------------------------------------------------------
function [e, dx] = tv(x,beta)
% --------------------------------------------------------------------
    if(~exist('beta', 'var'))
      beta = 1; % the power to which the TV norm is raized
    end
    d1 = x(:,[2:end end],:,:) - x ;
    d2 = x([2:end end],:,:,:) - x ;
    v = sqrt(d1.*d1 + d2.*d2).^beta ;
    e = sum(sum(sum(sum(v)))) ;
    if nargout > 1
      d1_ = (max(v, 1e-5).^(2*(beta/2-1)/beta)) .* d1;
      d2_ = (max(v, 1e-5).^(2*(beta/2-1)/beta)) .* d2;
      d11 = d1_(:,[1 1:end-1],:,:) - d1_ ;
      d22 = d2_([1 1:end-1],:,:,:) - d2_ ;
      d11(:,1,:,:) = - d1_(:,1,:,:) ;
      d22(1,:,:,:) = - d2_(1,:,:,:) ;
      dx = beta*(d11 + d22);
      if(any(isnan(dx)))
      end
    end
end