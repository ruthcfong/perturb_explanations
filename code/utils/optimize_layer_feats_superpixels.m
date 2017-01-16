function new_res = optimize_layer_feats_superpixels(net, img, target_class, layer, varargin)
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
    opts.loss = 'softmaxloss';
    opts.mask_init = 'rand';
    opts.error_stopping_threshold = 2.5e-3;
    opts.num_class_labels = 0;
    opts.num_superpixels = 500;

    opts.mask_dims = 2; % obsolete
    opts.denom_reg = false; % obsolete
    opts.num_masks = 1; % obsolete
    
    opts.adam.beta1 = 0.999;
    opts.adam.beta2 = 0.999;
    opts.adam.epsilon = 1e-8;

    opts.mask_transform_function = 'linear';
    opts.gen_sig_c = 0.5;
    opts.gen_sig_a = 10;
    
    opts = vl_argparse(opts, varargin);

    type = 'single';
    type_fh = @single;

    gen_sig_func = @(x) (1./(1+exp(-opts.gen_sig_a*(x-opts.gen_sig_c))));
    d_gen_sig_func = @(x) (opts.gen_sig_a*gen_sig_func(x).*(1-gen_sig_func(x)));
    linear_func = @(x) (x);
    d_linear_func = @(x) (ones(size(x), type));

    switch opts.mask_transform_function
        case 'linear'
            mask_transform = linear_func;
            d_mask_transform = d_linear_func;
        case 'generalized_sigmoid'
            mask_transform = gen_sig_func;
            d_mask_transform = d_gen_sig_func;
    end
        
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

    switch opts.mask_init
        case 'rand'
            mask_init_func = @rand;
        case 'ones'
            mask_init_func = @ones;
        case 'half'
            mask_init_func = @(s, t) (0.5 * ones(s, t));
    end
    
    assert(opts.mask_dims == 2);
    [superpixel_labels, num_superpixels] = superpixels(img, opts.num_superpixels);
    premask = mask_init_func([1 num_superpixels],type);
    mask = zeros([size_feats(1:2) 1], type);
    for i=1:num_superpixels
        mask(superpixel_labels == i) = premask(i);
    end
    mask_t = zeros([size_feats(1:2) opts.num_iters],type);
    
    E = zeros([5 opts.num_iters]);

    % for adam update
    m_t = zeros(size(premask), type);
    v_t = zeros(size(premask), type);

    tnet_cam = tnet;
    tnet_cam.layers = tnet_cam.layers(1:end-1); % exclude softmax loss layer
    gradient = zeros(size(res(end-1).x), type);
    gradient(target_class) = 1;
    res_orig_cam = vl_simplenn(tnet_cam, actual_feats, gradient);
    
    neg_gradient = zeros(size(res(end-1).x), type);

    tnet_mseloss = tnet;
    tnet_mseloss.layers{end} = struct('type', 'mseloss');
    tnet_mseloss.layers{end}.class = res_orig_cam(end).x;
    
    cam_weights_orig = sum(sum(res_orig_cam(1).dzdx,1),2) / prod(size_feats(1:2));
    cam_map_orig = bsxfun(@max, sum(bsxfun(@times, res_orig_cam(1).x, cam_weights_orig),3), 0);
    large_heatmap_orig = map2jpg(im2double(imresize(cam_map_orig, img_size(1:2))));

    display_im = normalize(img+net.meta.normalization.averageImage);

    [sorted_orig_scores, sorted_orig_class_idx] = sort(res(end-1).x, 'descend');
    num_top_scores = 5;
    interested_scores = zeros([num_top_scores+1 opts.num_iters]);
       
    if opts.num_class_labels ~= 0
        % don't include target class unless it's in the top K class labels
        gradient = zeros(size(res(end-1).x), type);
        neg_gradient = zeros(size(res(end-1).x), type);
        gradient(sorted_orig_class_idx(1:opts.num_class_labels)) = 1;
        neg_gradient(sorted_orig_class_idx(1:opts.num_class_labels)) = -1;
    end
    
    switch opts.loss
        case 'softmaxloss'
            orig_err = -log(exp(res(end-1).x(:,:,target_class))/(...
                sum(exp(res(end-1).x))));
        case 'max_softmaxloss'
            orig_err = -log(exp(res(end-1).x(:,:,target_class))/(...
                sum(exp(res(end-1).x))));
        case 'min_classlabel'
            orig_err = res_orig_cam(end).x(:,:,target_class);
        case 'max_classlabel'
            orig_err = res_orig_cam(end).x(:,:,target_class);
            neg_gradient(target_class) = -1;
        case 'preserve_class_vector'
            orig_err = 0;
        otherwise
            assert(false);
    end
    
    %null_feats = net.meta.normalization.averageImage;
    
    assert(opts.num_masks == 1);
    
    fig = figure('units','normalized','outerposition',[0 0 1 1]); % open a maxed out figure
    for t=1:opts.num_iters,
        mask_t(:,:,t) = mask;
        x = bsxfun(@times, actual_feats, mask_transform(mask));
        
        % L1 loss
        if strcmp(opts.loss, 'min_classlabel') || strcmp(opts.loss, 'max_softmaxloss')
            E(2,t) = opts.lambda * sum(abs(mask(:)-1));
        else
            E(2,t) = opts.lambda * sum(abs(mask(:)));
        end
        
        % use tv-norm for spatial mask only
        if opts.tv_lambda ~= 0
            assert(opts.beta ~= 0);
            [tv_err, tv_der_m] = tv(mask, opts.beta);
            tv_der = arrayfun(@(i) sum(tv_der_m(superpixel_labels == i)), 1:num_superpixels);
        else
            tv_err = 0;
            tv_der = zeros(size(premask), type);
        end
        
        E(3,t) = opts.tv_lambda* tv_err;
        
        switch opts.loss
            case 'softmaxloss'
                tres = vl_simplenn(tnet, x, 1);
                E(1,t) = tres(end).x;
            case 'max_softmaxloss';
                tres = vl_simplenn(tnet, x, -1);
                E(1,t) = tres(end).x;
            case 'min_classlabel'
                tres = vl_simplenn(tnet_cam, x, gradient);
                E(1,t) = tres(end).x(:,:,target_class);
            case 'max_classlabel'
                tres = vl_simplenn(tnet_cam, x, neg_gradient);
                E(1,t) = tres(end).x(:,:,target_class);
            case 'preserve_class_vector'
               tres = vl_simplenn(tnet_mseloss, x, 1);
               E(1,t) = tres(end).x;
            otherwise
                assert(false);
        end
        
        E(end,t) = sum(E(1:end-1,t));
                 
        if ~isempty(strfind(opts.loss, 'softmax')) || ~isempty(strfind(opts.loss, 'preserve'))
            interested_scores(1:num_top_scores,t) = tres(end-1).x(sorted_orig_class_idx(1:num_top_scores));
            interested_scores(end,t) = tres(end-1).x(target_class);
        else
            interested_scores(1:num_top_scores,t) = tres(end).x(sorted_orig_class_idx(1:num_top_scores));
            interested_scores(end,t) = tres(end).x(target_class);
        end
        
        softmax_der = bsxfun(@times, d_mask_transform(mask), sum(tres(1).dzdx.*actual_feats,3)); %.*d_gen_sig(mask, c, a);
        premask_der = arrayfun(@(i) sum(softmax_der(superpixel_labels == i)), 1:num_superpixels);
        
        if strcmp(opts.loss, 'min_classlabel') || strcmp(opts.loss, 'max_softmaxloss')
            reg_der = arrayfun(@(i) sum(sign(mask(superpixel_labels == i)-1)), 1:num_superpixels);
        else
            reg_der = arrayfun(@(i) sum(sign(mask(superpixel_labels == i))), 1:num_superpixels);
            % hinge regularization
            % reg_der = reg_der .* (softmax_der >= 0);
        end
%         
%         update_gradient = softmax_der + opts.lambda*reg_der + opts.tv_lambda*tv_der;

        update_gradient = premask_der + opts.lambda*reg_der + opts.tv_lambda*tv_der;
        
        % adam update
        if isfield(opts, 'adam')
            m_t = opts.adam.beta1*m_t + (1-opts.adam.beta1)*update_gradient;
            v_t = opts.adam.beta2*v_t + (1-opts.adam.beta2)*(update_gradient.^2);
            m_hat = m_t/(1-opts.adam.beta1^t);
            v_hat = v_t/(1-opts.adam.beta2^t);

            premask = premask - opts.learning_rate./(sqrt(v_hat)+opts.adam.epsilon).*m_hat;
        % vanilla gradient descent
        else
            premask = premask - opts.learning_rate*update_gradient;
        end

        % clip mask
        premask(premask > 1) = 1;
        premask(premask < 0) = 0;
        
        % update spatial mask
        for i=1:num_superpixels
            mask(superpixel_labels == i) = premask(i);
        end

        
% plotting
        if t == opts.num_iters || (mod(t-1,opts.plot_step) == 0)

            subplot(3,4,1);
            imshow(normalize(img));
            title('Orig Img');

            subplot(3,4,2);
            actual_max_feat_map = res(layer+1).x(:,:,max_feature_idx);
            curr_saliency_map = get_saliency_map_from_difference_map(...
                actual_max_feat_map - x(:,:,max_feature_idx), layer, rf_info, img_size);
            curr_saliency_map_rep = repmat(normalize(curr_saliency_map),[1 1 3]);
            imshow(display_im.*curr_saliency_map_rep);
            title('Diff Max Feat Saliency');

            subplot(3,4,3);
            curr_saliency_map = get_saliency_map_from_difference_map(mean(res(layer+1).x ...
                - x,3), layer, rf_info, img_size);
            curr_saliency_map_rep = repmat(normalize(curr_saliency_map),[1 1 3]);
            imshow(display_im.*curr_saliency_map_rep);
            title('Diff Avg Feats Saliency');

%                     imshow(normalize(x));
%                     title('Masked Img');

            subplot(3,4,7);
            hm_avg_diff_feats = map2jpg(im2double(imresize(...
                normalize(curr_saliency_map),img_size(1:2))));
            imshow(display_im*0.5 + hm_avg_diff_feats*0.5);
            title('Diff Avg Feats Saliency');

            subplot(3,4,4);
            curr_saliency_map = get_saliency_map_from_difference_map(mean(mask, 3), layer, rf_info, img_size);
            curr_saliency_map_rep = repmat(normalize(curr_saliency_map),[1 1 3]);
            imshow(display_im.*curr_saliency_map_rep);
            title('Mask Saliency');

%                     hm_new = map2jpg(im2double(imresize(bsxfun(@max,sum(...
%                         bsxfun(@times, x, mean(mean(tres(1).dzdx,1),2)),3),0), img_size(1:2))), [], 'jet');
%                     hm_old = map2jpg(im2double(imresize(bsxfun(@max,sum(...
%                         bsxfun(@times, actual_feats,...
%                         mean(mean(res(layer+1).dzdx(:,:,:,opts.img_i),1),2)),3),0), img_size(1:2))), [], 'jet');
%                     display_im = normalize(img+imdb.images.data_mean);

            hm_mask = map2jpg(im2double(imresize(mean(mask, 3),img_size(1:2))));

            res_new_cam = vl_simplenn(tnet_cam, x, gradient);
            cam_weights_new = sum(sum(res_new_cam(1).dzdx,1),2) / prod(size_feats(1:2));
            cam_map_new = bsxfun(@max, sum(bsxfun(@times, res_new_cam(1).x, cam_weights_new),3), 0);
            large_heatmap_new = map2jpg(im2double(imresize(cam_map_new, img_size(1:2))));

            if ~isempty(strfind(opts.loss, 'softmax')) || ~isempty(strfind(opts.loss, 'preserve'))
                [best_opt_score, best_opt_class_i] = max(tres(end-1).x);
            else
                [best_opt_score, best_opt_class_i] = max(tres(end).x);
            end

            subplot(3,4,5);
            imshow(display_im*0.5 + large_heatmap_new*0.5);
            title(sprintf('CAM Opt (%.3f: %s)', best_opt_score, ...
                get_short_class_name(net, best_opt_class_i, true)));
            subplot(3,4,6);
            imshow(display_im*0.5 + large_heatmap_orig*0.5);
            title(sprintf('CAM Orig (%.3f: %s)', sorted_orig_scores(1), ...
                get_short_class_name(net, sorted_orig_class_idx(1), true)));
                [sorted_orig_scores, sorted_orig_class_idx] = sort(res(end-1).x, 'descend');

            subplot(3,4,8);
            plot(transpose(interested_scores(:,1:t)));
            axis square;
            legend([get_short_class_name(net, [squeeze(sorted_orig_class_idx(1:num_top_scores)); target_class], true)]);
            title(sprintf('top %d scores', num_top_scores));

            subplot(3,4,10);
            imagesc(mean(mask, 3));
            colorbar;
            axis square;
            title('spatial mask');

%                     subplot(3,3,8);
%                     imagesc(tv_der);
%                     colorbar;
%                     axis square;
%                     title('tv der');

             subplot(3,4,11);
             imshow(display_im*0.5 + hm_mask*0.5);
             title('mask overlay');

            subplot(3,4,9);
            imagesc(mean(softmax_der, 3));
            colorbar;
            axis square;
            title('loss deriv');

            subplot(3,4,12);
            plot(transpose(E(1,1:t)));
            hold on;
            plot(transpose(E(end,1:t)));
            plot(repmat(orig_err, [1 t]));
            hold off;
            axis square;
            title(sprintf('log(lr) = %.2f, log(lambda) = %.2f, log(tv_lambda) = %.2f, beta = %.2f', ...
                log10(opts.learning_rate), log10(opts.lambda), ...
                log10(opts.tv_lambda), opts.beta));

            legend('Loss','Loss + Reg','Orig Loss');

            drawnow;
                
            fprintf(strcat('loss at epoch %d: orig: %f, softmax: %f, reg: %f, tv: %f, tot: %f\n', ...
                'derivs at epoch %d: softmax: %f, reg (unnorm): %f, reg (norm): %f, tv (unnorm): %f, tv (norm): %f\n'), ...
                t, orig_err, E(1,t), E(2,t), E(3,t), E(end,t), t, mean(softmax_der(:)), ...
                mean(reg_der(:)), opts.lambda * mean(reg_der(:)), ...
                mean(abs(tv_der(:))), opts.tv_lambda * mean(abs(tv_der(:))));
        end
        
    end

    if ~strcmp(opts.save_fig_path, ''),
        prep_path(opts.save_fig_path);
        print(fig, opts.save_fig_path, '-djpeg');
    end
    
    new_res = struct();
    
%     new_res.mask_t = mask_t;
    new_res.error = E;
%     new_res.optimized_feats = x;
%     new_res.actual_feats = actual_feats;
%     new_res.tnet = tnet;
%     new_res.tnet_cam = tnet_cam;
%     new_res.target_class = target_class;
%     new_res.img = img;
%     new_res.opts = opts;
    new_res.mask = mask;
    new_res.premask = premask;
    new_res.superpixel_labels = superpixel_labels;
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