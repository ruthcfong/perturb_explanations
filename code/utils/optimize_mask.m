function new_res = optimize_mask(net, img, gradient, varargin)
    opts.null_img = [];
    opts.num_iters = 500;
    opts.plot_step = 50;
    
    opts.save_fig_path = '';
    opts.save_res_path = '';
    
    % L1 regularization
    opts.lambda = 1e-7;
    %opts.lambda = 0;
    opts.l1_ideal = 0;
    
    % TV regularization
    opts.tv_lambda = 1e-6;
    %opts.tv_lambda = 0;
    opts.beta = 2;
    
    opts.mask_params.type = 'square_occlusion';
    opts.square_occlusion.opts.size = 75;
    opts.square_occlusion.opts.flip = false;
    opts.square_occlusion.opts.aff_idx = [1,4,5,6];
    opts.square_occlusion.opts.num_transforms = 1;
    
    opts.mask_params.type = 'superpixels';
    opts.superpixels.opts.num_superpixels = 500;
    
    opts.mask_params.type = 'direct';
    opts.direct.opts = struct();
    
    opts.noise.use = false;
    opts.noise.mean = 0;
    opts.noise.std = 1;
    
    opts.update_func = 'adam';
    opts.learning_rate = 1e1;
    opts.adam.beta1 = 0.9;
    opts.adam.beta2 = 0.999;
    opts.adam.epsilon = 1e-8;
    opts.nesterov.momentum = 0.9;
    
    opts.gpu = NaN; % NaN for CPU
    
    opts = vl_argparse(opts, varargin);
    
    switch opts.mask_params.type
        case 'square_occlusion'
            init_params = @init_square_occlusion_params;
            get_mask = @get_mask_from_square_occlusion_params;
            get_params_deriv = @dzdp_square_occlusion;
            clip_params = @clip_square_occlusion_params;
            param_opts = opts.square_occlusion.opts;
        case 'superpixels'
            init_params = @init_superpixels_params;
            get_mask = @get_mask_from_superpixels_params;
            get_params_deriv = @dzdp_superpixels;
            clip_params = @clip_direct_params;
            param_opts = opts.superpixels.opts;
        case 'direct'
            init_params = @init_direct_params;
            get_mask = @get_mask_from_direct_params;
            get_params_deriv = @dzdp_direct;
            clip_params = @clip_direct_params;
            param_opts = opts.direct.opts;
        otherwise
            assert(false);
    end
        
    % initialize mask parameters
    [params, param_opts] = init_params(img, param_opts);
    
    if isempty(opts.null_img)
        opts.null_img = zeros(net.meta.normalization.imageSize(1:3), 'single');
    end
    
    E = zeros([4 opts.num_iters], 'single'); % loss, L1, TV, sum
    
    % initialize variables for update function (adam, momentum)
    m_t = zeros(size(params), 'single');
    v_t = zeros(size(params), 'single');
    
    num_top_scores = 5;
    res = vl_simplenn(net, img);
    %[~, sorted_orig_class_idx] = sort(res(end-1).x, 'descend');
    [~, sorted_orig_class_idx] = sort(res(end).x, 'descend');
    interested_scores = zeros([num_top_scores opts.num_iters]);

    if ~isnan(opts.gpu)
        g = gpuDevice(opts.gpu + 1);
        
        img = gpuArray(img);
        net = vl_simplenn_move(net, 'gpu');
        opts.null_img = gpuArray(opts.null_img);
        params = gpuArray(params);
        E = gpuArray(E);
        m_t = gpuArray(m_t);
        v_t = gpuArray(v_t);
        interested_scores = gpuArray(interested_scores);
        gradient = gpuArray(gradient);
        
        switch opts.mask_params.type
            case 'superpixels'
                param_opts.superpixel_labels = gpuArray(param_opts.superpixel_labels);
        end
    end
    
    fig = figure('units','normalized','outerposition',[0 0 1 1]); % open a maxed out figure
    for t=1:opts.num_iters
        % inject noise (optionally)
        if opts.noise.use
            noise = opts.noise.mean + opts.noise.std*randn(size(params), 'like', params);
        else
            noise = 0;
        end
        
        % create mask (m) and modified input
        mask = get_mask(params + noise, param_opts);
        img_ = bsxfun(@times, img, mask) + bsxfun(@times, opts.null_img, 1-mask);
        
        mask_wo_noise = get_mask(params, param_opts);
        img_wo_noise = bsxfun(@times, img, mask_wo_noise) ...
            + bsxfun(@times, opts.null_img, 1-mask_wo_noise);
        
        % run black-box algorithm on modified input
        res = vl_simplenn(net, img_, gradient);
        
        % save top scores
        %interested_scores(:,t) = res(end-1).x(sorted_orig_class_idx(1:num_top_scores));
        interested_scores(:,t) = res(end).x(sorted_orig_class_idx(1:num_top_scores));
        
        % compute algo error
        err_ind = res(end).x .* gradient;
        E(1,t) = sum(err_ind(:));

        % compute algo derivatives w.r.t. mask parameters
        dzdx_ = res(1).dzdx;
        dzdm = sum(dzdx_.*(img-opts.null_img), 3);
        dzdp = get_params_deriv(dzdm, params, param_opts);
        
        % compute error and derivatives for regularization
        % L1 regularization
        if opts.lambda ~= 0
            E(2,t) = opts.lambda * sum(abs(mask(:)-opts.l1_ideal));
            dl1m = sign(mask-opts.l1_ideal);
            dl1dp = get_params_deriv(dl1m, params, param_opts);
        else
            E(2,t) = 0;
            dl1dp = 0;
        end

        % TV regularization
        if opts.tv_lambda ~= 0
            assert(opts.beta ~= 0);
            [tv_err, dTVdm] = tv(mask, opts.beta);
            E(3,t) = opts.tv_lambda*tv_err;
            dTVdp = get_params_deriv(dTVdm, params, param_opts);
        else
            E(3,t) = 0;
            dTVdp = 0;
        end
        update_gradient = dzdp + opts.lambda*dl1dp + opts.tv_lambda*dTVdp;
        
        E(end,t) = sum(E(1:end-1,t));
        
        % update mask parameters
        switch opts.update_func
            case 'adam'
                m_t = opts.adam.beta1*m_t + (1-opts.adam.beta1)*update_gradient;
                v_t = opts.adam.beta2*v_t + (1-opts.adam.beta2)*(update_gradient.^2);
                m_hat = m_t/(1-opts.adam.beta1^t);
                v_hat = v_t/(1-opts.adam.beta2^t);
                
                params = params - opts.learning_rate./(sqrt(v_hat)+opts.adam.epsilon).*m_hat;
            case 'nesterov_momentum'
                v_t = opts.nesterov.momentum*v_t - opts.learning_rate*update_gradient;
                params = params + opts.nesterov.momentum*v_t - opts.learning_rate*update_gradient;
            case 'gradient_descent'
                params = params - opts.learning_rate*update_gradient;
            otherwise
                assert(false);
        end
        
        params = clip_params(params, param_opts);
        
        % plot progress
        if mod(t, opts.plot_step) == 0
            subplot(3,3,1);
            imshow(uint8(cnn_denormalize(net.meta.normalization, gather(img))));
            title('Orig Img');
            
            subplot(3,3,2);
            imshow(uint8(cnn_denormalize(net.meta.normalization, gather(img_wo_noise))));
            title('Masked Img');
            
            subplot(3,3,3);
            plot(gather(E([1 end],1:t)'));
            axis square;
            legend({'loss','loss+reg'});
            title('Error');
            
            subplot(3,3,4);
            imagesc(gather(mask_wo_noise));
            axis square;
            colorbar;
            title('Mask');
            
            subplot(3,3,5);
            imagesc(gather(dzdm));
            axis square;
            colorbar;
            title('dzdm');
            
            subplot(3,3,6);
            plot(gather(transpose(interested_scores(:,1:t))));
            axis square;
            legend([get_short_class_name(net, [squeeze(sorted_orig_class_idx(1:num_top_scores))], true)]);
            title(sprintf('top %d scores', num_top_scores));
            
            drawnow;
            
            fprintf(strcat('loss at epoch %d: f(x) = %f, l1 = %f, TV = %f\n', ...
                'mean deriv at epoch %d: dzdp = %f, l1 = %f, tv = %f\n'), ...
                t, E(1,t), E(2,t), E(3,t), ...
                t, mean(abs(dzdp(:))), opts.lambda*mean(dl1dp(:)), ...
                opts.tv_lambda*mean(abs(dTVdp(:))));
        end
    end
    
    % save results
    if ~strcmp(opts.save_fig_path, ''),
        prep_path(opts.save_fig_path);
        print(fig, opts.save_fig_path, '-djpeg');
    end
    
    new_res = struct();
    
    %new_res.error = gather(E);
    new_res.mask = gather(mask);
    %new_res.params = params;
    %new_res.param_opts = param_opts;
    %new_res.opts = opts;
    %new_res.gradient = gradient;
    %new_res.num_layers = length(net.layers);
    
    if ~strcmp(opts.save_res_path, ''),
        [folder, ~, ~] = fileparts(opts.save_res_path);
        if ~exist(folder, 'dir')
            mkdir(folder);
        end

        save(opts.save_res_path, 'new_res');
        fprintf('saved to %s\n', opts.save_res_path);
    end
end

function [params, opts] = init_square_occlusion_params(img, varargin)
    opts.img_size = size(img);
    opts.size = 11;
    opts.flip = false;
    opts.aff_idx = [1,4,5,6];
    opts.num_transforms = 1;
    
    opts = vl_argparse(opts, varargin);
    
%     h = opts.img_size(1);
%     w = opts.img_size(2);
%     
%     x_center = floor(opts.size/2) + (w-opts.size)*rand('single');
%     y_center = floor(opts.size/2) + (h-opts.size)*rand('single');
%     
%     % TODO: remove debugging statement
%      x_center = 122;
%      y_center = 88;
    
    aff_nn = dagnn.DagNN();
    aff_nn.conserveMemory = false;
    aff_grid = dagnn.AffineGridGenerator('Ho',opts.img_size(1),'Wo',opts.img_size(2));
    aff_nn.addLayer('aff', aff_grid,{'aff'},{'grid'});
    sampler = dagnn.BilinearSampler();
    aff_nn.addLayer('samp',sampler,{'input','grid'},{'mask'});

    aff = zeros([1 1 6 opts.num_transforms], 'single');
    aff(:,:,1,:) = 1;
    aff(:,:,4,:) = 1;
    aff(:,:,5,:) = -0.5 + rand([1 opts.num_transforms], 'single'); % TODO -- hand-coded
    aff(:,:,6,:) = -0.5 + rand([1 opts.num_transforms], 'single');
    
    %input = zeros(opts.img_size(1:2), 'single');
    %input(1:opts.size,1:opts.size) = 1;
    x1 = 1:opts.img_size(1);
    x2 = 1:opts.img_size(2);
    [X1, X2] = meshgrid(x1,x2);
    F = mvnpdf([X1(:) X2(:)],[floor(opts.img_size(1)/2) floor(opts.img_size(2)/2)],...
        [opts.size^2 0; 0 opts.size^2]);
    F = reshape(F, length(x1), length(x2));
    F = normalize(F);
    input = repmat(single(F), [1 1 opts.num_transforms]);

    opts.aff_nn = aff_nn;
    opts.input = input;
    opts.aff = aff;
     
    params = aff(:,:,opts.aff_idx,:);
end

function [params, opts] = init_superpixels_params(img, varargin)
    opts.num_superpixels = 500;
    opts.img_size = size(img);
    
    opts = vl_argparse(opts, varargin);
    
    [superpixel_labels, opts.num_superpixels] = superpixels(double(img), opts.num_superpixels);

    params = rand([1 opts.num_superpixels], 'single');
    opts.superpixel_labels = superpixel_labels;
end

function [params, opts] = init_direct_params(img, varargin)
    opts.img_size = size(img);
    
    opts = vl_argparse(opts, varargin);
    
    params = rand(opts.img_size(1:2), 'single');
end

function mask = get_mask_from_square_occlusion_params(params, opts)
    aff = opts.aff;
    aff(:,:,opts.aff_idx,:) = params;
    inputs = {'input',opts.input,'aff', aff};
    aff_nn = opts.aff_nn;
    aff_nn.eval(inputs);
    mask = aff_nn.getVar('mask').value;
    if ndims(mask) > 2
        mask = squeeze(mask(:,:,1,:));
        mask = mean(mask, 3);
    end
%     figure;
%     imshow(mask);
%     drawnow;
%     
%     assert(true);
%     subplot(3,3,7);
%     imagesc(mask);
%     axis square;
%     colorbar;
%     drawnow;
%     mask = zeros(opts.img_size(1:2), 'single');
%     
%     x_center = params(1);
%     y_center = params(2);
%     
%     r_start = ceil(y_center - opts.size/2);
%     r_end = r_start + opts.size - 1;
%     c_start = ceil(x_center - opts.size/2);
%     c_end = c_start + opts.size - 1;
%     
%     if r_start < 1 || c_start < 1
%         assert(false);
%     end
%     mask(r_start:r_end,c_start:c_end) = 1;
%     
%     if opts.flip
%         mask = 1 - mask;
%     end
%     
%     assert(isequal(size(mask), opts.img_size(1:2)));
%     if ~isequal(size(mask), opts.img_size(1:2))
%         assert(false);
%     end
end

function mask = get_mask_from_superpixels_params(params, opts)
    mask = zeros(opts.img_size(1:2), 'like', params);
    
    for i=1:opts.num_superpixels
        mask(opts.superpixel_labels == i) = params(i);
    end
end

function mask = get_mask_from_direct_params(params, ~)
    mask = params;
end

function dzdp = dzdp_direct(dzdm, ~, ~)
    dzdp = dzdm;
end
% 
% function dzdp = dzdp_direction(dzdm, params)
%     assert(false);
%     dzdp = dzdm;
% end

function dzdp = dzdp_square_occlusion(dzdm, params, opts)
    aff = opts.aff;
    aff(:,:,opts.aff_idx,:) = params;
    inputs = {'input',opts.input,'aff',aff};
    aff_nn = opts.aff_nn;
    if ndims(opts.input) > 2
        [~,~,N] = size(opts.input);
        dzdm = repmat(dzdm/N, [1 1 N N]);
    end
    derOutputs = {'mask', dzdm};
    aff_nn.eval(inputs, derOutputs);
    dzdp = aff_nn.getVar('aff').der;

%     subplot(3,3,8);
%     imagesc(squeeze(dzdp));
%     colorbar;
%     drawnow;
    if ndims(opts.input) == 2
        dzdp = dzdp(opts.aff_idx); % only allow translation (dof 5 and 6)
    else
        dzdp = dzdp(:,:,opts.aff_idx,:);
    end
%     x_center = params(1);
%     y_center = params(2);
%     
%     r_start = ceil(y_center - opts.size/2);
%     r_end = r_start + opts.size - 1;
%     c_start = ceil(x_center - opts.size/2);
%     c_end = c_start + opts.size - 1;
% 
%     dzdm_occ = dzdm(r_start:r_end,c_start:c_end);
%     
%     dx = dzdm_occ(:,2:end) - dzdm_occ(:,1:end-1);
%     dzdx_center = sum(dx(:));
%     dy = dzdm_occ(2:end,:) - dzdm_occ(1:end-1,:);
%     dzdy_center = sum(dy(:));
%     
%     if ~opts.flip
%         dzdx_center = -dzdx_center;
%         dzdy_center = -dzdy_center;
%     end
%     
%     subplot(3,3,7);
%     imagesc(dzdm_occ);
%     colorbar;
%     axis square;
%     
%     subplot(3,3,8);
%     imagesc(dx);
%     colorbar;
%     axis square;
%     
%     subplot(3,3,9);
%     imagesc(dy);
%     colorbar;
%     axis square;
%     
%     drawnow;
%     
%     fprintf('x = %f, y = %f, dzdx = %f, dzdy = %f\n', ...
%         x_center, y_center, dzdx_center, dzdy_center);
%     
%     dzdp = [dzdx_center dzdy_center];
end

function dzdp = dzdp_superpixels(dzdm, params, opts)
    dzdp = zeros(size(params), 'like', params);
    for i=1:opts.num_superpixels
        dzdp(i) = sum(dzdm(opts.superpixel_labels == i));
    end
%       dzdp = arrayfun(@(i) sum(dzdm(opts.superpixel_labels == i)), 1:opts.num_superpixels);
end

function params = clip_square_occlusion_params(params, opts)
    translation_idx = [5, 6];
    for ti = translation_idx
        if(~isempty(find(opts.aff_idx == ti, 1)))
            i = find(opts.aff_idx == ti);
            params(:,:,i,:) = min(max(params(:,:,i,:), -0.5), 0.5);
        end
    end
%     params(1) = min(max(params(1), -1.30), 0); % TODO: fix hardcoding
%     params(2) = min(max(params(2), -1.30), 0);

%     params(5) = min(max(params(5), -0.5), 0.5); % TODO: fix hardcoding
%     params(6) = min(max(params(6), -0.5), 0.5);
end

function params = clip_direct_params(params, ~)
    params(params > 1) = 1;
    params(params < 0) = 0;
end

function [e, dx] = tv(x,beta)
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
%       if(any(isnan(dx)))
%       end
    end
end
