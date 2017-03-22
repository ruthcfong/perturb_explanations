function new_res = optimize_zoom(net, img, gradient, target_class, varargin)
    opts.null_img = [];
    opts.num_iters = 500;
    opts.plot_step = 50;
    
    opts.save_fig_path = '';
    opts.save_res_path = '';
            
    opts.noise.use = false;
    opts.noise.mean = 0;
    opts.noise.std = 1e-3;
    
    opts.update_func = 'adam';
    opts.learning_rate = 1e1;
    opts.adam.beta1 = 0.9;
    opts.adam.beta2 = 0.999;
    opts.adam.epsilon = 1e-8;
    opts.nesterov.momentum = 0.9;
    
    opts = vl_argparse(opts, varargin);
    
    % initialize parameters
    [params, aff_nn] = init_zoom_params(img);
    
    params_t = zeros([length(params) opts.num_iters], 'single');
    
    if isempty(opts.null_img)
        opts.null_img = zeros(net.meta.normalization.imageSize(1:3), 'single');
    end
    
    E = zeros([4 opts.num_iters], 'single'); % loss, L1, TV, sum
    
    % initialize variables for update function (adam, momentum)
    m_t = zeros(size(params), 'single');
    v_t = zeros(size(params), 'single');
    
    num_top_scores = 5;
    res = vl_simplenn(net, img);
    [~, sorted_orig_class_idx] = sort(res(end).x, 'descend');
    interested_idx = [squeeze(sorted_orig_class_idx(1:num_top_scores)); target_class];
    interested_scores = zeros([num_top_scores+1 opts.num_iters]);

    fig = figure('units','normalized','outerposition',[0 0 1 1]); % open a maxed out figure
    for t=1:opts.num_iters
        params_t(:,t) = params;

        % inject noise (optionally)
        if opts.noise.use
            noise = opts.noise.mean + opts.noise.std*randn(size(params), 'single');
        else
            noise = zeros(size(params), 'single');
        end
        
        % create zoomed input img
        img_ = get_zoom_img(params + noise, img, aff_nn);
                
        % run black-box algorithm on modified input
        res = vl_simplenn(net, img_, gradient);
        
        % save top scores
        interested_scores(:,t) = res(end).x(interested_idx);
        
        % compute algo error
        err_ind = res(end).x .* gradient;
        E(1,t) = sum(err_ind(:));

        % compute algo derivatives w.r.t. zoom parameters
        dzdx_ = res(1).dzdx;
        dzdp = dzdp_zoom(dzdx_, params, img, aff_nn);

        update_gradient = dzdp;
        
        E(end,t) = sum(E(1:end-1,t));
        
        % update parameters
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
        
        params = clip_zoom_params(params);
        
        % plot progress
        if mod(t, opts.plot_step) == 0
            subplot(3,3,1);
            imshow(uint8(cnn_denormalize(net.meta.normalization, img)));
            title('Orig Img');
            
            subplot(3,3,2);
            imshow(uint8(cnn_denormalize(net.meta.normalization, img_)));
            title('Zoomed Img');
            
            subplot(3,3,3);
            plot(E([1 end],1:t)');
            axis square;
            legend({'loss','loss+reg'});
            title('Error');
            
            subplot(3,3,4);
            plot(params_t(:,1:t)');
            legend({'scale','xt','yt'});
            title('Params');
            
%             subplot(3,3,5);
%             imagesc(dzdx);
%             axis square;
%             colorbar;
%             title('dzdm');
            
            subplot(3,3,6);
            plot(transpose(interested_scores(:,1:t)));
            axis square;
            legend([get_short_class_name(net, interested_idx, true)]);
            title(sprintf('top %d+target scores', num_top_scores));
            
            drawnow;
            
            fprintf(strcat('epoch %d: f(x) = %f, s = %f, xt = %f, yt = %f\n', ...
                'derivs: s = %f, xt = %f, yt = %f\n'), ...
                t, E(1,t), params(1), params(2), params(3), ...
                dzdp(1), dzdp(2), dzdp(3));
        end
    end
    
    % save results
    if ~strcmp(opts.save_fig_path, ''),
        prep_path(opts.save_fig_path);
        print(fig, opts.save_fig_path, '-djpeg');
    end
    
    new_res = struct();
    
    new_res.error = E;
    new_res.params = params;
    new_res.opts = opts;
    new_res.gradient = gradient;
    new_res.num_layers = length(net.layers);
    
    if ~strcmp(opts.save_res_path, ''),
        [folder, ~, ~] = fileparts(opts.save_res_path);
        if ~exist(folder, 'dir')
            mkdir(folder);
        end

        save(opts.save_res_path, 'new_res');
    end
end


function [params, aff_nn] = init_zoom_params(img)    
    img_size = size(img);
    aff_nn = dagnn.DagNN();
    aff_nn.conserveMemory = false;
    aff_grid = dagnn.AffineGridGenerator('Ho',img_size(1),'Wo',img_size(2));
    aff_nn.addLayer('aff', aff_grid,{'aff'},{'grid'});
    sampler = dagnn.BilinearSampler();
    aff_nn.addLayer('samp',sampler,{'input','grid'},{'output'});

    params = zeros([1 3], 'single'); % scaling factor, x_translation, y_translation
    params(1) = 0.9;
end

function aff = get_aff_from_params(params)
    aff = zeros([1 1 6], 'single');
    aff(1) = params(1);
    aff(4) = params(1);
    aff(5) = params(2);
    aff(6) = params(3);
end

function zoomed_img = get_zoom_img(params, img, aff_nn)
    aff = get_aff_from_params(params);
    inputs = {'input',img,'aff', aff};
    aff_nn.eval(inputs);
    zoomed_img = aff_nn.getVar('output').value;
%     figure;
%     imshow(normalize(zoomed_img));
end

function dzdp = dzdp_zoom(dzdx, params, img, aff_nn)
    aff = get_aff_from_params(params);
    inputs = {'input', img,'aff',aff};
    derOutputs = {'output', dzdx};
    aff_nn.eval(inputs, derOutputs);
    dzdaff = aff_nn.getVar('aff').der;
    dzdp = zeros(size(params), 'single');
    dzdp(1) = (dzdaff(1) + dzdaff(2))/2; % not sure if we need to divide
    dzdp(2) = dzdaff(5);
    dzdp(3) = dzdaff(6);
end

function params = clip_zoom_params(params)
% do nothing for now
end

% function [e, dx] = tv(x,beta)
%     if(~exist('beta', 'var'))
%       beta = 1; % the power to which the TV norm is raized
%     end
%     d1 = x(:,[2:end end],:,:) - x ;
%     d2 = x([2:end end],:,:,:) - x ;
%     v = sqrt(d1.*d1 + d2.*d2).^beta ;
%     e = sum(sum(sum(sum(v)))) ;
%     if nargout > 1
%       d1_ = (max(v, 1e-5).^(2*(beta/2-1)/beta)) .* d1;
%       d2_ = (max(v, 1e-5).^(2*(beta/2-1)/beta)) .* d2;
%       d11 = d1_(:,[1 1:end-1],:,:) - d1_ ;
%       d22 = d2_([1 1:end-1],:,:,:) - d2_ ;
%       d11(:,1,:,:) = - d1_(:,1,:,:) ;
%       d22(1,:,:,:) = - d2_(1,:,:,:) ;
%       dx = beta*(d11 + d22);
%       if(any(isnan(dx)))
%       end
%     end
% end
