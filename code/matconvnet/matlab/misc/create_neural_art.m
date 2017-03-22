function x_art = create_neural_art(net, x_content, x_style, varargin)
    opts.content_layer = floor(length(net.layers)/2);
    opts.style_layers = length(net.layers)-1;
    opts.style_weights = 1/length(opts.style_layers)*ones(...
        [1 length(opts.style_layers)], 'single');
    opts.alpha = 1;
    opts.beta = 1e-3;
    
    opts.adam.beta1 = 0.999;
    opts.adam.beta2 = 0.999;
    opts.adam.epsilon = 1e-8;
    
    opts.gpu = 0;
    
    opts.debug = false;
    
    opts.learning_rate = 1;
    
    opts.num_iters = 5000;
    opts.plot_step = 100;
    opts.output_image = 'out.png';
    opts.output_fig = 'out.jpg';
    
    opts = vl_argparse(opts, varargin);
    
    assert(length(opts.style_layers) == length(opts.style_weights));
    
    num_style_layers = length(opts.style_layers);
    
    E = zeros([3, opts.num_iters]); 
    
    % change all pool layers to avg pool layers
    for i=1:length(net.layers)
        if strcmp(net.layers{i}.type, 'pool') ...
                && ~strcmp(net.layers{i}.method, 'avg')
            net.layers{i}.method = 'avg';
        end
    end
    
    x_art = rand(size(x_content), 'single');
    
    res_content = vl_simplenn(net, x_content);
    content_target = res_content(opts.content_layer+1).x;
    net_content = truncate_net(net, 1, opts.content_layer);
    net_content.layers{end+1} = struct('type', 'mseloss', ...
        'class', content_target);
    
    res_style = vl_simplenn(net, x_style);
    content_style = {};
    net_styles = {};
    for i=1:num_style_layers
        l=opts.style_layers(i);
        content_style{i} = res_style(l+1).x;
        net_styles{i} = truncate_net(net, 1 , l);
        net_styles{i}.layers{end+1} = struct('type', 'gramsimloss', ...
            'class', content_style{i});
    end
    
    % for adam update
    m_t = zeros(size(x_art), 'single');
    v_t = zeros(size(x_art), 'single');
    
    % Move to img and net to gpu.
    if ~isnan(opts.gpu)
        g = gpuDevice(opts.gpu + 1);

        x_art = gpuArray(x_art);
        net_content = vl_simplenn_move(net_content, 'gpu');
        for i=1:num_style_layers
            net_styles{i} = vl_simplenn_move(net_styles{i}, 'gpu');
        end
    end

    f = figure;
    for t=1:opts.num_iters
        res_content_sim = vl_simplenn(net_content, x_art, 1);
        E(1,t) = gather(res_content_sim(end).x);
        content_der = gather(res_content_sim(1).dzdx);
        
        styles_loss = zeros(size(opts.style_weights));
        styles_ders = zeros([size(x_art), num_style_layers]);
        weighted_style_der = zeros(size(x_art));
        
        for i=1:num_style_layers 
            res_style_sim = vl_simplenn(net_styles{i}, x_art, 1);
            styles_loss(i) = gather(res_style_sim(end).x);
            styles_ders(:,:,:,i) = gather(res_style_sim(1).dzdx);
            weighted_style_der = weighted_style_der ...
                + opts.style_weights(i)*styles_ders(:,:,:,i);
        end
        
        E(2,t) = sum(opts.style_weights.*styles_loss);
        
        E(3,t) = opts.alpha*E(1,t) + opts.beta*E(2,t);
        
        gradient = opts.alpha*content_der + opts.beta*weighted_style_der;
df
        % adam update
        if isfield(opts, 'adam')
            m_t = opts.adam.beta1*m_t + (1-opts.adam.beta1)*gradient;
            v_t = opts.adam.beta2*v_t + (1-opts.adam.beta2)*(gradient.^2);
            m_hat = m_t/(1-opts.adam.beta1^t);
            v_hat = v_t/(1-opts.adam.beta2^t);

            x_art = x_art - opts.learning_rate/(sqrt(v_hat)+opts.adam.epsilon).*m_hat;
        % vanilla gradient descent
        else
            x_art = x_art - opts.learning_rate*gradient;
        end
        
        if t == opts.num_iters || mod(t-1, opts.plot_step) == 0
            subplot(2,3,1);
            imshow(normalize(gather(x_art)));
            title('Generated');
            
            subplot(2,3,2);
            imshow(normalize(x_content));
            title('Content');
            
            subplot(2,3,3);
            imshow(normalize(x_style));
            title('Style');
            
            subplot(2,3,4);
            imagesc(mean(gather(content_der),3));
            colorbar;
            axis square;
            title('Content Der');
            
            subplot(2,3,5);
            imagesc(mean(gather(weighted_style_der),3));
            colorbar;
            axis square;
            title('Weighted Styles Der');
            
            subplot(2,3,6);
            plot(opts.alpha*E(1,1:t)');
            hold on;
            plot(opts.beta*E(2,1:t)');
            plot(E(3,1:t)');
            hold off;
            title('Loss');
            legend('Content', 'Style','Tot');
            
            drawnow;
        end
        
        if opts.debug
            fprintf('weighted loss @ epoch %d: content = %f, style = %f, = %f\n', ...
                t, opts.alpha*E(1,t), opts.beta*E(2,t), E(3,t));
            fprintf('avg unweighted derivs @ epoch %d: conten = %f, style = %f\n', ...
                t, mean(content_der(:)), mean(weighted_style_der(:)));
        end
    end
    
    print(f, opts.output_fig, '-djpeg');
    imwrite(normalize(x_art), opts.output_image);
   
end