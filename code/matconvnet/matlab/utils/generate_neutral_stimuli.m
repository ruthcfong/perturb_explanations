function generate_neutral_stimuli(net)
    x = rand(net.meta.normalization.imageSize(1:3), 'single');
    res = vl_simplenn(net, x);
    learning_rate = 1e5;
    num_iters = 500;
    E = zeros([1 num_iters], 'single');
    
    % add loss layer
    net.layers{end+1} = struct(...
        'type','mseloss', ...
        'class', zeros(size(res(end).x), 'single'));
    
    figure;
    for t=1:num_iters
        res = vl_simplenn(net, x, 1);
        E(t) = res(end).x;
        x = x - learning_rate*res(1).dzdx;
        if mod(t,50) == 0
            subplot(1,2,1);
            imshow(uint8(cnn_denormalize(net.meta.normalization, x)));
            subplot(1,2,2);
            plot(E(1:t));
            axis square;
            drawnow;
        end
    end
end