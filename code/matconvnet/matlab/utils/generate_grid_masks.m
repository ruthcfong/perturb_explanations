function new_res = generate_grid_masks(net, img, gradient, varargin)
    opts.flip = false;
    opts.sigma = 500;
    opts.num_centers = 36;
    opts.gpu = NaN;
    opts.save_fig_path = '';
    opts.save_res_path = '';
    
    opts = vl_argparse(opts, varargin);
    
    img_size = net.meta.normalization.imageSize;
    
    x1 = 1:img_size(1);
    x2 = 1:img_size(2);
    [X1, X2] = meshgrid(x1,x2);

    assert(img_size(1) == img_size(2));
    
    num_side = ceil(sqrt(opts.num_centers));
    centers = linspace(1,img_size(1),num_side);
    
    opts.num_centers = num_side^2;
    
    gaussians = zeros([img_size(1:3) num_side^2], 'single');

    for i = 1:num_side
        for j = 1:num_side
            F = mvnpdf([X1(:) X2(:)],[centers(i) centers(j)],...
                [opts.sigma opts.sigma]);
            gaussians(:,:,:,i+(j-1)*num_side) = repmat(normalize(reshape(F, length(x1), length(x2))), [1 1 3]);
        end
    end

%     figure;
%     idx = randperm(num_side^2,9);
%     for i=1:9
%         subplot(3,3,i);
%         imagesc(gaussians(:,:,1,idx(i)));
%         axis square;
%     end
    
    if opts.flip
        gaussians = 1 - gaussians;
    end
    
    isDag = isfield(net, 'params') || isprop(net, 'params');

    if isDag
        net = dagnn.DagNN.loadobj(net);
        net.mode = 'test';
        order = net.getLayerExecutionOrder();
        input_i = net.layers(order(1)).inputIndexes;
        output_i = net.layers(order(end)).outputIndexes;
        assert(length(input_i) == 1);
        assert(length(output_i) == 1);
        input_name = net.vars(input_i).name;
        net.vars(input_i).precious = 1;
        softmax_i = find(arrayfun(@(l) isa(l.block, 'dagnn.SoftMax'), net.layers));
        assert(length(softmax_i) == 1);
        softmax_i = net.layers(softmax_i).outputIndexes;
        net.vars(softmax_i).precious = 1;
    else
        softmax_i = find(cellfun(@(l) strcmp(l.type, 'softmax'), net.layers));
        assert(length(softmax_i) == 1);
%         if length(softmax_i) == 0
%             softmax_i = length(net.layers) - 1;
%         end
    end

    
%     figure;
%     imagesc(sum(gaussians(:,:,1,:),4));
%     colorbar;
%     axis square;
    
    input = bsxfun(@times, gaussians, img);
    
    if ~isnan(opts.gpu)
        g = gpuDevice(opts.gpu+1);
        input = gpuArray(input);
        if isDag
            inputs = {input_name, input};
            net.move('gpu');
        else
            net = vl_simplenn_move(net, 'gpu');
        end
    end
    
    if isDag
        net.eval(inputs);
        scores = net.vars(softmax_i).value;
    else
        res = vl_simplenn(net, input);
        scores = res(end).x;
    end
    
    target_scores = squeeze(sum(bsxfun(@times, scores, gradient),3));
    if opts.flip % flip again so the gaussian peaks are weighted
        gaussians = 1 - gaussians;
    end

    weighted_scores = permute(bsxfun(@times, permute(gaussians, [4 1 2 3]), target_scores), ...
        [2 3 4 1]);
    heatmap = sum(weighted_scores(:,:,1,:), 4)./sum(gaussians(:,:,1,:), 4);
    
%     [~, sorted_idx] = sort(target_scores);
    if opts.flip % flip again to display & save the originally used gaussian peaks
        gaussians = 1 - gaussians;
    end

    fig = figure;
    subplot(1,3,1);
    imshow(uint8(cnn_denormalize(net.meta.normalization, img)));
    title('Orig Img');
    subplot(1,3,2);
    imagesc(normalize(heatmap));
    colorbar;
    axis square;
    title('Norm Mask');
    subplot(1,3,3);
    imshow(normalize(img)*0.5 + 0.5*normalize(map2jpg(gather(heatmap))));
    title('Mask Overlay');
%     subplot(3,3,4);
%     imshow(gaussians(:,:,:,sorted_idx(1)));
%     title('Bottom1');
%     subplot(3,3,5);
%     imshow(gaussians(:,:,:,sorted_idx(2)));
%     title('Bottom2');
%     subplot(3,3,6);
%     imshow(gaussians(:,:,:,sorted_idx(3)));
%     title('Bottom3');
%     subplot(3,3,7);
%     imshow(gaussians(:,:,:,sorted_idx(end)));
%     title('Top1');
%     subplot(3,3,8);
%     imshow(gaussians(:,:,:,sorted_idx(end-1)));
%     title('Top2');
%     subplot(3,3,9);
%     imshow(gaussians(:,:,:,sorted_idx(end-2)));
%     title('Top3');

    if ~isempty(opts.save_fig_path)
        prep_path(opts.save_fig_path);
        print(fig, opts.save_fig_path, '-djpeg');
    end
    
    new_res = struct();    
    new_res.mask = normalize(gather(heatmap));
    new_res.opts = opts;
    %new_res.gaussians = gaussians;
    new_res.target_scores = gather(target_scores);
    
    assert(isequal(size(new_res.mask),net.meta.normalization.imageSize(1:2)));
        
    if ~isempty(opts.save_res_path)
        [folder, ~, ~] = fileparts(opts.save_res_path);
        if ~exist(folder, 'dir')
            mkdir(folder);
        end

        save(opts.save_res_path, 'new_res');
        fprintf('saved to %s\n', opts.save_res_path);
    end
end
