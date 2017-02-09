function generate_grid_masks(net, img, target_class, varargin)
    opts.flip = false;
    opts.sigma = 500;
    opts.num_centers = 36;
    
    opts = vl_argparse(opts, varargin);
    
    img_size = net.meta.normalization.imageSize;
    
    x1 = 1:img_size(1);
    x2 = 1:img_size(2);
    [X1, X2] = meshgrid(x1,x2);

    assert(img_size(1) == img_size(2));
    centers = linspace(1,img_size(1),ceil(sqrt(opts.num_centers)));
    num_side = length(centers);
    gaussians = zeros([img_size(1:3) num_side^2], 'single');

    for i = 1:num_side
        for j = 1:num_side
            F = mvnpdf([X1(:) X2(:)],[centers(i) centers(j)],...
                [opts.sigma opts.sigma]);
            gaussians(:,:,:,i+(j-1)*num_side) = repmat(normalize(reshape(F, length(x1), length(x2))), [1 1 3]);
        end
    end

    figure;
    idx = randperm(num_side^2,9);
    for i=1:9
        subplot(3,3,i);
        imagesc(gaussians(:,:,1,idx(i)));
        axis square;
    end
    
    if opts.flip
        gaussians = 1 - gaussians;
    end
    
    figure;
    imagesc(sum(gaussians(:,:,1,:),4));
    colorbar;
    axis square;
    
    res = vl_simplenn(net, bsxfun(@times, gaussians, img));
    target_scores = squeeze(res(end).x(1,1,target_class,:));
    weighted_scores = permute(bsxfun(@times, permute(gaussians, [4 1 2 3]), target_scores), ...
        [2 3 4 1]);
    figure;
    subplot(1,2,1);
    imshow(uint8(cnn_denormalize(net.meta.normalization, img)));
%     subplot(1,3,2);
    heatmap = sum(weighted_scores(:,:,1,:), 4)./sum(gaussians(:,:,1,:), 4);
%     imagesc(heatmap);
%     colorbar;
%     axis square;
    subplot(1,2,2);
    imshow(normalize(img)*0.5 + 0.5*normalize(map2jpg(heatmap)));
    
    %input = repmat(single(F), [1 1 opts.num_transforms]);
%end
