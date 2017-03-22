function new_res = generate_random_masks(net, img, target_class, varargin)    
    opts.num_transforms = 100;
    opts.size = 25;
    opts.flip = false;
    opts.num_top = 10;
    opts.gpu = NaN;
    opts.null_img = [];
    opts.save_fig_path = '';
    opts.save_res_path = '';
    
    opts = vl_argparse(opts, varargin);
    
    if isempty(opts.null_img)
        opts.null_img = zeros(size(img), 'like', img);
    end
    
    img_size = size(img);
    aff_nn = dagnn.DagNN();
    aff_nn.conserveMemory = false;
    aff_grid = dagnn.AffineGridGenerator('Ho',img_size(1),'Wo',img_size(2));
    aff_nn.addLayer('aff', aff_grid,{'aff'},{'grid'});
    sampler = dagnn.BilinearSampler();
    aff_nn.addLayer('samp',sampler,{'input','grid'},{'mask'});

    aff = zeros([1 1 6 opts.num_transforms], 'single');
    aff(:,:,1,:) = 2*rand([1 opts.num_transforms], 'single');
    aff(:,:,2,:) = 0;%-2 + 4*rand([1 opts.num_transforms], 'single');
    aff(:,:,3,:) = 0;%-2 + 4*rand([1 opts.num_transforms], 'single');
    aff(:,:,4,:) = 2*rand([1 opts.num_transforms], 'single');
    aff(:,:,5,:) = -0.5 + rand([1 opts.num_transforms], 'single');
    aff(:,:,6,:) = -0.5 + rand([1 opts.num_transforms], 'single');

    x1 = 1:img_size(1);
    x2 = 1:img_size(2);
    [X1, X2] = meshgrid(x1,x2);
    F = mvnpdf([X1(:) X2(:)],[floor(img_size(1)/2) floor(img_size(2)/2)],...
        [opts.size^2 0; 0 opts.size^2]);
    F = reshape(F, length(x1), length(x2));
    F = normalize(F);
    input = repmat(single(F), [1 1 opts.num_transforms]);
    
    imgs = repmat(img, [1 1 1 opts.num_transforms]);
    null_imgs = repmat(opts.null_img, [1 1 1 opts.num_transforms]);
    if ~isnan(opts.gpu)
        g = gpuDevice(opts.gpu+1);
        aff = gpuArray(aff);
        input = gpuArray(input);
        aff_nn.move('gpu');
        imgs = gpuArray(imgs);
        null_imgs = gpuArray(null_imgs);
        net = vl_simplenn_move(net, 'gpu');
    end
    inputs = {'input',input,'aff', aff};
    
    aff_nn.eval(inputs);
    mask = aff_nn.getVar('mask').value;
    if opts.flip
        mask = 1 - mask;
    end
    
    composite_imgs = imgs .* mask(:,:,1:3,:) + null_imgs .* (1 - mask(:,:,1:3,:));
    res = vl_simplenn(net, composite_imgs);
    
    if opts.flip
        [~,best_idx] = sort(gather(res(end).x(:,:,target_class,:)));
    else
        [~,best_idx] = sort(gather(res(end).x(:,:,target_class,:)),'descend');
    end
    
    fig = figure;
    subplot(2,3,1);
    imshow(uint8(cnn_denormalize(net.meta.normalization, img)));
    title('Orig Img');
    
    subplot(2,3,2);
    imshow(uint8(cnn_denormalize(net.meta.normalization, gather(composite_imgs(:,:,:,best_idx(1))))));
    title('Composite Img');
    
    subplot(2,3,3);
    imshow(0.5*normalize(img) ...
        + 0.5*map2jpg(im2double(gather(mask(:,:,1,best_idx(1))))));
    title('Mask Overlay');
    
    subplot(2,3,4);
    imshow(0.5*normalize(img) ...
        + 0.5*map2jpg(im2double(mean(gather(mask(:,:,1,best_idx(1:opts.num_top))),4))));
    title(sprintf('Avg Top%d Masks', opts.num_top));
    
    
    % save results
    if ~isempty(opts.save_fig_path),
        prep_path(opts.save_fig_path);
        print(fig, opts.save_fig_path, '-djpeg');
    end
    
    new_res = struct();
    new_res.mask = gather(mask(:,:,1,best_idx(1)));
    new_res.mean_top_mask = mean(gather(mask(:,:,1,best_idx(1:opts.num_top))),4);
    new_res.top_masks = squeeze(gather(mask(:,:,1,best_idx(1:opts.num_top))));
    new_res.num_top = opts.num_top;
    new_res.sorted_idx = best_idx;
    new_res.end_res = gather(res(end).x);
    
    if ~strcmp(opts.save_res_path, ''),
        [folder, ~, ~] = fileparts(opts.save_res_path);
        if ~exist(folder, 'dir')
            mkdir(folder);
        end

        save(opts.save_res_path, 'new_res');
        fprintf('saved to %s\n', opts.save_res_path);
    end
%     figure; 
%     for i=1:9
%         subplot(3,3,i);
%         imagesc(gather(mask(:,:,1,i))); axis square; colorbar;
%     end
%     figure;
%     imagesc(sum(gather(mask(:,:,1,:)), 4));
%     axis square;
%     colorbar;
end