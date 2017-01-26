%function iterative_mask(
    opts.num_superpixels= 100;
    opts.plot_step = 20;
    opts.null_img = [];
    % TODO: move out
    net = load('/home/ruthfong/packages/matconvnet/data/models/imagenet-caffe-alex.mat');
    imdb_paths = load('/data/ruthfong/ILSVRC2012/val_imdb_paths.mat');
    img_i = 12; %img_idx(3);
    img = cnn_normalize(net.meta.normalization, ...
        imread(imdb_paths.images.paths{img_i}), true);
    %null_img = zeros(size(img), 'single');
    null_img = imgaussfilt(img, 10);

    target_class = imdb_paths.images.labels(img_i);
    res = vl_simplenn(net, img);
    
    gradient = zeros(size(res(end).x), 'single');
    gradient(target_class) = 1;

    [superpixels_labels, num_superpixels] = superpixels(img, opts.num_superpixels);
    
%     figure;
%     BW = boundarymask(superpixels_labels);
%     imshow(imoverlay(uint8(display_img), BW, 'cyan'));

    [~,top_score_i] = max(squeeze(res(end).x));
    modified_img = img;
    available_superpixels = 1:num_superpixels;
    
    running = zeros([1 num_superpixels], 'single');
    mod_imgs = zeros([size(img) num_superpixels], 'single');
    best_score = 0;
    best_t = -1;
    figure;
    for t=1:num_superpixels
        prev = modified_img;
        X = zeros([size(img) length(available_superpixels)], 'single');
        for j=1:length(available_superpixels);
            mask = ones(net.meta.normalization.imageSize(1:2), 'single');
            mask(superpixels_labels == available_superpixels(j)) = 0;
            X(:,:,:,j) = bsxfun(@times, modified_img, mask) ...
                + bsxfun(@times, null_img, 1-mask);
        end
        res = vl_simplenn(net, X);
        target_scores = squeeze(res(end).x(:,:,target_class,:));
        [~,best_i] = max(target_scores);
        scores_best = squeeze(res(end).x(:,:,:,best_i));
        [~,top_score_i] = max(scores_best);
        if top_score_i ~= target_class
            break;
        end
        modified_img = X(:,:,:,best_i);
        available_superpixels = available_superpixels(...
            available_superpixels~=available_superpixels(best_i));
        running(t) = target_scores(best_i);
        mod_imgs(:,:,:,t) = modified_img;
        if mod(t, opts.plot_step) == 0
            subplot(2,1,1);
            imshow(uint8(cnn_denormalize(net.meta.normalization, modified_img)));
            subplot(2,1,2);
            plot(running(1:t));
            drawnow;
        end
    end
    
    subplot(2,1,1);
    imshow(uint8(cnn_denormalize(net.meta.normalization, prev)));
    subplot(2,1,2);
    plot(running(1:t-1));
    
    tt = 80;
    figure;
    imshow(uint8(cnn_denormalize(net.meta.normalization, mod_imgs(:,:,:,tt))));
    title(sprintf('Score %.2f at t=%d', running(tt), tt));
%end