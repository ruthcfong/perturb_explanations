function %function iterative_mask(
    opts.num_superpixels= 50;
    % TODO: move out
    net = load('/home/ruthfong/packages/matconvnet/data/models/imagenet-caffe-alex.mat');
    imdb_paths = load('/data/ruthfong/ILSVRC2012/val_imdb_paths.mat');
    img_i = 3; %img_idx(3);
    img = cnn_normalize(net.meta.normalization, ...
        imread(imdb_paths.images.paths{img_i}), true);
    null_img = zeros(size(img), 'single');
    display_img = cnn_denormalize(net.meta.normalization, img);
    display_null_img = cnn_denormalize(net.meta.normalization, null_img);
    [Ah, Av] = imgrad(display_img);
    [Bh, Bv] = imgrad(display_null_img);
    X = display_null_img;
    Fh = Ah;
    Fv = Av;
    mask = 1 - mask;
    m = find(mask);
    for i=1:length(m)
        [r,c] = ind2sub([227 227], m(i));
        X(r,c,:) = display_img(r,c,:);
        Fh(r,c,:) = Bh(r,c,:);
        Fv(r,c,:) = Bv(r,c,:);
    end
        
    Y = PoissonJacobi(X,Fh,Fv,repmat(mask,[1 1 3]));
    
    figure; 
    subplot(1,2,1); imshow(uint8(X));
    subplot(1,2,2); imshow(uint8(Y));
    
    boxSrc = [92 178 50 139];
    posDest = [92 50];
    
    imr = poissonSolver(display_null_img(:,:,1), display_img(:,:,1), boxSrc, posDest);
    img = poissonSolver(display_null_img(:,:,2), display_img(:,:,2), boxSrc, posDest);
    imb = poissonSolver(display_null_img(:,:,3), display_img(:,:,3), boxSrc, posDest);

    imnew = composeRGB(imr, img, imb);
    figure;
    imshow(mat2gray(imnew))
    
    imr = poissonSolver(display_img(:,:,1), display_null_img(:,:,1), boxSrc, posDest);
    img = poissonSolver(display_img(:,:,2), display_null_img(:,:,2), boxSrc, posDest);
    imb = poissonSolver(display_img(:,:,3), display_null_img(:,:,3), boxSrc, posDest);

    imnew = composeRGB(imr, img, imb);
    figure;
    imshow(mat2gray(imnew))

    
    target_class = imdb_paths.images.labels(img_i);
    res = vl_simplenn(net, img);
    gradient = zeros(size(res(end).x), 'single');
    gradient(target_class) = 1;

    [superpixels_labels, num_superpixels] = superpixels(img, 100);
    figure;
    BW = boundarymask(superpixels_labels);
    imshow(imoverlay(uint8(display_img), BW, 'cyan'));
    
    mask = zeros([227 227], 'single');
    mask(superpixels_labels ==35) = 1;
    figure; imshow(normalize(bsxfun(@times, img, mask)));
    
    figure; imagesc(superpixels_labels); colorbar;
%     for t=1:num_superpixels
%         res = vl_simplenn(net, X);
%     end
%end