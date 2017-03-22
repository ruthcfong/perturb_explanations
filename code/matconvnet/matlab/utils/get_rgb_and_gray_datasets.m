function [rgb_norm_imgs, gray_norm_imgs] = get_rgb_and_gray_datasets(paths, ...
    normalization)
    num_images = length(paths);
    rgb_norm_imgs = zeros([normalization.imageSize(1:3) num_images], 'single');
    gray_norm_imgs = zeros([normalization.imageSize(1:3) num_images], 'single');
    for i=1:num_images
        rgb_orig_img = imread(paths{i});
        rgb_norm_imgs(:,:,:,i) = cnn_normalize(normalization, rgb_orig_img, true);
        if ndims(rgb_orig_img) == 3,
            gray_orig_img = repmat(rgb2gray(rgb_orig_img), [1 1 3]);
            gray_norm_imgs(:,:,:,i) = cnn_normalize(normalization, gray_orig_img, true);
        else
            sprintf('%s is already grayscale', paths{i});
            gray_norm_imgs(:,:,:,i) = rgb_norm_imgs(:,:,:,i);
        end
    end
end