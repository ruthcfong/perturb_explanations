function img_ = denormalize_img(img, mean_img)
    img_ = img + imresize(mean_img, [size(img,1) size(img,2)]);
end