function img_ = normalize_img(img, mean_img, new_size)
    if ismatrix(img)
        img_ = imresize(repmat(img, [1 1 3]), new_size(1:2)) - imresize(mean_img, new_size(1:2));
    else
        img_ = imresize(img, new_size(1:2)) - imresize(mean_img, new_size(1:2));
    end
%     if ismatrix(img)
%         img_ = repmat(img, [1 1 3]);
%     else
%         img_ = img;
%     end
%     img_ = imresize(img_, new_size(1:2));
%     mean_c = squeeze(mean(mean(mean_img)));
%     img_ = permute(bsxfun(@minus, permute(img_, [3 1 2]), mean_c), [2 3 1]);
end