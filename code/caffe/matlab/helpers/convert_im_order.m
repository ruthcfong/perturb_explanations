function imgs_ = convert_im_order(imgs)
    if ~ismatrix(imgs) && size(imgs,3) == 3
        imgs_ = imgs(:,:,[3 2 1],:);
    else
        imgs_ = imgs;
    end
    imgs_ = permute(imgs_, [2 1 3 4]);
end