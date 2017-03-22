%% compute null reference
null = mean(imdb.images.data(:,:,:,batch_range),4);
figure; imshow(normalize(null));

%% comparison image
img_i = 83;
layer = 13;
class_offset = 1;
% figure; imshow(normalize(imdb.images.data(:,:,:,batch_range(img_i)) + ...
%     imdb.images.data_mean));

net.layers{end}.class = imdb.images.labels(batch_range(img_i)) + class_offset;
res_null = vl_simplenn(net, null, 1);

[~, max_feature_idx] = max(sum(sum(res(layer+1).x(:,:,:,img_i),1),2));

% plot max feature map
% figure; imagesc(res(layer+1).x(:,:,max_feature_idx,img_i));

test_image_idx = [1 12 21 32 42 52 61 79 83];

%
null_diff = res(layer+1).x(:,:,max_feature_idx,img_i) - res_null(layer+1).x(:,:,max_feature_idx);
img_size = size(net.meta.normalization.averageImage);
saliency_map_for_null = zeros([img_size(1), img_size(2)]);
for r=1:size(null_diff,1)
    for c=1:size(null_diff,2)
        [r_start,r_end, c_start, c_end] = get_patch_coordinates(...
            rf_info, layer, img_size, [r,c]);
        saliency_map_for_null(r_start:r_end,c_start:c_end) = ...
            saliency_map_for_null(r_start:r_end,c_start:c_end) + null_diff(r,c);
    end
end

img = imdb.images.data(:,:,:,batch_range(img_i));
figure;
subplot(2,3,1);
imagesc(res(layer+1).x(:,:,max_feature_idx,img_i));
colorbar; 
title('Orig Img');
subplot(2,3,2);
imagesc(res_null(layer+1).x(:,:,max_feature_idx));
title('Null Ref');
colorbar;
subplot(2,3,3);
imagesc(res(layer+1).x(:,:,max_feature_idx,img_i) - res_null(layer+1).x(:,:,max_feature_idx));
colorbar;
title('Diff');
subplot(2,3,4);
imshow(normalize(img+imdb.images.data_mean));
title('Orig Img')
subplot(2,3,5);
imshow(normalize(saliency_map_for_null));
title('Sal Map from Ref Diff')
subplot(2,3,6);
imshow(normalize((img+imdb.images.data_mean).*repmat(normalize(saliency_map_for_null),[1 1 3])));
title('Dot Product');

%%
img = imdb.images.data(:,:,:,batch_range(img_i));
occ_img = img;
occ_range = 1:10;
occ_img(occ_range, occ_range, :) = 0;
res_occ = vl_simplenn(net, occ_img, 1);

%% plot difference in max feature map between that of image and occluded image
figure; 
subplot(3,1,1);
imagesc(res(layer+1).x(:,:,max_feature_idx,img_i));
colorbar;
title('Orig Img');
subplot(3,1,2);
imagesc(res_occ(layer+1).x(:,:,max_feature_idx));
title('Null Ref');
colorbar;
subplot(3,1,3);
imagesc(res(layer+1).x(:,:,max_feature_idx,img_i) - res_occ(layer+1).x(:,:,max_feature_idx));
colorbar;

%%
orig_feature_map = res(layer+1).x(:,:,max_feature_idx,img_i);
map = compute_occlusion_map(net, img, layer, max_feature_idx, orig_feature_map);

%%
figure;
subplot(1,3,1);
imagesc(normalize(null_diff));
colorbar;
title('Null Diff');
subplot(1,3,2);
imagesc(normalize(mean(map,3)));
title('Mean Occ Diff');
colorbar;
subplot(1,3,3);
imagesc(normalize(null_diff) - normalize(mean(map,3)));
colorbar;
title('Diff btwn Null and Mean Occ Diff');

%%
saliency_map_for_occ = zeros([img_size(1),img_size(2)]);
for n=1:num_steps*num_steps,
    r = 1 + step_size*ceil(n/num_steps);
    c = 1 + step_size*(mod(n-1,num_steps)+1);
    if c + step_size - 1 > img_size(2) || r + step_size -1 > img_size(1)
        continue
    end
    saliency_map_for_occ(r:r+step_size-1,c:c+step_size-1) = ...
        saliency_map_for_occ(r:r+step_size-1,c:c+step_size-1) + sum(sum(map(:,:,n)));
end

figure;
imagesc(saliency_map_for_occ);
colorbar;

%%
figure;
subplot(1,3,1);
imshow(normalize(img+imdb.images.data_mean));
title('Orig Img')
subplot(1,3,2);
imshow(normalize(saliency_map_for_occ));
title('Sal Map from Occ')
subplot(1,3,3);
imshow(normalize((img+imdb.images.data_mean).*repmat(normalize(saliency_map_for_occ),[1 1 3])));
title('Dot Product');

%%
saliency_map_same_method = zeros([img_size(1), img_size(2)]);
for r=1:size(map,1)
    for c=1:size(map,2)
        [r_start,r_end, c_start, c_end] = get_patch_coordinates(...
            rf_info, layer, img_size, [r,c]);
        saliency_map_same_method(r_start:r_end,c_start:c_end) = ...
            saliency_map_same_method(r_start:r_end,c_start:c_end) + mean(map(r,c,:),3);
    end
end

figure;
subplot(1,3,1);
imshow(normalize(img+imdb.images.data_mean));
title('Orig Img')
subplot(1,3,2);
imshow(normalize(saliency_map_same_method));
title('Sal Map from Occ')
subplot(1,3,3);
imshow(normalize((img+imdb.images.data_mean).*repmat(normalize(saliency_map_same_method),[1 1 3])));
title('Dot Product');

%%
figure;
subplot(1,3,1);
imagesc(normalize(saliency_map_for_null));
colorbar;
title('Null Saliency Map');
subplot(1,3,2);
imagesc(normalize(saliency_map_same_method));
colorbar;
title('Occ Saliency Map Same Method');
subplot(1,3,3);
imagesc(normalize(saliency_map_for_null)-normalize(saliency_map_same_method));
colorbar;
title('Diff');