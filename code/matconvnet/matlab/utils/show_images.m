function show_images(images)
    figure;
    num_images = size(images,4);
    side_length = ceil(sqrt(num_images));
    for i=1:num_images
        subplot(side_length,side_length,i)
        imshow(normalize(images(:,:,:,i)));
    end
end