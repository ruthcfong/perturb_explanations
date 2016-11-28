function data = get_data_from_paths(paths, normalization)
    num_images = length(paths);
    data = zeros([normalization.imageSize(1:3) num_images],'single');
    for i=1:num_images
        image_path = paths{i};
        try 
            img = imread(image_path);
        catch
            % copy original file to temp file and convert it to RGB
            [~,image_name, ext] = filepaths(image_path);
            temp_image_path = fullfile(tmp_folder, strcat(image_name, ext));
            disp(fprintf('copying %s to %s and converting it to a RGB file\n', ...
                image_path, temp_image_path));
            assert(system(sprintf('cp %s %s', image_path, temp_image_path)) == 0);
            assert(system(sprintf('convert %s -colorspace RGB %s', ...
                temp_image_path, temp_image_path)) == 0);

            img = imread(temp_image_path);
        end
        norm_img = cnn_normalize(normalization, img, true);
        data(:,:,:,i) = norm_img;
    end
end