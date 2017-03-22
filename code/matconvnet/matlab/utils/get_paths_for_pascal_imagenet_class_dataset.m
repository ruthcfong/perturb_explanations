function get_paths_for_pascal_imagenet_class_dataset(set_file, images_dir)
    set_file = '/ramdisk/ruthfong/PASCAL3D+_release1.1/Image_sets/aeroplane_imagenet_train.txt';
    images_dir = '/ramdisk/ruthfong/PASCAL3D+_release1.1/Images/aeroplane_imagenet';
    
    img_names = strsplit(fileread(set_file), '\n');
    if isempty(img_names{end})
        img_names = img_names(1:end-1);
    end
    
    num_images = length(img_names);
    paths = cell([1 num_images]);
    for i=1:num_images,
        paths{i} = fullfile(images_dir, strcat(img_names{i}, '.JPEG'));
    end
end