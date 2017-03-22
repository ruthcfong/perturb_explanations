function imdb = build_imagenet_class_imdb(class_imdb_paths, normalization)
    data = get_data_from_paths(class_imdb_paths.images.paths, normalization);
    images = struct();
    images.data = data;
    images.labels = class_imdb_paths.images.labels;
    
    imdb = struct();
    imdb.images = images;
    imdb.meta = class_imdb_paths.meta;
end