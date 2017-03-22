imdb_paths = load('/data/ruthfong/ILSVRC2012/annotated_train_imdb_paths.mat');
load('/data/ruthfong/ILSVRC2012/annotated_train_heldout_idx.mat');
f = fopen('/home/ruthfong/packages/caffe/data/ilsvrc12/annotated_train_imdb.txt', 'w');
for i=1:length(imdb_paths.images.labels)
    fprintf(f, '%s %d\n', imdb_paths.images.paths{i}, imdb_paths.images.labels(i) - 1);
end
fclose(f);

f = fopen('/home/ruthfong/packages/caffe/data/ilsvrc12/annotated_train_heldout_imdb.txt', 'w');
for i=heldout_idx
    fprintf(f, '%s %d\n', imdb_paths.images.paths{i}, imdb_paths.images.labels(i) - 1);
end
fclose(f);