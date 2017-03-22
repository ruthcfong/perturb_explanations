function imdb = create_places_imdb(split_csv_filepath, images_dirpath)
%split_csv_filepath = '/data/datasets/places205/trainvalsplit_places205/val_places205.csv';
%images_dirpath = '/data/datasets/places205/images256';

split_csv_file = fopen(split_csv_filepath);
info = textscan(split_csv_file, '%s %d\n'); 

relative_img_paths = info{1};
labels = single(info{2});

assert(length(relative_img_paths) == length(labels));
num_examples = length(labels);

default_img_size = [256,256,3];
data = zeros([default_img_size,num_examples], 'single');
for i=1:num_examples
    fp = fullfile(images_dirpath, relative_img_paths{i});
    img = single(imread(fp));
    switch ndims(img)
        case 2
            img = repmat(img, [1,1,3]);
        case 3
            % do nothing
        otherwise
            assert(false);
    end
    assert(isequal(size(img), default_img_size));
    data(:,:,:,i) = img;
end

data_mean = mean(data, 4);
imdb.images.data = data;
imdb.images.data_mean = data_mean; % note: this is the mean for the given set split
imdb.images.labels = labels;

% works only if the split set has examples for each label
[unique_labels,idx,~] = unique(labels);
classes = cell(size(unique_labels));
for i=1:length(idx)
    ss = strsplit(relative_img_paths{idx(i)},'/');
    classes{unique_labels(i)+1} = ss{2};
end

imdb.meta.classes = classes;

% save('/data/datasets/places205/imdb_val.mat', '-struct', 'imdb','-v7.3');

% the below method missed a few classes:
% classes = {};
% class_i = 1;
% 
% alphabet_dirnames = dir(images_dirpath);
% alphabet_dirnames = alphabet_dirnames(3:end);
% 
% for i=1:length(alphabet_dirnames)
%     letter_dirnames = dir(fullfile(images_dirpath, alphabet_dirnames(i).name));
%     letter_dirnames = letter_dirnames(3:end);
%     num_classes_in_letter = length(letter_dirnames);
%     classes(class_i:class_i+num_classes_in_letter-1) = {letter_dirnames.name};
%     class_i = class_i + num_classes_in_letter;
% end


end