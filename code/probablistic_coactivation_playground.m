%% set parameters
load_network = true;
load_data = true;
show_figures = true;
is_local = false;
dataset = 'places';


switch dataset
    case 'mnist' 
        end_l = 7;
        start_l = 1;
        batch_range = 1:1000;
        class_offset = 0;
    case 'places' % conv5 in places alexnet
        end_l = 20;
        start_l = 13;
        batch_range = 1:10:20500;
        class_offset = 1; % needed for places alexnet
        if exist('places_net')
            net = places_net;
        end
        if exist('places_imdb')
            imdb = places_imdb;
        end
        if exist('places_res')
            res = places_res;
        end
    otherwise
        assert(false);
end

%% load variables

% load network
if load_network
    if strcmp(dataset, 'mnist')
        if is_local
            load('/Users/brian/matconvnet-1.0-beta21/data/trained_nets/mnist-net.mat');
        else
            load('/home/ruthfong/packages/matconvnet/data/mnist-baseline-simplenn/net-final.mat');
        end
    else
        net = load('/home/ruthfong/neural_coding/models/places-caffe-ref-upgraded-tidy-with-classes.mat');
    end
end

% load images
if load_data
    if strcmp(dataset, 'mnist')
        if is_local
            imdb = load('/Users/brian/matconvnet-1.0-beta21/data/mnist/imdb.mat');
        else
            imdb = load('/home/ruthfong/packages/matconvnet/data/mnist-baseline-simplenn/imdb.mat');
        end
    else
        imdb = load('/data/datasets/places205/imdb_val_resized_227.mat');
    end
end

%% forward and backward pass

% for places alexnet, add a loss layer
if strcmp(dataset, 'places') && ~isequal(net.layers{end}.type, 'softmaxloss')
    net.layers{end+1} = struct('type', 'softmaxloss') ;
end

net.layers{end}.class = imdb.images.labels(batch_range) + class_offset;
res = vl_simplenn(net, imdb.images.data(:,:,:,batch_range),1);

%%
threshold = 0;
size_x = size(res(start_l+1).x);
assert(length(size_x) == 4);
num_hus = size_x(3);

acts = res(start_l+1).x;
% acts = reshape(res(start_l+1).x, [prod(size_x(1:end-1)) size_x(end)]);
% acts = squeeze(mean(mean(res(start_l+1).x,1),2)); % take the average activation -- i know this is a bit problematic
% acts = squeeze(mean(mean(res(start_l+1).x .* res(start_l+1).dzdx,1),2));
num_pos_acts = sum(acts > threshold, 1);
mu_pos_acts = mean(num_pos_acts);
std_pos_acts = std(num_pos_acts);

if false
   figure; 
   h = histogram(num_pos_acts);
   h.Normalization = 'probability';
   title(sprintf('>%f Activations at Layer %d (mean = %.2f, std = %.2f, BS = %d)', ...
       threshold, start_l, mu_pos_acts, std_pos_acts, batch_size));
   xlabel('Number of Positive Activations');
   ylabel('% of HUs');
end

%%
curr_hu = 1;
threshold = mean(acts(curr_hu,:));
curr_hu_threshold_idx = find(acts(curr_hu,:) > threshold);
figure;
for h=1:num_hus
    if h == curr_hu
        continue;
    end
    h_threshold_idx = find(acts(h, curr_hu_threshold_idx) > threshold);
    histogram(acts(h, curr_hu_threshold_idx(h_threshold_idx)));
    hold on;
end
hold off;

%% PCA on activations
[coeff, score, latent, tsquared, explained, mu] = pca(acts');
exp_threshold = .90*100;
explained_cumsum = cumsum(explained);
exp_thres_id = find(explained_cumsum > exp_threshold,1);

figure;
plot(1-explained_cumsum/100);
xlabel('Number of Components Included');
ylabel('% of Variance Unexplained by first K components');
title('Variance Unexplained by the first K PCA components');

%%
scoeff = repmat(sum(coeff,1),[256,1]); % some sums are negative...
i = 1;
[~,sorted_idx] = sort(coeff(i,:),'descend');
j = 66;
k = 25;
t1 = 195;
t2 = 199;

%% CC and clustering
[R,P] = corrcoef(acts');
RR = 1 - abs(R);
rr = squareform(RR);
Z = linkage(rr);
figure;
[H,T,outperm] = dendrogram(Z,0);

figure; 
imagesc(R(outperm,outperm)); 
colorbar;
axis square;
title('Clustered HU Correlations');

%% investigating individual dendrogram clusters
start_i = 1;
end_i = 12;

if false
    display_range = max(1,start_i):min(length(outperm),end_i+1);
    figure;
    imagesc(R(outperm(display_range,display_range)));
end

for i=start_i:end_i
    for j=start_i:end_i
        if i == j
            continue;
        end
        % TODO finish
    end
end

%% explore relationship between two hidden unuits
i = 180;
j = 188;
bv = vertcat(acts(i,:), acts(j,:));
bv = bv';

plot_heatmap_of_two_units(bv)
plot_fitted_mvn_pdf(bv);

%% plot responsible activations
opts = struct();
opts.batch_range = batch_range;
opts.space_type = 'all';
opts.all_step = 4;
opts.num_images = 64;
show_images_sorted_by_activation(imdb, acts(1,:), opts);


%% check the number of positive values in the analysis layer
max_acts = squeeze(max(max(res(start_l+1).x, [], 1), [], 2));
pos_max_acts_per_example = sum(max_acts > 0, 1);
disp(sprintf('Number of Positive Maximum Activations across all Batch Examples: Max: %d, Mean: %.2f, Std: %.4f', ...
    max(pos_max_acts_per_example), mean(pos_max_acts_per_example), std(pos_max_acts_per_example)));

%% Add a constrained layer and observe how error varies with the number of top filters used
% num_top_filters = 20;
constraint_type = 'mask';

feature_map_size = size(res(start_l+1).x);
switch dataset
    case 'mnist'
        num_top_filters_range = 10:10:feature_map_size(3);
    case 'places'
        num_top_filters_range = 8:8:feature_map_size(3);
end
errors = zeros(size(num_top_filters_range), 'single');

for i=1:length(num_top_filters_range)
    num_top_filters = num_top_filters_range(i);
    net_c = add_constrained_layer(net, start_l, num_top_filters, constraint_type);

    net_c.layers{end}.class = imdb.images.labels(batch_range) + class_offset; 
    res_c = vl_simplenn(net_c, imdb.images.data(:,:,:,batch_range),1);
    errors(i) = res_c(end).x;
end

figure; 
plot(num_top_filters_range, errors);
xlabel('Number of Top Filters Used');
ylabel('Error');
title(sprintf('Error wrt Constraining the Number of Top Filters Used (# HU = %d, mean # HU w pos acts = %d)', ...
    feature_map_size(3), mean(pos_max_acts_per_example)));