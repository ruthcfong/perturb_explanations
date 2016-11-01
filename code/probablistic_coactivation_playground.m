%% set parameters
load_network = true;
load_data = true;
show_figures = true;

end_l = 7;
start_l = 5;

batch_range = 1:1000;
batch_size = length(batch_range);

%% load variables

% load network
if load_network
    load('/Users/brian/matconvnet-1.0-beta21/data/trained_nets/mnist-net.mat');
end

% load images
if load_data
    imdb = load('/Users/brian/matconvnet-1.0-beta21/data/mnist/imdb.mat');
end

%% forward and backward pass
net.layers{end}.class = imdb.images.labels(batch_range);
res = vl_simplenn(net, imdb.images.data(:,:,:,batch_range),1);

%%
threshold = 0;
size_x = size(res(start_l+1).x);
assert(length(size_x) == 4);
num_hus = size_x(3);
acts = reshape(res(start_l+1).x, [prod(size_x(1:end-1)) size_x(end)]);
num_pos_acts = sum(acts > threshold, 1);
mu_pos_acts = mean(num_pos_acts);
std_pos_acts = std(num_pos_acts);

if true
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
threshold = 0;
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

%%
[coeff, score, latent, tsquared, explained, mu] = pca(acts);
exp_threshold = .90*100;
explained_cumsum = cumsum(explained_cumsum);
exp_thres_id = find(explained_cumsum > exp_threshold,1);

%%
layernum = 5;
% num_top_filters = 20;
constraint_type = 'mask';

num_top_filters_range = 5:5:500;
errors = zeros(size(num_top_filters_range), 'single');

for i=1:length(num_top_filters_range)
    num_top_filters = num_top_filters_range(i);
    net_c = add_constrained_layer(net, layernum, num_top_filters, constraint_type);

    net_c.layers{end}.class = imdb.images.labels(batch_range); 
    res_c = vl_simplenn(net_c, imdb.images.data(:,:,:,batch_range),1);
    errors(i) = res_c(end).x;
end

%%
figure; 
plot(num_top_filters_range, errors);
xlabel('Number of Top Filters Used');
ylabel('Error');
title('Error wrt Constraining the Number of Top Filters Used');
