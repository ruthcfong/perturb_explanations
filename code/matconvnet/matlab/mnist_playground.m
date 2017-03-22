% imdb = load('/Users/brian/matconvnet-1.0-beta21/data/mnist/imdb.mat');
% load('/Users/brian/matconvnet-1.0-beta21/data/trained_nets/mnist-net.mat');
% 
% batch_range = 1:1000;
net.layers{end}.class = imdb.images.labels(batch_range);
res = vl_simplenn(net, imdb.images.data(:,:,:,batch_range),1);

%%
layernum = 5; 
num_top = 9;
activity_l = res(layernum+1).x;
size_activity_l = size(activity_l);
num_filters = size_activity_l(3);

filternum = 1;

activity_filt = res(layernum+1).x(:,:,filternum,:);
[~,sorted_idx] = sort(activity_filt);

filter_basis = zeros(size_activity_l);
filter_basis(:,:,filternum,:) = 1;

filter_basis_r = reshape(filter_basis, [prod(size_activity_l(1:end-1)), size_activity_l(end)]);
activity_l_r = reshape(activity_l, [prod(size_activity_l(1:end-1)), size_activity_l(end)]);

dot_res = dot(activity_l_r, filter_basis_r, 1);
[~,sorted_dot_idx] = sort(dot_res);

%% show results from top three "closest" random basis
for n=1:3
    rand_filter_basis = randn([1 1 num_filters]);
    rand_filter_basis_rep = repmat(rand_filter_basis, size_activity_l(1), ...
        size_activity_l(2), 1, size_activity_l(end));
    rand_filter_basis_r = reshape(rand_filter_basis_rep, ...
        [prod(size_activity_l(1:end-1)), size_activity_l(end)]);

    dot_rand_res = dot(activity_l_r, rand_filter_basis_r, 1);
    [~,sorted_dot_rand_idx] = sort(dot_rand_res);

    [r_activity,p_activity] = corrcoef(squeeze(activity_filt(sorted_idx(end-100:end))), ...
        dot_rand_res(sorted_dot_rand_idx(end-100:end)));
    [r_basis,p_basis] = corrcoef(filter_basis_r(:,1), rand_filter_basis_r(:,1));

    p_comparator = p_basis;
    p_threshold = 0.05;
    
    while abs(p_comparator) > p_threshold
        rand_filter_basis = randn([1 1 num_filters]);
        rand_filter_basis_rep = repmat(rand_filter_basis, size_activity_l(1), ...
            size_activity_l(2), 1, size_activity_l(end));
        rand_filter_basis_r = reshape(rand_filter_basis_rep, ...
            [prod(size_activity_l(1:end-1)), size_activity_l(end)]);

        dot_rand_res = dot(activity_l_r, rand_filter_basis_r, 1);

    end
    disp(['activity correlation: r = ', num2str(r_activity(1,2)), ', p = ', num2str(p_activity(1,2))])
    disp(['basis correlation: r = ', num2str(r_basis(1,2)), ', p = ', num2str(p_basis(1,2))])

    figure;
    %         text(0.75,1.25,sprintf('Top %d Activations for HU %d in Layer %s', num_top, filternum, layernum));
    for i=1:num_top
        subplot(sqrt(num_top), sqrt(num_top), i);
        %imshow(imdb.images.data(:,:,:,batch_range(sorted_idx(end-i+1))));
        %title(num2str(activity_filt(sorted_idx(end-i+1))));
        imshow(imdb.images.data(:,:,:,batch_range(sorted_dot_rand_idx(end-i+1))));
        title(num2str(dot_rand_res(sorted_dot_rand_idx(end-i+1))));

    end
end

%% show top activations for current HU
figure;
%         text(0.75,1.25,sprintf('Top %d Activations for HU %d in Layer %s', num_top, filternum, layernum));
for i=1:num_top
    subplot(sqrt(num_top), sqrt(num_top), i);
    imshow(imdb.images.data(:,:,:,batch_range(sorted_idx(end-i+1))));
    title(num2str(activity_filt(sorted_idx(end-i+1))));
end

%% show histogram distribution of either random basis activations or real activiations
filternum = 1;
activity_filt = res(layernum+1).x(:,:,filternum,:);

figure;
num_classes = length(net.meta.classes.name);
for c=1:num_classes
    class_idx = find(imdb.images.labels(batch_range) == c);
    %subplot(2,5,c);
    histogram(activity_filt(:,:,:,class_idx));
    hold on;
    %histogram(dot_rand_res(class_idx));
    %title(num2str(c-1));
end
hold off;
legend(net.meta.classes.name);
title(sprintf('Activation Distribution for L%d HU %d (BS = %d)', ...
    layernum, filternum, length(batch_range)));
xlabel('Activation');
ylabel('# of HUs');

%% show range of activations for a HU
filternum = 1;
activity_filt = res(layernum+1).x(:,:,filternum,:);
[~,sorted_idx] = sort(activity_filt);
space_range = 1:10:length(batch_range);
figure;
for i=1:length(space_range)
    subplot(10,10,i);
    id = batch_range(sorted_idx(space_range(i)));
    imshow(imdb.images.data(:,:,:,id));
    title(sprintf('%.2f',activity_filt(:,:,:,id)));
end

%% show top or bottom sorted activations for a HU
filternum = 1;
activity_filt = res(layernum+1).x(:,:,filternum,:);
[~,sorted_idx] = sort(activity_filt);
num_top = 100;
show_top = false;
figure;
for i=1:num_top
    subplot(10,10,i);
    if show_top
        id = batch_range(sorted_idx(end-i+1));
    else
        id = batch_range(sorted_idx(i));
    end
    imshow(imdb.images.data(:,:,:,id));
    title(sprintf('%.1f',activity_filt(:,:,:,id)));
end

%% test the normal-ness of all filters in a layer

pvs_gaussian = zeros([1 num_filters]);
for f=1:num_filters
    activity_filt = res(layernum+1).x(:,:,f,:);
    [~,pvs_gaussian(f)] = jbtest(squeeze(activity_filt));
end

%%
figure;
h = histogram(pvs_gaussian);
h.Normalization = 'probability';
title(sprintf('L%d all HUs pvs from JB test', layernum));
xlabel('p-value');
ylabel(sprintf('%% of HUs (%d tot.)', num_filters));
% set(gca,'xscale','log');

%non_gaussian_idx = find(pvs_gaussian > 0.01);
%non_gaussian_idx
%pvs_gaussian(non_gaussian_idx)

%%
[~, sorted_pv_idx] = sort(pvs_gaussian);
filter_range = 1:25;

figure;
for i=1:length(filter_range)
    filternum = sorted_pv_idx(end-i+1);
    activity_filt = res(layernum+1).x(:,:,filternum,:);
    subplot(5,5,i);
    histogram(activity_filt);
    title(sprintf('L%d HU%d (pv = %.2f)', layernum, filternum, pvs_gaussian(filternum)));
end

%% correlate activity and derivative for all filters
rs = zeros([1 num_filters]);
pvs = zeros([1 num_filters]);
num_top = 50;

for f=1:num_filters
    dzdx_lh = res(layernum+1).dzdx(:,:,f,:);
    activity_filt = res(layernum+1).x(:,:,f,:);
    [~,sorted_activity_idx] = sort(activity_filt);
    [r,p] = corrcoef(activity_filt(sorted_activity_idx(end-num_top:end)), ...
        dzdx_lh(sorted_activity_idx(end-num_top:end)));
    rs(f) = r(1,2);
    pvs(f) = p(1,2);
end
[~,sorted_r_idx] = sort(rs);

figure; 
h = histogram(pvs);
h.Normalization = 'probability';
title(sprintf('L%d p-values of cc btwn activity and dzdx', layernum));
xlabel('p-value');
ylabel('% of HUs');
%% show distributions for several HUs

filter_range = 1:25;

figure;
for i=1:length(filter_range)
    filternum = filter_range(i);
    activity_filt = res(layernum+1).x(:,:,filternum,:);
    [~,sorted_idx] = sort(activity_filt);
    subplot(5,2,i);
    imshow(imdb.images.data(:,:,:,batch_range(sorted_idx(end))));
    title(sprintf('Top1 L%d HU%d', layernum, filternum));
end

%%
filter_range = 1:25;

figure;
num_classes = length(net.meta.classes.name);

separate_by_class = false;
for i=1:length(filter_range)
    filternum = filter_range(i);
    activity_filt = res(layernum+1).x(:,:,filternum,:);
    subplot(5, 5, filternum);
    if separate_by_class
        for c=1:num_classes
            class_idx = find(imdb.images.labels(batch_range) == c);
            histogram(activity_filt(:,:,:,class_idx));
            hold on;
        end
        hold off;
    else
        histogram(activity_filt);
    end
    title(sprintf('L%d HU %d (BS = %d)', ...
        layernum, filternum, length(batch_range)));
    % test normalness
    %[h,p] = jbtest(squeeze(activity_filt));
    %title(sprintf('L%d HU %d (p = %.f)', ...
    %    layernum, filternum, p));
    if i == length(filter_range) - 1
        xlabel('Activation');
        ylabel('# of HUs');
    end
end

%% show images and their labels
% figure;
% for i=batch_range
%     subplot(1,batch_range(end),i);
%     imshow(imdb.images.data(:,:,:,batch_range(i)));
%     [~,best_class_i] = max(res(end-1).x(:,:,:,i));
%     title(['target: ', num2str(imdb.images.labels(batch_range(i))), ', pred: ', ...
%         mnist_net.net.meta.classes.name{best_class_i}]);
% end
