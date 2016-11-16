function plot_heatmap_of_two_units(bv)
    figure;
    % hist3(bv);
    n = hist3(bv);
    n1 = n';
    n1(size(n,1) + 1, size(n,2) + 1) = 0;
    xb = linspace(min(bv(:,1)),max(bv(:,1)),size(n,1)+1);
    yb = linspace(min(bv(:,2)),max(bv(:,2)),size(n,1)+1);
    h = pcolor(xb, yb, n1);
    h.ZData = ones(size(n1)) * -max(max(n));
    colormap(hot)
    grid on
    %view(3);
%     title(sprintf('HUs %d and %d L%d activations', i, j, start_l));
%     xlabel(sprintf('HU %d L%d activations', i, start_l));
%     ylabel(sprintf('HU %d L%d activations', i, start_l));
end