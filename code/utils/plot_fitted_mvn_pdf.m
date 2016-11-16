function plot_fitted_mvn_pdf(bv)
    mu = [mean(bv(:,1)), mean(bv(:,2))];
    Sigma = cov(bv(:,1), bv(:,2));
    x1 = min(bv(:,1)):(max(bv(:,1))-min(bv(:,1)))/50:max(bv(:,1));
    x2 = min(bv(:,2)):(max(bv(:,2))-min(bv(:,2)))/50:max(bv(:,2));
    [X1,X2] = meshgrid(x1,x2);
    F = mvnpdf([X1(:) X2(:)],mu,Sigma);
    F = reshape(F,length(x2),length(x1));
    figure;
    surf(x1,x2,F);
    caxis([min(F(:))-.5*range(F(:)),max(F(:))]);
    %xlabel(sprintf('L%d HU %d', start_l, i)); 
    %ylabel(sprintf('L%d HU %d', start_l, j)); 
    zlabel('Probability Density');
    %title(sprintf('PDF of Bivariate Normal Distribution for L%d HUs %d and %d', start_l, ...
    %    i, j));
end
