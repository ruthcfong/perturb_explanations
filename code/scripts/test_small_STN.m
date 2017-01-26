%function test_small_STN()
    lr = 1e-3;
    nn = dagnn.DagNN();
    nn.conserveMemory = false;
    aff_grid = dagnn.AffineGridGenerator('Ho',50,'Wo',50);
    nn.addLayer('aff', aff_grid,{'aff'},{'grid'});
    sampler = dagnn.BilinearSampler();
    nn.addLayer('samp',sampler,{'input','grid'},{'xST'});

    aff = zeros([1 1 6], 'single');
    aff(1) = 1;
    aff(4) = 1;
    input = zeros([50 50], 'single');
    input(5:15,5:15) = 1;
    inputs = {'input',input,'aff',aff};
    x1 = 1:50;
    x2 = 1:50;
    [X1, X2] = meshgrid(x1,x2);
    F = mvnpdf([X1(:) X2(:)],[25 25],[100 0; 0 100]);
    F = reshape(F, length(x1), length(x2));
    F = normalize(F);
    xSTder = 1-single(F);
    derOutputs = {'xST',xSTder};
    
    figure;
    for t=1:500
        nn.eval(inputs, derOutputs);
        aff_der = nn.getVar('aff').der;
        
        subplot(2,2,1);
        imshow(input);
        
        subplot(2,2,2);
        imshow(nn.getVar('xST').value);
        
        subplot(2,2,3);
        imagesc(squeeze(aff));
        colorbar;
        
        subplot(2,2,4);
        imagesc(squeeze(aff_der));
        colorbar;
        
        drawnow;
        
        aff(5:6) = aff(5:6) - lr*aff_der(5:6);

        inputs = {'input',input,'aff',aff};
    end
%end