function [error, dzdy] = mse_loss(pred, target)
    error = sum((target - pred).^2);
    dzdy = -2*(target-pred);
end