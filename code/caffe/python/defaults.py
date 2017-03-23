caffe_dir = '/home/ruthfong/sample_code/Caffe-ExcitationBP/'

# default hyperparameters for optimize_mask.py
num_iters = 300
lr = 1e-1
l1_lambda = 1e-4
l1_ideal = 1
l1_lambda_2 = 0
tv_lambda = 1e-2
tv_beta = 3
jitter = 4
num_top = 5
noise = 0
null_type = 'mean_img'
given_gradient = True
norm_score = False
end_layer = 'prob'
use_conv_norm = False
blur_mask = 0
mask_scale = 1
