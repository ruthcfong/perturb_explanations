import caffe

import numpy as np
import pylab
import matplotlib.pyplot as plt
import sys, os, time, argparse
import scipy

from PIL import ImageFilter, Image

from helpers import *
from defaults import caffe_dir

def generate_learned_mask(net, path, label, given_gradient = True, norm_score = False, num_iters = 300, lr = 1e-1, l1_lambda = 1e-4, 
        l1_ideal = 1, l1_lambda_2 = 0, tv_lambda = 1e-2, tv_beta = 3, mask_scale = 8, use_conv_norm = False, blur_mask = 5, 
        jitter = 4, noise = 0, null_type = 'blur', gpu = None, start_layer = 'data', end_layer = 'prob', 
        plot_step = None, debug = False, fig_path = None, mask_path = None, verbose = False, show_fig = True, mask_init_type = 'circle', num_top = 0, 
        labels = np.loadtxt(os.path.join(caffe_dir, 'data/ilsvrc12/synset_words.txt'), str, delimiter='\t')):
    ''' 
    num_iters = 300
    lr = 1e-1
    l1_lambda = 1e-4
    l1_ideal = 1
    l1_lambda_2 = 0
    tv_lambda = 1e-2
    tv_beta = 3
    jitter = 4
    num_top = 0
    noise = 0
    null_type = 'blur'
    given_gradient = True
    norm_score = False
    end_layer = 'prob'
    use_conv_norm = False
    blur_mask = 5
    mask_scale = 8
    '''

    if mask_path is not None and os.path.exists(mask_path):
        print "%s already exists; cancel if you don't want to overwrite it" % mask_path
    start = time.time()
    net_transformer = get_ILSVRC_net_transformer(net)

    img = net_transformer.preprocess('data', caffe.io.load_image(path))
    net.blobs['data'].data[...] = img
    net.forward()
    scores = np.squeeze(net.blobs['prob'].data)
    sorted_idx = np.argsort(scores)
    if given_gradient:
        target = np.zeros(scores.shape)
        if num_top == 0:
            target[label] = 1
        else:
            target[sorted_idx[:-(num_top+1):-1]] = 1
    else:
        if num_top == 0:
            target = np.array([label])
        else:
            target = sorted_idx[:-(num_top+1):-1]

    if mask_init_type == 'circle':
        mask_radius = test_circular_masks(net, path, label, plot = False)
        mask_init = 1-create_blurred_circular_mask((net.blobs['data'].data.shape[2], net.blobs['data'].data.shape[3]),
                                         mask_radius, center = None, sigma = 10)
    elif mask_init_type == None:
        mask_init = None

    if show_fig:
       plt.ion() 
    else:
        plt.ioff()

    mask = optimize_mask(net, path, target, labels = labels, given_gradient = given_gradient, norm_score = norm_score,
                        num_iters = num_iters, lr = lr, l1_lambda = l1_lambda, l1_ideal = l1_ideal,
                        l1_lambda_2 = l1_lambda_2, tv_lambda = tv_lambda, tv_beta = tv_beta, mask_scale = mask_scale,
                        use_conv_norm= use_conv_norm, blur_mask = blur_mask, jitter = jitter,
                        null_type = null_type, mask_init = mask_init, gpu = gpu, start_layer = None, end_layer = end_layer,
                        plot_step = plot_step, debug = debug, fig_path = fig_path, mask_path = mask_path, verbose = verbose)

    plt.ion()
    end = time.time()
    if verbose:
        print 'Time elapsed:', (end-start)
    #plt.close()
    return mask

def optimize_mask(net, path, target, labels, given_gradient = False, norm_score = False, num_iters = 300, lr = 1e-1, l1_lambda = 1e-4, 
                  l1_ideal = 1, l1_lambda_2 = 0, tv_lambda = 1e-2, tv_beta = 3, mask_scale = 8, use_conv_norm = False, blur_mask = 5, 
                  jitter = 4, noise = 0, null_type = 'blur', mask_init = None, gpu = None, start_layer = None, 
                  end_layer = None, plot_step = None, debug = False, fig_path = None, mask_path = None, verbose = False):
    # adam parameters
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8
    
    if start_layer is None:
        start_layer = net.blobs.keys()[0]
    
    if end_layer is None:
        end_layer = net.blobs.keys()[-1]

    if plot_step is None: 
        plot_step = num_iters

    if given_gradient:
        gradient = target
    
    assert(start_layer == 'data')
    
    net_transformer = get_ILSVRC_net_transformer(net)
    net_shape = net.blobs[start_layer].data.shape
    assert(len(net_shape) == 4)
    if jitter > 0:
        jitter_shape = (1, 3, net_shape[2]+jitter, net_shape[3]+jitter)
    else:
        jitter_shape = net_shape
    jitter_transformer = get_ILSVRC_net_transformer_with_shape(jitter_shape)

    if norm_score:
        assert(given_gradient)
        #orig_score = forward_pass(net, net_transformer.preprocess('data', caffe.io.load_image(path)),
        #    target = target, last_layer = end_layer)
        orig_score = forward_pass(net, net_transformer.preprocess('data', caffe.io.load_image(path)),
                target = None, last_layer = end_layer)

    orig_output = np.squeeze(forward_pass(net, net_transformer.preprocess('data', caffe.io.load_image(path)), 
            target = None, last_layer = end_layer))
    orig_max_i = np.argmax(orig_output)

    if null_type == 'blur':
        null_img = jitter_transformer.preprocess('data', get_blurred_img(path, radius = 10))
        if norm_score:
            null_score = forward_pass(net, net_transformer.preprocess('data', get_blurred_img(path, radius = 10)), 
                target = target, last_layer = end_layer)
            #print orig_score, null_score
    elif null_type == 'blur_sample':
        null_lookup = transform_batch(jitter_shape, get_blurred_pyramid(path))
    elif null_type == 'random_noise':
        pass
    elif null_type == 'avg_blur_blank_noise':
        null_blur_img = jitter_transformer.preprocess('data', get_blurred_img(path, radius = 10))
        if norm_score:
            #null_score = forward_pass(net, net_transformer.preprocess('data', get_blurred_img(path, radius = 10)),
            #        target = target, last_layer = end_layer)
            null_score = forward_pass(net, net_transformer.preprocess('data', get_blurred_img(path, radius = 10)),
                    target = None, last_layer = end_layer)
        null_blur_img = null_blur_img.reshape(jitter_shape)
        null_blank_img = np.zeros(jitter_shape)
        null_rand_img = np.random.random(jitter_shape)*255
        null_img = np.concatenate((null_blur_img, null_blank_img, null_rand_img))
        net.blobs['data'].reshape(3,3,net_shape[2],net_shape[3])
        if given_gradient:
            gradient = np.tile(gradient, [3, len(gradient)])
    elif null_type == 'random_sample':
        assert(false)
    elif null_type == 'mean_img':
        null_img = np.zeros(jitter_shape[1:])
    else:
        assert(False)
    
    if mask_init is not None:
        if mask_scale == 1:
            mask = mask_init
        else:
            mask = scipy.misc.imresize(mask_init, (int(net_shape[2]/float(mask_scale)), int(net_shape[3]/float(mask_scale))), 
                    'nearest')/float(255)
    else:
        if mask_scale == 1:
            mask = np.random.rand(net_shape[2], net_shape[3])
        else:
            assert(int(net_shape[2]/float(mask_scale)) == net_shape[2]/float(mask_scale))
            assert(int(net_shape[3]/float(mask_scale)) == net_shape[3]/float(mask_scale))
            mask = np.random.rand(net_shape[2]/mask_scale, net_shape[3]/mask_scale)

    m_t = np.zeros(mask.shape)
    v_t = np.zeros(mask.shape)

    
    if gpu is None:
        caffe.set_mode_cpu()
    else:
        caffe.set_device(gpu)
        caffe.set_mode_gpu()
    
    E = np.empty((num_iters, 5))
    pylab.rcParams['figure.figsize'] = (12.0,12.0)
    f,ax = plt.subplots(4,2)
    #plt.ion()
    for t in range(num_iters):
        img = jitter_transformer.preprocess(start_layer, caffe.io.load_image(path))
        if jitter != 0:
            j1 = np.random.randint(jitter)
            j2 = np.random.randint(jitter)
        else:
            j1 = 0
            j2 = 0
        img_ = img[:,j1:(net_shape[2]+j1),j2:(net_shape[3]+j2)]
        
        if null_type == 'blur' or null_type == 'mean_img':
            null_img_ = null_img[:,j1:(net_shape[2]+j1),j2:(net_shape[3]+j2)]
        elif null_type == 'random_noise':
            null_img_ = np.random.rand(net_shape[1], net_shape[2], net_shape[3])*255
        elif null_type == 'avg_blur_blank_noise':
            null_rand_img = np.random.random(jitter_shape)*255
            null_img[2] = null_rand_img
            null_img_ = null_img[:,:,j1:(net_shape[2]+j1),j2:(net_shape[3]+j2)]
            img_ = img_.reshape(net_shape)
            img_ = np.concatenate((img_,img_,img_))
        else:
            assert(false)

        if noise != 0:
            noisy = np.random.normal(loc=0.0, scale=noise, size=mask.shape)
        else:
            noisy = 0
         
        mask_w_noise = mask + noisy
        mask_w_noise[mask_w_noise > 1] = 1
        mask_w_noise[mask_w_noise < 0] = 0
        if mask_scale > 1:
            mask_w_noise = resize(mask_w_noise, mask_scale)

        if blur_mask > 0:
            mask_w_noise = blur(mask_w_noise, radius=blur_mask)

        x = img_ * mask_w_noise + null_img_ * (1 - mask_w_noise)
        net.blobs[start_layer].data[...] = x
        net.forward(start = start_layer, end = end_layer)
        if not given_gradient:
            output = np.squeeze(net.blobs[end_layer].data)
            gradient = np.squeeze(np.zeros(net.blobs[end_layer].data.shape))
            if null_type == 'avg_blur_blank_noise':
                gradient[:,target] = output[:,target]
            else:
                gradient[target] = output[target]
        net.blobs[end_layer].diff[...] = gradient
        net.backward(start = end_layer, end = start_layer)
       
        summed_score = (net.blobs[end_layer].data * gradient).sum()
        if norm_score:
            #print (np.exp(null_score)*gradient).sum(), (np.exp(net.blobs[end_layer].data)*gradient).sum()
            E[t,0] = max((np.exp(null_score) * gradient).sum(), (np.exp(net.blobs[end_layer].data) * gradient).sum())
            if (np.exp(null_score)*gradient).sum() > (np.exp(net.blobs[end_layer].data)*gradient).sum():
                der = 0
            else:
                der = np.exp(net.blobs[end_layer].data)
            #der = np.maximum(np.exp(null_score), np.exp(net.blobs[end_layer].data))
            net.blobs[end_layer].diff[...] = gradient * der
            '''
            if null_type == 'avg_blur_blank_noise':
                norm_s = (summed_score/float(3) - null_score)/float(orig_score - null_score)
            else:
                norm_s = (summed_score - null_score)/float(orig_score - null_score)
            E[t,0] = max(0,norm_s)
            #a = np.abs(norm_s)
            #E[t,0] = np.exp(a) - 1
            #der = np.exp(a)*np.sign(norm_s)
            der = max(0,np.sign(norm_s))
            net.blobs[end_layer].diff[...] = gradient * der
            #print summed_score, norm_s, a, E[t,0], der 
            '''
        else:
            E[t,0] = summed_score
            net.blobs[end_layer].diff[...] = gradient
            #assert(np.array_equal(net.blobs[end_layer].diff[0], net.blobs[end_layer].diff[1]))

        net.backward(start = end_layer, end = start_layer)

        dx = np.squeeze(net.blobs[start_layer].diff)
        if null_type == 'avg_blur_blank_noise':
            dm = (dx * img_).sum((0,1)) - (dx * null_img_).sum((0,1))
        else:
            dm = (dx * img_).sum(0) - (dx * null_img_).sum(0)

        # L1 regularization
        if l1_lambda > 0:
            E[t,1] = l1_lambda*(np.abs(mask - l1_ideal).sum())
            dl1 = np.sign(mask-l1_ideal)
        else:
            E[t,1] = 0
            dl1 = 0
        
        if l1_lambda_2 > 0:
            E[t,2] = l1_lambda_2*((0.5-np.abs(mask - 0.5)).sum())
            dl1_2 = -np.sign(mask - 0.5)
        else:
            E[t,2] = 0
            dl1_2 = 0

        
        # TV regularization
        if tv_lambda > 0:
            assert(tv_beta > 0)
            (err, dtv) = tv(mask, tv_beta)
            E[t,3] = tv_lambda*err
        else:
            E[t,3] = 0
            dtv = 0

        if use_conv_norm:
            (err, dtv) = conv_norm(mask)
            E[t,3] = tv_lambda*err
        
        E[t,4] = E[t,:-1].sum()
        
        if mask_scale > 1:
            dm = resize(dm, mask_scale, diff = True)
        
        update_gradient = dm + l1_lambda*dl1 + l1_lambda_2*dl1_2 + tv_lambda*dtv
        
        m_t = beta1*m_t + (1-beta1)*update_gradient
        v_t = beta2*v_t + (1-beta2)*(update_gradient**2)
        m_hat = m_t/float(1-beta1**(t+1))
        v_hat = v_t/float(1-beta2**(t+1))
        
        mask -= (float(lr)/(np.sqrt(v_hat)+epsilon))*m_hat
        
        mask[mask > 1] = 1
        mask[mask < 0] = 0
        
        if debug or ((t+1) % plot_step == 0):
            if null_type == 'avg_blur_blank_noise':
                ax[0,0].imshow(net_transformer.deprocess('data', img_[0]))
                rand_i = np.random.randint(3)
                ax[0,1].imshow(net_transformer.deprocess('data', x[rand_i]))
                max_i = np.argmax(np.squeeze(net.blobs[end_layer].data[rand_i]))
                ax[0,1].set_title('%s %.2f' % (labels[max_i], np.squeeze(net.blobs[end_layer].data[rand_i])[max_i]))
            else:
                ax[0,0].imshow(net_transformer.deprocess('data', img_))
                ax[0,1].imshow(net_transformer.deprocess('data', x))
                max_i = np.argmax(np.squeeze(net.blobs[end_layer].data))
                ax[0,1].set_title('%s %.2f' % (labels[max_i], np.squeeze(net.blobs[end_layer].data)[max_i]))
            ax[0,0].set_title('%s %.2f' % (labels[orig_max_i], orig_output[orig_max_i]))
            ax[1,0].imshow(mask*255)
            ax[1,1].imshow(mask_w_noise*255)
            ax[3,1].plot(E[:(t+1),0])
            ax[3,1].plot(E[:(t+1),-1])
            #print E[t,-1]
            #ax[3,1].semilogy(E[:(t+1),0])
            #ax[3,1].semilogy(E[:(t+1),-1])
            ax[2,0].imshow(dm)
            ax[2,1].imshow(dtv)
            #plt.ion()
            #plt.clf()
            f.canvas.draw()
            time.sleep(1e-2)
            #plt.pause(1e-3)
            if debug:
                print 'loss at epoch %d: f(x) = %f, l1 = %f, l1_2 = %f, TV = %f' % (t, E[t,0], E[t,1], E[t,2], E[t,3])
                print 'mean |deriv| at epoch %d: dm = %f, dl1 = %f, dl1_2 = %f, dtv = %f' % (t, 
                                                                                 np.abs(dm).mean(), 
                                                                                 l1_lambda*np.abs(dl1).mean(),
                                                                                 l1_lambda_2*np.abs(dl1_2).mean(),
                                                                                 tv_lambda*np.abs(dtv).mean())
            
    if fig_path is not None:
        directory = os.path.dirname(os.path.abspath(fig_path))
        if not os.path.exists(directory):
            os.makedirs(directory)
        plt.savefig(fig_path)

    if mask_path is not None:
        directory = os.path.dirname(os.path.abspath(mask_path))
        if not os.path.exists(directory):
            os.makedirs(directory)
        if mask_scale > 1:
            mask = resize(mask, mask_scale)
        if blur_mask > 0:
            mask = blur(mask, radius=blur_mask)
        np.save(mask_path, mask)
        if verbose:
            print 'saved mask to %s' % mask_path
    
    net.blobs[start_layer].reshape(net_shape[0], net_shape[1], net_shape[2], net_shape[3])
    return mask

def resize(img, scale, interp = 'nearest', diff = False):
    assert(len(img.shape) == 2)
    assert(interp == 'nearest')
    if diff:
        assert(int(img.shape[0]/scale) == img.shape[0]/scale)
        assert(int(img.shape[1]/scale) == img.shape[1]/scale)
        img_ = np.zeros((img.shape[0]/scale, img.shape[1]/scale))
    else:
        assert(int(scale) == scale)
        img_ = np.zeros((scale*img.shape[0], scale*img.shape[1]))
    
    for i in range(img_.shape[0]):
        for j in range(img_.shape[1]):
            if diff:
                for r in range(scale):
                    for c in range(scale):
                        img_[i][j] += img[i*scale+r][j*scale+c]
            else:
                img_[i][j] = img[int(i/scale)][int(j/scale)]
    return img_

def tv(x, beta = 1):
    d1 = np.zeros(x.shape)
    d2 = np.zeros(x.shape)
    d1[:-1,:] = np.diff(x, axis=0)
    d2[:,:-1] = np.diff(x, axis=1)
    v = np.sqrt(d1*d1 + d2*d2)**beta
    e = v.sum()
    d1_ = (np.maximum(v, 1e-5)**(2*(beta/float(2)-1)/float(beta)))*d1
    d2_ = (np.maximum(v, 1e-5)**(2*(beta/float(2)-1)/float(beta)))*d2
    d11 = -d1_
    d22 = -d2_
    d11[1:,:] = -np.diff(d1_, axis=0)
    d22[:,1:] = -np.diff(d2_, axis=1)
    dx = beta*(d11 + d22)
    return (e,dx)

def conv_norm(x):
    conv_net = caffe.Net('conv.prototxt', caffe.TEST)
    weights = np.ones(conv_net.params['conv'][0].data.shape)/float(5**2-1)
    weights[0,:,5/2,5/2] = -1
    conv_net.blobs['data'].data[...] = x
    conv_net.forward()
    output = conv_net.blobs['conv'].data
    output_abs = np.abs(output)
    conv_net.blobs['conv'].diff[...] = output_abs 
    conv_net.backward()
    diff = conv_net.blobs['data'].diff
    return (output_abs.sum(),np.squeeze(diff))

def blur(img, radius = 10):
    img = Image.fromarray(np.uint8(img*255))
    blurred_img = img.filter(ImageFilter.GaussianBlur(radius))
    return np.array(blurred_img)/float(255)

def get_blurred_img(path, radius = 10):
    img = Image.open(path).convert('RGB')
    blurred_img = img.filter(ImageFilter.GaussianBlur(10))
    return np.array(blurred_img)/float(255)


def get_blurred_pyramid(path, radii=np.arange(10,0,-0.1)):
    N = radii.shape[0]
    [H,W,D] = caffe.io.load_image(path).shape
    blurred_pyramid = np.empty([H,W,D,N+1])
    for i in range(N):
        blurred_pyramid[:,:,:,i] = get_blurred_img(path, radii[i])
    blurred_pyramid[:,:,:,-1] = np.array(Image.open(path))/float(255)
    return blurred_pyramid

def forward_pass(net, img, target = None, last_layer = 'prob'):
    net.blobs['data'].data[...] = img
    net.forward(end=last_layer)
    scores = np.squeeze(np.copy(net.blobs[last_layer].data))
    if target is None:
        return scores
    else:
        return (scores * target).sum()

def create_blurred_circular_mask(mask_shape, radius, center = None, sigma = 10):
    assert(len(mask_shape) == 2)
    if center is None:
        x_center = int(mask_shape[1]/float(2))
        y_center = int(mask_shape[0]/float(2))
        center = (x_center, y_center)
    y,x = np.ogrid[-y_center:mask_shape[0]-y_center, -x_center:mask_shape[1]-x_center]
    mask = x*x + y*y <= radius*radius
    grid = np.zeros(mask_shape)
    grid[mask] = 1
    if sigma is not None:
        grid = scipy.ndimage.filters.gaussian_filter(grid, sigma)
    return grid

def create_blurred_circular_mask_pyramid(mask_shape, radii, sigma = 10):
    assert(len(mask_shape) == 2)
    num_masks = len(radii)
    masks = np.zeros((num_masks, 3, mask_shape[0], mask_shape[1]))
    for i in range(num_masks):
        masks[i,:,:,:] = create_blurred_circular_mask(mask_shape, radii[i], sigma = sigma)
    return masks

def test_circular_masks(net, img_path, label, end_layer = 'prob', radii = np.arange(0,175,5), thres = 1e-2, plot = True):
    net_transformer = get_ILSVRC_net_transformer(net)
    masks = create_blurred_circular_mask_pyramid((net.blobs['data'].data.shape[-2], net.blobs['data'].data.shape[-1]), 
                                                 radii)
    masks = 1 - masks
    num_masks = len(radii)
    img = net_transformer.preprocess('data', caffe.io.load_image(img_path))
    null_img = net_transformer.preprocess('data', get_blurred_img(img_path))
    gradient = np.zeros(net.blobs['prob'].data.shape)
    gradient[0][label] = 1
    
    '''
    net.blobs['data'].reshape(num_masks, 3, net.blobs['data'].data.shape[-2], net.blobs['data'].data.shape[-1])
    imgs = np.zeros(net.blobs['data'].data.shape)
    imgs[...] = img
    null_imgs = np.zeros(net.blobs['data'].data.shape)
    null_imgs[...] = null_img
        
    masked_imgs = imgs * masks + null_imgs * (1 - masks)
    net.blobs['data'].data[...] = masked_imgs
    net.forward(end = end_layer)

    
    percs = ((net.blobs[end_layer].data[:,label] - net.blobs[end_layer].data[-1,label])/
        float(net.blobs[end_layer].data[0,label]-net.blobs[end_layer].data[-1,label]))
    try:
        first_i = np.where(percs < thres)[0][0]
    except:
        first_i = -1
    '''

    scores = np.zeros(num_masks)
    for i in range(num_masks):
        masked_img = img*masks[i] + null_img * (1 - masks[i])
        net.blobs['data'].data[...] = masked_img
        net.forward(end = end_layer)
        scores[i] = net.blobs[end_layer].data[0,label]

    net.blobs['data'].data[...] = img
    net.forward(end = end_layer)
    orig_score = net.blobs[end_layer].data[0,label]

    percs = (scores - scores[-1])/float(orig_score - scores[-1]) 
    try:
        first_i = np.where(percs < thres)[0][0]
    except:
        first_i = -1
        
    if plot:
        f, ax = plt.subplots(1,2)
        ax[0].imshow(net_transformer.deprocess('data', masked_imgs[first_i]))
        ax[0].set_title(radii[first_i])
        #ax[1].plot(radii, net.blobs[end_layer].data[:,label])
        ax[1].plot(radii, percs)
        plt.show()
    
    #net.blobs['data'].reshape(1,3,net.blobs['data'].data.shape[2], net.blobs['data'].data.shape[3])

    return radii[first_i]

'''
Possibilities:
[X] Binarize mask
Sigmoid mask
Use a different blur
Jitter image underneath mask
[X] Use random noise or gray/null background'''
def check_mask_generalizability(net, img_path, target, mask_path, null_type = 'blur', last_layer = 'prob', fig_path = None):
    transformer = get_ILSVRC_net_transformer(net)
    img = transformer.preprocess('data', caffe.io.load_image(img_path))
    mask = np.load(mask_path)
    if mask.shape != net.blobs['data'].shape[2:]:
        mask = scipy.misc.imresize(mask, net.blobs['data'].shape[2:], 'nearest')/float(255) 
    binarize_mask = np.copy(mask)
    binary_thres = 1 - 1e-1
    binarize_mask[binarize_mask >= binary_thres] = 1
    binarize_mask[binarize_mask < binary_thres] = 0
    blur_mask = np.array(Image.fromarray(np.uint8(mask*255)).filter(
        ImageFilter.GaussianBlur(10)))/float(255)
    blur_mask = (blur_mask-blur_mask.min())/float(blur_mask.max() - blur_mask.min())
    bin_blur_mask = np.copy(blur_mask)
    bin_blur_mask[bin_blur_mask >= binary_thres] = 1
    bin_blur_mask[bin_blur_mask < binary_thres] = 0
    if null_type == 'blur':
        null_img = transformer.preprocess('data', get_blurred_img(img_path, radius = 10))
    null_rand_img = np.random.rand(img.shape[0], img.shape[1], img.shape[2])*255
    grey_img = np.zeros(img.shape)
    orig_score = forward_pass(net, img, target, last_layer = last_layer)
    blur_score = forward_pass(net, null_img, target, last_layer = last_layer)
    grey_score = forward_pass(net, grey_img, target, last_layer = last_layer)
    rand_score = forward_pass(net, null_rand_img, target, last_layer = last_layer)
    mask_score = forward_pass(net, img * mask + null_img * (1 - mask), target, last_layer = last_layer)
    bin_score = forward_pass(net, img * binarize_mask + null_img * (1 - binarize_mask), 
                             target, last_layer = last_layer)
    blur_mask_score = forward_pass(net, img * blur_mask + null_img * (1- blur_mask), target, last_layer = last_layer)
    bin_blur_score = forward_pass(net, img * bin_blur_mask + null_img * (1- bin_blur_mask), target, last_layer = last_layer)
    grey_mask_score = forward_pass(net, img * mask, target, last_layer = last_layer)
    grey_bin_score = forward_pass(net, img * binarize_mask, target, last_layer = last_layer)
    grey_blur_score = forward_pass(net, img * blur_mask, target, last_layer = last_layer)
    grey_bin_blur_score = forward_pass(net, img * bin_blur_mask, target, last_layer = last_layer)
    rand_mask_score = forward_pass(net, img * mask + null_rand_img * (1 - mask), target, last_layer = last_layer)
    rand_bin_score = forward_pass(net, img * binarize_mask + null_rand_img * (1 - binarize_mask), 
                             target, last_layer = last_layer)
    rand_blur_score = forward_pass(net, img * blur_mask + null_rand_img * (1- blur_mask), target, 
                                   last_layer = last_layer)
    rand_bin_blur_score = forward_pass(net, img * bin_blur_mask + null_rand_img * (1- bin_blur_mask), 
                                       target, last_layer = last_layer)

    #print orig_score, mask_score, bin_score, blur_score, grey_score, grey_bin_score, grey_blur_score

    f, ax = plt.subplots(4,5)
    ax[0,0].imshow(transformer.deprocess('data', img))
    ax[0,0].set_title('%.3f' % orig_score)
    ax[0,1].imshow(mask)
    ax[0,1].set_title('mask')
    ax[0,2].imshow(binarize_mask)
    ax[0,2].set_title('bin mask')
    ax[0,3].imshow(blur_mask)
    ax[0,3].set_title('blur mask')
    ax[0,4].imshow(bin_blur_mask)
    ax[0,4].set_title('bin blur mask')
    ax[1,0].imshow(transformer.deprocess('data', null_img))
    ax[1,0].set_title('%.3f' % blur_score)
    ax[1,1].imshow(transformer.deprocess('data', img * mask + null_img * (1 - mask)))
    ax[1,1].set_title('%.3f' % mask_score)
    ax[1,2].imshow(transformer.deprocess('data', img * binarize_mask + null_img * (1 - binarize_mask)))
    ax[1,2].set_title('%.3f' % bin_score)
    ax[1,3].imshow(transformer.deprocess('data', img * blur_mask + null_img * (1 - blur_mask)))
    ax[1,3].set_title('%.3f' % blur_mask_score)
    ax[1,4].imshow(transformer.deprocess('data', img * bin_blur_mask + null_img * (1 - bin_blur_mask)))
    ax[1,4].set_title('%.3f' % bin_blur_score)
    ax[2,0].imshow(transformer.deprocess('data', grey_img))
    ax[2,0].set_title('%.3f' % grey_score)
    ax[2,1].imshow(transformer.deprocess('data', img * mask))
    ax[2,1].set_title('%.3f' % grey_mask_score)
    ax[2,2].imshow(transformer.deprocess('data', img * binarize_mask))
    ax[2,2].set_title('%.3f' % grey_bin_score)
    ax[2,3].imshow(transformer.deprocess('data', img * blur_mask))
    ax[2,3].set_title('%.3f' % grey_blur_score)
    ax[2,4].imshow(transformer.deprocess('data', img * bin_blur_mask))
    ax[2,4].set_title('%.3f' % grey_bin_blur_score)
    ax[3,0].imshow(transformer.deprocess('data', null_rand_img))
    ax[3,0].set_title('%.3f' % rand_score)
    ax[3,1].imshow(transformer.deprocess('data', img * mask + null_rand_img * (1 - mask)))
    ax[3,1].set_title('%.3f' % rand_mask_score)
    ax[3,2].imshow(transformer.deprocess('data', img * binarize_mask + null_rand_img * (1 - binarize_mask)))
    ax[3,2].set_title('%.3f' % rand_bin_score)
    ax[3,3].imshow(transformer.deprocess('data', img * blur_mask + null_rand_img * (1 - blur_mask)))
    ax[3,3].set_title('%.3f' % rand_blur_score)
    ax[3,4].imshow(transformer.deprocess('data', img * bin_blur_mask + null_rand_img * (1 - bin_blur_mask)))
    ax[3,4].set_title('%.3f' % rand_bin_blur_score)
    plt.show()
    
    if fig_path is not None:
        directory = os.path.dirname(fig_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        plt.savefig(fig_path)

def main(argv):
    parser = argparse.ArgumentParser(description='Learn perturbation masks for ImageNet examples.')

    parser.add_argument('data_desc', default='train_heldout', help="choose from ['train_heldout', 'val', 'animal_parts']")

    parser.add_argument('-n', '--net_type', default='googlenet', help="choose from ['googlenet', 'vgg16', 'alexnet']")
    parser.add_argument('-g', '--gpu', default=None, type=int, help="zero-indexed gpu to use [i.e. 0-3]") 
    parser.add_argument('-s', '--start', default=0, type=int, help="start index")
    parser.add_argument('-e', '--end', default=10, type=int, help="end index")
    parser.add_argument('-f', '--fig_dir', default=None)
    parser.add_argument('-m', '--mask_dir', default=None)
    parser.add_argument('--show_fig', action='store_true')

    #gpu = 0 
    #net_type = 'googlenet'
    #data_desc = 'train_heldout'

    args = parser.parse_args(argv)
    data_desc = args.data_desc
    gpu = args.gpu
    net_type = args.net_type
    start = args.start
    end = args.end
    fig_dir = args.fig_dir
    mask_dir = args.mask_dir
    show_fig = args.show_fig

    if gpu is not None:
        caffe.set_device(gpu)
        caffe.set_mode_gpu()
    else:
        caffe.set_mode_cpu()

    assert(data_desc == 'train_heldout' or data_desc == 'val' or data_desc == 'animal_parts')

    if data_desc == 'train_heldout':
        (paths, labels) = read_imdb('/home/ruthfong/packages/caffe/data/ilsvrc12/annotated_train_heldout_imdb.txt')
    elif data_desc == 'val':
        (paths, labels) = read_imdb('/home/ruthfong/packages/caffe/data/ilsvrc12/val_imdb.txt')
    elif data_desc == 'animal_parts':
        (paths, labels) = read_imdb('/home/ruthfong/packages/caffe/data/ilsvrc12/animal_parts_require_both_min_10_imdb.txt')
    
    labels_desc = np.loadtxt(os.path.join(caffe_dir, 'data/ilsvrc12/synset_words.txt'), str, delimiter='\t')

    from defaults import (num_iters, lr, l1_lambda, l1_ideal, l1_lambda_2, tv_lambda, tv_beta, jitter, num_top, noise, null_type, 
            given_gradient, norm_score, end_layer, use_conv_norm, blur_mask, mask_scale)
    '''
    # default parameters
    num_iters = 300
    lr = 1e-1
    l1_lambda = 1e-4
    l1_ideal = 1
    l1_lambda_2 = 0
    tv_lambda = 1e-2
    tv_beta = 3
    jitter = 4
    num_top = 0
    noise = 0
    null_type = 'blur'
    given_gradient = True
    norm_score = False
    end_layer = 'prob'
    use_conv_norm = False
    blur_mask = 5
    mask_scale = 8
    '''
    

    '''
    # localization parameters
    num_iters = 300
    lr = 1e-1
    l1_lambda = 1e-3
    l1_ideal = 1
    l1_lambda_2 = 0
    tv_lambda = 1e-2
    tv_beta = 2
    jitter = 4
    num_top = 5
    noise = 0
    null_type = 'blur'
    given_gradient = True
    norm_score = False
    end_layer = 'prob'
    use_conv_norm = False
    blur_mask = 5
    mask_scale = 8
    '''

    net = get_net(net_type)
    net_transformer = get_ILSVRC_net_transformer(net)
    for i in range(start, end):
        fig_path = None
        mask_path = None
        if fig_dir is not None:
            fig_path = os.path.join(fig_dir, '%d.png' % i)
        if mask_dir is not None:
            mask_path = os.path.join(mask_dir, '%d.npy' % i)

        if mask_dir is not None and os.path.exists(mask_path):
            print '%s already exists so skipping' % mask_path
            continue

        start = time.time()
        
        generate_learned_mask(net, paths[i], labels[i], fig_path = fig_path, mask_path = mask_path, gpu = gpu, show_fig = show_fig)
        #generate_learned_mask(net, paths[i], labels[i], fig_path = fig_path, mask_path = mask_path, gpu = gpu, show_fig = show_fig, 
        #        num_iters = num_iters, lr = lr, l1_lambda = l1_lambda, l1_ideal = l1_ideal, l1_lambda_2 = l1_lambda_2, 
        #        tv_lambda = tv_lambda, tv_beta = tv_beta, mask_scale = mask_scale, use_conv_norm = use_conv_norm, blur_mask = blur_mask,
        #        jitter = jitter, noise = noise, null_type = null_type, end_layer = end_layer, num_top = num_top)

        end = time.time()
        print 'Time elapsed:', (end-start)

    del net

if __name__ == '__main__':
    main(sys.argv[1:])
