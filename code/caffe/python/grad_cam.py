import caffe

net_type = 'vgg16'
gpu = 0

caffe_dir = '/users/ruthfong/sample_code/Caffe-ExcitationBP/'
alexnet_prototxt = '/users/ruthfong/packages/caffe/models/bvlc_reference_caffenet/deploy_force_backward.prototxt'
alexnet_model = '/users/ruthfong/packages/caffe/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'
vgg16_prototxt = '/users/ruthfong/packages/caffe/models/vgg16/VGG_ILSVRC_16_layers_deploy_force_backward.prototxt'
vgg16_model = '/users/ruthfong/packages/caffe/models/vgg16/VGG_ILSVRC_16_layers.caffemodel'
googlenet_prototxt = '/users/ruthfong/packages/caffe/models/bvlc_googlenet/deploy_force_backward.prototxt'
googlenet_model = '/users/ruthfong/packages/caffe/models/bvlc_googlenet/bvlc_googlenet.caffemodel'

labels_desc = np.loadtxt('/home/ruthfong/packages/caffe/data/ilsvrc12/synset_words.txt', str, delimiter='\t')

def get_grad_cam(net, transformer, path, label, top_layer, bottom_layer, show_fig = False):
	# set input data
	net.blobs['data'].data[...] = transformer.preprocess('data', 
	                                                caffe.io.load_image(path))

	# forward pass
	net.forward()

	# set gradient to backpropagate
	net.blobs[top_layer].diff[0][...] = 0
	net.blobs[top_layer].diff[0][label] = 1

	# backward pass to interested layer
	net.backward(start = top_layer, end = bottom_layer)

	activations = net.blobs[bottom_layer].data
	gradient = net.blobs[bottom_layer].diff

	# compute per-kernel weights based on gradient at interested layer
	alphas = np.mean(gradient,(2,3))

	# take the weighted sum of activations and pass through ReLU
	heatmap = np.squeeze(np.maximum(np.sum(activations * np.broadcast_to(
	    np.expand_dims(np.expand_dims(alphas, 2),3), activations.shape), 1), 0))

	# normalize to be within [0,1]
	heatmap = np.true_divide(heatmap-np.min(heatmap),np.max(heatmap) - np.min(heatmap))

	if show_fig:
		f, ax = plt.subplots(1,1)
		ax.imshow(transformer.deprocess('data', img))
		cax = ax.imshow(imresize(heatmap, img.shape[1:], 'bilinear'), alpha = 0.5, cmap = 'jet')
		f.colorbar(cax)
		ax.set_xticks([])
		ax.set_yticks([])
		ax.set_title('%s: %s %s (%s)' % (labels_desc[label], top_layer, bottom_layer, net_type))
		plt.show()

def get_net(net_type):
    if net_type == 'alexnet':
        net = caffe.Net(alexnet_prototxt, alexnet_model, caffe.TEST)
    elif net_type == 'vgg16':
        net = caffe.Net(vgg16_prototxt, vgg16_model, caffe.TEST)
    elif net_type == 'googlenet':
        net = caffe.Net(googlenet_prototxt, googlenet_model, caffe.TEST)
    else:
        assert(False)
   
    net_shape = net.blobs['data'].data.shape
    net.blobs['data'].reshape(1,3,net_shape[2],net_shape[3])
    return net

 def get_ILSVRC_mean(print_mean = False):
    mu = np.load(os.path.join(caffe_dir, 'python/caffe/imagenet/ilsvrc_2012_mean.npy'))
    mu = mu.mean(1).mean(1)
    if print_mean:
        print 'mean-subtracted values:', zip('BGR', mu)
    return mu

def get_ILSVRC_net_transformer(net):
    transformer = caffe.io.Transformer({'data':net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2,0,1))
    mu = get_ILSVRC_mean()
    transformer.set_mean('data', mu)
    transformer.set_raw_scale('data', 255)
    transformer.set_channel_swap('data', (2,1,0))
    return transformer

def main():
	gpu = 0
	net_type = 'vgg16'
	top_layer = 'prob'
	bottom_layers = ['conv5_3','conv5_2','conv5_1', 'conv4_3','conv4_2', 'conv4_1', 'conv3_3', 'conv3_2', 'conv3_1',
                'conv2_2', 'conv2_1','conv1_2', 'conv1_1']

	if gpu is not None:
		caffe.set_device(gpu)
		caffe.set_mode_gpu()

	if net_type == 'alexnet':
	    net = alexnet
	elif net_type == 'vgg16':
	    net = vgg_net
	elif net_type == 'googlenet':
	    net = googlnet

	transformer = get_ILSVRC_net_transformer(net)

	path = '../../../images/COCO_train2014_000000114269.jpg'
	label = 282 # tiger cat

	if net_type == 'vgg16':
	    f, ax = plt.subplots(2,7)
	    f.set_size_inches(14,6)
	elif net_type == 'alexnet':
	    f, ax = plt.subplots(1,6)
	    f.set_size_inches(12,4)
	else:
		assert(False)

	for i in range(len(bottom_names)+1):
	    curr_ax = ax[i/7][i%7] if net_type == 'vgg16' else ax[i]
	    curr_ax.set_xticks([])
	    curr_ax.set_yticks([])
	    curr_ax.imshow(transformer.deprocess('data', img))
	    
	    if i == 0:
	        curr_ax.set_title('%s %s' % (net_type, top_name))
	        continue
	        
	    bottom_name = bottom_names[i-1]
	    heatmap = get_grad_cam(net, transformer, path, label, top_layer, bottom_layer, show_fig = False)	    
	    curr_ax.imshow(imresize(heatmap, img.shape[1:]), alpha = 0.75, cmap = 'jet')
	    #f.colorbar(cax)
	    curr_ax.set_title('%s' % bottom_name)
	plt.show()


