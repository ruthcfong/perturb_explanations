import numpy as np
import caffe
from skimage import transform, filters

topLayerName='fc8'
topBlobName='fc8'
outputLayerName = 'relu5_3'
outputBlobName = 'relu5_3'

caffe.set_device(0)

def Normalize(a):
    return (a -a.min())/(a.max()-a.min())

def doGradCAM(net, img, tagID, top = topLayerName, bottom = outputLayerName):
    caffe.set_mode_gpu()
    # forward pass
    out = net.forward(end = top)

    net.blobs[top].diff[0][...] = 0

    net.blobs[top].diff[0][tagID] = 1
    fprop_maps = net.blobs[bottom].data[0]

    # backward pass till last convolution layer (after ReLU i.e. relu5_3 in case of VGG-16)
    out = net.backward(start = top, end = bottom)

    # get Grad-CAM weights of maps
    map_weights = net.blobs[bottom].diff[0].sum(1).sum(1)
    map_weights = map_weights.repeat(fprop_maps.shape[1]*fprop_maps.shape[2]).reshape(map_weights.shape[0],fprop_maps.shape[1],fprop_maps.shape[2])

    gradCAM_beforeReLU = np.multiply(fprop_maps,map_weights).sum(0)
    gradCAM = Normalize(np.maximum(gradCAM_beforeReLU,0))
    gradCAM = transform.resize(gradCAM, (224,224))

    return gradCAM
