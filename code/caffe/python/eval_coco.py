import numpy as np
import matplotlib.pyplot as plt
from skimage import transform, filters
import sys, time, argparse
import shapely.geometry
import coco_util as util

# COCO API
coco_root = '/data/datasets/coco'  # modify to point to your COCO installation
sys.path.insert(0, coco_root + '/PythonAPI')
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import pycocotools.mask as mask

# CAFFE
#caffe_root = '..'
#sys.path.insert(0, caffe_root + '/python')
import caffe

from helpers import get_COCO_net_transformer

# PARAMS
tags, tag2ID = util.loadTags(caffe_root + '/models/COCO/catName.txt')
imgScale = 224
topBlobName = 'loss3/classifier'
topLayerName = 'loss3/classifier'
secondTopLayerName = 'pool5/7x7_s1'
secondTopBlobName = 'pool5/7x7_s1'
outputLayerName = 'pool2/3x3_s2'
outputBlobName = 'pool2/3x3_s2'

def parseArgs():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Excitation Backprop')
    parser.add_argument('-g', '--gpu', dest='gpu_id', help='GPU device ID to use [0]',
                        default=0, type=int)
    parser.add_argument('-s', '--saliency', default='saliency', type=str, 
        help="saliency heatmap method ['center', 'saliency', 'guided_backprop', 'excitation_backprop', 'contrast_excitation_backprop']")
    args = parser.parse_args()
    return args

'''
# CAFFE
def initCaffe(args):
    caffe.set_mode_gpu()
    caffe.set_device(args.gpu_id)
    net = caffe.Net(caffe_root+'/models/COCO/deploy.prototxt',
                    caffe_root+'/models/COCO/GoogleNetCOCO.caffemodel',
                    caffe.TRAIN)
    return net

def doExcitationBackprop(net, img, tagName):
    # load image, rescale
    minDim = min(img.shape[:2])
    newSize = (int(img.shape[0]*imgScale/float(minDim)), int(img.shape[1]*imgScale/float(minDim)))
    imgS = transform.resize(img, newSize)

    # reshape net
    net.blobs['data'].reshape(1,3,newSize[0],newSize[1])
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_mean('data', np.array([103.939, 116.779, 123.68]))
    transformer.set_transpose('data', (2,0,1))
    transformer.set_channel_swap('data', (2,1,0))
    transformer.set_raw_scale('data', 255.0)

    # forward pass
    net.blobs['data'].data[...] = transformer.preprocess('data', imgS)
    out = net.forward(end = topLayerName)

    # switch to the excitation backprop mode
    caffe.set_mode_eb_gpu() 

    tagID = tag2ID[tagName]
    net.blobs[topBlobName].diff[0][...] = 0
    net.blobs[topBlobName].diff[0][tagID] = np.exp(net.blobs[topBlobName].data[0][tagID].copy())
    net.blobs[topBlobName].diff[0][tagID] /= net.blobs[topBlobName].diff[0][tagID].sum()

    # invert the top layer weights
    net.params[topLayerName][0].data[...] *= -1
    out = net.backward(start = topLayerName, end = secondTopLayerName)
    buff = net.blobs[secondTopBlobName].diff.copy()

    # invert back
    net.params[topLayerName][0].data[...] *= -1 
    out = net.backward(start = topLayerName, end = secondTopLayerName)

    # compute the contrastive signal
    net.blobs[secondTopBlobName].diff[...] -= buff

    # get attention map
    out = net.backward(start = secondTopLayerName, end = outputLayerName)
    attMap = np.maximum(net.blobs[outputBlobName].diff[0].sum(0), 0)

    # resize back to original image size
    attMap = transform.resize(attMap, (img.shape[:2]), order = 3, mode = 'nearest')
    return attMap
'''

def evalPointingGame(cocoAnn, cat, caffeNet, imgDir, transformer, heatmapType, topName = 'loss3/classifier', 
    bottomName = 'data', normDeg = np.inf, naiveMax = True, gpu = None):
    imgIds  = cocoAnn.getImgIds(catIds=cat['id'])
    imgList = cocoAnn.loadImgs(ids=imgIds)
    hit  = 0
    miss = 0
    hitDiff = 0
    missDiff = 0
    t0 = time.time()
    numImgs = len(imgList)
    for i in range(numImgs):
        I = imgList[i]
        # run EB on img, get max location on attMap
        imgName = os.path.join(imgDir, I['file_name'])
        img     = caffe.io.load_image(imgName)
        if heatmapType != 'center':
            #attMap  = doExcitationBackprop(caffeNet, img, cat['name'])
            attMap = compute_heatmap(net = caffeNet, transformer = transformer, paths = imgName, labels = cat['id'],
                               heatmap_type = heatmapType, topBlobName = topName, topLayerName = topName,
                               outputBlobName = bottomName, outputLayerName = bottomName, norm_deg = normDeg,
                               gpu = gpu)

            # reshape to original image
            attMap = = transform.resize(attMap, (img.shape[:2]), order = 3, mode = 'nearest')
            
            if naiveMax:
                # naively take argmax
                maxSub = np.unravel_index(np.argmax(attMap), attMap.shape)
            else:
                # take center of max locations
                maxAtt = np.max(attMap)
                maxInd = np.where(attMap == maxAtt)
                maxSub = (np.mean(maxInd[0]), np.mean(maxInd[1]))
        else:
            # choose center of image
            maxSub = (img.shape[0]/float(2), img.shape[1]/float(2))

        # determine if it's a difficult image (1) sum of the area of bounding boxes is less than 1/4 of image area,
        # 2) at least one distractor category
        allAnnList = cocoAnn.loadAnns(cocoAnn.getAnnIds(imgIds=I['id']))
        bbsArea = np.sum([a['area'] for a in allAnnList])
        imgArea = np.prod(img.shape[:2])
        numCats = len(np.unique([a['category_id'] for a in allAnnList]))
        isDiff  = bbsArea < 0.25*imgArea and numCats > 1

        # load annotations (for target category)
        annList = cocoAnn.loadAnns(cocoAnn.getAnnIds(imgIds=I['id'], catIds=cat['id']))

        # hit/miss?
        isHit = 0
        for ann in annList:
            # create a radius-15 circle around max location and see if it 
            # intersects with segmentation mask
            if type(ann['segmentation']) == list:
                # polygon
                for seg in ann['segmentation']:
                    polyPts = np.array(seg).reshape((len(seg)/2, 2))
                    poly    = shapely.geometry.Polygon(polyPts)
                    circ    = shapely.geometry.Point(maxSub[::-1]).buffer(15)
                    isHit  += poly.intersects(circ)
            else:
                # RLE
                if type(ann['segmentation']['counts']) == list:
                    rle = mask.frPyObjects([ann['segmentation']], I['height'], I['width'])
                else:
                    rle = [ann['segmentation']]
                m = mask.decode(rle)
                m = m[:, :, 0]
                ind  = np.where(m>0)
                mp   = shapely.geometry.MultiPoint(zip(ind[0], ind[1]))
                circ = shapely.geometry.Point(maxSub).buffer(15)
                isHit += circ.intersects(mp)

            if isHit:
                break

        if isHit: 
            hit += 1
            hitDiff += 1 if isDiff else 0
        else:
            miss += 1
            missDiff += 1 if isDiff else 0
        accuracy = (hit+0.0)/(hit+miss)
        try:
            accuracyDiff = (hitDiff+0.0)/(hitDiff+missDiff)
        except:
            accuracyDiff = None

        if time.time() - t0 > 10: 
            print cat['name'], '(', i, '/', numImgs, '): Hit =', hit, 'Miss =', miss, ' Acc =', accuracy, ' Diff Hit =', hitDiff, ' Diff Miss =', missDiff, ' Diff Acc =', accuracyDiff
            t0 = time.time()

    return (accuracy, accuracyDiff)


if __name__ == '__main__':
    args = parseArgs()
    print args

    # load COCO val2014
    imset   = 'val2014'
    imgDir  = '%s/images/%s/'%(coco_root, imset)
    annFile = '%s/annotations/instances_%s.json'%(coco_root, imset)
    cocoAnn = COCO(annFile)
    cocoAnn.info()
    catIds  = cocoAnn.getCatIds()
    catList = cocoAnn.loadCats(catIds)

    # init caffe
    #caffeNet = initCaffe(args)
    gpu = args.gpu
    caffe.set_device(gpu)
    caffe.set_mode_gpu()
	    
    caffeNet = get_net('googlenet_coco')
    transformer = get_COCO_net_transformer(caffeNet)

    # set heatmap type and parameters
    heatmapType = args.saliency
    if heatmap == 'excitation_backprop':
        normDeg = -1
        bottomName = 'pool2/3x3_s2'
    elif heatmap == 'contrast_excitation_backprop':
        normDeg = -2
        bottomName = 'pool2/3x3_s2'
    else:
        normDeg = np.inf
        bottomName = 'data'

    naiveMax = True # TODO: take as input via parser

    # get per-category accuracies
    accuracy = []
    accuracyDiff = []
    for cat in catList:
        (catAcc, catAccDiff) = evalPointingGame(cocoAnn, cat, caffeNet, imgDir, transformer, 
            heatmapType, topName = 'loss3/classifier', bottomName = bottomName, normDeg = normDeg, 
            naiveMax = naiveMax, gpu = gpu):
        print cat['name'], ' Acc =', catAcc, ' Diff Acc =', catAccDiff 
        accuracy.append(catAcc)
        accuracyDiff.append(catAccDiff)

    # report
    for c in range(len(catList)):
        print catList[c]['name'], ': Acc =', accuracy[c], ' Diff Acc =', accuracyDiff[c]
    print 'mean Acc =', np.mean(accuracy), 'mean Diff Acc =', np.mean(accuracyDiff)

