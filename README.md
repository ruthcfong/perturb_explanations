# Interpretable Explanations of Black Boxes by Meaningful Perturbation
Ruth Fong and Andrea Vedaldi

## Instructions
1. Install Jianming Zhang's [version of Caffe](https://github.com/jimmie33/Caffe-ExcitationBP), which includes implementations for the gradient, guided backprop, and excitation backprop saliency methods.
2. Include your local copy of the above repo in your python path by running `export PYTHONPATH=$PYTHONPATH:$CAFFEDIR` (or including it in your .bash_profile or .bashrc)
3. Run `sh $CAFFEDIR/data/ilsvrc12/get_ilsvrc_aux.sh` to download ILSVRC support files.
4. Change `caffe_dir` to $CAFFEDIR in [defaults.py](code/caffe/python/defaults.py).
3. (Optional but quite useful) Download ImageNet Annotations: [http://image-net.org/Annotation/Annotation.tar.gz](http://image-net.org/Annotation/Annotation.tar.gz) (found [here](http://image-net.org/download-bboxes))
4. (Optional, for animal parts experiment only) Download animal parts annotations from [here](http://www.robots.ox.ac.uk/~vgg/data/animal_parts)

## Usage
All pycaffe files are in [code/caffe/python](code/caffe/python). Public support for matconvnet will come soon (see dev branch for some experimental code).

### Learn Perturbation Masks
See [optimize_mask.py](code/caffe/python/optimize_mask.py) (the functions `optimize_mask` and `generate_learned_mask` provide most of the functionality).

To generate masks (and debugging figures) for the the first 10 images in the heldout set, run:
`python optimize_mask.py train_heldout -f $FIG_DIR -m $MASK_DIR -g [$GPU]`

### Reproduce Figures
See [figures.ipynb](code/caffe/python/figures.ipynb) to reproduce all figures in the main text. Public support for reproducing all supplementary figures will come soon.

### Weak Localization
See [localization.ipynb](code/caffe/python/localization.ipynb) to reproduce numbers in Table 1.

To produce weak localization bounding boxes, use the [heatmaps.py](code/caffe/python/heatmaps.py):
`python heatmaps.py OUT_PATH/OUT_DIR DATA_DESC HEATMAP_TYPE THRESHOLD_TYPE [MASK_DIR] [GPU]`
* `OUT_PATH/OUT_DIR` is where to save the output file with bounding box information (a directory is taken as input when `DATA_DESC` == `annotated_train_heldout`
* `DATA_DESC` $\in$ [`annotated_train_heldout`, `val`, `animal_parts`]
* `HEATMAP_TYPE` $\in$ [`mask`, `saliency` (a.k.a. gradient), `guided_backprop`, `excitation_backprop`], and 
* `THRESHOLD_TYPE` $\in$ [`min_max_diff` (a.k.a. value), `energy`, `mean`]
* `MASK_DIR` is where the masks for the 50k val images are saved (only include when `HEATMAP_TYPE` == `mask`)
