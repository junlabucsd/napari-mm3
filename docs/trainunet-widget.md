## Train U-Net.

<img width="1392" alt="train_unet" src="https://user-images.githubusercontent.com/40699438/230508707-af476b11-c7d5-4a81-a03f-6ce0b4576d08.png">

This widget trains a U-Net CNN on training data generated by the Annotate widget.

The current version is indebted to [DeLTA](https://gitlab.com/dunloplab/delta) for several of the image augmentation functions as well as the implementation of the pixelwise weighted loss.

If you prefer, you can alternately train your model by adapting the Jupyter notebook [here](https://github.com/junlabucsd/napari-mm3/blob/main/notebooks/train_Unet_from_GUI.ipynb).

**Parameters**

* `Image directory` : Location of raw images.
* `Mask directory` : Location of segmentation masks.
* `Weights directory` : Location of pixelwise weights for training (will be auto-generated if not found).
* `Test data directory` : (Optional) location of data to test saved model's performance on.
* `Model file` : Path to save trained model.
* `Load pre-trained weights` : Option to load pre-trained model.
* `Source model`: Pre-trained model location.
* `Training / validation split` : Fraction of images to use for training vs. validation.
* `Batch size` : Number of samples per epoch (should be a power of 2).
* `Epochs` : Number of epochs to train for.
* `Patience` : Number of epochs to wait with no improvement in validation loss before early stopping.
* `Image height` : Height to rescale images to.
* `Image width` : Width to rescale images to.
