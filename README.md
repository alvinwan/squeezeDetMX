# SqueezeDetMX

This repository contains a SqueezeDet port from Tensorflow to MXNet. Note that this *only* runs on Python3. It additionally holds utilities for reading and writing compact binary data, on top of deserialization functions for the KITTI dataset.

You can find the original squeezeDet implementation, using Tensorflow, [here](https://github.com/BichenWuUCB/squeezeDet). Parts of the codebase were taken from the original repository; all such instances have been cited accordingly.

# 1. Install

(Optional) We recommend setting up a virtual environment.

```
virtualenv squeezeDetMX --python=python3
source activate squeezeDetMX/bin/activate
```

Say `$SDMX_ROOT` is the root of your repository. Navigate to your root repository.

```
cd $SDMX_ROOT
```

We need to setup our Python dependencies.

```
pip install -r requirements.txt
```

# 2. Setup KITTI

*The first two steps were taken nearly word-for-word from the [original README](https://github.com/BichenWuUCB/squeezeDet#trainingvalidation).*

First, obtain the KITTI object detection dataset links: [images](http://www.cvlibs.net/download.php?file=data_object_image_2.zip) and [labels](http://www.cvlibs.net/download.php?file=data_object_label_2.zip). Start by creating and changing into a directory for KITTI.

```
mkdir $SDMX_ROOT/data/KITTI
cd $SDMX_ROOT/data/KITTI
```

Download and then unzip both files.

```
wget <link to object zip>
unzip data_object_image_2.zip
wget <link to label zip>
unzip data_object_label_2.zip
```

## Train-Val Splitting

Now we need to split the training data into a training set and a validation set. Create a directory to hold both.

```Shell
mkdir ImageSets
cd ./ImageSets
```

Then, create a new file `trainval.txt`, containing indices to all the images in the training data. In our experiments, we randomly split half of indices in `trainval.txt` into `train.txt` to form a training set and rest of them into `val.txt` to form a validation set.

```
ls ../training/image_2/ | grep ".png" | sed s/.png// > trainval.txt
```

For your convenience, we provide a script to split the train-val set automatically. Navigate to the data directory's root.

```
cd $SDMX_ROOT/data
```

Simply run

```Shell
python random_split_train_val.py
```

You should get both `train.txt` and `val.txt` under `$SQDT_ROOT/data/KITTI/ImageSets`.

When above two steps are finished, the structure of `$SQDT_ROOT/data/KITTI/` should at least contain:

```Shell
$SQDT_ROOT/data/KITTI/
                  |->training/
                  |     |-> image_2/00****.png
                  |     L-> label_2/00****.txt
                  |->testing/
                  |     L-> image_2/00****.png
                  L->ImageSets/
                        |-> trainval.txt
                        |-> train.txt
                        L-> val.txt
```

## Converting into RecordIO

This repository additionally contains a conversion script, from KITTI to RecordIO file objects. Navigate to the repository root, and run `convert.py`.

```
cd $SDMX_ROOT
python convert.py
```

If your data was downloaded and setup per the above `Data` section, the script will require no flags. Otherwise, run `python convert.py --help` for more information.

# 3. Train

The repository will support any RecordIO binary, written using the `squeezeDetMX.utils.Writer` object. By default, the script will look for RecordIO objects in `./data/KITTI`. To start running, use

```
python train.py
```