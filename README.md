# xbd-classification-training
Code preparation for Training xbd dataset for classification task

## Environment Setup


Before seting up the environment, clone this git repository. The first step involves preparing a Python conda environment. You can download Anaconda from [here](https://www.anaconda.com/distribution/).
Note that the corresponding python version required is **Python3.7**. 
Once downloaded and installed use the following codes to setup and activate the environment
```
cd \path\to\code\directory(i-e downloaded git repo --> to be refered as code directory ahead)
conda -V
conda update conda
conda create -n xview2 python=3.7 anaconda
conda activate xview2 && conda install pip
```
Once done use the following command to install tensorflow backend for GPU support. It is to be noted that tensorflow requires a compatiblity 
between tensorflow-gpu version , cuda version , cudnn version and installed drivers version. Please refer to [this](https://towardsdatascience.com/tensorflow-gpu-installation-made-easy-ubuntu-version-4260a52dd7b0)
article for further details. It is recommended to validate the correct installation of gpu support before moving ahead.
```
pip install numpy tensorflow-gpu && conda install cupy
```
for non-gpu version you can use
```
pip install numpy tensorflow
```
Once completed install the following dependencies 
```
pip install opencv-contrib-python
conda install -n xview2 shapely
conda install -n xview2 -c conda-forge keras
```
Finally you can run the following command to install the dependencies of requirements.txt file avalible in the code directory.
```
pip install -r requirements.txt
```
## xBD Dataset Download

The next step involves downloading the dataset. The same dataset in availble [here](https://xview2.org/download). Once downloaded, unzip and 
place the train dataset in the code directory in a folder named **Dataset**. The directory structure must be as following
```
Dataset\
  train\
    images
    labels
```
## Generating dataset for classification

This step involves extraction of damaged building instances from post damaged train images in the **Dataset** folder through the given annotations, cropping the same and saving them in **processed_data**
folder along with labels in **csv** folder. Note that lablels are already availble in the **csv** folder, you can either use the same by creating 
their backup at this stage as otherwise they will be overwritten. 
In the conda environment run the following script
```
python process_data.py --input_dir ./Dataset --output_dir ./processed_data --output_dir_csv ./csv --val_split_pct 0.2
```
On successful completion, a total number of **1,62,788** image files would be created occupying a harddisk space of **~3.06 GB** with 20% as validation data and 80% as test data

## Training Classifier on Dataset

Before using the classifier training script, it is noted that this script prepared as a pilot code just for meant to evaluate different aspects of the classifier
training as well as the dataset itself. Therefore the log created by conda during execution is critical to evaluate the correct implementation
of code. Important parts of log includes whether code validates all images taken from **processed_data** folder, runs on gpu, etc. Furthermore 
it would also help in identifying any error or bugs in codes. A backup of log even if all runs smoothly will be highly recommended.

Use the following script to run this code:
```
python xview2_training_v5.py --train_data ./processed_data --train_csv ./csv --test_data ./processed_data --test_csv ./csv --model_out ./output
```
The model uses different inbuild as well as external callbacks for logging and saving models in **Output** folder. It currently uses Resnet101
model available in **model** folder. Note that the **output** folder contains a live performance monitoring via an image file. Currently model is
set to train for only 100 epochs. The idea was to monitor the performance and set the learning rate and other parameters accordingly.

### Additional Note
Jupyter Colab notebook has been added [here](./Notebook.ipynb). Refer the same for implementation guidance.

### Citation Details
If you find this code useful in your research, please consider citing

	@misc{xbd-classification,
	author = {{Rafique, Hamza}},
	title = {Experimenting with xBD classification task for xview2 challenge using Keras},
	year = {2019},
	version = {1.0}
	address = {Air University, ISB. ham952@hotmail.com},
	url = {https://github.com/ham952/xview2-pytorch-firstrun}
	} 
