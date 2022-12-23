

## Project Overview

In this notebook, we will build a deep neural network that functions as part of an end-to-end automatic speech recognition (ASR) pipeline!  

![ASR Pipeline][image1]

We begin by investigating the [LibriSpeech dataset](http://www.openslr.org/12/) that will be used to train and evaluate our models. Our algorithm will first convert any raw audio to feature representations that are commonly used for ASR. You will then move on to building neural networks that can map these audio features to transcribed text. After learning about the basic types of layers that are often used for deep learning-based approaches to ASR, 
## Project Instructions


1. Obtain the appropriate subsets of the LibriSpeech dataset, and convert all flac files to wav format.
```
wget http://www.openslr.org/resources/12/dev-clean.tar.gz
tar -xzvf dev-clean.tar.gz
wget http://www.openslr.org/resources/12/test-clean.tar.gz
tar -xzvf test-clean.tar.gz
mv flac_to_wav.sh LibriSpeech
cd LibriSpeech
./flac_to_wav.sh
```

2. Create JSON files corresponding to the train and validation datasets.
```
cd ..
python create_desc_json.py LibriSpeech/dev-clean/ train_corpus.json
python create_desc_json.py LibriSpeech/test-clean/ valid_corpus.json
```

3. Start Jupyter:
```
jupyter notebook --ip=0.0.0.0 --no-browser
```

### Local Environment Setup

You should run this project with GPU acceleration for best performance.



1. Create (and activate) a new environment with Python 3.6 and the `numpy` package.

	
	- __Windows__: 
	```
	conda create --name aind-vui python=3.5 numpy scipy
	activate aind-vui
	```

2. Install TensorFlow.
	- Option 1: __To install TensorFlow with GPU support__, follow [the guide](https://www.tensorflow.org/install/) to install the necessary NVIDIA software on your system.  If you are using an EC2 GPU instance, you can skip this step and only need to install the `tensorflow-gpu` package:
	```
	pip install tensorflow-gpu==1.1.0
	```
	- Option 2: __To install TensorFlow with CPU support only__,
	```
	pip install tensorflow==1.1.0
	```

3. Install a few pip packages.
```
pip install -r requirements.txt
```

4. Switch [Keras backend](https://keras.io/backend/) to TensorFlow.
	
	- __Windows__: 
	```
	set KERAS_BACKEND=tensorflow
	python -c "from keras import backend"
	```
	
5. Obtain the `libav` package.
	
	- __Windows__: Browse to the [Libav website](https://libav.org/download/)
		- Scroll down to "Windows Nightly and Release Builds" and click on the appropriate link for your system (32-bit or 64-bit).
		- Click `nightly-gpl`.
		- Download most recent archive file.
		- Extract the file.  Move the `usr` directory to your C: drive.
		- Go back to your terminal window from above.
	```
	rename C:\usr avconv
    set PATH=C:\avconv\bin;%PATH%
	```

6. Obtain the appropriate subsets of the LibriSpeech dataset, and convert all flac files to wav format.
	
	- __Windows__: Download two files ([file 1](http://www.openslr.org/resources/12/dev-clean.tar.gz) and [file 2](http://www.openslr.org/resources/12/test-clean.tar.gz)) via browser and save in the `AIND-VUI-Capstone` directory.  Extract them with an application that is compatible with `tar` and `gz` such as [7-zip](http://www.7-zip.org/) or [WinZip](http://www.winzip.com/). Convert the files from your terminal window.
	```
	move flac_to_wav.sh LibriSpeech
	cd LibriSpeech
	powershell ./flac_to_wav.sh
	```

7. Create JSON files corresponding to the train and validation datasets.
```
cd ..
python create_desc_json.py LibriSpeech/dev-clean/ train_corpus.json
python create_desc_json.py LibriSpeech/test-clean/ valid_corpus.json
```

8. Create an [IPython kernel](http://ipython.readthedocs.io/en/stable/install/kernel_install.html) for the `aind-vui` environment.  Open the notebook.
```
python -m ipykernel install --user --name aind-vui --display-name "aind-vui"
jupyter notebook vui_notebook.ipynb
```

9. Before running code, change the kernel to match the `aind-vui` environment by using the drop-down menu.  Then, follow the instructions in the notebook.

![select aind-vui kernel][image2]

_
### Project Submission

When you are ready to submit your project, collect the following files and compress them into a single archive for upload:
- The `vui_notebook.ipynb` file with fully functional code, all code cells executed and displaying outpu.
- An HTML or PDF export of the project notebook with the name `report.html` or `report.pdf`.
- The `sample_models.py` file with all model architectures that were trained in the project Jupyter notebook.
- The `results/` folder containing all HDF5 and pickle files corresponding to trained models.



#### Files Submitted

| Criteria       		|     Meets Specifications	        			            | 
|:---------------------:|:---------------------------------------------------------:| 
| Submission Files      | The submission includes all required files.		|

#### STEP 2: Model 0: RNN

| Criteria       		|     Meets Specifications	        			            | 
|:---------------------:|:---------------------------------------------------------:| 
| Trained Model 0         		| The submission trained the model for at least 20 epochs, and none of the loss values in `model_0.pickle` are undefined.  The trained weights for the model specified in `simple_rnn_model` are stored in `model_0.h5`.   	|

#### STEP 2: Model 1: RNN + TimeDistributed Dense

| Criteria       		|     Meets Specifications	        			            | 
|:---------------------:|:---------------------------------------------------------:| 
| Completed `rnn_model` Module         		| The submission includes a `sample_models.py` file with a completed `rnn_model` module containing the correct architecture.   	|
| Trained Model 1         		| The submission trained the model for at least 20 epochs, and none of the loss values in `model_1.pickle` are undefined.  The trained weights for the model specified in `rnn_model` are stored in `model_1.h5`.   	|

#### STEP 2: Model 2: Bidirectional RNN + TimeDistributed Dense(Our Model)
| Criteria       		|     Meets Specifications	        			            | 
|:---------------------:|:---------------------------------------------------------:| 
| Completed `bidirectional_rnn_model` Module         		| The submission includes a `sample_models.py` file with a completed ` bidirectional_rnn_model` module containing the correct architecture.   	|
| Trained Model 2         		| The submission trained the model for at least 20 epochs, and none of the loss values in `model_2.pickle` are undefined.  The trained weights for the model specified in ` bidirectional_rnn_model` are stored in `model_2.h5`.   	|

#### STEP 2: Model 3: Deeper RNN + TimeDistributed Dense

| Criteria       		|     Meets Specifications	        			            | 
|:---------------------:|:---------------------------------------------------------:| 
| Completed `deep_rnn_model` Module         		| The submission includes a `sample_models.py` file with a completed `deep_rnn_model` module containing the correct architecture.   	|
| Trained Model 3         		| The submission trained the model for at least 20 epochs, and none of the loss values in `model_3.pickle` are undefined.  The trained weights for the model specified in `deep_rnn_model` are stored in `model_3.h5`.   	|

#### STEP 2: Model 4:  CNN + RNN + TimeDistributed Dense


| Criteria       		|     Meets Specifications	        			            | 
|:---------------------:|:---------------------------------------------------------:| 
| Completed `cnn_rnn_model` Module         		| The submission includes a `sample_models.py` file with a completed `cnn_rnn_model` module containing the correct architecture.   	|
| Trained Model 4         		| The submission trained the model for at least 20 epochs, and none of the loss values in `model_4.pickle` are undefined.  The trained weights for the model specified in `cnn_rnn_model` are stored in `model_4.h5`.   	|

#### STEP 2: Compare the Models

| Criteria       		|     Meets Specifications	        			            | 
|:---------------------:|:---------------------------------------------------------:| 
| Question 1         		| The submission includes a detailed analysis of why different models might perform better than others.   	|

