# Max Fusion U-Net (MFU-Net) for multi-modal cardiac anatomy and pathology segmentation

Implementation of the **NFU-Net** model to perform multi-modal caridac anatomy and pathology segmentation. For further details please see our [paper], accepted in [MICCAI-2020 Workshop: DART].

Python dependencies to run the code is listed in the file 'requirements.txt'.

The structure of this project is the following:

* **configuration**: package containing configuration parameters for running an experiment.
* **layers**: package with custom Keras layers
* **loaders**: package with data loaders
* **models**: package with the SDNet model and other Keras models
* **model_executors**: package with scripts for running an experiment
* **callbacks**: package with Keras callbacks for printing images and losses during training
* **DataProcess**: package with some of the data preprocess codes for some public datasets


To define a new data loader, extend class `base_loader.Loader`, and register the loader in `loader_factory.py`. The datapath is specified in `parameters.py`.

To run an experiment, execute `experiment.py`, passing the configuration filename, the split number as runtime parameters, and the testmode:
```
--test True --config unet_multi_modal_cardiac --split 0 --testmode feature-concat-attention-maxfuseall-keeporg
```

The test mode is defined as follows:
pixel-concat: multi-modal information is merged by pixel-concatenation
feature-concat: multi-modal information is merged by feature-concatenation
maxfuseall: multi-modal information is merged by maximum operator at all the encoder-decoder skipping connections;
keeporg: the original concatenated features across different modalities are kept in the skipping connections;
attention: dedicate a spatial attention module at the end of the decoder

Details can be referred to the paper and the ./models/unet.py
In addition, other module instructions can be referred to from the shell-scripts in the folder: 
64-TrainingScriptPaper-MaxFuse-NoPretrained
TrainingScriptsChallenge


To run an test, execute `experiment.py` as follows:
```
--test True --config unet_multi_modal_cardiac --split 0 --testmode feature-concat-attention-maxfuseall-keeporg --test True
```

Citation

If you use this code for your research, please cite our paper:

```
@incollection{jiang2020maxfusion,
  title={Max-Fusion U-Net for Multi-Modal Pathology Segmentation with Attention and Dynamic Resampling},
  author={Haochuan, Jiang and Wang, Chengjia and Chartsias, Agisilaos and Tsaftaris, Sotirios A},
  booktitle={Myocardial pathology segmentation combining multi-sequence SMR - MyoPS 2020},
  year={2020},
  publisher={Springer}
}
```
 
[Keras]: https://keras.io/
[tensorflow]: https://www.tensorflow.org/
[MICCAI-2020]: https://miccai2020.org/en/
[MyoPS-2020]: http://www.sdspeople.fudan.edu.cn/zhuangxiahai/0/MyoPS20/index.html
