# 0xf3cd-UNet

* Dataset
    * Download from here: https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/data
    * Or download the input folder here: https://drive.google.com/file/d/1vJHLEkHA_cVOU8uVvldgBwLNmXfiAY0G/view?usp=sharing
    * All the data is stored in the folder "input"
    * If you get the dataset from kaggle
        * Create a folder called "input"
        * Unzip the datasets downloaded from kaggle, and put them into "./input"
        * Remember to mkdir for following dirs:
            * `mkdir input/segmentation`
            * `mkdir input/segmentation/test`
            * `mkdir input/segmentation/train`
            * `mkdir input/segmentation/train/augmentation`
            * `mkdir input/segmentation/train/image`
            * `mkdir input/segmentation/train/mask`
            * `mkdir input/segmentation/train/dilate`
        * Remember to run `process_dataset.py` for at least one time to process the dataset
    * The structure of folder "input" is just like:

    ```
    .
    ├── pulmonary-chest-xray-abnormalities
    │   ├── ChinaSet_AllFiles
    │   │   ├── ClinicalReadings
    │   │   ├── CXR_png
    │   │   └── NLM-ChinaCXRSet-ReadMe.docx
    │   └── Montgomery
    │       └── MontgomerySet
    ├── segmentation
    │   ├── test
    │   └── train
    │       ├── augmentation
    │       ├── dilate
    │       ├── image
    │       └── mask
    └── shcxr-lung-mask
    └── mask
    └── mask
    ```

* References
    * https://www.kaggle.com/eduardomineo/u-net-lung-segmentation-montgomery-shenzhen
    * https://github.com/zhixuhao/unet/
    * https://github.com/jocicmarko/ultrasound-nerve-segmentation/
    * https://github.com/lixiaolei1982/Keras-Implementation-of-U-Net-R2U-Net-Attention-U-Net-Attention-R2U-Net.-
