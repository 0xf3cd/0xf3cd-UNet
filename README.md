# 0xf3cd-UNet

* Dataset

    * https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/data

    * stored in the folder input

    * the structure of folder input is like:
    
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

    * Remember to mkdir for following dirs:

        * mkdir input/segmentation

        * mkdir input/segmentation/test

        * mkdir input/segmentation/train

        * mkdir input/segmentation/train/augmentation

        * mkdir input/segmentation/train/image

        * mkdir input/segmentation/train/mask

        * mkdir input/segmentation/train/dilate
    ```


* References

    * https://www.kaggle.com/eduardomineo/u-net-lung-segmentation-montgomery-shenzhen

    * https://github.com/zhixuhao/unet/

    * https://github.com/jocicmarko/ultrasound-nerve-segmentation/
