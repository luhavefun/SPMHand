# SPMHand: Segmentation-guided Progressive Multi-path 3D Hand Pose and Shape Estimation

## Introduction

Pytorch implementation of "SPMHand: Segmentation-guided Progressive Multi-path 3D Hand Pose and Shape Estimation"
- paper download: https://ieeexplore.ieee.org/abstract/document/10404038


# Usage
## 1. Dependencies
This code is tested on [Ubuntu 20.04 LTS，python 3.9，pytorch 1.12.0]. 
```
1. conda activate [your_enviroment]
2. pip install -r requirments.txt
```

## 2. Pytorch MANO layer
* Download `MANO_RIGHT.pkl` from [here](https://mano.is.tue.mpg.de/) and place at `common/utils/manopth/mano/models`. 
* We use the pytorch MANO layer [manopth](https://github.com/hassony2/manopth).

## 3. Data
We use the annotation files in [HandOccNet](https://github.com/namepllet/HandOccNet). Download and place the annotation files following the instructions in HandOccNet.

## 4. Test
Place the [pretrained model](https://1drv.ms/u/s!AsjOlKfg2ljb2S9uWqzfu6fEUiSD?e=058TZV) at `output/model_dump/` and run
```
python main/test.py --test_epoch {test epoch}  
```  
The result will generate at `output/result/`. Upload the .zip file to [ho3d-v2 test server](https://codalab.lisn.upsaclay.fr/competitions/4318).

For HO3D-v3, use the [pretrained model](https://1drv.ms/u/s!AsjOlKfg2ljb9WEzjwNvZidvKrs6?e=QeLNJr) and upload the results to [ho3d-v3 test server](https://codalab.lisn.upsaclay.fr/competitions/4393).

## 5. Train
For HO3D-v2 dataset, we use the visible hand segmentations in the original dataset. Download whole hand segmentations from [here](https://1drv.ms/u/c/db58dae0a794cec8/EbWcvQGZRidFuWKPaCzY7dQB7KB80xnyXy18-Gx8sCZ_eA?e=SkciNM) and update lines 56-57 in `config.py`.

To train the model with Obman pretraining, download the pretrained weights fron [here](https://1drv.ms/u/c/db58dae0a794cec8/EbxYgcIRNvRKng3RXOS6eQEBElmYCCbXw5dC-hP1fZtteQ?e=u8eUYE) and run
```
python main/train.py  --pretrain --pretrain_cpt {cpt_path}
```

To train the model without pretraining, download the stage 1 weights from [here](https://1drv.ms/u/c/db58dae0a794cec8/Ee8AKE44bq5BkUpuX4VmMCYBhuwGKpb8tJwJvgbdS7ZXrA?e=1cVS9g), or train the stage 1 weights from scratch, and run
```
python main/train.py  --stage_seg --stage_seg_cpt {cpt_path}
``` 


## Citation

If you find our work useful in your research, please consider citing:

```
@article{lu2024spmhand,
  title={SPMHand: Segmentation-guided Progressive Multi-path 3D Hand Pose and Shape Estimation},
  author={Lu, Haofan and Gou, Shuiping and Li, Ruimin},
  journal={IEEE Transactions on Multimedia},
  year={2024},
  publisher={IEEE}
}
```

## Acknowledgements
We relied on the project framework and research codes of [HandOccNet](https://github.com/namepllet/HandOccNet).
