<h1 align="center"><ins>S</ins>patio-<ins>T</ins>emporal <ins>G</ins>raph <ins>C</ins>onvolutional <ins>N</ins>etworks: <br> A Deep Learning Framework for Traffic Forecasting</h1>

This is the offical implementation of 
## Abstract

## Requirements
Our code is based on Python3 (>= 3.6). There are a few dependencies to run the code. The major libraries are listed as follows:
* TensorFlow (>= 1.9.0)
* NumPy (>= 1.15)
* SciPy (>= 1.1.0)
* Pandas (>= 0.23)

The implementation of Spatio-Temporal Graph Convolutional Layer with PyTorch is available in [PyG Temporal](https://github.com/benedekrozemberczki/pytorch_geometric_temporal/blob/master/torch_geometric_temporal/nn/attention/stgcn.py). You might refer to [STConv](https://pytorch-geometric-temporal.readthedocs.io/en/latest/modules/root.html#temporal-graph-attention-layers) that supports ChebConv Graph Convolutions.

## Dataset
### Data Source
**[PeMSD7](http://pems.dot.ca.gov/)** was collected from Caltrans Performance Measurement System (PeMS) in real-time by over 39, 000 sensor stations, deployed across the major metropolitan areas of California state highway system. The dataset is also aggregated into 5-minute interval from 30-second data samples. We randomly select a medium and a large scale among the District 7 of California containing **228** and **1, 026** stations, labeled as PeMSD7(M) and PeMSD7(L), respectively, as data sources. The time range of PeMSD7 dataset is in the weekdays of **May and June of 2012**. We select the first month of historical speed records as training set, and the rest serves as validation and test set respectively. 

Dataset PeMSD7(M/L) is now available under `dataset` folder (station list included). Please refer [issue #6](https://github.com/VeritasYin/STGCN_IJCAI-18/issues/6) for how to download metadata from PeMS.

### Data Format
You can make your customized dataset by the following format:  
- PeMSD7_V_{`$num_route`}.csv : Historical Speed Records with shape of [len_seq * num_road] (len_seq = day_slot * num_dates).
- PeMSD7_W_{`$num_route`}.csv : Weighted Adjacency Matrix with shape of [num_road * num_road].

Note: please replace the `$num_route` with the number of routes in your dataset. '*.csv' should not contain any index or header in the file.

### Data Preprocessing
The standard time interval is set to 5 minutes. Thus, every node of the road graph contains **288** data points per day (day_slot = 288). The linear interpolation method is used to fill missing values after data cleaning. In addition, data input are normalized by Z-Score method.  
In PeMSD7, the adjacency matrix of the road graph is computed based on the distances among stations in the traffic network. The weighted adjacency matrix W can be formed as,  
<p align="center">
  <img src="https://latex.codecogs.com/svg.image?\large&space;w_{ij}=\left\{&space;\begin{aligned}&space;&\exp(-\frac{{d_{ij}^2}}{{\sigma^2}}),~i&space;\neq&space;j~\text{and}~\exp(-\frac{{d_{ij}^2}}{{\sigma^2}})&space;\geq&space;\epsilon\\&space;&0\qquad\qquad,~\text{otherwise}.&space;\end{aligned}&space;\right."/>
</p>
  
All of our experiments use 60 minutes as the historical time window, a.k.a. 12 observed data points (M = 12) are used to forecast traffic conditions in the next 15, 30, and 45 minutes (H = 3, 6, 9).

## Model Details
### Training
python main.py --n_route {`$num_route`} --graph {`$weight_matrix_file`} 

**Default settings**:  
* Training configs: argparse is used for passing parameters. 
    * n_route=228, graph='default', ks=3, kt=3, n_his=12, n_pred=9 
    * batch_size=50, epoch=50, lr=0.001, opt='RMSProp', inf_mode='merge', save=10
* Data source will be searched in dataset_dir = './dataset', including speed records and the weight matrix.
* Trained models will be saved in save_path = './output/models' every `args.save=10` epochs.
* Training logs will be saved in sum_path = './output/tensorboard'.  

Note: it normally takes around 6s on a NVIDIA TITAN Xp for one epoch with the batch size of 50 and n_route of 228.

### Folder structure
```
├── data_loader
│   ├── data_utils.py
│   └── __init__.py
├── dataset
│   ├── PeMSD7_V_228.csv
│   ├── PeMSD7_W_228.csv
│   ├── PeMSD7_V_1026.csv
│   └── PeMSD7_W_1026.csv
├── main.py
├── models
│   ├── base_model.py
│   ├── __init__.py
│   ├── layers.py
│   ├── tester.py
│   └── trainer.py
├── output
│   ├── models
│   └── tensorboard
├── README.md
└── utils
    ├── __init__.py
    ├── math_graph.py
    └── math_utils.py
```

## Updates
**Feb. 22, 2022**:
* Sensor Station List of PeMSD7-M released.

**Feb. 11, 2022**:
* Dataset PeMSD7-L (1,026 nodes) released. 
* Fix the issue in size calculation of temporal channel. Thanks to @KingWang93 and @cheershuaizhao.

**Apr. 18, 2019**: 
* Dataset PeMSD7-M (228 nodes) released.  
  
**Jan. 14, 2019**: 
* Code refactoring based on the [Tensorflow-Project-Template](https://github.com/MrGemy95/Tensorflow-Project-Template), following the PEP 8 code style; 
* Function model_save(), model_test() and tensorboard support are added; 
* The process of model training and inference is optimized;
* Corresponding code comments are updated.

## Citation
Please refer to our paper. Bing Yu*, Haoteng Yin*, Zhanxing Zhu. [Spatio-temporal Graph Convolutional Networks: A Deep Learning Framework for Traffic Forecasting](https://www.ijcai.org/proceedings/2018/0505). In *Proceedings of the 27th International Joint Conference on Artificial Intelligence (IJCAI)*, 2018

    @inproceedings{yu2018spatio,
        title={Spatio-temporal Graph Convolutional Networks: A Deep Learning Framework for Traffic Forecasting},
        author={Yu, Bing and Yin, Haoteng and Zhu, Zhanxing},
        booktitle={Proceedings of the 27th International Joint Conference on Artificial Intelligence (IJCAI)},
        year={2018}
    }

