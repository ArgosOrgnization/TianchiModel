# TianchiModel

A repo to learn tianchi competition's model and a desktop application built on `PySide6`.

[toc]

## How to run

### Train

First you need install the training environment. A conda `yaml` file [envtrain.yaml](envtrain.yaml) is provided. You can create the environment by running:

```bash
conda env create -f envtrain.yaml
```

Second you need download the [dataset](https://tianchi.aliyun.com/dataset/140666). And put the `.zip` files under the `data` directory. An unzip bash script is provided to unzip the dataset:

```bash
cd data
sh unzip.sh
```

Third the label and test data has been preprocessed and the `.csv` files are stored at [data/label.csv](data/label.csv) and [data/test.csv](data/test.csv). Check the `data` directory to see if the files are there.

Fourth you can check the `model/recipe/demo.json` to check the traing configuration. You may modify the configuration to fit your needs. As the configuration is straightforward, you can modify it by yourself.

Finally you can run the training script and enter the `json` file path. Don't forget to activate the environment before running the script.

```bash
python scripttrain.py
please enter the train recipe file name: model/recipe/demo.json # take the demo.json as an example
```

### Application

First you need install the application environment. A conda `yaml` file [envapp.yaml](envapp.yaml) is provided. You can create the environment by running:

```bash
conda env create -f envapp.yaml
```

And then activate the environment.

Second you need to ensure a model has been trained. A trained model's path which is named as `demo` is supposed to look like:

```bash
.
└── demo
    ├── check_points
    │   ├── check_point.pth.tar
    │   ├── lowest_loss.pth.tar
    │   └── model_best.pth.tar
    ├── demo.json
    ├── log
    │   ├── best_accuracy.log
    │   └── train_log.log
    └── result
        └── result.csv
```

Third, you can run the application by running:

```bash
python scriptapp.py
```

And choose the model you want to use. The path should be `demo` in this case.

Finally you can use the user-interface to predict the result.

- currently supports single picture prediction. -- 2024/10/03

## Problems you may meet

### Pretrained-Model problem

Pretrained model is usually downloaded from the internet by a link. However, as the network is not stable, downloading the model may fail. A broken model may cause the program to crash. To solve this problem, you can delete the broken model and download it again.

On linux, the broken model is usually stored in `~/.cache/torch/hub/checkpoints/`. Just delete the broken model and run the script again.