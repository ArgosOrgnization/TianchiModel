conda install -c conda-forge numpy pandas pyside6 pillow qt-material
conda install pytorch torchvision torchaudio cpuonly -c pytorch # for CPU
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia # for CUDA 12.1
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia