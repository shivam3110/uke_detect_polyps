conda create -y --name polyps python==3.12.2
conda activate polyps

conda install -y -c pytorch -c pytorch==2.5.1 torchvision=0.20.1 cpuonly
conda install -y -c conda-forge pandas=2.2.3
conda install -y -c conda-forge segmentation-models-pytorch=0.3.4 

pip install opencv-python==4.10.0.84
pip install flask==3.1.0