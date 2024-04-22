# FeddepWithEM
We make an attempt to adjust feddep Dgen to get better results and consider more about the privacy of each client.

# Installation
First of all, users need to clone the source code and install the required packages (we suggest python version >= 3.9).

 ```
 git clone https://github.com/GalaxyBangBang/FeddepWithEM.git
 cd FeddepWithEM
 ```
## Use Conda
We recommend using a new virtual environment to install FederatedScope:
```
conda create -n fs python=3.9
conda activate fs
```
If your backend is torch, please install torch in advance (torch-get-started). For example, if your cuda version is 11.3 please execute the following command:

```
conda install -y pytorch=1.10.1 torchvision=0.11.2 torchaudio=0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
```
 ## From source
 Finally, after the backend is installed, you can install FederatedScope from source:
```
 pip install .
```
Now, you have successfully installed the minimal version of FederatedScope. For application version including graph run:
```
conda install -y pyg==2.0.4 -c pyg
conda install -y rdkit=2021.09.4=py39hccf6a74_0 -c conda-forge
conda install -y nltk
```

After all the above steps are completed, you can run 
```
#pwd PATH/TO/FeddepWithEM
python ./federatedscope/main.py --cfg federatedscope/gfl/feddep/feddep_on_cora5.yaml
```
# Tips
If you encouter a problem of the version of scipy, just use 
```
pip uninstall scipy
pip install scipy
```
to solove the compatiblity problem.