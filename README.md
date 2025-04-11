
## Install dependencies   
```bash
cd CoMER
# install project   
conda create -y -n CoMER python=3.7
conda activate CoMER
conda install pytorch=1.8.1 torchvision=0.9.1 cudatoolkit=11.1 pillow=8.4.0 -c pytorch -c nvidia
# training dependency
conda install pytorch-lightning=1.4.9 torchmetrics=0.6.0 -c conda-forge
# evaluating dependency
conda install pandoc=1.19.2.1 -c conda-forge
pip install -e .
 ```

## Training
```bash
# train CoMER(Fusion) model using 4 gpus and ddp
python train.py --config config.yaml  
```



For single gpu user, you may change the `config.yaml` file to
```yaml
gpus: 1
# gpus: 4
# accelerator: ddp
```

## Evaluation

```bash
perl --version  # make sure you have installed perl 5

unzip -q data.zip

bash eval_all.sh 0
```