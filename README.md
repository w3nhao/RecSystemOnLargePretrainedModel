

```bash
conda create -n plmrs python=3.8
conda activate plmrs

wget https://mirror.sjtu.edu.cn/pytorch-wheels/cu113/torch-1.12.1+cu113-cp38-cp38-linux_x86_64.whl
pip install torch-1.12.1+cu113-cp38-cp38-linux_x86_64.whl
pip install https://github.com/Lightning-AI/lightning/archive/refs/heads/master.zip -U
pip install -r requirements.txt

cd ./data/setup_scripts
python hm.py
python MIND_large.py
python MIND_small.py
cd ../../

python run.py --input_type "text" --plm_name "facebook/opt-125m" --dataset "MIND_large"
```