# Install 

```bash
git clone https://github.com/baixiaok/WDLCD.git
cd WDLCD
conda env create -f environment.yml
cd apex-master
python setup.py install
```

# Train

```bash
python -m torch.distributed.launch --nproc_per_node=1 main.py --data dog/  --num-classes 120  --epochs 1000 --model wdlcd_tiny_super  -b 16 --lr 1e-3 --weight-decay .03 --img-size 224 --amp
```

# Validate

```bash
python validate.py "dog/test/"  --model wdlcd_tiny_super  --eval_checkpoint  "output/train/.../model_best.pth.tar"   --num-classes 120

