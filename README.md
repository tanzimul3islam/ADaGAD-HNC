# AdaGAD-HNC

**Ada**ptive **G**raph **A**nomaly **D**etection with **H**ard-**N**egative **C**urriculum

---

## Project Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install torch torchvision
pip install torch-geometric
pip install -r requirements.txt
```

This will install CPU-only PyTorch. For GPU support, replace the PyTorch install with the appropriate CUDA version from https://pytorch.org.

---

## Data Preparation

Supported datasets: Amazon, YelpHotel, YelpNYC, YelpRes.

Place raw `.mat` files under `dataset/` or `raw_dataset/`. Run:

```bash
python scripts/prepare_data.py --dataset amazon
python scripts/prepare_data.py --dataset yelphotel
python scripts/prepare_data.py --dataset yelpnyc
python scripts/prepare_data.py --dataset yelpres
```

---

train:

python -m src.main train model=adagad_hnc_medium data=yelphotel_cpu train=tuned experiment_name=yelphotel > train_yelphotel.log
python -m src.main train model=adagad_hnc_medium data=yelpnyc_cpu train=tuned experiment_name=yelpnyc > train_yelpnyc.log
python -m src.main train model=adagad_hnc_medium data=yelpres_cpu train=tuned experiment_name=yelpres > train_yelpres.log

test:
python -m src.main eval --ckpt outputs/yelphotel/best.pt model=adagad_hnc_medium data=yelphotel_cpu
python -m src.main eval --ckpt outputs/yelpnyc/best.pt model=adagad_hnc_medium data=yelpnyc_cpu
python -m src.main eval --ckpt outputs/yelpres/best.pt model=adagad_hnc_medium data=yelpres_cpu```
