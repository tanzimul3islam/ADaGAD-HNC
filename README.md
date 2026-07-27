# AdaGAD-HNC

**Ada**ptive **G**raph **A**nomaly **D**etection with **H**ard-**N**egative **C**urriculum

---

## Project Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```


```

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

## Training

### Single training run

```bash
python -m src.main train model=adagad_hnc data=amazon train=curriculum
python -m src.main train model=adagad_hnc data=yelphotel train=curriculum
python -m src.main train model=adagad_hnc data=yelpnyc train=curriculum
python -m src.main train model=adagad_hnc data=yelpres train=curriculum
```

### Multi-seed evaluation

```bash
python -m src.experiments.run_multiseed --config configs/default.yaml --seeds 1 2 3 4 5
```

### Evaluation

```bash
python -m src.main eval --ckpt outputs/<run>/checkpoints/best.pt
```

### Ablations

```bash
python -m src.experiments.ablations --config configs/default.yaml --suite fusion_curriculum
```
