# Cross-lingual Aspect-based Sentiment Analysis (EN to VI/DE/ZH)

> [!Attention]
> This project is a group project, not individial, our members: 
> [civi0411](https://github.com/civi0411) (Chí Vĩ)
> tantdna3@gmail.com (Anh Tấn)
> gaymap2005@gmail.com (Huỳnh An)
> dattran112212@gmail.com (Đạt)
> levokhanhtoan05@gmail.com (Khánh Toàn)
> This repository is re-posted from a deleted repository that was upload by our leader: [civi0411](https://github.com/civi0411)

---

This project is a complete source code study for researching the transfer of Emotional Assessment Knowledge (ABSA) from resource-rich languages ​​(English - EN) to less resource-rich languages ​​(Vietnamese - VI, German - DE, Chinese - ZH).

The project evaluates three modeling approaches:
1. **AG-CAN:** ​​Specialized architecture (mBERT + Aspect-Guided Attention + Gated Residuals).
2. **XLM-R:** Large-scale encoder with Exact Masked Mean Pooling.
3. **mT5:** Generating model (Generative Seq2Seq with Constrained Decoding/Label Scoring).

> How much target-language data do you really need for Aspect-Based Sentiment Analysis?  
> EN → VI / EN → DE · Restaurant & Phone domains · 3 model architectures.

[![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square)](https://python.org)
[![HuggingFace](https://img.shields.io/badge/🤗-M--ABSA-green?style=flat-square)](https://huggingface.co/datasets/Multilingual-NLP/M-ABSA)

---

## 📌 What is this?

We study **cross‑lingual transfer for Aspect‑Based Sentiment Analysis** under few‑shot conditions.  
Given an English training set, how many Vietnamese or German labeled examples are needed to reach good performance? Does the answer change between a restaurant domain (cultural, implicit aspects) and a phone domain (technical, explicit aspects)?

- **Task**: Oracle Aspect Sentiment Classification (aspect category provided, predict sentiment).
- **Languages**: English (source), Vietnamese, German (targets).
- **Models**: AG‑CAN (self‑built), XLM‑R, mT5.
- **Data**: [M-ABSA](https://huggingface.co/datasets/Multilingual-NLP/M-ABSA).

---

## 🧩 Repository structure (simplified)

```
.
├── configs/               # YAML configuration (hyperparameters, paths)
├── src/                   # Core modules (data, models, trainers, evaluation)
├── scripts/               # Utility scripts (prepare data, run experiments, aggregate)
├── notebooks/eda/         # Jupyter notebooks for exploratory data analysis
├── outputs/               # Generated results (checkpoints, figures, csv)
├── docs/                  # Detailed documentation (optional)
├── main.py                # Entry point for a single experiment
├── environment.yml        # Conda environment definition
├── requirements.txt       # Pip dependencies
└── README.md
```
---

## 🌟 Process Summary (A-Z)
The entire experimental process revolves around **3 main settings**:
* **S1 (Zero-shot):** Train 100% in English (Only need to run ONCE).
* **S2 (Few-shot):** Reload the "brain" of S1, then learn a small number of samples (50, 100, 200) of the target language (Run 3 seeds to get an average).
* **S3 (Full-target):** Reload the "brain" of S1, then learn the entire training set of the target language (Run 3 seeds).

---

## Step 1: Environment Preparation (Setup)
### Method 1: Using Pure Python (For Personal Computers / Colab)
You need to install the Python libraries. Make sure you are using a GPU-enabled environment (Nvidia/CUDA) to save runtime (Google Colab or Kaggle are best).

```bash
# Install core libraries
pip install torch transformers datasets matplotlib scikit-learn pyyaml
```
*(If you are running on a PC, make sure you have installed the CUDA driver for PyTorch.)*

### Method 2: Using Docker
Install **Docker Desktop** and type the following two commands:

```bash
# 1. Build a virtual environment (Run only once)
docker-compose build

# 2. Start running the code (Insert "docker-compose run --rm absa" before every python command)
# For example, instead of typing "python scripts/eval.py", type:
docker-compose run --rm absa python scripts/eval.py
```

---

## Step 2: Training S1 (Initializing the Root Checkpoint)
**THIS STEP MUST BE RUN FIRST BEFORE RUNNING S2/S3!**
S1 will train the models based on the English dataset (`seed = 42`). Then, the system will automatically save the Checkpoint and test it on all three datasets: Vietnamese (vi), German (de), and Chinese (zh).

**This command runs the entire project (3 Models x 2 Domains):**
```bash
python scripts/train.py --setting s1 --targets vi de zh
```

**Or run separately:**
```bash
# Run AG-CAN
python scripts/train.py --setting s1 --models ag_can --domains restaurant phone --targets vi de zh

# Run XLM-R
python scripts/train.py --setting s1 --models xlmr --domains restaurant phone --targets vi de zh

# Run mT5
python scripts/train.py --setting s1 --models mt5 --domains restaurant phone --targets vi de zh
```
After running, the best model will be saved at: `outputs/checkpoints/<model>/<domain>/s1/s1_seed42/best.pt`. **(You can send this `best.pt` file to your friends so they can run S2/S3 directly without having to rework S1).**

---

## Step 3: Training S2 & S3 (Knowledge Transfer)
After obtaining the S1 Checkpoint file in the `outputs/checkpoints/` directory, proceed to run S2 (Few-shot) or S3 (Full-target).
*The system will automatically find Checkpoint S1, load the payload, and repeat the training on 3 levels of Random Seeds (42, 123, 456).*

**Run S2 (Few-shot learning):**
```bash
python scripts/train.py --setting s2 --targets vi de zh
```
*(By default, it will automatically run at N = 50, 100, and 200 samples.)*

**Run S3 (Full-target learning):**
```bash
python scripts/train.py --setting s3 --targets vi de zh
```

---

## Step 4: Evaluation & Report Charting
After the 3 Settings (S1, S2, S3) have finished running, all the score results (`Macro F1`, `Accuracy`) will be compiled into `.json` files inside the `outputs/results/` folder.

To read data and convert it into **charts**:
```bash
python scripts/eval.py
```

Now, in the `outputs/figures/` folder, and you will find:
1. `macro_f1_comparison.png`: A graph comparing the F1 scores of the three models.
2. `recovery_curves.png`: A line graph demonstrating how few shots (S2) help the model improve quickly.
3. `gap_matrix.png`: A heatmap showing multilingual inertia.
4. `error_taxonomy.png`: A graph classifying errors (Which type of sentence does the model make the most mistakes in?).

---

## 🛠️ Some Advanced Customization Commands

**1. Run only one specific language in S2:**
```bash
python scripts/train.py --setting s2 --targets vi --models ag_can --domains restaurant
```

**2. Overwrite (Delete the old result and run again from the beginning):**
Add the `--force` flag to the end of the command:
```bash
python scripts/train.py --setting s1 --targets vi de zh --force
```

**3. Editing Hyperparameters:**
Open the `config.yml` file and change the following directly:
* `batch_size`: Decrease if the GPU runs out of VRAM (OOM - Out of memory).
* `epochs`: Increase if the model has not converged.
* `mt5_lr`: A very small Learning Rate specifically for mT5.
