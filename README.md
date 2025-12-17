# TSorchestra

[On going Project]

Time Series Orchestra (TSorchestra) is a novel ensemble framework designed for zero-shot time series forecasting. It's a curated collection of time series foundation models (TSFMs) that leverages each TSFM's strengths to create something greater than the sum of its parts, yielding SOTA performance.

## Set Up

---

1. Create a new conda environment named `tso` from our .yml file:

```bash
conda env create -f environment.yml
```

2. Download the [GIFT-Eval benchmark](https://huggingface.co/spaces/Salesforce/GIFT-Eval) from Hugging Face:

```bash
mkdir data
huggingface-cli download Salesforce/GiftEval --repo-type=dataset --local-dir data
```

3. Set up the environment variable for loading the datasets:
``bash 
echo "GIFT_EVAL=data" >> .env
``

## Usage

---

Run our evaluation script to reproduce our results:

```bash
chmod +x ./cli/eval.sh
./cli/eval.sh
```
