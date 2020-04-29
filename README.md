# wearsys20
Codes and data analysis for workshop wearsys 2020: "Human-In-The-Loop RL with an EEG Wearable Headset: On Effective Use of Brainwaves to Accelerate Learning"

## Running codes
Put codes in `train/` folder, with a separate folder for each subject
- Run `errp_riemann.ipynb` for running the baseline (state-of-the-art) results
- Run `errp_proposed.ipynb` for running the proposed algorithm

Note:
1. Choose `flag_test=False` for running 10-fold CV results (Paper includes only 10-fold CV)

## Data Analysis
All the results and analysis is in `data_analysis/` folder

## Motivation Results
- Activate conda environment `ml-py37-gpu` and run `rl_errp.py`
- Change `p_err_sim` flag to the decoding accuracy of errp (e.g., 0.6,0.7, etc.)
- Copy/paste the values to plot folder, and run `plot_erp_arl_all.py`



