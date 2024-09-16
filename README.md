This is the code used in the following paper:

https://www.biorxiv.org/content/10.1101/2023.11.28.569131v1

- `inspect_interleaved_vs_blocked_regress.py`: Performs the
  regression analysis reported in the paper and generates
  figure 1.

- `inspect_interleaved_vs_blocked_models_results_single_state.py`:
  Generate figure 2 from the paper and report some stats
  about fit parameters.

- `inspect_interleaved_vs_blocked_models_single_state.py`:
  fit the single-state state-space model.

- `inspect_interleaved_vs_blocked_models.py`: fit a variety
  of other state-space models not reported in the paper.

- Every other file in the code directory is a utility file
  that is used by the above scripts.
