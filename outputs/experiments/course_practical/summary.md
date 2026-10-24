# Experiment Campaign: course_practical

Validation metrics decide the ablation winner. Test metrics are present only for final runs.

Selected configuration: `E4_sgd_momentum` by `highest val_macro_f1, tie lowest best_val_loss`.

| experiment | phase | task | model | params | optimizer | aug | val_f1 | test_f1 |
| --- | --- | --- | --- | ---: | --- | --- | ---: | ---: |
| E0_overfit_100 | sanity | multiclass | scratch_cnn | 390986 | adam | none | 0.7395 | - |
| E1_baseline_no_aug | ablation | multiclass | scratch_cnn | 390986 | adam | none | 0.9419 | - |
| E2_augmentation | ablation | multiclass | scratch_cnn | 390986 | adam | hflip+vflip+rot15 | 0.9456 | - |
| E3_regularized | ablation | multiclass | scratch_cnn | 390986 | adam | hflip+vflip+rot15 | 0.9489 | - |
| E4_sgd_momentum | ablation | multiclass | scratch_cnn | 390986 | sgd | hflip+vflip+rot15 | 0.9557 | - |
| E5_depthwise_separable | ablation | multiclass | depthwise_separable_cnn | 48424 | sgd | hflip+vflip+rot15 | 0.9104 | - |
| Final_multiclass | final | multiclass | scratch_cnn | 390986 | sgd | hflip+vflip+rot15 | 0.9557 | 0.9519 |
| Final_binary | final | binary | scratch_cnn | 388930 | sgd | hflip+vflip+rot15 | 0.9795 | 0.9795 |

## Interpretation Guide

- Sanity rows check whether the model and training loop can learn a tiny subset.
- Ablation rows are validation-only and decide model selection.
- Final rows use the selected validation configuration and include test metrics.
