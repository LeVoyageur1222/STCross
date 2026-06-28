# Experiment Summary

> Generated: 2026-06-28T19:15:14.006609

| Model        | Setting      | Subset   |  Count |      MAE |     RMSE |    MAPE% |      PCC |       R² |
|--------------|--------------|----------|--------|----------|----------|----------|----------|----------|
| TimesNet     | CHI->CHI     | full     |   5761 |   0.8656 |   2.4884 |   72.58% |   0.5342 |   0.2715 |
| TimesNet     | CHI->CHI     | anomaly  |    398 |   0.6900 |   2.0075 |   73.95% |   0.5532 |   0.3029 |
| TimesNet     | CHI->CHI     | normal   |   5363 |   0.8786 |   2.5204 |   72.50% |   0.5333 |   0.2695 |
| TimesNet     | NYC->CHI     | full     |   5761 |   0.8916 |   2.4959 |   74.02% |   0.5248 |   0.2671 |
| TimesNet     | NYC->CHI     | anomaly  |    398 |   0.7317 |   2.0593 |   75.52% |   0.5380 |   0.2664 |
| TimesNet     | NYC->CHI     | normal   |   5363 |   0.9034 |   2.5253 |   73.94% |   0.5243 |   0.2667 |
| TimesNet     | NYC->NYC     | full     |   5761 |   1.8764 |   4.2241 |   78.72% |   0.6102 |   0.3613 |
| TimesNet     | NYC->NYC     | anomaly  |    507 |   1.5945 |   3.4724 |   81.52% |   0.5556 |   0.2923 |
| TimesNet     | NYC->NYC     | normal   |   5254 |   1.9036 |   4.2897 |   78.47% |   0.6129 |   0.3644 |
| TimesNet     | CHI->NYC     | full     |   5761 |   2.0244 |   4.5329 |   76.94% |   0.5392 |   0.2645 |
| TimesNet     | CHI->NYC     | anomaly  |    507 |   1.6462 |   3.6060 |   77.02% |   0.5060 |   0.2368 |
| TimesNet     | CHI->NYC     | normal   |   5254 |   2.0609 |   4.6125 |   76.93% |   0.5405 |   0.2651 |
