## Ablation experiments on the PIE-RGB-SAR datase.

| Modality |   SFM    |   CFM    |    PWA    |   mIoU    |   Kappa   |    OA     |
| :------: | :------: | :------: | :-------: | :-------: | :-------: | :-------: |
|   RGB    |          |          |           |   76.90   |   84.13   |   88.84   |
|   SAR    |          |          |           |   66.73   |   74.16   |   83.82   |
| RGB+SAR  |          |          |  &#x2714; |   77.81   |   84.88   |   89.36   |
| RGB+SAR  |          | &#x2714; |           |   78.05   |   85.03   |   89.47   |
| RGB+SAR  | &#x2714; |          |  &#x2714; |   78.01   |   85.03   |   89.46   |
| RGB+SAR  | &#x2714; | &#x2714; |           | **78.31** | **85.27** | **89.64** |