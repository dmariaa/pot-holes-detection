# Cross Validation

## Current model

- Cross validate differente sensor/session configurations
    - ** S1+S3 / 10/02/2026 (TRAIN) - S1 (11/02/2026) (validation) ** 
    - ** S1+S3 / 10/02/2026 (TRAIN) - S3 (11/02/2026) (validation) **
    - ** S1 / 10/02/2026 (TRAIN) - S3 (11/02/2026) (validation) **
    - S3 / 10/02/2026 (TRAIN) - S1 (11/02/2026) (validation) 

- ** Run individual channel experiments to try to determine the best channels for the task **

- FINETUNE THE TRANSFORMER ENCODER with new data
- Evolve the CNN encoding with CONTRASTIVE training

## Next things to do

- TRANSFORMER encoder
    - Pretrained (frozen)
    - Pretrained + fine tuning
    - Trained from scratch


- ALTERNATIVE encoder
  - CONTRASTIVE learning based