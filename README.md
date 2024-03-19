# Temporal Feature Mix
Official code for "Enhancing Robustness of Multi-Object Trackers with Temporal Feature Mix"

## Environment
Developed in python3.8, pytorch 1.11

## Prepare Codes and Datasets
Link for MOT17: https://motchallenge.net/data/MOT17.zip

Link for MOT20: https://motchallenge.net/data/MOT20.zip

```
- code
  - frost_images
  - generate_corruption_dataset.py
  - temporal_feature_mix.py

- dataset
  - MOT-C
    - train
      - MOT17-02-FRCNN
      - MOT17-10-FRCNN
      - MOT17-11-FRCNN
      - MOT17-13-FRCNN
      - MOT20-02
      - MOT20-05
    - test
      - MOT17-04-FRCNN
      - MOT17-05-FRCNN
      - MOT17-09-FRCNN
      - MOT20-01
      - MOT20-03


## Usage
