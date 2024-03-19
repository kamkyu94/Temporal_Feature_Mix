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
  - feature_mix.py
  - generate_corruption_dataset.py

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
```

## Usage
```
# To generating corruption dataset
python generate_corruption_dataset.py
```

```
# To apply temporal feature mix (Example code)

# Define model and loss
model = Model()
model = FeatureMix(model, batch_size // device_num,  tfm_p, tfm_r_max)
loss = Loss()

# Get two randomly adjacent frames and labels for frame_1
frame_1, frame_2, labels = sample_adjacent_frames_labels()

# Save features to be mixed
model.eval()
with torch.no_grad():
    model.start_feature_record()
    _ = model(frame_2, targets)
    model.end_feature_record()
model.train()

# Inference and mixing the features
model.start_feature_mix()
outputs = model(frame_1)
model.end_feature_mix()

# Get loss and back-propagation
total_loss = loss(outputs, labels)
optimizer.zero_grad()
total_loss.backward()
optimizer.step()

```
