# Temporal Feature Mix
Official code for "Enhancing Robustness of Multi-Object Trackers with Temporal Feature Mix", TCSVT, 2024
  - https://ieeexplore.ieee.org/document/10535304

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

## Usage (Example codes)
### 1. Set temporal feature mix in the model
```python
# Define temporal feature mix
class TFM(nn.Module):
    @staticmethod
    def forward(x):
        return x

# Apply temporal feature mix in every layer of the model
# For example, add temporal feature mix in the basic convolution module of the model
class BaseConv(nn.Module):
    def __init__(self, in_channels, out_channels, ksize, stride, groups=1, bias=False, act="silu", tfm=False):
        super().__init__()
        # Define a convolution layer
        pad = (ksize - 1) // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=ksize, stride=stride,
                              padding=pad, groups=groups, bias=bias,)

        # Define others
        self.tfm = TFM() if tfm else None
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = get_activation(act, inplace=True)

    def forward(self, x):
        if self.tfm is not None:
            return self.act(self.bn(self.tfm(self.conv(x))))
        else:
            self.act(self.bn(self.conv(x)))
```

### 2. Train the model with temporal feature mix
```python
# Define model and loss
model = Model()
model = FeatureMix(model, batch_size // device_num,  tfm_p, tfm_r_max)
loss = Loss()

for idx in range(data_len):
    # Get two randomly adjacent frames and labels for frame_1
    frame_1, frame_2, labels = sample_adjacent_frames_labels(idx)
    
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

# Remove temporal feautre mix after finishing all training
model.remove_hooks()
model = model.model
```
