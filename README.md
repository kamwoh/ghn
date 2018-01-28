## Implementation of Generalized Hamming Distance Network

This is an re-implementation of the Generalized Hamming Distance Network published in NIPS 2017. 

### Requirements to run
1. python 2.7
2. keras (for dataset)
3. tensorflow

### How to run
```
python mnist_main.py
```

### Conv2D GHD
```
L = reduce_prod(weights.shape[:3])
hout = 2/L * conv2d(x, w) - mean(weights) - mean(avgpool2d(x, w))

for more informations, refer to nets/tf_interface.py
```

### FC GHD
```
L = weights.shape[0]
hout = 2/L * matmul(x, w) - mean(weights) - mean(x)

for more informations, refer to nets/tf_interface.py
```

### Mnist image classification with GHD
```
Layers=[
    GHD_Conv2D [kernel_size=5],
    MaxPool2D,
    GHD_Conv2D [kernel_size=5],
    MaxPool2D,
    Flatten,
    GHD_FC,
    GHD_FC
]

loss=CrossEntropy

optimizer=Adam
```

### Experiment Results (Mnist dataset)
At the end of first epoch with `learning rate = 0.1, r = 0`, validation and testing accuracy reaches 97%

As stated in the paper, at `log(48000) = 4.68`, accuracy is around 97~98%

### Reference
[1] Fan, L. (2017). Revisit Fuzzy Neural Network: Demystifying Batch Normalization and ReLU with Generalized Hamming Network. Nokia Technologies Tampere, Finland.

### Feedback
Suggestions and opinions of this implementation are greatly welcome. Please contact the us by sending email to Kam Woh Ng at `kamwoh at gmail.com` or Chee Seng Chan at `cs.chan at um.edu.my`
