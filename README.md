# small-cnn
Some experiments to get small pretrained CNNs working on a not so big machine

## SqueezeNet, Alexnet with 50x less parameters

### Installation
I installed [pytorch](pytorch.org).  For my linux machine running python 2.7, using pip, and not requiring GPU support it was 2 lines of code. Get an ethernet cord, it was about 350mb.

```
pip install https://s3.amazonaws.com/pytorch/whl/cu75/torch-0.1.9.post2-cp27-none-linux_x86_64.whl 
pip install torchvision
```

### Usage

See explore.py

### Comments
I timed the loop of extracting screenshots and processing the image with squeezenet in PyTorch. Over 100 trials, this was the average time:

```
Average Time (us): 481556.81
```

So on average, it takes 0.5 seconds to process each frame of video, if you resize to 277 x 277 pixels
