# DeepCO3: Deep Instance Co-segmentation by Co-peak Search and Co-saliency

This is a python re-implementation of a paper published in CVPR 2019, which can be find [here](https://github.com/KuangJuiHsu/DeepCO3). My whole algorithm is based on this and the variable name is the same as him, And all parameters and implementation details are as similar to the original as possible.

Environment you need is: Python 3, cv2, PyTorch and so on.

First, you need to download the Datasets.

Then, get VGG16 pretrained weights from PyTorch official version [here](https://pytorch.org/docs/stable/_modules/torchvision/models/vgg.html#vgg16), and we remove the last few layers. Click [here](https://drive.google.com/open?id=1Ar3pF4bzNWX-CSXaWcQqSoTRS_46KSLl) to download in Google Drive.

Modify and run the RunDeepInstCoseg.py.

This is the comparation between DeepCO3(author) and DeepCO3-python(this) in four datasets from original cvpr19 paper.

![](https://github.com/cj4L/DeepCO3-python/raw/master/pic/comp.png)


By the way, if you use the code, please cite the following reference:
```
@inproceedings{HsuCVPR19,
  author = {Kuang-Jui Hsu and Yen-Yu Lin and Yung-Yu Chuang},
  booktitle = {IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPR)},
  title = {DeepCO$^3$: Deep Instance Co-segmentation by Co-peak Search and Co-saliency Detection},
  year = {2019}
}
```
