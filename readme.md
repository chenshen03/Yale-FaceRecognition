# Face recognition on Yale

> Version 1.0, Last updated :2018.10.20

We solved the face recognition task on Yale database by using HOG+Softmax model. The model achieves the best performance in both test set and noise set.

## Dataset

- [The Yale Face Database](https://pan.baidu.com/s/1iumeE?qq-pf-to=pcqq.c2c)

- Yale database has a total of 165 human face images. The database contains 15 people, each with 11 pictures of faces.

## Requirements

- python: 3.6.5
- scikit-learn: 0.19.2
- scikit-image: 0.14.0
- opencv: 3.4.2
- pyqt: 5.9.2

## Usage

Model training:

```python
python exam.py
```

Demo presentation:

```python
python demo.py
```

## Results

The acc_class and err_in of model on the test set and noise set:

|           | acc_class(%) | err_in(%) |
| :-------: | :----------: | :-------: |
| test set  |     100      |     0     |
| noise set |     100      |     0     |

Prediction of data sets on demo:

![demo1](https://ws4.sinaimg.cn/large/006tNbRwgy1fwemb4o8a5j30m80cutcd.jpg)

Prediction of noise sets on demo:

![demo2](https://ws2.sinaimg.cn/large/006tNbRwgy1fwembcawmnj30m80cuq8u.jpg)

## References

- **ResNet:** Kaiming He, Xiangyu Zhang, Shaoqing Ren, et al. Deep Residual Learning for Image Recognition[J]. 2015:770-778.
- **PCA:** Wold S, Esbensen K, Geladi P. Principal component analysis[J]. Chemometrics & Intelligent Laboratory Systems, 1987, 2(1):37-52.
- **SIFT:** Lowe D G. Distinctive Image Features from Scale-Invariant Keypoints[C]// International Journal of Computer Vision. 2004:91-110.
- **HOG:** Dalal, Navneet, Triggs, et al. Histograms of Oriented Gradients for Human Detection[C]// IEEE Computer Society Conference on Computer Vision and Pattern Recognition. IEEE, 2005:886-893.
- **SVM:** Adankon M M, Cheriet M. Support Vector Machine[J]. Computer Science, 2002, 1(4):1-28.