# Ingredient Recognition

# Dataset

* [Recipe1M+](http://im2recipe.csail.mit.edu/)
    * [Ingredients + Image URLs](https://www.kaggle.com/datasets/kmader/layer-urls)
* [ISIA200](http://123.57.42.89/FoodComputing-Dataset/FoodComputing-ISIA200.html)

## Experiments

| Dataset | Neural Network | Accuracy | Accuracy Top 5 |
| :----: | :----: | :----: | :----: |
| FoodSeg103 | [Xception](https://drive.google.com/file/d/1Arh19kfTIld90I101tjuXvsqYmZ3IxDt/view?usp=share_link) | 65.58% | 87.77% |
| FoodSeg103 | [DenseNet121](https://drive.google.com/file/d/1bI8-vNbQR2YXOabKa50TLtXGlTKzVl5Q/view?usp=share_link) | 63.66% | 86.86% |
| FoodSeg103 | [DenseNet201](https://drive.google.com/file/d/1xRrqro_ER6QQRTNoXB8crgbXUpKwzx9w/view?usp=share_link) | 64.60% | 87.42% |

* SAM means the the test dataset is segmented by Meta Segment Anything

| Dataset | Neural Network | SAM F-Score | SAM Precision | SAM Recall |
| FoodSeg103 | [DenseNet201](https://drive.google.com/file/d/1xRrqro_ER6QQRTNoXB8crgbXUpKwzx9w/view?usp=share_link) | 36.38% | 49.03% | 32.83% |

## Categories need to remove or not exist in empower dataset

* CNY love letter
* Sweets
* bakso
* begedil
* chawanmushi
* pitaya
* tumpeng
* buckwheat
