print(__doc__)
from sklearn import datasets,svm,metrics
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


digits = datasets.load_digits()
print(digits.data.shape)


_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, label in zip(axes, digits.images, digits.target):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    ax.set_title('Training: %i' % label)
    plt.show()