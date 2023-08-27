import tensorflow as tf
import new_evaluation as eva
md = 'vgg'
d = 18
w = 4.0
x_train, y_train, x_test, y_test = eva.load_data()
epochs = 1
