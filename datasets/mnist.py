from sklearn.datasets import fetch_mldata
import os

dirname = os.path.dirname(os.path.abspath(__file__))
mnist = fetch_mldata('MNIST original', data_home='%s/%s' % (dirname, 'mnist'))

mnist.data = mnist.data.reshape((-1, 28, 28))
