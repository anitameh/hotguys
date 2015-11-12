from flask import Flask
import numpy as np
from sklearn.cluster import AgglomerativeClustering


def cluster(data, number_of_clusters):
    """
    Cluster hot guys by attributes.
    :param numpy.matrix matrix:
    :param int number_of_clusters: number of clusters to "find"

    :rtype: list of lists
    :return: clusters of hot guys' names
    """
    # matrix = np.array([[1.,  2.,  0.,  2.,  5.,  2.,  0.,  5.],
    #                    [3.,  0.,  0.,  2.,  4.,  2.,  2.,  1.],
    #                    [0.,  3.,  1.,  3.,  0.,  0.,  1.,  1.],
    #                    [2.,  4.,  0.,  0.,  0.,  0.,  1.,  4.],
    #                    [0.,  4.,  0.,  0.,  0.,  1.,  1.,  5.],
    #                    [2.,  1.,  1.,  2.,  3.,  0.,  2.,  2.],
    #                    [2.,  1.,  0.,  2.,  1.,  1.,  2.,  5.],
    #                    [0.,  1.,  0.,  1.,  1.,  2.,  0.,  3.],
    #                    [1.,  2.,  0.,  2.,  1.,  2.,  1.,  4.],
    #                    [0.,  1.,  0.,  1.,  2.,  0.,  3.,  0.],
    #                    [0.,  1.,  0.,  2.,  3.,  2.,  2.,  2.],
    #                    [0.,  0.,  0.,  2.,  1.,  0.,  2.,  5.]])
    matrix = []
    for each_row in data:
        matrix.append(list(each_row))
    clustering_model = AgglomerativeClustering(n_clusters=number_of_clusters).fit(matrix)
    labels = clustering_model.labels_
    return str(labels)


app = Flask(__name__)


@app.route('/')
def index():
    data = np.loadtxt('database.csv', dtype={'names': ('face_shape', 'skin_tone', 'hair_length', 'hair_type',
                                                       'hair_color', 'lips', 'eye_color', 'nose_shape'),
                                             'formats': ('f4', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4')},
                      delimiter=',', skiprows=1, usecols=(1, 2, 3, 4, 5, 6, 7, 8))
    return cluster(data, 3)


if __name__ == '__main__':
    app.run(debug=True)
