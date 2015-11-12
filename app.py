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
