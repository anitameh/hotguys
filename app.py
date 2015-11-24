from flask import Flask
import numpy as np
from sklearn.cluster import AgglomerativeClustering


def read_data_from_database(filename_database):
    # def read_data_from_database(filename_database, filename_user_inputs):
    """
    Returns matrix with both reserved and user-selected hot guys
    :param String filename_database:
    :param String filename_user_inputs:

    :rtype: triple
    :return: matrix of values, list of urls for images, row indices of matrix corresponding to user_inputs
    """
    database = np.loadtxt(filename_database, dtype={'names': ('face_shape', 'skin_tone', 'hair_length', 'hair_type',
                                                              'hair_color', 'lips', 'eye_color', 'nose_shape', 'url'),
                                                    'formats': ('f4', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4',
                                                                'S1000')},
                          delimiter=',', skiprows=1, usecols=(1, 2, 3, 4, 5, 6, 7, 8, 9))

    # user_inputs = np.loadtxt(filename_user_inputs, dtype={'names': ('face_shape', 'skin_tone', 'hair_length',
    #                                                                 'hair_type', 'hair_color', 'lips', 'eye_color',
    #                                                                 'nose_shape', 'url'),
    #                                                       'formats': ('f4', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4',
    #                                                                   'S1000')},
    #                          delimiter=',', skiprows=1, usecols=(1, 2, 3, 4, 5, 6, 7, 8, 9))

    matrix, urls = [], []
    for each_row in database:
        matrix.append(list(each_row)[:-1])
        urls.append(each_row[-1])

    # for each_row in user_inputs:
    #     matrix.append(list(each_row)[:-1])
    #     urls.append(each_row[-1])

    # indices_of_user_selected = range(len(database) + len(user_inputs))[-len(user_inputs):]

    # return matrix, urls, indices_of_user_selected
    return matrix

def cluster(matrix, number_of_clusters):
    """
    Cluster hot guys by attributes.
    :param numpy.matrix matrix:
    :param int number_of_clusters: number of clusters to "find"

    :rtype: list of lists
    :return: clusters of hot guys' names
    """
    clustering_model = AgglomerativeClustering(n_clusters=number_of_clusters).fit(matrix)
    labels = clustering_model.labels_
    return labels


def recommend(urls, indices_of_user_selected):
    """
    :param list urls:
    :param list indices_of_user_selected:

    :rtype: String
    :return: url pointing to image of recommended hot guy
    """
    # TODO
    # This is where the heuristics for determining the most similar guy goes


app = Flask(__name__)


@app.route('/')
def index():
    # matrix, urls, indices_of_user_selected = read_data_from_database('database.csv', 'user_inputs.csv')
    matrix = read_data_from_database('database.csv')
    # clusters = str(cluster(matrix, 3))
    clusters = 'https://s-media-cache-ak0.pinimg.com/236x/e3/72/84/e372847cd39daa517b3ebead7228f94c.jpg'
    return clusters

if __name__ == '__main__':
    app.run(debug=True)
