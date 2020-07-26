
import pandas as pd
from pandas.plotting import radviz
import numpy as np
import scipy

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px

import sys
import argparse
import io

import unittest

import warnings
warnings.filterwarnings("ignore")

def encode_svg(svg_file):
    encoded = base64.b64encode(open(svg_file,'rb').read())
    return 'data:image/svg+xml;base64,{}'.format(encoded.decode())

def loading_data():
    try:
    # Getting the data from the link
        df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', 
                    header=None, names = ['sepal length cm', 'sepal width cm', 
                                        'petal length cm','petal width cm', 'class'], 
                        index_col=False
                        )
        print('Loading online data')
    except:
        df = pd.read_csv('Data/iris_data.csv', 
                        index_col=False
                        ).drop('Unnamed: 0',axis=1)
        print('Loading local data')
    # Defining the columns as target and features
    target='class'
    features=['sepal length cm', 'sepal width cm', 'petal length cm','petal width cm']

    return df, target, features

class IrisQuality(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', 
                header=None, names = ['sepal length cm', 'sepal width cm', 
                                        'petal length cm','petal width cm', 'class'], 
                        index_col=False
                        )

        # Defining the columns as target and features
        cls.target='class'
        cls.features=['sepal length cm', 'sepal width cm', 'petal length cm','petal width cm']

    def test_data_completeness(self):
    # Check on the numbers of records and attributes
      self.assertEqual(self.df.shape[0], 150)
      self.assertEqual(self.df.shape[1], 5)

    def test_missing_data(self):
    # Check for missing/empty records from the data source
      self.assertEqual(self.df.isna().any(axis=None), False)

    def test_duplication(self):
    # Check for number of duplicates within in the data source
    # on prior investigation, it is found that there are 5 duplicate records in the data
      self.assertEqual(len(self.df.drop_duplicates(keep=False))+5, len(self.df))

    def test_positive(self):
    # Check that all recorded attributes (width and length) is positive.
        for i in self.features:
            self.assertEqual(all(i >= 0 for i in self.df[i]), True)

def data_quality(df):
    # capturedOutput = io.StringIO()                  # Create StringIO object
    # sys.stdout = capturedOutput                     #  and redirect stdout.
    # unittest.main(argv=['first-arg-is-ignored'], exit=False)                                     # Call function.
    # sys.stdout = sys.__stdout__                     # Reset redirect.

    return str(unittest.main(argv=['first-arg-is-ignored'], exit=False))


def selection_data(df, selection):
    # selection = 'All'
    # selection_x = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica', 'All']
    if selection != 'All':
        sub_df = df.loc[df['class']==selection]
    else:
        sub_df = df
    return sub_df

def prep_model(df, target):
    # Seperating the features and target
    Y = df.pop(target)
    X = df
    return Y,X

def sort_and_augment(X, Y, input, dist, neighbor_slider):
    # Identifying the those with lowest distance/diff
    Top_n_counts = neighbor_slider
    # Adding the calculated series into df
    df_final = X.copy()
    df_final['dist'] = dist
    df_final.sort_values(by=['dist'], inplace=True)
    df_final = df_final.reset_index(drop=True).reset_index().rename({'index':'similarity_ranked'}, axis=1)
    df_final['Class'] = Y.values
    df_final['Top_n_similar'] = ['similar' if i<=Top_n_counts else 'dissimilar' for i in df_final['similarity_ranked']]
    df_final['similar_alpha'] = [2 if i<=Top_n_counts else 1 for i in df_final['similarity_ranked']] # for colouring
    # # Adding test/input into the df_final
    tmp_test = [0]
    tmp_test = tmp_test+list(input[0])
    for i in [0,"input","input", 3]: #note the hard coded rank/columns
        tmp_test.append(i)
    dict_to_add = {j:i for i,j in zip(tmp_test,list(df_final.columns))}
    df_final = df_final.append(dict_to_add, ignore_index=True)
    return df_final

def dist_measure(X, Y, input, pipeline_selection, pca_n_dimension, neighbor_slider):
    # Distance Measure
    # pipeline_selection = 'Euclidean-Space'
    # pca_n_dimension = 2 #2
    # Dist_Pipeline_x = ['Euclidean-Space','PCA-Space']
    if pipeline_selection == 'Euclidean-Space':
        # euclidean space
        # dist
        dist = scipy.spatial.distance.cdist(X, input, 'euclidean')
        df_final = sort_and_augment(X, Y, input, dist, neighbor_slider)
    else:
        # pca space
        pca = PCA(n_components = pca_n_dimension)
        std_scaler = StandardScaler()
        X_std = std_scaler.fit_transform(X)
        pca.fit(X_std)
        pca_X = pca.transform(X_std)
        # pca test
        test = std_scaler.transform(input)
        pca_test = pca.transform(input)
        # dist
        dist = scipy.spatial.distance.cdist(pca_X, pca_test, 'euclidean')
        mod_X = pd.DataFrame(pca_X)
        if pca_n_dimension == 2:
            mod_X.columns =['PCA1','PCA2']
        elif  pca_n_dimension == 3:
            mod_X.columns =['PCA1','PCA2','PCA3']
        df_final = sort_and_augment(mod_X, Y, pca_test, dist, neighbor_slider)
    return df_final

def visualisation(df_final, visual_ctl, Y, X, scatter_x, features, test, Top_n_counts, export_ctl=False, dim_ctl=3):

    ### Raw
    if visual_ctl in ['all', 'raw']:
        if dim_ctl == 2:
            fig = px.scatter(df_final, x=scatter_x[0], y=scatter_x[1],
                        color='Top_n_similar', size_max=18,
                        symbol='Class', opacity=0.5)
        # 3d scatter plot
        elif dim_ctl == 3:
            fig = px.scatter_3d(df_final, x=scatter_x[0], y=scatter_x[1], z=scatter_x[2],
                        color='Top_n_similar', size_max=18,
                        symbol='Class', opacity=0.5)

        # Layout
        fig.update_layout(margin=dict(l=0, r=0, b=0, t=0),  
            title="Scatter Plot",)
        # Output
        if export_ctl=='True':
            print(f'Export Raw to Output/Raw-3D.png')
            fig.write_image("Output/Raw-3D.png")


    ### Radviz
    if visual_ctl in ['all', 'radviz']:
        radviz_fig, ax = plt.subplots( nrows=1, ncols=1 )
        # Turn off tick labels
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        # Plot
        ax = radviz(df_final[scatter_x], "Top_n_similar", color =['Red','Orange','Blue','Green'], alpha=0.5)
        ax.title.set_text('Radviz Plot of the Features')

        if export_ctl=='True':
            print(f'Export Radviz to Output/MultiDimension_Radviz.png')
            radviz_fig.savefig('Output/MultiDimension_Radviz.png', bbox_inches='tight')

    ### Parallel Coordinates
    if visual_ctl in['all', 'paral_coor']:
        parallel_x = scatter_x.copy()
        parallel_x.append('similar_alpha')
        # parallel plot
        par_fig = px.parallel_coordinates(df_final[parallel_x],
                                    color="similar_alpha", 
                                    dimensions=scatter_x,
                                    color_continuous_scale=px.colors.diverging.Tealrose, 
                                    color_continuous_midpoint=2)
        # Layout
        par_fig.update_layout(
            title={
                'text': "Parrallel Plot for Iris Neighbors",
                'y':0.1,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'}
        )
        # Output
        if export_ctl=='True':
            print(f'Export Parallel Coordinates to Output/MultiDimension_ParrCoor.png')
            par_fig.write_image("Output/MultiDimension_ParrCoor.png")

    return raw_fig, radviz_fig, par_fig


    