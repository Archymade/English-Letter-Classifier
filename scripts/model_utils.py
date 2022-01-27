''' Model utilities. '''

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import QuantileTransformer

from sklearn.decomposition import PCA

from sklearn.svm import SVC
from sklearn.pipeline import Pipeline


def import_model(C_param = 1.0, probability = True,
                 n_jobs = -1, shrinking = False, random_state = 42):
    ''' Return model object. '''
    
    return SVC(probability = probability, C = C_param, random_state = random_state,
               shrinking = shrinking)


def create_pipeline(model, n_components = 7, dist_type = 'normal', whiten = True):
    ''' Returns an Estimator pipeline. '''
    
    qtrans = QuantileTransformer(output_distribution = dist_type)
    pca = PCA(n_components = n_components, whiten = whiten)
    
    estimator = Pipeline(steps = [('qtrans', qtrans),
                                  ('PCA', pca),
                                  ('model', model)
                                 ]
                        )
    
    return estimator


def train_model(model, X, y):
    ''' Fit model to dataset. '''
    
    model.fit(X, y)
    
    return model


