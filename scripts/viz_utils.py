import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import plot_confusion_matrix
    
    
# Univariate visualization
def univariate_plot(data, path, save = True):
    
    ''' Plot the data univariately. '''
    
    for col in data.columns:
        plt.figure(figsize = (10, 8))
        sns.displot(data[col])

        plt.title(f'Distribution plot for Feature {col}')
        
        if save:
            plt.savefig(f'{path} - Feature {col}.png', dpi = 300)
        
        plt.show()
        plt.close('all')
        
    return None


def correlogram(data, path, palette = 'inferno', h = 10, w = 10, save = True):
    ''' Plot and save correlogram. '''
    
    plt.figure(figsize = (h, w))
    sns.pairplot(data = data, palette = palette)
    
    plt.title('Bivariate visual relationships in data')
    
    if save:
        plt.savefig(f'{path}.png', dpi = 300)
    
    plt.show()
    plt.close('all')
    
    return None

          
def get_correlation_map(data, path, save = True, h = 20, w = 10):
    ''' Visualize feature correlation. '''
    
    plt.figure(figsize = (h, w))
    sns.heatmap(data.corr(), annot = True, fmt = '.3g')
    plt.title('Feature collinearity heatmap')
    
    if save:
        plt.savefig(f'{path}.png', dpi = 300)
    
    plt.show(); plt.close('all')
          
    return None

          
def visualize_confusion_matrix(model, X, y, split, path, save = True):
    """ Display Confusion Matrix visually."""

    fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (20, 20))
    plt.title(f'Confusion matrix for {split.upper()}', fontsize = 30, pad = 30)
    plot_confusion_matrix(model, X, y, ax = ax)
    
    if save:
        plt.savefig(os.path.join(path, f'{split}-confusion-matrix.png'), dpi = 300)
    
    plt.show()
    plt.close('all')

    return None


def class_distribution(data, path, save = True, h = 10, w = 10):
    ''' Visualize class distribution. '''
    
    plt.figure(figsize = (w, h))
    sns.countplot(x = data)
    
    plt.title('Class Distribution', pad = 20, fontsize = 20)
    
    plt.xlabel('Class Label', fontsize = 20)
    plt.ylabel('Class Population', fontsize = 20)
    
    if save:
        plt.savefig(f'{path}.png', dpi = 300)
    
    plt.show(); plt.close('all')
    
    return None



