""" Train model from command line. """

import argparse
import os
import time

import joblib
import random

import numpy as np
from copy import deepcopy
import seaborn as sns

from data_ops import load_dataset
from data_ops import manual_describe

from metrics import get_precision_score, get_accuracy_score
from metrics import get_recall_score, get_f1_score

from model_utils import import_model
from model_utils import create_pipeline
from model_utils import train_model

from viz_utils import get_correlation_map
from viz_utils import correlogram
from viz_utils import univariate_plot

from viz_utils import class_distribution
from viz_utils import visualize_confusion_matrix


def configure_args():
    """ Configure cli arguments. """

    args = argparse.ArgumentParser(description='Arguments for training English letter classifier.')
    
    args.add_argument('--n_jobs', default=-1, type=int, help='Number of threads')

    args.add_argument('--C', default = 1.5, type = int, help = 'Regularization Factor')
    
    args.add_argument('--r_state', default=42, type=int, help='Random state')

    args.add_argument('--data_dir', type=str, default = os.path.join(os.getcwd().replace('scripts', 'data'),
                                                                     'dataset'),
                      help='Data directory')

    args.add_argument('--train', default=True, type=bool, choices=[True, False], help='Show train scores')
    args.add_argument('--valid', default=True, type=bool, choices=[True, False], help='Show valid scores')
    args.add_argument('--test', default=True, type=bool, choices=[True, False], help='Show test scores')

    args.add_argument('--visualize', default=False, type=bool, choices=[True, False],
                      help='Visualise data?')
    
    args.add_argument('--whiten', default=True, type=bool, help='Apply PCA whitening?')
    
    args.add_argument('--n_components', default=14, type=int, help='Number of principal components')
    
    args.add_argument('--dist_type', default='normal', type=str, choices=['normal', 'uniform'],
                      help='Output distribution for quantile transformer')
    
    args.add_argument('--probs', default=True, type=bool, help='Allow for probability estimation via SVC?')

    args.add_argument('--matrix', default=True, type=bool, choices=[True, False],
                      help= 'Show confusion matrix?')

    args.add_argument('--acc', default=True, type=bool, choices=[True, False], help='Show accuracy score')
    args.add_argument('--rec', default=True, type=bool, choices=[True, False], help='Show recall score')
    args.add_argument('--pre', default=True, type=bool, choices=[True, False], help='Show precision score')
    args.add_argument('--f1', default=True, type=bool, choices=[True, False], help='Show f1 score')

    args.add_argument('--avg', default='macro', choices=['micro', 'macro', 'samples', 'weighted', 'binary'],
                      help='Metric aggregation')

    args.add_argument('--text', default=True, choices=[True, False], help='Display text with diagnostics')

    args.add_argument('--dp', default=7, type=int, help='Rounding precision for report metrics')

    args.add_argument('--save', default=True, type=bool, choices=[True, False], help='Save trained model')

    args.add_argument('--model_name', default='trained_letter_classifier.bin', type=str,
                      help='Name for trained model')

    args.add_argument('--model_dir', default=os.getcwd().replace('scripts', 'artefacts'),
                      type=str, help='Storage location for saved model')
    
    args.add_argument('--report_dir', default=os.getcwd().replace('scripts', 'reports'),
                      type=str, help='Storage location for generated reports and visuals')

    return args


def main():
    ### CLI arguments

    start_time = time.time()
    origin_time = deepcopy(start_time)

    print('>>> Parsing CLI arguments...')
    start_time = time.time()
    args = configure_args().parse_args()
    print(f'>>> CLI arguments parsed! Time elapsed : {time.time() - start_time:.5f} secs.')
    print()
    
    try:
        from jupyterthemes import jtplot
    except:
        print('Jupyterthemes installation missing.')
        sns.set()
        print()
    else:
        jtplot.style('gruvboxd')


    ### Dataset
    print('>>> Importing dataset...')
    start_time = time.time()

    X_train, y_train = load_dataset(path = os.path.join(args.data_dir, 'train', 'letter.scale.tr'))
    
    if args.valid:
        X_valid, y_valid = load_dataset(path = os.path.join(args.data_dir, 'valid', 'letter.scale.val'))
        
    if args.test:
        X_test, y_test = load_dataset(path = os.path.join(args.data_dir, 'test', 'letter.scale.t'))

    print(f'>>> Dataset successfully imported! Time elapsed : {time.time() - start_time:.5f} secs.',
          f'\n\t> Number of data observations (Train): [{X_train.shape[0]}]',
          f'\n\t> Feature dimensions (Train): [{X_train.shape[1]}]')
    print()
    
    if args.valid:
        print(f'\n\t> Number of data observations (Valid): [{X_valid.shape[0]}]',
              f'\n\t> Feature dimensions (Valid): [{X_valid.shape[1]}]')
    print()
    
    if args.test:
        print(f'\n\t> Number of data observations (Test): [{X_test.shape[0]}]',
              f'\n\t> Feature dimensions (Test): [{X_test.shape[1]}]')
    print()

    ### Reproducibility
    print('>>> Ensuring reproducibility...')
    print('\t> Setting global and local random seeds...')

    start_time = time.time()

    random.seed(args.r_state)
    os.environ['PYTHONHASHSEED'] = str(args.r_state)
    np.random.default_rng(args.r_state)

    print('\t> Random seeds set!')
    print(f'>>> Reproducibility ensured! Time elapsed : {time.time() - start_time:.5f} secs.')
    print()
    
    if not os.path.exists(args.report_dir):
        os.makedirs(os.path.join(args.report_dir, 'images', '1'))
        os.makedirs(os.path.join(args.report_dir, 'images', '2'))
        os.makedirs(os.path.join(args.report_dir, 'images', '3'))

        os.makedirs(os.path.join(args.report_dir, 'text'))
    else:
        pass
    
    if args.visualize:
        print(f'>>> Visualizing dataset...')
        univariate_plot(X_train, path = os.path.join(args.report_dir, 'images', '1', 'dist_plot'),
                        save = args.save)

        correlogram(X_train, path = os.path.join(args.report_dir, 'images', '2', 'pair_plot'),
                    save = args.save)
        
        get_correlation_map(X_train, path = os.path.join(args.report_dir, 'images', '2', 'Feature correlation'),
                            save = args.save)
        
        class_distribution(y_train, path = os.path.join(args.report_dir, 'images', '2', 'Target distribution'))
    
    print()
    
    print('>>> Describing dataset...')
    print(manual_describe(X_train, path = os.path.join(args.report_dir, 'text'), save = args.save))
    print()


    ### Model fitting
    print('>>> Importing model; Training model...')
    start_time = time.time()
    
    model = import_model(C_param=args.C, probability = args.probs,
                         random_state = args.r_state)
    
    model = create_pipeline(model = model, n_components = args.n_components,
                            whiten = args.whiten, dist_type = args.dist_type)
    
    model = train_model(model, X_train, y_train)
    
    print(f'>>> Model trained successfully! Time elapsed : {time.time() - start_time:.5f} secs.')
    print()

    if args.save:
        print('>>> Saving artefacts...')
        start_time = time.time()

        if not os.path.exists(args.model_dir):
            os.mkdir(args.model_dir)
        else:
            pass

        print('\t> Saving model artefacts...')

        with open(os.path.join(args.model_dir, args.model_name), 'wb') as f:
            joblib.dump(model, f)

        print('\t> Model artefact saved!')
        print()

        print(f'>>> Artefact redundancy achieved! Time elapsed : {time.time() - start_time:.5f} secs.')
        print()

    if args.matrix:
        if args.train:
            visualize_confusion_matrix(model, X_train, y_train, split = 'train',
                                       path = os.path.join(args.report_dir, 'images', '3'))

        if args.valid:
            visualize_confusion_matrix(model, X_valid, y_valid, split = 'valid',
                                       path = os.path.join(args.report_dir, 'images', '3'))

        if args.test:
            visualize_confusion_matrix(model, X_test, y_test, split = 'test',
                                       path = os.path.join(args.report_dir, 'images', '3'))

    
    if args.train:
        train_preds = model.predict(X_train)

        print('>' * 10, 'Train Diagnostics', '<' * 10)

        print(get_accuracy_score(y_train, train_preds, num_places=args.dp, text=args.text))
        print(get_precision_score(y_train, train_preds, num_places=args.dp, text=args.text))
        print(get_recall_score(y_train, train_preds, num_places=args.dp, text=args.text))
        print(get_f1_score(y_train, train_preds, num_places=args.dp, text=args.text))
        print()

    if args.valid:
        valid_preds = model.predict(X_valid)

        print('>' * 10, 'Valid Diagnostics', '<' * 10)

        print(get_accuracy_score(y_valid, valid_preds, num_places=args.dp, text=args.text))
        print(get_precision_score(y_valid, valid_preds, num_places=args.dp, text=args.text))
        print(get_recall_score(y_valid, valid_preds, num_places=args.dp, text=args.text))
        print(get_f1_score(y_valid, valid_preds, num_places=args.dp, text=args.text))
        print()

    if args.test:
        test_preds = model.predict(X_test)

        print('>' * 10, 'Test Diagnostics', '<' * 10)

        print(get_accuracy_score(y_test, test_preds, num_places=args.dp, text=args.text))
        print(get_precision_score(y_test, test_preds, num_places=args.dp, text=args.text))
        print(get_recall_score(y_test, test_preds, num_places=args.dp, text=args.text))
        print(get_f1_score(y_test, test_preds, num_places=args.dp, text=args.text))
        print()

    print(f'>>> Program run successfully! Total Time elapsed : {time.time() - origin_time :.5f} secs.')




if __name__ == '__main__':
    main()




