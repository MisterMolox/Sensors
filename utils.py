import itertools
import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from IPython.display import clear_output

from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, make_scorer, roc_auc_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.manifold import MDS
from sklearn.manifold import TSNE

def show_sensors(num, x, y):
    print(num, y[num])
    plt.figure(figsize=(30, 20))
    for sensor in range(9):
        plt.subplot(911+sensor)
        plt.title(f'Sensor={sensor}')
        plt.plot(x[num, :, sensor])
        plt.grid()
    plt.show()

def print_metrics(y_true, y_prob, average_mode='macro'):
    y_pred = y_prob.argmax(axis=1)
    print(f"F1 score: {np.round(f1_score(y_true, y_pred, average=average_mode), 3)}")
    print(f"Presicion score: {np.round(precision_score(y_true, y_pred, average=average_mode), 3)}")
    print(f"Recall score: {np.round(recall_score(y_true, y_pred, average=average_mode), 3)}")
    print(f"ROC-AUC-OVO score: {np.round(roc_auc_score(y_true, y_prob, multi_class='ovo', average=average_mode), 3)}")
    print(f"ROC-AUC-OVR score: {np.round(roc_auc_score(y_true, y_prob, multi_class='ovr', average=average_mode), 3)}")

def plot_confusion_matrix(x, y, cm, classes, cmap=plt.cm.Greens):
    plt.figure(figsize=(x, y))
    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title('Confusion matrix', size=30)
    plt.colorbar()
    tick_marks = np.arange(6)- 0.5
    
    plt.xticks(tick_marks, classes, horizontalalignment="left")
    plt.yticks(tick_marks, classes, rotation=90)

    thresh = (cm.max()+cm.min()) / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center", 
                 verticalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylim(5.5, -0.5)
    plt.grid()
    plt.ylabel('True label', size=20)
    plt.xlabel('Predicted label', size=20)
    plt.tight_layout()

def show_separable(x, data):
    tsne_model    = TSNE(verbose      = 1,
                         random_state = 2021,
                         n_jobs       = 3)
    
    mds_model     = MDS(verbose      = 2,
                        random_state = 2021,
                        n_jobs       = 4)
    
    mds_cos_model = MDS(verbose       = 2,
                        random_state  = 2021,
                        n_jobs        = 4, 
                        dissimilarity = "precomputed")

    x_mds_train, _, y_mds_train, _ = train_test_split(x, data, 
                                                      train_size=2000, 
                                                      stratify=data, 
                                                      shuffle=True, 
                                                      random_state=2021)
    
    print('Fit TSNE')
    tsne_representation = tsne_model.fit_transform(x)
    clear_output()
    
    print('Fit MDS')
    MDS_transformed = mds_model.fit_transform(x_mds_train)
    clear_output()
    
    print('Fit cosine MDS')
    MDS_transformed_cos = mds_cos_model.fit_transform(pairwise_distances(x_mds_train, metric='cosine'))
    clear_output()
    

    plt.figure(figsize=(15, 12))
    plt.subplot(221)
    plt.title('TSNE')
    colors = cm.rainbow(np.linspace(0, 1, 6))
    for y, c in zip(set(data), colors):
        c = c.reshape(1,-1)
        plt.scatter(tsne_representation[data==y, 0], 
                    tsne_representation[data==y, 1], c=c, alpha=0.5, label=str(y))
    plt.legend()
    plt.grid()
    
    plt.subplot(223)
    plt.title('MDS')
    for y, c in zip(set(data), colors):
        c = c.reshape(1,-1)
        plt.scatter(MDS_transformed[y_mds_train==y, 0], 
                    MDS_transformed[y_mds_train==y, 1], 
                    c=c, alpha=0.5, label=str(y))
    plt.legend()
    plt.grid()
    
    plt.subplot(224)
    plt.title('MDS cosine')
    for y, c in zip(set(data), colors):
        c = c.reshape(1,-1)
        plt.scatter(MDS_transformed_cos[y_mds_train==y, 0], 
                    MDS_transformed_cos[y_mds_train==y, 1], 
                    c=c, alpha=0.5, label=str(y))
    plt.legend()
    plt.grid()
    
    plt.show()