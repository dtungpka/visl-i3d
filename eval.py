
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import sklearn.metrics as metrics
import os
import sys
import time
import datetime
import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
class Eval:
    def __init__(self,run_name):
        #create a folder "dd-hh-mm-ss_[run_name]" in the current directory
        save_folder = datetime.datetime.now().strftime("%d-%H-%M-%S") + "_" + run_name
        os.makedirs(save_folder, exist_ok=True)
        self.save_folder = save_folder
        os.makedirs(os.path.join(save_folder, "raw"), exist_ok=True)
        os.makedirs(os.path.join(save_folder, "confusion_matrix"), exist_ok=True)
        #create a csv file in the folder to save loss
        self.loss_file = os.path.join(save_folder, "loss.csv")
        self.loss_file_handle = open(self.loss_file, "w")
        self.add_loss_entry("epoch,loss,loc_loss,cls_loss,total_loss")
    def set_labels(self,labels):
        self.labels = labels
    def add_loss_entry(self,entry):
        self.loss_file_handle.write(f"{entry}\n")
        #flush the buffer to write to the file
        self.loss_file_handle.flush()
    def evaluate(self,phase,ep,result,class_labels):
        #result is [[logit, classif,predicted, labels]]
        #save [phase]_[ep]_raw.pkl in raw folder
        raw_file = os.path.join(self.save_folder, "raw", f"{phase}_{ep}_raw.pkl")
        with open(raw_file, "wb") as f:
            pickle.dump(result, f)
        #if file "[phase]_result.csv" does not exist, create it and write the header
        result_file = os.path.join(self.save_folder, f"{phase}_result.csv")
        #combine [Precision,Recall,F1 Score] for each class to make a list of [Precision_classname1,Recall_classname1,F1_classname1,Precision_classname2,Recall_classname2,F1_classname2,...]
        met_str = ["Precision","Recall","F1 Score"]
        comb_str = [f"{x}_{y}" for y in class_labels for x in met_str]
        if not os.path.exists(result_file):
            with open(result_file, "w") as f:
                f.write(f"Accuracy,{','.join(comb_str)}\n")
                f.write(f"{self.calculate_metrics(result)}\n")
        else:
            with open(result_file, "a") as f:
                f.write(f"{self.calculate_metrics(result)}\n")
        #save the confusion matrix as a png file in the confusion_matrix folder as name [phase]_[ep]_confusion_matrix.png
        self.save_confusion_matrix(phase,ep,result,class_labels)
    def get_path(self):
        return self.save_folder
    def save_confusion_matrix(self, phase,ep, result, class_labels):
        predicted = [x[2] for x in result]
        labels = [x[3] for x in result]
        
        cm = np.zeros((len(class_labels), len(class_labels)), dtype=np.float32)
        for i in range(len(predicted)):
            cm[labels[i], predicted[i]] += 1
        #cm /= np.sum(cm, axis=1, keepdims=True)
        #cm = np.round(cm, 2)
        ds_type = f"{phase} set"
        print('Confusion matrix of ' + ds_type)
        print(f'Classes: {labels}')
        print(cm)
        ax = sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', cbar=False)
        sns.set(rc={'figure.figsize': (16, 16)})  # Increase figure size
        sns.set(font_scale=1.2)  # Decrease font scale
        ax.set_title('Confusion matrix of ' + ds_type)
        ax.set_xlabel('Predicted Action')
        ax.set_ylabel('Actual Action')
        plt.xticks(rotation=90, fontsize=6)  # Decrease font size and rotate x-axis labels
        plt.yticks(rotation=0, fontsize=6)  # Decrease font size and rotate y-axis labels
        ax.xaxis.set_ticklabels(class_labels)
        ax.yaxis.set_ticklabels(class_labels)
        #save the confusion matrix as a png file in the confusion_matrix folder as name [phase]_[ep]_confusion_matrix.png
        cm_file = os.path.join(self.save_folder, "confusion_matrix", f"{phase}_{ep}_confusion_matrix.svg")
        plt.savefig(cm_file, dpi=300, format='svg',bbox_inches='tight')  # Increase dpi for higher resolution


    def calculate_metrics(self,result):
        return_str = ""
        #calculate all the metrics, save as dec 4
        predicted = [x[2] for x in result]
        labels = [x[3] for x in result]
        # Check if the number in predicted and labels is consistent
        

        


        acc = metrics.accuracy_score(labels,predicted)
        return_str += f"{acc:.4f},"

        #mean average precision
        #mAP = metrics.label_ranking_average_precision_score(labels,predicted)
        #return_str += f"{mAP:.4f},"
        #precision, recall, f1 score for each class
        prec = metrics.precision_score(labels,predicted,average=None)
        rec = metrics.recall_score(labels,predicted,average=None)
        f1 = metrics.f1_score(labels,predicted,average=None)
        for i in range(len(prec)):
            return_str += f"{prec[i]:.4f},{rec[i]:.4f},{f1[i]:.4f},"
        return return_str
        



