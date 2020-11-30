import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
from sklearn.metrics import confusion_matrix

class Plotter:

      def __init__(self, output_name):
              self.graphs_directory = os.makedirs(os.path.join(output_name, "graphs"))
              self.graphs_path = os.path.join(output_name, "graphs")
      def density(self, data):
            """Plot distrbution after filling missing data"""
            dimensions = (11.7, 8.27)
            fig, ax = plt.subplots(figsize=dimensions)

            for column in data:
                sns.distplot(data[column], hist = False, ax=ax)

            plt.title("Sample Distributions")
            plt.xlabel("Expression Level")
            plt.ylabel("Density")
            plt.savefig(os.path.join(self.graphs_path,"density.pdf"))

      def accuracy(self, train_stats, val_stats, graphs_title="Training vs Testing"):
          """Plot training/validation accuracy/loss"""

          # Plot Accuracy
          figure = plt.figure()
          figure.set_figheight(11.7) # A4 size
          figure.set_figwidth(8.27)
          plot1 = figure.add_subplot(211)
          plot1.plot(train_stats['accuracy'])
          plot1.plot(val_stats['accuracy'])
          plot1.set_title(graphs_title)
          plot1.set_ylabel("Accuracy")
          plot1.set_xlabel("Epoch")
          plt.legend(["Training", "Validation"], loc="upper left")

          # Plot Loss 
          plot2 = figure.add_subplot(212)
          plot2.plot(train_stats['loss'])
          plot2.plot(val_stats['loss'])
          plot2.set_title(graphs_title)
          plot2.set_ylabel("Loss")
          plot2.set_xlabel("Epoch")
          plot2.legend(["Training", "Validation"], loc="upper left")
      
          # Save plots into pdf
          plt.savefig(os.path.join(self.graphs_path, 'stats.pdf'))
      
      def confusion(self, y_target, y_predict, labels, cm_title="Confusion Matrix"):
          """Plots confusion matrix after the model is trained"""
          # Set up dimensions for plots
          dimensions = (8.27,11.7)
          fig, axis = plt.subplots(figsize=dimensions)
      
          # Plot CM and save to pdf
          confusion_matrix_df = pd.DataFrame(confusion_matrix(y_target, y_predict))
          sns_heatmap=sns.heatmap(confusion_matrix_df, ax=axis, annot=True, cbar=False,
                                      square=True, xticklabels=labels, yticklabels=labels)
          axis.set_ylabel("Actual")
          axis.set_xlabel("Predicted")
          axis.set_title(cm_title)
          sns_heatmap.figure.savefig(os.path.join(self.graphs_path, "confusion_matrix.pdf"))
