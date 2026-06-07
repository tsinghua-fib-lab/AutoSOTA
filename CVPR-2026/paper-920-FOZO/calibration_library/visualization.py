import numpy as np
#import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pyplot as plt

from . import metrics

class ConfidenceHistogram(metrics.MaxProbCELoss):

    def plot(self, output, labels, n_bins = 15, logits = True, title = None):
        super().loss(output, labels, n_bins, logits)
        #scale each datapoint
        n = len(labels)
        w = np.ones(n)/n

        plt.rcParams["font.family"] = "serif"
        #size and axis limits 
        plt.figure(figsize=(5,5))
        plt.xlim(0,1)
        plt.ylim(0,1)
        plt.xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], ['0.0', '0.2', '0.4', '0.6', '0.8', '1.0'])
        plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], ['0.0', '0.2', '0.4', '0.6', '0.8', '1.0'])
        #plot grid
        plt.grid(color='tab:grey', linestyle=(0, (1, 5)), linewidth=1,zorder=0)    
        #plot histogram
        plt.hist(self.confidences,n_bins,weights = w,color='b',range=(0.0,1.0),edgecolor = 'k')

        #plot vertical dashed lines
        acc = np.mean(self.accuracies)
        conf = np.mean(self.confidences)                
        plt.axvline(x=acc, color='tab:grey', linestyle='--', linewidth = 3)
        plt.axvline(x=conf, color='tab:grey', linestyle='--', linewidth = 3)
        if acc > conf:
            plt.text(acc+0.03,0.9,'Accuracy',rotation=90,fontsize=11)
            plt.text(conf-0.07,0.9,'Avg. Confidence',rotation=90, fontsize=11)
        else:
            plt.text(acc-0.07,0.9,'Accuracy',rotation=90,fontsize=11)
            plt.text(conf+0.03,0.9,'Avg. Confidence',rotation=90, fontsize=11)

        plt.ylabel('% of Samples',fontsize=13)
        plt.xlabel('Confidence',fontsize=13)
        plt.tight_layout()
        if title is not None:
            plt.title(title,fontsize=16)
        return plt

def add_label(plt, x, y):
    for i in range(len(x)):
        plt.text(x[i], np.around(y[i]+0.01,3), np.around(y[i], 3), zorder = 16)

class ReliabilityDiagram(metrics.MaxProbCELoss):

    def plot(self, output, labels, n_bins = 15, logits = True, title = None):
        super().loss(output, labels, n_bins, logits)

        #computations
        delta = 1.0/n_bins
        x = np.arange(0,1,delta)
        mid = np.linspace(delta/2,1-delta/2,n_bins)
        error = np.abs(np.subtract(mid, self.bin_acc))

        plt.rcParams["font.family"] = "serif"
        #size and axis limits
        plt.figure(figsize=(5,5))
        plt.xlim(0,1)
        plt.ylim(0,1)
        #plot grid
        plt.grid(color='tab:grey', linestyle=(0, (1, 5)), linewidth=1,zorder=0)
        #plot bars and identity line
        plt.bar(x, self.bin_acc, color = 'b', width=delta,align='edge',edgecolor = 'k',label='Outputs',zorder=5)
        plt.bar(x, error, bottom=np.minimum(self.bin_acc,mid), color = 'mistyrose', alpha=0.5, width=delta,align='edge',edgecolor = 'r',hatch='/',label='Gap',zorder=10)
        ident = [0.0, 1.0]
        plt.plot(ident,ident,linestyle='--',color='tab:grey',zorder=15)
        #labels and legend
        plt.ylabel('Accuracy',fontsize=13)
        plt.xlabel('Confidence',fontsize=13)
        plt.legend(loc='upper left',framealpha=1.0,fontsize='medium')
        if title is not None:
            plt.title(title,fontsize=16)
        plt.tight_layout()

        return plt

class ReliabilityDiagramPerClass(metrics.CELoss):

    def __draw(self, title = None):
        #computations
        delta = 1.0/self.n_bins
        x = np.arange(0,1,delta)
        mid = np.linspace(delta/2,1-delta/2,self.n_bins)
        error = np.abs(np.subtract(mid, self.bin_acc))

        plt.rcParams["font.family"] = "serif"
        #size and axis limits
        plt.figure(figsize=(5,5))
        plt.xlim(0,1)
        plt.ylim(0,1)
        #plot grid
        plt.grid(color='tab:grey', linestyle=(0, (1, 5)), linewidth=1,zorder=0)
        #plot bars and identity line
        plt.bar(x, self.bin_acc, color = 'b', width=delta,align='edge',edgecolor = 'k',label='Outputs',zorder=5)
        plt.bar(x, error, bottom=np.minimum(self.bin_acc,mid), color = 'mistyrose', alpha=0.5, width=delta,align='edge',edgecolor = 'r',hatch='/',label='Gap',zorder=10)
        ident = [0.0, 1.0]
        plt.plot(ident,ident,linestyle='--',color='tab:grey',zorder=15)
        #labels and legend
        plt.ylabel('Accuracy',fontsize=13)
        plt.xlabel('Confidence',fontsize=13)
        plt.legend(loc='upper left',framealpha=1.0,fontsize='medium')
        if title is not None:
            plt.title(title,fontsize=16)
        plt.tight_layout()
        plt.savefig("plots/"+title+"-v2.png")

    def plot(self, output, labels, n_bins = 15, logits = True, title = None):
        self.n_bins = n_bins
        self.n_data = len(output)
        self.n_class = len(output[0])

        super().compute_bin_boundaries()
        super().get_probabilities(output, labels, logits)
        super().binary_matrices()

        for i in range(self.n_class):
            super().compute_bins(i)
            self.__draw(title + f"-class{i}-v2")