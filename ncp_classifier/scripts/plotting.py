import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import rc
import matplotlib.colors as colors
from matplotlib.ticker import FormatStrFormatter
import matplotlib.pylab as pl
import pickle
import os

plt.rcParams['text.usetex'] = True
plt.rcParams['axes.axisbelow'] = True
plt.rc('font', family = 'sans')
plt.rcParams['font.size'] = 9
plt.rcParams['pgf.texsystem'] = 'pdflatex'
#FNULL = open(os.devnull, 'w')





def run_plotting(EXPERIMENT_NAME):
    """
    Run the plotting analysis.
    
    We will run one of three modes based on the experiment type.
    """
    
    print('Running plotting script with experiment type')
    print(EXPERIMENT_NAME)
    
    #EXPERIMENT_NAME = 'digout' #'alpha' #'rotate' #'digout'
    MODE = 'ncp_on' #'ncp_off'
    
    
    savedir = os.path.join('ncp_classifier', 'output', EXPERIMENT_NAME)
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    prefix = 'log_ncp_on' if MODE == 'ncp_on' else 'log_ncp_off'
    logdir = os.path.join('ncp_classifier', 'logs')
    data_list = [os.path.join(logdir,i) for i in os.listdir(logdir) if i.endswith('.p') and (i.find(EXPERIMENT_NAME)>=0) and (i.find(prefix)>=0)]
    data_list.sort()
    print(data_list)
    
    
    
    
    
    # =============================================================================
    # The "rotate" experiment:
    # For each distribution of allowed angles, look at the entropies over training
    # =============================================================================
    if EXPERIMENT_NAME=='rotate':
        for dd in data_list:
            
            s = dd.find('__')+2
            e = dd.find('.p')
            exp = dd[s:e]
            logging = pickle.load(open(dd, 'rb'), encoding='latin1')
            
            x_axis = range(len(logging['id_loss']))
            markersize = 7
            
            fig = plt.figure(figsize = (6, 4))
            ax0 = fig.add_subplot(111)
            ax0.plot(
                    x_axis,
                    logging['om_ncp_loss'],
                    markersize = markersize,
                    markerfacecolor = 'r',
                    marker = '.',
                    markeredgecolor = 'black',
                    linewidth = 0.5,
                    linestyle = '--',
                    color = 'k',
                    markeredgewidth = 0.4,
                    label = 'OOD entropy'
                    )
            ax0.plot(
                    x_axis,
                    logging['id_ncp_loss'],
                    markersize = markersize,
                    markerfacecolor = 'b',
                    marker = '.',
                    markeredgecolor = 'black',
                    linewidth = 0.5,
                    linestyle = '--',
                    color = 'k',
                    markeredgewidth = 0.4,
                    label = 'in-distribution entropy'
                    )
            #ax0.yaxis.set_major_formatter(FormatStrFormatter('%g'))
            ax0.xaxis.set_major_formatter(FormatStrFormatter('%g'))
            ax0.get_yaxis().set_tick_params(which='both', direction='in')
            ax0.get_xaxis().set_tick_params(which='both', direction='in')
            ax0.set_ylabel('Entropy')
            ax0.set_xlabel('Iterations')
            ax0.legend()
            fig.savefig(os.path.join(savedir,f'ncp_loss_{exp}.pgf'), bbox_inches = 'tight')
            fig.savefig(os.path.join(savedir,f'ncp_loss_{exp}.pdf'), bbox_inches = 'tight')
            
            fig = plt.figure(figsize = (6, 4))
            ax0 = fig.add_subplot(111)
            ax0.plot(
                    x_axis,
                    logging['id_loss'],
                    markersize = markersize,
                    markerfacecolor = 'r',
                    marker = '.',
                    markeredgecolor = 'black',
                    linewidth = 0.5,
                    linestyle = '--',
                    color = 'k',
                    markeredgewidth = 0.4,
                    label = 'in-distribution loss'
                    )
            #ax0.yaxis.set_major_formatter(FormatStrFormatter('%g'))
            ax0.xaxis.set_major_formatter(FormatStrFormatter('%g'))
            ax0.get_yaxis().set_tick_params(which='both', direction='in')
            ax0.get_xaxis().set_tick_params(which='both', direction='in')
            ax0.legend()
            ax0.set_ylabel('Loss')
            ax0.set_xlabel('Iterations')
            fig.savefig(os.path.join(savedir,f'losses_{exp}.pgf'), bbox_inches = 'tight')
            fig.savefig(os.path.join(savedir,f'losses_{exp}.pdf'), bbox_inches = 'tight')
        
        
        
    
    
    
    # =============================================================================
    # The "alpha" loss experiment:
    # For each value of alpha, for each of the 3 dataset partitions (in, out, omitted),
    # look at the uncertainty over the entire partition, as a function of alpha.
    # =============================================================================
    if EXPERIMENT_NAME=='alpha' or EXPERIMENT_NAME=='digout':
        
        digouts = []
        if EXPERIMENT_NAME=='digout':
            for jj in data_list:
                s = jj.find('digout_') + len('digout_') - 1
                e = jj.find('out_',s) + len('out_')
                digouts.extend([jj[s:e]])
            digouts = set(digouts)
        elif EXPERIMENT_NAME=='alpha':
            digouts = ('__alpha_')
    
        for ds in digouts:
            
            data_list_ = [i for i in data_list if i.find(ds)>-1]
            alphas = []
            id_entropy_mean = []
            id_entropy_std = []
            od_entropy_mean = []
            od_entropy_std = []
            om_entropy_mean = []
            om_entropy_std = []
            id_acc = []
            od_acc = []
            for dd in data_list_:
                s = dd.find('__')+2
                e = dd.find('.p')
                exp = dd[s:e]
                logging = pickle.load(open(dd, 'rb'), encoding='latin1')    
                
                alphas.extend([logging['alpha']])
                id_entropy_mean.extend([logging['id_entropy_mean']])
                id_entropy_std.extend([logging['id_entropy_std']])
                od_entropy_mean.extend([logging['od_entropy_mean']])
                od_entropy_std.extend([logging['od_entropy_std']])
                #For om, we test on whole om set at every iteration, so just use last (fully trained) value [-1]
                om_entropy_mean.extend([logging['om_ncp_loss'][-1]])
                om_entropy_std.extend([logging['om_ncp_std'][-1]])
                #And look at accuracies
                id_acc.extend([logging['id_acc']])
                od_acc.extend([logging['od_acc']])
                #(om_acc would just be 0 always)
        
        
            # =============================================================================
            # Entropies
            # =============================================================================
            markersize = 7
            fig = plt.figure(figsize = (6, 4))
            plt.title('Entropy as a function of alpha\n(inverse weighting of entropy term)')
            ax0 = fig.add_subplot(111)
            #In-distribution entropy
            print(id_entropy_mean)
            print(alphas)
            ax0.semilogx(alphas, id_entropy_mean,
                    markersize = markersize,
                    markerfacecolor = 'b',
                    markeredgecolor = 'black',
                    marker = '.',
                    linestyle = '--',
                    color = 'k',
                    label = 'In-distribution'
                    )
            #Out-of-distribution entropy
            ax0.semilogx(alphas, od_entropy_mean,
                    markersize = markersize,
                    markerfacecolor = 'g',
                    markeredgecolor = 'black',
                    marker = '.',
                    linestyle = '--',
                    color = 'k',
                    label = 'OOD'
                    )
            #Omitted digit entropy
            ax0.semilogx(alphas, om_entropy_mean,
                    markersize = markersize,
                    markerfacecolor = 'r',
                    markeredgecolor = 'black',
                    marker = '.',
                    linestyle = '--',
                    color = 'k',
                    label = 'Omitted digits'
                    )
            
            #1 SD Error bars:
            opacity = .5
            for nn, a in enumerate(alphas):
                plt.vlines(a, id_entropy_mean[nn] - id_entropy_std[nn], id_entropy_mean[nn] + id_entropy_std[nn], colors='b', alpha=opacity)
                plt.vlines(a, od_entropy_mean[nn] - od_entropy_std[nn], od_entropy_mean[nn] + od_entropy_std[nn], colors='g', alpha=opacity)
                plt.vlines(a, om_entropy_mean[nn] - om_entropy_std[nn], om_entropy_mean[nn] + om_entropy_std[nn], colors='r', alpha=opacity)
                           
            ax0.xaxis.set_major_formatter(FormatStrFormatter('%g'))
            ax0.get_yaxis().set_tick_params(which='both', direction='in')
            ax0.get_xaxis().set_tick_params(which='both', direction='in')
            ax0.legend()
            ax0.set_ylabel('Entropy')
            ax0.set_xlabel('alpha')
            fig.savefig(os.path.join(savedir,f'experiment_entropy{ds}.pgf'), bbox_inches = 'tight')
            fig.savefig(os.path.join(savedir,f'experiment_entropy{ds}.pdf'), bbox_inches = 'tight')    
            
            
            # =============================================================================
            # Accuracies
            # =============================================================================
            fig = plt.figure(figsize = (6, 4))
            plt.title('Accuracy as a function of alpha\n(inverse weighting of entropy term)')
            ax0 = fig.add_subplot(111)
            #In-distribution
            ax0.semilogx(alphas, id_acc, markerfacecolor='b', 
                     markeredgecolor='black', marker='.', linestyle='--', color='k', 
                     label='In-distribution')
            #OOD
            ax0.semilogx(alphas, od_acc, markerfacecolor='g', 
                     markeredgecolor='black', marker='.', linestyle='--', color='k', 
                     label='OOD')
            #Omitted digits
            ax0.semilogx(alphas, [0.]*len(alphas), markerfacecolor='r', 
                     markeredgecolor='black', marker='.', linestyle='--', color='k', 
                     label='Omitted digits')    
                           
            ax0.xaxis.set_major_formatter(FormatStrFormatter('%g'))
            ax0.get_yaxis().set_tick_params(which='both', direction='in')
            ax0.get_xaxis().set_tick_params(which='both', direction='in')
            ax0.legend()
            ax0.set_ylabel('Accuracy')
            ax0.set_xlabel('alpha')
            plt.ylim(None,1.)
            fig.savefig(os.path.join(savedir,f'experiment_accuracy{ds}.pgf'), bbox_inches = 'tight')
            fig.savefig(os.path.join(savedir,f'experiment_accuracy{ds}.pdf'), bbox_inches = 'tight')       
    
