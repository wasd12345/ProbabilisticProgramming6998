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


# =============================================================================
# PARAMETERS
# =============================================================================
MODE = 'ncp_on' #'ncp_off'
EXPERIMENT_NAME = 'rotate'





# =============================================================================
# MAIN PLOTTING
# =============================================================================
prefix = 'log_ncp_on' if MODE == 'ncp_on' else 'log_ncp_off'
data_list = [i for i in os.listdir() if i.endswith('.p') and (i.find(EXPERIMENT_NAME)>=0) and (i.find(prefix)>=0)]
print(data_list)


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
    fig.savefig(f'ncp_loss_{exp}.pgf', bbox_inches = 'tight')
    fig.savefig(f'ncp_loss_{exp}.pdf', bbox_inches = 'tight')
    
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
    fig.savefig(f'losses_{exp}.pgf', bbox_inches = 'tight')
    fig.savefig(f'losses_{exp}.pdf', bbox_inches = 'tight')
