import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import rc
import matplotlib.colors as colors
from matplotlib.ticker import FormatStrFormatter
import matplotlib.pylab as pl
import pickle
plt.rcParams['text.usetex'] = True
plt.rcParams['axes.axisbelow'] = True
plt.rc('font', family = 'sans')
plt.rcParams['font.size'] = 9
plt.rcParams['pgf.texsystem'] = 'pdflatex'
#FNULL = open(os.devnull, 'w')

ncp_on = pickle.load(open('log_ncp_on.p', 'rb'), encoding='latin1')
ncp_off = pickle.load(open('log_ncp_off.p', 'rb'), encoding='latin1')
x_axis = range(len(logging['id_loss']))

markersize = 7


fig = plt.figure(figsize = (6, 4))
ax0 = fig.add_subplot(111)
ax0.plot(
        x_axis,
        ncp_off['om_ncp_loss'],
        markersize = markersize,
        markerfacecolor = 'r',
        marker = '.',
        markeredgecolor = 'black',
        linewidth = 0.5,
        linestyle = '--',
        color = 'k',
        markeredgewidth = 0.4,
        label = ''
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
fig.savefig('ncp_loss.pgf', bbox_inches = 'tight')
fig.savefig('ncp_loss.pdf', bbox_inches = 'tight')

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
fig.savefig('losses.pgf', bbox_inches = 'tight')
fig.savefig('losses.pdf', bbox_inches = 'tight')
