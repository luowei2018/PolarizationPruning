#!/usr/bin/python

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
labelsize_b = 13
linewidth = 2
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16
plt.rcParams["font.family"] = "Times New Roman"
colors = ['#DB1F48','#FF9636','#1C4670','#9D5FFB','#21B6A8','#D65780']
# colors = ['#D00C0E','#E09C1A','#08A720','#86A8E7','#9D5FFB','#D65780']
labels = ['LSVC','H.264','H.265','DVC','RLVC']
markers = ['o','P','s','>','D','^']
linestyles = ['solid','dotted','dashed','dashdot', (0, (3, 5, 1, 5, 1, 5))]


def line_plot(XX,YY,label,color,path,xlabel,ylabel,lbsize=labelsize_b,lfsize=labelsize_b,legloc='best',
				xticks=None,yticks=None,ncol=None, yerr=None,
				use_arrow=False,arrow_coord=(0.4,30)):
	fig, ax = plt.subplots()
	ax.grid(zorder=0)
	# if use_arrow:
	# 	plt.hlines(94.1,0,90,colors='k',linestyles='--',label='Best')
	for i in range(len(XX)):
		xx,yy = XX[i],YY[i]
		if yerr is None:
			plt.plot(xx, yy, color = color[i], marker = markers[i], 
				# linestyle = linestyles[i], 
				label = label[i], 
				linewidth=1, markersize=4)
		else:
			plt.errorbar(xx, yy, yerr=yerr[i], color = color[i], 
				marker = markers[i], label = label[i], 
				linewidth=1, markersize=4)
	plt.xlabel(xlabel, fontsize = lbsize)
	plt.ylabel(ylabel, fontsize = lbsize)
	if xticks is not None:
		plt.xticks(xticks,fontsize=lfsize)
	if yticks is not None:
		plt.yticks(yticks,fontsize=lfsize)
	plt.tight_layout()
	# if ncol!=0:
	# 	if ncol is None:
	# 		plt.legend(loc=legloc,fontsize = lfsize)
	# 	else:
	# 		plt.legend(loc=legloc,fontsize = lfsize,ncol=ncol)

	plt.legend(bbox_to_anchor=(0.46, 1.28), fancybox=True,
	           loc='upper center', ncol=5, fontsize=lfsize)

	plt.xlim((0,100))
	# plt.ylim((-40,90))

	# inset axes....
	axins = ax.inset_axes([0.01, 0.04, 0.35, 0.48])
	for i in range(len(XX)):
		xx,yy = XX[i][5:],YY[i][5:]
		axins.plot(xx, yy, color = color[i], marker = markers[i], 
			label = label[i], 
			linewidth=1, markersize=4)
	# sub region of the original image
	x1, x2, y1, y2 = 0, 26, 90, 95
	axins.set_xlim(x1, x2)
	axins.set_ylim(y1, y2)
	axins.set_xticklabels([])
	axins.set_yticklabels([])
	axins.tick_params(
	    axis='x',          # changes apply to the x-axis
	    which='both',      # both major and minor ticks are affected
	    bottom=False,      # ticks along the bottom edge are off
	    top=False,         # ticks along the top edge are off
	    labelbottom=False) # labels along the bottom edge are off

	axins.tick_params(
	    axis='y',          # changes apply to the x-axis
	    which='both',      # both major and minor ticks are affected
	    right=False,      # ticks along the bottom edge are off
	    left=False,         # ticks along the top edge are off
	    labelbottom=False) # labels along the bottom edge are off


	ax.indicate_inset_zoom(axins, edgecolor="black")

	ratio = 0.3
	xleft, xright = ax.get_xlim()
	ybottom, ytop = ax.get_ylim()
	ax.set_aspect(abs((xright-xleft)/(ybottom-ytop))*ratio)


	plt.tight_layout()
	fig.savefig(path,bbox_inches='tight')
	plt.close()

# x_coords = []
# for layer in range(4,9):
# 	x_coords.append([1.0*l/layer for l in range(layer)])
# y_coords = [[94.24,93.95,91.71,10.42],
# 			[94.14,93.90,92.92,89.96,9.92],
# 			[93.90,93.78,93.57,90.69,83.71,10.25],
# 			[93.98,93.87,93.70,92.16,89.35,46.07,10.00],
# 			[94.14,94.08,93.97,92.93,89.94,84.61,10.32,14.22]]
# line_plot(x_coords, y_coords,[str(i) for i in range(4,9)],colors,
# 		'/home/bo/Dropbox/Research/CVPR23/images/diff_layer.eps',
# 		'Pruned Neurons (%)','Top1 Prec.')

# x_coords = [[0.25*i for i in range(4)]for j in range(4)]
# y_coords = [[94.24,93.95,91.71,10.42],
# 			[93.93,93.73,91.21,10.72],
# 			[93.96,93.86,90.67,11.83],
# 			[94.00,93.94,90.47,11.85]]
# line_plot(x_coords, y_coords,[str(i) for i in [4,8,16,32]],colors,
# 		'/home/bo/Dropbox/Research/CVPR23/images/diff_superbatch.eps',
# 		'Pruned Neurons (%)','Top1 Prec.')		

# with open('/home/bo/Dropbox/Research/CVPR23/exp/resnet56.log','r') as f:
# 	count = 0
# 	loss = []
# 	prec1 = [[] for _ in range(4)]
# 	x_coords = []
# 	for epoch,line in enumerate(f.readlines()):
# 		line = line.strip()
# 		line = line.split(' ')
# 		x_coords += [epoch+1]
# 		loss += [float(line[0])]
# 		for i in range(4):
# 			prec1[i] += [float(line[i+1])]
# line_plot([x_coords for i in range(4)], prec1,[f'{i*25}% pruned' for i in range(4)],colors,
# 		'/home/bo/Dropbox/Research/CVPR23/images/resnet56_lc.eps',
# 		'Epoch','Top1 Prec.')	


x = []
y = []
# original
x += [[92.52,87.53,80.16,71.11,61.96,51.42,40.52,30.26,20.15,0.00]]
y += [[0.1040,0.1216,0.1160,0.1655,0.1820,0.2294,0.4102,0.8530,0.9135,0.9411]]
# sr
x += [[93.66,86.21,77.18,67.75,58.50,51.87,45.76,34.85,18.31,0.00]]
y += [[0.1000,0.1004,0.1049,0.1795,0.3598,0.7431,0.9096,0.9332,0.9353,0.9370]]
# zol
x += [[94.25,86.15,76.8,66.29,59.31,53.22,38.14,24.72,8.41,0.00]]
y += [[0.1000,0.0927,0.1090,0.2734,0.7647,0.8648,0.9242,0.9310,0.9310,0.9322]]
# ofa
x += [[90.70,86.88,80.53,72.31,64.04,57.53,50.84,40.20,22.81,0.00]]
y += [[0.3587,0.6448,0.8675,0.9009,0.9103,0.9081,0.9082,0.9084,0.9086,0.9086]]
x = np.array(x)
y = np.array(y)*100
line_plot(x, y,['Original','Lasso','Polar','OFANet'],colors,
		'/home/bo/Dropbox/Research/CVPR23/images/compare_scalability.eps',
		'Pruned FLOPS (%)','Top1 Prec. (%)',yticks=[50,90],xticks=[i*10 for i in range(1,10)],use_arrow=True)	