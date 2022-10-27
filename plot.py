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
# colors = ['#DB1F48','#FF9636','#1C4670','#9D5FFB','#21B6A8','#D65780']
colors = ['#d53e4f',
'#f46d43',
'#fdae61',
'#fee08b',
'#3288bd',
'#abdda4',
'#66c2a5',
'#e6f598',
]
labels = ['LSVC','H.264','H.265','DVC','RLVC']
# markers = ['o','P','s','>','D','^']
markers = ['v','^','<','>','o','P','s','D']
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
		xx = np.array(xx)
		yy = np.array(yy)*100
		if yerr is None:
			plt.plot(xx, yy, color = color[i], marker = markers[i], 
				# linestyle = linestyles[i], 
				label = label[i], 
				linewidth=2, markersize=6)
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
	           loc='upper center', ncol=4, fontsize=lfsize)

	plt.xlim((0,100))
	# plt.ylim((0,95))

	# inset axes....
	axins = ax.inset_axes([0.02, 0.04, 0.30, 0.65])
	for i in range(len(XX)):
		xx,yy = XX[i],YY[i]
		xx = np.array(xx)
		yy = np.array(yy)*100
		axins.plot(xx, yy, color = color[i], marker = markers[i], 
			label = label[i], 
			linewidth=2, markersize=6)
	# sub region of the original image
	x1, x2, y1, y2 = 0, 60, 90, 95
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
	# for axis in ['top','bottom','left','right']:
	# 	axins.spines[axis].set_linestyle('--')	


	ax.indicate_inset_zoom(axins, edgecolor="gray")

	# ratio = 1
	# xleft, xright = ax.get_xlim()
	# ybottom, ytop = ax.get_ylim()
	# ax.set_aspect(abs((xright-xleft)/(ybottom-ytop))*ratio)


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
# onion 4
y += [[0.8254,0.9216,0.9398,0.9425]]
x += [[84.77,62.79,36.12,0.00]]
# ONN8
y += [[0.4266,0.8285,0.8929,0.9195,0.9300,0.9370,0.9402,0.9427]]
x += [[93.22,86.11,76.02,64.22,51.33,36.68,23.35,0.00]]
# ONN12
y += [[0.1067,0.5651,0.8049,0.8701,0.8972,0.9159,0.9288,0.9330,0.9382,0.9399,0.9423,0.9429]]
x += [[92.67,90.62,86.42,80.16,72.02,64.39,55.75,46.84,36.59,27.71,18.18,0.00]]
# ONN16
y += [[0.1320,0.1590,0.5718,0.8065,0.8238,0.8444,0.8881,0.9162,0.9228,0.9283,0.9355,0.9357,0.9375,0.9391,0.9427,0.9433]]
x += [[92.95,92.78,88.97,86.40,81.69,76.06,70.21,64.26,58.40,51.57,44.47,36.67,30.07,23.33,14.82,0.00]]
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
line_plot(x, y,['ONN-L4','ONN-L8','ONN-L12','ONN-L16','Original','Lasso','Polar','OFA'],colors,
		'/home/bo/Dropbox/Research/CVPR23/images/compare_scalability.eps',
		'Pruned FLOPS (%)','Top1 Accuracy (%)',yticks=[i*10 for i in range(10)],xticks=[i*10 for i in range(1,10)],use_arrow=True)	