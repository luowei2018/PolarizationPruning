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
colors = ['#d53e4f',
'#f46d43',
'#fdae61',
'#fee08b',
'#3288bd',
'#abdda4',
'#66c2a5',
'#e6f598',
]
markers = ['v','^','<','>','o','P','s','D']


# def line_plot(XX,YY,label,color,path,xlabel,ylabel,lbsize=labelsize_b,lfsize=labelsize_b,legloc='best',
# 				xticks=None,yticks=None,ncol=None, yerr=None,
# 				use_arrow=False,arrow_coord=(0.4,30)):
# 	fig, ax = plt.subplots()
# 	ax.grid(zorder=0)
# 	# if use_arrow:
# 	# 	plt.hlines(94.1,0,90,colors='k',linestyles='--',label='Best')
# 	for i in range(len(YY)):
# 		yy = YY[i]
# 		if XX is None:
# 			xx = range(len(yy))
# 		else:
# 			xx = XX[i]
# 		xx = np.array(xx)
# 		yy = np.array(yy)*100
# 		if yerr is None:
# 			plt.plot(xx, yy, color = color[i], marker = markers[i], 
# 				# linestyle = linestyles[i], 
# 				label = label[i], 
# 				linewidth=2, markersize=6)
# 		else:
# 			plt.errorbar(xx, yy, yerr=yerr[i], color = color[i], 
# 				marker = markers[i], label = label[i], 
# 				linewidth=1, markersize=4)
# 	plt.xlabel(xlabel, fontsize = lbsize)
# 	plt.ylabel(ylabel, fontsize = lbsize)
# 	if xticks is not None:
# 		plt.xticks(xticks,fontsize=lfsize)
# 	if yticks is not None:
# 		plt.yticks(yticks,fontsize=lfsize)
# 	plt.tight_layout()
# 	# if ncol!=0:
# 	# 	if ncol is None:
# 	# 		plt.legend(loc=legloc,fontsize = lfsize)
# 	# 	else:
# 	# 		plt.legend(loc=legloc,fontsize = lfsize,ncol=ncol)

# 	plt.legend(bbox_to_anchor=(0.46, 1.28), fancybox=True,
# 	           loc='upper center', ncol=4, fontsize=lfsize)

# 	plt.xlim((0,100))
# 	# plt.ylim((0,95))

# 	# inset axes....
# 	axins = ax.inset_axes([0.02, 0.04, 0.30, 0.65])
# 	for i in range(len(YY)):
# 		yy = YY[i]
# 		if XX is None:
# 			xx = range(len(yy))
# 		else:
# 			xx = XX[i]
# 		xx = np.array(xx)
# 		yy = np.array(yy)*100
# 		axins.plot(xx, yy, color = color[i], marker = markers[i], 
# 			label = label[i], 
# 			linewidth=2, markersize=6)
# 	# sub region of the original image
# 	x1, x2, y1, y2 = 0, 60, 90, 95
# 	axins.set_xlim(x1, x2)
# 	axins.set_ylim(y1, y2)
# 	axins.set_xticklabels([])
# 	axins.set_yticklabels([])
# 	axins.tick_params(
# 	    axis='x',          # changes apply to the x-axis
# 	    which='both',      # both major and minor ticks are affected
# 	    bottom=False,      # ticks along the bottom edge are off
# 	    top=False,         # ticks along the top edge are off
# 	    labelbottom=False) # labels along the bottom edge are off

# 	axins.tick_params(
# 	    axis='y',          # changes apply to the x-axis
# 	    which='both',      # both major and minor ticks are affected
# 	    right=False,      # ticks along the bottom edge are off
# 	    left=False,         # ticks along the top edge are off
# 	    labelbottom=False) # labels along the bottom edge are off
# 	# for axis in ['top','bottom','left','right']:
# 	# 	axins.spines[axis].set_linestyle('--')	


# 	ax.indicate_inset_zoom(axins, edgecolor="gray")

# 	# ratio = 1
# 	# xleft, xright = ax.get_xlim()
# 	# ybottom, ytop = ax.get_ylim()
# 	# ax.set_aspect(abs((xright-xleft)/(ybottom-ytop))*ratio)


# 	plt.tight_layout()
# 	fig.savefig(path,bbox_inches='tight')
# 	plt.close()

# x = []
# y = []
# # onion 4
# y += [[0.8254,0.9216,0.9398,0.9425]]
# x += [[84.77,62.79,36.12,0.00]]
# # ONN8
# y += [[0.4266,0.8285,0.8929,0.9195,0.9300,0.9370,0.9402,0.9427]]
# x += [[93.22,86.11,76.02,64.22,51.33,36.68,23.35,0.00]]
# # ONN12
# y += [[0.1067,0.5651,0.8049,0.8701,0.8972,0.9159,0.9288,0.9330,0.9382,0.9399,0.9423,0.9429]]
# x += [[92.67,90.62,86.42,80.16,72.02,64.39,55.75,46.84,36.59,27.71,18.18,0.00]]
# # ONN16
# y += [[0.1320,0.1590,0.5718,0.8065,0.8238,0.8444,0.8881,0.9162,0.9228,0.9283,0.9355,0.9357,0.9375,0.9391,0.9427,0.9433]]
# x += [[92.95,92.78,88.97,86.40,81.69,76.06,70.21,64.26,58.40,51.57,44.47,36.67,30.07,23.33,14.82,0.00]]
# # original
# x += [[92.52,87.53,80.16,71.11,61.96,51.42,40.52,30.26,20.15,0.00]]
# y += [[0.1040,0.1216,0.1160,0.1655,0.1820,0.2294,0.4102,0.8530,0.9135,0.9411]]
# # sr
# x += [[93.66,86.21,77.18,67.75,58.50,51.87,45.76,34.85,18.31,0.00]]
# y += [[0.1000,0.1004,0.1049,0.1795,0.3598,0.7431,0.9096,0.9332,0.9353,0.9370]]
# # zol
# x += [[94.25,86.15,76.8,66.29,59.31,53.22,38.14,24.72,8.41,0.00]]
# y += [[0.1000,0.0927,0.1090,0.2734,0.7647,0.8648,0.9242,0.9310,0.9310,0.9322]]
# # ofa
# x += [[90.70,86.88,80.53,72.31,64.04,57.53,50.84,40.20,22.81,0.00]]
# y += [[0.3587,0.6448,0.8675,0.9009,0.9103,0.9081,0.9082,0.9084,0.9086,0.9086]]
# line_plot(x, y,['ONN-L4','ONN-L8','ONN-L12','ONN-L16','Original','Lasso','Polar','OFA'],colors,
# 		'/home/bo/Dropbox/Research/CVPR23/images/compare_scalability.eps',
# 		'Pruned FLOPS (%)','Top1 Accuracy (%)',yticks=[i*10 for i in range(10)],xticks=[i*10 for i in range(1,10)],use_arrow=True)	

# --------------------------------------------

def line_plot(XX,YY,label,color,path,xlabel,ylabel,lbsize=labelsize_b,lfsize=labelsize_b,legloc='best',
				xticks=None,yticks=None,ncol=None, yerr=None,
				use_arrow=False,arrow_coord=(0.4,30)):
	fig, ax = plt.subplots()
	ax.grid(zorder=0)
	for i in range(len(YY)):
		yy = YY[i]
		if XX is None:
			xx = range(len(yy))
		else:
			xx = XX[i]
		xx = np.array(xx)
		yy = np.array(yy)*100
		plt.scatter(xx, yy, color = color[i], marker = markers[i], 
			label = label[i], 
			linewidth=1)
	plt.xlabel(xlabel, fontsize = lbsize)
	plt.ylabel(ylabel, fontsize = lbsize)
	if xticks is not None:
		plt.xticks(xticks,fontsize=lfsize)
	if yticks is not None:
		plt.yticks(yticks,fontsize=lfsize)
	plt.tight_layout()
	plt.legend(bbox_to_anchor=(0.46, 1.28), fancybox=True,
	           loc='upper center', ncol=4, fontsize=lfsize)

	ratio = .3
	xleft, xright = ax.get_xlim()
	ybottom, ytop = ax.get_ylim()
	ax.set_aspect(abs((xright-xleft)/(ybottom-ytop))*ratio)

	plt.tight_layout()
	fig.savefig(path,bbox_inches='tight')
	plt.close()



r50_means=[[0.002290084958076477, -0.962398111820221, 0.10927463322877884, 0.0026021632365882397, -0.18301846086978912, 0.0740204006433487, -0.17628717422485352, -0.0007581686368212104, -0.07687775790691376, -0.05563865602016449, -0.014795482158660889, -0.21798792481422424, -0.034169457852840424, -0.007116457913070917, -0.06855857372283936, 0.11071239411830902, -0.05976599454879761, -0.005744374822825193, -0.10917907953262329, -0.09111443161964417, -0.010958501137793064, -0.2637360990047455, -0.0700397938489914, -0.022291583940386772, -0.3519327640533447, -0.07580901682376862, -0.014093825593590736, -0.039119377732276917, -0.09748006612062454, -0.09527412056922913, -0.0010333503596484661, -0.19566184282302856, -0.11679476499557495, -0.00278111407533288, -0.1711583286523819, -0.17094393074512482, -0.0064657824113965034, -0.23097378015518188, -0.11335508525371552, -0.014148380607366562, -0.24118351936340332, -0.13230116665363312, -0.026485729962587357, -0.26362743973731995, -0.15814052522182465, -0.027796611189842224, -0.0289810411632061, -0.5085495710372925, -0.17985932528972626, -0.01625940203666687, -0.3514408469200134, -0.214706689119339, -0.009003974497318268], [-0.0001537876669317484, -0.9071130752563477, 0.0810493528842926, 0.003968825563788414, -0.1827617883682251, 0.06790988892316818, -0.14468953013420105, -0.0014172722585499287, -0.0540904626250267, -0.04723609238862991, -0.019197072833776474, -0.2090425193309784, -0.02549234963953495, -0.017417864874005318, -0.06831424683332443, 0.11060310900211334, -0.0689195990562439, -0.00790325179696083, -0.09858225286006927, -0.08938335627317429, -0.017796792089939117, -0.22312356531620026, -0.08474281430244446, -0.027952883392572403, -0.3382217288017273, -0.05302819609642029, -0.03263234719634056, -0.03932086378335953, -0.0965222716331482, -0.04952177032828331, -0.008282230235636234, -0.1819535195827484, -0.12267110496759415, -0.012192565947771072, -0.1506955772638321, -0.13199938833713531, -0.027982641011476517, -0.20790719985961914, -0.10125921666622162, -0.021688226610422134, -0.22047321498394012, -0.10796099901199341, -0.0426865853369236, -0.2484147995710373, -0.13764327764511108, -0.049654241651296616, -0.03721322864294052, -0.44972944259643555, -0.18544916808605194, -0.03705831617116928, -0.37953677773475647, -0.22223715484142303, -0.012049444019794464], [0.0014006902929395437, -0.811802089214325, 0.04514002799987793, 0.010403444990515709, -0.18291181325912476, 0.04161807894706726, -0.06694188714027405, 0.0009768899763002992, -0.05308933183550835, -0.07438099384307861, -0.017836567014455795, -0.20279602706432343, -0.03794720768928528, -0.011002498678863049, -0.06883472204208374, 0.09518449753522873, -0.11829286068677902, -0.01642134226858616, -0.0391220822930336, -0.03130460903048515, -0.027509354054927826, -0.19483540952205658, -0.09977836906909943, -0.02905331738293171, -0.3288798928260803, -0.0324515774846077, -0.048598453402519226, -0.04062924534082413, -0.08996459096670151, -0.04554259777069092, -0.021211059764027596, -0.14891891181468964, -0.10029588639736176, -0.022419024258852005, -0.12602192163467407, -0.1147325336933136, -0.03774075582623482, -0.19305749237537384, -0.1255052387714386, -0.030506938695907593, -0.2112475037574768, -0.11441588401794434, -0.0611475333571434, -0.2706674039363861, -0.131624236702919, -0.07111983746290207, -0.043467164039611816, -0.3993421792984009, -0.26626816391944885, -0.045708585530519485, -0.3856489956378937, -0.24845926463603973, -0.01615070179104805], [0.001136920414865017, -0.821086585521698, 0.08531845360994339, 0.011506522074341774, -0.1823384314775467, 0.03903050720691681, -0.1454218029975891, 0.008703507483005524, -0.04979851096868515, -0.0759437307715416, -0.02038741298019886, -0.20671489834785461, -0.04743171110749245, -0.00589616596698761, -0.0691274106502533, 0.07334238290786743, -0.06639233231544495, -0.026855209842324257, -0.04708512872457504, -0.051486290991306305, -0.012812959030270576, -0.18733176589012146, -0.11331570148468018, -0.026058249175548553, -0.34011515974998474, -0.0430002436041832, -0.05182783678174019, -0.04275326058268547, -0.06606875360012054, -0.11449927091598511, -0.02516813576221466, -0.1285754144191742, -0.1330518126487732, -0.024289753288030624, -0.11548824608325958, -0.17062193155288696, -0.034397855401039124, -0.18340763449668884, -0.15368521213531494, -0.034512780606746674, -0.20925408601760864, -0.1306755244731903, -0.06768888980150223, -0.28184470534324646, -0.13752509653568268, -0.0786244124174118, -0.04514646530151367, -0.39290258288383484, -0.2635403275489807, -0.045952703803777695, -0.387648344039917, -0.25716474652290344, -0.02554829604923725]]
r50_vars=[[7.188324451446533, 1.4733529090881348, 0.29812899231910706, 0.010335711762309074, 0.1730586737394333, 0.3073308765888214, 0.10100433230400085, 0.0020679098088294268, 0.2878609001636505, 0.18354201316833496, 0.0033904616720974445, 0.2549631595611572, 0.20029790699481964, 0.0028325810562819242, 0.08603799343109131, 0.6182273030281067, 0.25935620069503784, 0.00324944406747818, 0.30228954553604126, 0.14980459213256836, 0.0027814575005322695, 0.21630488336086273, 0.062132395803928375, 0.003972809761762619, 0.28157004714012146, 0.274664044380188, 0.0009016380645334721, 0.030361881479620934, 0.29652202129364014, 0.14555580914020538, 1.8619197108533339e-19, 0.20337721705436707, 0.1383809894323349, 0.0007864565122872591, 0.20322436094284058, 0.13497868180274963, 0.0014409484574571252, 0.1747989058494568, 0.09853790700435638, 0.002120410557836294, 0.1875949651002884, 0.08431866019964218, 0.0043889619410037994, 0.1887180060148239, 0.14502303302288055, 0.0029665993060916662, 0.009406034834682941, 0.46527761220932007, 0.11237895488739014, 0.0031956101302057505, 0.3055599331855774, 0.0973263680934906, 0.005817086435854435], [7.397983551025391, 1.419447898864746, 0.25543877482414246, 0.011283221654593945, 0.17209230363368988, 0.28333890438079834, 0.08942501991987228, 0.003774186596274376, 0.20326262712478638, 0.12824474275112152, 0.005255251191556454, 0.19275692105293274, 0.15845254063606262, 0.007111229933798313, 0.08943462371826172, 0.6191472411155701, 0.22917689383029938, 0.00580039294436574, 0.2957257032394409, 0.11426860839128494, 0.0050496188923716545, 0.16605618596076965, 0.07825757563114166, 0.005685871466994286, 0.2378963977098465, 0.22387373447418213, 0.00415052380412817, 0.0325559601187706, 0.29027891159057617, 0.11461643129587173, 0.003184499451890588, 0.19933286309242249, 0.10287605226039886, 0.0023409295827150345, 0.17725646495819092, 0.09904968738555908, 0.003600910771638155, 0.1377505660057068, 0.07557640969753265, 0.0040217917412519455, 0.15409226715564728, 0.06661366671323776, 0.005905043799430132, 0.13758233189582825, 0.11911969631910324, 0.007595080882310867, 0.017251506447792053, 0.3759780526161194, 0.06920583546161652, 0.0051027280278503895, 0.2890310287475586, 0.10151410102844238, 0.009231433272361755], [7.169163227081299, 1.2983720302581787, 0.14592345058918, 0.029520856216549873, 0.1728520691394806, 0.21546737849712372, 0.057362571358680725, 0.006088660564273596, 0.152449369430542, 0.1569073349237442, 0.0077225989662110806, 0.1878524124622345, 0.15942956507205963, 0.013564459048211575, 0.10310649871826172, 0.549042820930481, 0.05845649912953377, 0.007974918000400066, 0.16741709411144257, 0.061025410890579224, 0.0067056757397949696, 0.12634563446044922, 0.11702093482017517, 0.007151404395699501, 0.21155348420143127, 0.18362931907176971, 0.013375476002693176, 0.0408908985555172, 0.2102140486240387, 0.04493240267038345, 0.003841258818283677, 0.14851222932338715, 0.050962526351213455, 0.004425863269716501, 0.1223459392786026, 0.07751431316137314, 0.006071729119867086, 0.11074243485927582, 0.09794864803552628, 0.006190402898937464, 0.13125377893447876, 0.06670242547988892, 0.007265955675393343, 0.15769615769386292, 0.10612227022647858, 0.014731474220752716, 0.027132226154208183, 0.30968940258026123, 0.12247125804424286, 0.0066927345469594, 0.32668405771255493, 0.10843189060688019, 0.009767686016857624], [7.468334197998047, 1.2176684141159058, 0.14744853973388672, 0.03187411651015282, 0.17310090363025665, 0.16183722019195557, 0.06555332243442535, 0.009493974968791008, 0.14388395845890045, 0.14993910491466522, 0.010633638128638268, 0.1959417164325714, 0.14640140533447266, 0.017829623073339462, 0.10562388598918915, 0.20991867780685425, 0.1209055632352829, 0.008977938443422318, 0.11866968870162964, 0.10563929378986359, 0.010287413373589516, 0.11189810931682587, 0.15173111855983734, 0.007722535170614719, 0.23828110098838806, 0.13556712865829468, 0.022246412932872772, 0.050616901367902756, 0.15883387625217438, 0.09952215105295181, 0.0061311181634664536, 0.11266996711492538, 0.10695983469486237, 0.006466712802648544, 0.11067728698253632, 0.12677423655986786, 0.008478602394461632, 0.09729653596878052, 0.11510705202817917, 0.007631044834852219, 0.12233829498291016, 0.07367715984582901, 0.008619549684226513, 0.176730215549469, 0.10135887563228607, 0.01579923927783966, 0.030494969338178635, 0.29571908712387085, 0.1294260174036026, 0.006831655278801918, 0.3318614363670349, 0.10998199880123138, 0.009820826351642609]]
r56_means=[[0.009795176796615124, 0.07316964864730835, -0.004168625921010971, -0.09321027994155884, 0.0001543387770652771, -0.1500256061553955, -0.0006201551295816898, -0.1581169217824936, -0.003483300097286701, -0.048333995044231415, -0.0011426499113440514, -0.10524696111679077, 0.010356634855270386, -0.20468680560588837, -0.003910634201020002, -0.17200759053230286, -0.0015564914792776108, -0.03183125704526901, 0.0027709081768989563, -0.13682544231414795, -0.004443794954568148, -0.06366464495658875, -0.007161667570471764, -0.10864880681037903, -0.0031317060347646475, -0.10658780485391617, -0.008152063004672527, -0.19601118564605713, -0.0025358772836625576, -0.19094382226467133, -0.0005718475440517068, -0.22699624300003052, -0.0016851041000336409, -0.18590393662452698, -0.005961225368082523, -0.03798497095704079, -0.003083116840571165, -0.038457728922367096, -0.021407384425401688, -0.089304119348526, -0.009794358164072037, -0.14429780840873718, -0.0062317755073308945, -0.1564766764640808, -0.004648004658520222, -0.15842004120349884, -0.006521353963762522, -0.15859389305114746, -0.005332272965461016, -0.17543494701385498, -0.002557708416134119, -0.18235093355178833, -0.00379821565002203, -0.10088160634040833, 0.003444305155426264], [0.00976688601076603, 0.056463487446308136, -0.007210662588477135, -0.09323246777057648, 0.0024880431592464447, -0.10106375813484192, 0.003775473451241851, -0.12135936319828033, -0.001854486996307969, -0.04544336721301079, -0.0012592484708875418, -0.09428836405277252, 0.010936787351965904, -0.18439359962940216, -0.003530167043209076, -0.1613256335258484, 0.0033236476592719555, -0.03712715953588486, 0.0008772136643528938, -0.11841139942407608, 0.00034157000482082367, -0.053860172629356384, -0.007723410613834858, -0.06831037998199463, -0.004412123002111912, -0.08486699312925339, -0.0047551426105201244, -0.13656045496463776, -0.002714234869927168, -0.14050185680389404, 0.00013585819397121668, -0.12989798188209534, -0.0014188755303621292, -0.11458148062229156, -0.005274980794638395, -0.03099754825234413, -0.0032789306715130806, -0.048303477466106415, -0.021977288648486137, -0.04812793806195259, -0.01471068523824215, -0.11682069301605225, -0.006699824705719948, -0.11569751799106598, -0.006494814530014992, -0.13294237852096558, -0.007181750610470772, -0.1292710155248642, -0.005392732564359903, -0.15087249875068665, -0.0020980604458600283, -0.15208494663238525, -0.004683528561145067, -0.10527171194553375, 0.002599662635475397], [0.009832956828176975, 0.04377536475658417, -0.02378511242568493, -0.09768642485141754, -0.00038729002699255943, -0.10474702715873718, 0.0044791908003389835, -0.10580458492040634, -0.0023286775685846806, -0.03755496069788933, -0.001490466995164752, -0.10569845139980316, 0.011004531756043434, -0.14318734407424927, -0.0035476083867251873, -0.12652470171451569, 0.003556673415005207, -0.05822807550430298, 0.0022225757129490376, -0.11566168814897537, -0.001138370018452406, -0.05363260582089424, -0.008990488946437836, -0.0812484622001648, -0.005628623533993959, -0.08607325702905655, -0.006200803443789482, -0.11328177154064178, -0.0024787145666778088, -0.14555977284908295, -0.0005284989019855857, -0.1389760971069336, -0.0009486021008342505, -0.1295143961906433, -0.00516662560403347, -0.028636876493692398, -0.0030044117011129856, -0.05938539281487465, -0.02050507441163063, -0.06464514136314392, -0.017278410494327545, -0.11523641645908356, -0.007283878978341818, -0.11597844213247299, -0.007267346139997244, -0.12743768095970154, -0.007798411883413792, -0.11763404309749603, -0.0050683100707829, -0.13979966938495636, -0.003079324495047331, -0.1457676887512207, -0.005490848794579506, -0.12426656484603882, 0.0015991737600415945], [0.006747809238731861, 0.04376348853111267, -0.024640047922730446, -0.10187415778636932, -0.0034905122593045235, -0.10758431255817413, 0.005267485976219177, -0.11452944576740265, -0.00269608898088336, -0.04315515235066414, -0.0005062490236014128, -0.13196823000907898, 0.009367450140416622, -0.16184526681900024, -0.0028007747605443, -0.15837764739990234, 0.002950384747236967, -0.07682862877845764, -0.001534629613161087, -0.1424248069524765, -0.002735862508416176, -0.061635054647922516, -0.007702953182160854, -0.10141933709383011, -0.004812159109860659, -0.11153102666139603, -0.005827703513205051, -0.1343918740749359, -0.0019764546304941177, -0.17015883326530457, -0.00017031983588822186, -0.16647833585739136, -0.00017640582518652081, -0.1497698873281479, -0.0051545435562729836, -0.03254900500178337, -0.0022892700508236885, -0.06808464229106903, -0.02009427174925804, -0.07098729908466339, -0.01664908416569233, -0.11928226053714752, -0.006605998612940311, -0.11780551075935364, -0.007001074030995369, -0.12509554624557495, -0.007503660395741463, -0.11710728704929352, -0.004852957092225552, -0.134890615940094, -0.00320423417724669, -0.14294025301933289, -0.004900689236819744, -0.13513383269309998, 0.0015390886692330241]]
r56_vars =[[0.5190925598144531, 0.028423044830560684, 0.008469484746456146, 0.02042321488261223, 0.009162338450551033, 0.04145069047808647, 0.0022574919275939465, 0.023776262998580933, 0.0021402679849416018, 0.02876548282802105, 0.0008298380998894572, 0.034102410078048706, 0.0032209742348641157, 0.05246394872665405, 0.0008850122103467584, 0.03522835671901703, 0.0007921452634036541, 0.04750063642859459, 0.006068226415663958, 0.05872233957052231, 0.004438820760697126, 0.0677216649055481, 0.0013639205135405064, 0.0712452083826065, 0.000969139626249671, 0.052210330963134766, 0.0009164139628410339, 0.06350814551115036, 0.000575247744563967, 0.05243567377328873, 0.00045496877282857895, 0.08104798197746277, 0.0005574483657255769, 0.08087734878063202, 0.0006592640420421958, 0.01138315163552761, 0.00020087033044546843, 0.05488494038581848, 0.0026221792213618755, 0.12049531936645508, 0.0005049105384387076, 0.09910020232200623, 0.0003799956466536969, 0.08985184133052826, 0.0003144968650303781, 0.07752540707588196, 0.0002946815802715719, 0.0723080188035965, 0.0002456373767927289, 0.06722629815340042, 0.00016254442743957043, 0.08306913077831268, 0.00012071241508238018, 0.11336992681026459, 0.00030528532806783915], [0.535454273223877, 0.021205363795161247, 0.00683086272329092, 0.01788719743490219, 0.009703060612082481, 0.024859381839632988, 0.0020050746388733387, 0.02078396826982498, 0.0018008800689131021, 0.01815466769039631, 0.0007535513723269105, 0.02474268153309822, 0.003578690579161048, 0.04554297775030136, 0.0007355160778388381, 0.035373248159885406, 0.0008020887617021799, 0.05606615170836449, 0.00514580262824893, 0.04895331338047981, 0.005412193015217781, 0.04074426740407944, 0.0009727354045026004, 0.03669530153274536, 0.0009179054759442806, 0.03474605828523636, 0.0007346405764110386, 0.04094415530562401, 0.0005101787974126637, 0.028361022472381592, 0.0004774624831043184, 0.039977606385946274, 0.00047233173972927034, 0.034954775124788284, 0.000586612441111356, 0.007659362629055977, 0.00018514471594244242, 0.04452838376164436, 0.004853155463933945, 0.06901811063289642, 0.0006320412503555417, 0.0624539852142334, 0.0004116224590688944, 0.048761166632175446, 0.0003958356101065874, 0.052641257643699646, 0.0003101279435213655, 0.04584970697760582, 0.0002469785395078361, 0.04678032547235489, 0.00014807515253778547, 0.056781381368637085, 0.0003145757655147463, 0.1060626357793808, 0.0008865119889378548], [0.516187310218811, 0.011372940614819527, 0.008766966871917248, 0.018818797543644905, 0.009050327353179455, 0.01973831281065941, 0.0019199904054403305, 0.023394789546728134, 0.0015818907413631678, 0.011596261523663998, 0.0007123535615392029, 0.03438407927751541, 0.0033525649923831224, 0.02862289547920227, 0.000828656367957592, 0.03301544487476349, 0.0008178226999007165, 0.08210036158561707, 0.004439433570951223, 0.04808275029063225, 0.005347147583961487, 0.028036046773195267, 0.001100161112844944, 0.03265535086393356, 0.0008104399894364178, 0.02490953542292118, 0.000790417892858386, 0.024768244475126266, 0.0005217831349000335, 0.02636079303920269, 0.00046880042646080256, 0.03465813770890236, 0.00047736664419062436, 0.033644236624240875, 0.000529000535607338, 0.005816694814711809, 0.0001809830719139427, 0.060199327766895294, 0.005089934915304184, 0.057428762316703796, 0.0007119205547496676, 0.04891037940979004, 0.0003995330771431327, 0.04871998727321625, 0.0003982622001785785, 0.04657883197069168, 0.00034449738450348377, 0.03820979967713356, 0.00025152694433927536, 0.03843878582119942, 0.00021047219343017787, 0.04861529916524887, 0.000615272088907659, 0.11624323576688766, 0.0009677016641944647], [0.5216809511184692, 0.011296105571091175, 0.006890091113746166, 0.02047758176922798, 0.007914235815405846, 0.019426017999649048, 0.0016380565939471126, 0.025156719610095024, 0.0013555532786995173, 0.01249754149466753, 0.00035146813024766743, 0.03967882692813873, 0.0029073371551930904, 0.030415765941143036, 0.0005521698622033, 0.0347573459148407, 0.0006932439864613116, 0.10319364070892334, 0.0043203020468354225, 0.05626619979739189, 0.005063327960669994, 0.03490426763892174, 0.000756244407966733, 0.04000062495470047, 0.0005186735070310533, 0.03100290708243847, 0.0005432907491922379, 0.030048584565520287, 0.00031971308635547757, 0.03263193741440773, 0.0002916174125857651, 0.043316714465618134, 0.0003419616841711104, 0.04214581847190857, 0.0004053572192788124, 0.007311119697988033, 0.00010139141522813588, 0.07443453371524811, 0.0051180594600737095, 0.06615777313709259, 0.0006332488264888525, 0.0557217001914978, 0.0003311325272079557, 0.05326053500175476, 0.00034136773319914937, 0.05036836490035057, 0.0003328931634314358, 0.040281884372234344, 0.00023911605239845812, 0.03892131149768829, 0.00021744206605944782, 0.04969486966729164, 0.0006292560137808323, 0.12054229527711868, 0.0009712517494335771]]

markers = ['o','P','s','>','D','^']
colors = ['#d53e4f',
'#3288bd',
'#66c2a5',
'#e6f598',
]

line_plot(None, r50_means,['SubNet0','SubNet1','SubNet2','SubNet3'],colors,
		'/home/bo/Dropbox/Research/CVPR23/images/compare_r50means.eps','# Batch Normalization Layer','Running Mean')
line_plot(None, r50_vars,['SubNet0','SubNet1','SubNet2','SubNet3'],colors,
		'/home/bo/Dropbox/Research/CVPR23/images/compare_r50vars.eps','# Batch Normalization Layer','Running Var')
line_plot(None, r56_means,['SubNet0','SubNet1','SubNet2','SubNet3'],colors,
		'/home/bo/Dropbox/Research/CVPR23/images/compare_r56means.eps','# Batch Normalization Layer','Running Mean')
line_plot(None, r56_vars,['SubNet0','SubNet1','SubNet2','SubNet3'],colors,
		'/home/bo/Dropbox/Research/CVPR23/images/compare_r56vars.eps','# Batch Normalization Layer','Running Var')

for arr in [r50_means,r50_vars,r56_means,r56_vars]:
	arr = np.array(arr)
	norm_var_per_layer = np.std(arr,axis=0)#/np.mean(arr,axis=0)
	print(norm_var_per_layer.mean())


