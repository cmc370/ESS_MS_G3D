"""
import copy
import numpy as np
import os
import torch

a = torch.zeros((1, 2, 2, 2))
a[0, 0, :, :] = torch.eye(2, dtype=torch.float32)
a[0, 1, :, :] = torch.ones((2, 2), dtype=torch.float32)
ZeroPad = torch.nn.ZeroPad2d((1, 2, 3, 4))
input = ZeroPad(a)
print(input)

unfold = torch.nn.Unfold(kernel_size=(3, 1), dilation=(1, 1), stride=(1, 1), padding=(0, 0))
input = unfold(input)
print(input)

input = input.view(1, 2, 3, 7, 5).permute(0, 1, 3, 2, 4).contiguous()
print(input)

a = torch.tensor([[2, 3],
                  [3, 4],
                  [1, 2],
                  [5, 5],
                  [6, 4],
                  [8, 8]])
print(a.view(3, 4))
"""

# from ptflops import get_model_complexity_info
#
# model = 'model.ESS_msg3d.Model'
# component = model.split('.')
# name = __import__(component[0])
# for i in range(1, len(component)):
#     name = getattr(name, component[i])
#
# config = {'num_class': 2, 'num_point': 18, 'num_person': 1, 'num_gcn_scales': 8, 'num_g3d_scales': 8,
#           'graph': 'model.graph.AdjMatrixGraph', 'g3d_graph': 'same_in_Temporal_Graph',
#           'extension_graph': 'extension_with_influence', 'g3d_style': 'now', 'shift': True, 'in_channels': 3}
# Model = name(**config)
# macs, params = get_model_complexity_info(Model, (3, 32, 18, 1), as_strings=True,
#                                          print_per_layer_stat=True, verbose=True)
# print(macs)
# print(params)

# from sklearn.metrics import recall_score
# from sklearn.metrics import precision_score
# from sklearn.metrics import f1_score
# from sklearn.metrics import roc_curve, auc
# from torch.nn import functional as F
#
# a = np.array([[0.2, 0.3], [0.4, 0.5], [0.6, 0.1]])
# b = np.array([0, 1, 0])
# predict = a.argmax(axis=1)
# result = recall_score(y_true=b, y_pred=predict, labels=[0], average='micro')
# result_2 = precision_score(y_true=b, y_pred=predict, labels=[0], average='micro')
# result_3 = f1_score(y_true=b, y_pred=predict, labels=[0], average='micro')
# a = torch.tensor(a)
# b = torch.tensor(b)
# a_2 = F.softmax(a, dim=1)
# a_2 = a_2[:, 0]
# print(a_2)
#
# fpr, tpr, threshold = roc_curve(y_true=b, y_score=a_2, pos_label=0)
# print(fpr)
# print(tpr)
# print(threshold)
# result_4 = auc(fpr, tpr)
# print(result_4)
# print(result)
# print(result_2)
# print(result_3)

import matplotlib.pyplot as plt

ax = plt.gca()
plt.tick_params(labelsize=10)
labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
x1 = [1, 1.1, 1.2, 1.3, 1.4]
x2 = [1.2, 1.21, 1.22, 1.23, 1.25]
color = ['red', 'blue']
# plt.boxplot(x1, labels=['x1'], patch_artist=True, showfliers=False, positions=[1], showmeans=False,
#             boxprops={'color': 'black', 'facecolor': 'bisque'}, medianprops={'linestyle': '-', 'color': 'black'})
# plt.boxplot(x2, labels=['x2'], patch_artist=True, showfliers=False, positions=[2], showmeans=False,
#             boxprops={'color': 'black', 'facecolor': 'paleturquoise'}, medianprops={'linestyle': '-', 'color': 'black'})
# plt.grid(False)
# plt.show()

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.boxplot(x1, labels=['x1'], patch_artist=True, showfliers=False, positions=[1], showmeans=False,
            boxprops={'color': 'black', 'facecolor': 'bisque'}, medianprops={'linestyle': '-', 'color': 'black'})
ax2 = ax1.twinx()
ax2.boxplot(x2, labels=['x2'], patch_artist=True, showfliers=False, positions=[2], showmeans=False,
            boxprops={'color': 'black', 'facecolor': 'paleturquoise'}, medianprops={'linestyle': '-', 'color': 'black'})

plt.show()

import numpy as np
a = np.array([[2, 3],
              [5, 9]])
print(np.sum(a, axis=1))


