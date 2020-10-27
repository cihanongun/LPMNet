import matplotlib.pyplot as plt
import os
import shutil
import numpy as np
import torch
        
def clear_folder(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.mkdir(path)

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

def plotPC(pcList, show = True, save = False, name=None, figCount=9 , sizex = 12, sizey=3):
    
    if (len(np.shape(pcList)) == 3) :  pcList = [pcList] # if single array
    listCount = len(pcList)
    pIndex = 1
    
    fig=plt.figure(figsize=(sizex, sizey))
    
    for l in range(listCount):
        pc = pcList[l]
        
        for f in range(figCount):

            ax = fig.add_subplot(listCount, figCount, pIndex, projection='3d')
        
            if(np.shape(pcList[0])[2] == 4): # colors
                c_values = [colors[x-3] for x in pc[f,:,3].astype(int)]
            else:
                c_values = 'b'
            
            ax.scatter(pc[f,:,0], pc[f,:,2], pc[f,:,1], c=c_values, marker='.', alpha=0.8, s=8)

            ax.set_xlim3d(-0.25, 0.25)
            ax.set_ylim3d(-0.25, 0.25)
            ax.set_zlim3d(-0.25, 0.25)
            
            plt.axis('off')
            
            pIndex += 1
        
        plt.subplots_adjust(wspace=0, hspace=0)
        
    if(save):
        fig.savefig(name + '.png')
        plt.close(fig)
    
    if(show):
        plt.show()
    else:
        return fig
    
def interpolate(pointA, pointB, inter):
    temp = pointB - pointA
    return pointA + (inter*temp)

def interpolateArray(pointA, pointB, arraysize):
    interpolateList = []
    for i in range(arraysize):
        interpolateList.append( interpolate(pointA, pointB, (1/(arraysize-1))*(i) ) )
    return torch.stack(interpolateList)