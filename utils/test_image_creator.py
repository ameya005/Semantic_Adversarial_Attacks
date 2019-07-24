import numpy as np
import os
import sys
import argparse
from matplotlib import pyplot as plt
from PIL import Image 

if  __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='Path to input file')
    parser.add_argument('-o', '--outdir', help='Path to output_dir')

    args = parser.parse_args()

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    
    files = os.listdir(args.input)

    for file in files:
        if file.endswith(".npy"):
            data = np.load(os.path.join(args.input,file))
            bname = os.path.splitext(os.path.basename(os.path.join(args.input,file)))[0]
            tr_data = data[:,:256,:]
            tr_data  = (tr_data - tr_data.min()) /tr_data.ptp()
            fig = plt.figure(figsize=(5,5))
            ax = plt.Axes(fig, [0.,0.,1.,1.])
            ax.set_axis_off()
            fig.add_axes(ax)
            ax.imshow(tr_data)
            plt.savefig(os.path.join(args.outdir,bname)+'.png')
            plt.close()
