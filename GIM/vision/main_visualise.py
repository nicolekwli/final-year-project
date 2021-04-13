import torch
import numpy as np
import time
import os
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from torchvision import utils as u
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from PIL import Image
import torchvision
import math as m
import csv
import pandas as pd
import json
import io
import sys
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import colors


## own modules
from GIM.vision.data import get_dataloader
from GIM.vision.arg_parser import arg_parser
from GIM.vision.models import load_vision_model
from GIM.utils import logger, utils

def visTensor(name, tensor, ch=0, allkernels=False, nrow=4, padding=1): 
    n,c,w,h = tensor.shape

    if allkernels: tensor = tensor.view(n*c, -1, w, h)
    elif c != 3: tensor = tensor[:,ch,:,:].unsqueeze(dim=1)

    rows = np.min((tensor.shape[0] // nrow + 1, 64))    
    grid = u.make_grid(tensor, nrow=nrow, normalize=True, padding=padding)
    plt.figure( figsize=(nrow,rows) )

    plt.imsave(name, grid.cpu().numpy().transpose((1, 2, 0)))



def visEncoder(model, test_loader):
    examples = enumerate(test_loader)
    batch_idx, (example_data, example_targets) = next(examples)

    number = example_data[6]
    number.unsqueeze_(0)
    number = Variable(number, requires_grad=True)

    n_patches_x, n_patches_y = None, None

    # conv1_out = model.conv1.forward(number.cuda())
    # output = model.module.encoder[0].model.Conv1.conv1.forward(number)
    for step, (img, label) in enumerate(train_loader):
        if (step == 1):
            model_input = img
            visTensor("original.png", img )
            for idx, module in enumerate(model.module.encoder[: 2+1]):
                print(idx)
                print(module)
                h, z, cur_loss, cur_accuracy, n_patches_x, n_patches_y = module(
                    model_input, n_patches_x, n_patches_y, label
                )
                model_input = z.detach()

                name = "encode" + str(idx) + ".png"
                visTensor(name, h.detach().clone() )

def visEncoderGIM(model, test_loader):
    examples = enumerate(test_loader)
    batch_idx, (example_data, example_targets) = next(examples)

    number = example_data[6]
    number.unsqueeze_(0)
    number = Variable(number, requires_grad=True)

    n_patches_x, n_patches_y = None, None
    patch_size = 16
    overlap = 2

    # conv1_out = model.conv1.forward(number.cuda())
    # output = model.module.encoder[0].model.Conv1.conv1.forward(number)
    for step, (img, label) in enumerate(train_loader):
        if (step == 1):
            x = img
            x = (
                x.unfold(2, patch_size, patch_size // overlap)
                .unfold(3, patch_size, patch_size // overlap)
                .permute(0, 2, 3, 1, 4, 5)
            )
            n_patches_x = x.shape[1]
            n_patches_y = x.shape[2]
            x = x.reshape(
                x.shape[0] * x.shape[1] * x.shape[2], x.shape[3], x.shape[4], x.shape[5]
            )

            model_input = x
            visTensor("original.png", x )

            filter1_data = model.module.encoder[0].model.Conv1.forward(img)
            # filter1_data = F.relu(model.module.encoder[0].model.Conv1.conv1.forward(img))
            
            model_input = filter1_data.detach()
            name = "res1_out_1.png" 
            visTensor(name, filter1_data.detach().clone() )

            _,_,filter2_data,_ = model(img, label)
            # filter2_data = model.module.encoder[1].model.Conv2.forward(model_input)
            # filter2_data = F.relu(model.module.encoder[1].model.Conv2.conv1.forward(filter1_data))
            name = "res1_out_2.png"
            visTensor(name, filter2_data.detach().clone())  

def visEncoderGreedy(model, test_loader):
    examples = enumerate(test_loader)
    batch_idx, (example_data, example_targets) = next(examples)

    number = example_data[6]
    number.unsqueeze_(0)
    number = Variable(number, requires_grad=True)

    n_patches_x, n_patches_y = None, None
    patch_size = 16
    overlap = 2

    # conv1_out = model.conv1.forward(number.cuda())
    # output = model.module.encoder[0].model.Conv1.conv1.forward(number)
    for step, (img, label) in enumerate(train_loader):
        if (step == 1):
            x = img
            x = (
                x.unfold(2, patch_size, patch_size // overlap)
                .unfold(3, patch_size, patch_size // overlap)
                .permute(0, 2, 3, 1, 4, 5)
            )
            n_patches_x = x.shape[1]
            n_patches_y = x.shape[2]
            x = x.reshape(
                x.shape[0] * x.shape[1] * x.shape[2], x.shape[3], x.shape[4], x.shape[5]
            )
            model_input = x
            visTensor("original.png", x )

            filter1_data = model.module.encoder[0].model.Conv1.forward(img)
            #filter1_data = model.module.encoder[0].model.Conv1.conv1.forward(img)


            
            model_input = filter1_data.detach()
            name = "res1_out_1.png" 
            visTensor(name, filter1_data.detach().clone() )

            filter2_data = model.module.encoder[1].model.Conv2.forward(filter1_data)
            # filter2_data = model.module.encoder[1].model.Conv2.conv1.forward(filter1_data)
            #patches
            model_input = filter2_data.detach()
            name = "res1_out_2.png"
            visTensor(name, filter2_data.detach().clone() )  
                
def visEncoderEnd(model, test_loader):
    examples = enumerate(test_loader)
    batch_idx, (example_data, example_targets) = next(examples)

    number = example_data[6]
    number.unsqueeze_(0)
    number = Variable(number, requires_grad=True)

    n_patches_x, n_patches_y = None, None
    patch_size = 16
    overlap = 2

    for step, (img, label) in enumerate(train_loader):
        if (step == 1):
            x = img
            x = (
                x.unfold(2, patch_size, patch_size // overlap)
                .unfold(3, patch_size, patch_size // overlap)
                .permute(0, 2, 3, 1, 4, 5)
            )
            n_patches_x = x.shape[1]
            n_patches_y = x.shape[2]
            x = x.reshape(
                x.shape[0] * x.shape[1] * x.shape[2], x.shape[3], x.shape[4], x.shape[5]
            )
            model_input = x
            visTensor("original.png", x )

            filter1_data = model.module.encoder[0].model.Conv1.forward(img)    
            #filter1_data = F.relu(model.module.encoder[0].model.Conv1.conv1.forward(img))
            model_input = filter1_data.detach()
            name = "res1_out_1.png" 
            visTensor(name, filter1_data.detach().clone() )

            # gives black

            filter2_data = model.module.encoder[0].model.Conv2.forward(model_input) 
            # gives numbers
            #filter2_data = model.module.encoder[0].model.Conv2.conv1.forward(model_input)
            model_input = filter2_data.detach()
            name = "res1_out_2.png"
            visTensor(name, filter2_data.detach().clone() )  

def getWeightsAndActivations():
    # 0 corresponds fully black color, and 255 corresponds to the white color
    # dark patches have a lower weight than the brighter patches

    # code to visualise weights of each layer - gim  -------------------------------
    filter1_data = model.module.encoder[0].model.Conv1.conv1.weight.data.clone()
    filter2_data = model.module.encoder[1].model.Conv2.conv1.weight.data.clone()

    visTensor("filter1_visual.png", filter1_data, ch=0, allkernels=False)
    visTensor("filter2_visual.png", filter2_data, ch=0, allkernels=False)

    visEncoderGIM(model, test_loader)

    # code to visualise weights of each layer - gim supervised -------------------------------
    # filter1_data = model.module.encoder[0].model.Conv1.conv1.weight.data.clone()
    # filter2_data = model.module.encoder[1].model.Conv2.conv1.weight.data.clone()

    # visTensor("filter1_visual_30_gs.png", filter1_data, ch=0, allkernels=False)
    # visTensor("filter2_visual_30_gs.png", filter2_data, ch=0, allkernels=False)


    # visEncoderGreedy(model, test_loader)

    # code to visualise cpc filters -------------------------------
    # filter1_data = model.module.encoder[0].model.Conv1.conv1.weight.data.clone()
    # filter2_data = model.module.encoder[0].model.Conv2.conv1.weight.data.clone()

    # visTensor("filter1_visual_30.png", filter1_data, ch=0, allkernels=False)
    # visTensor("filter2_visual_30.png", filter2_data, ch=0, allkernels=False)

    # code to visualise fully supervised filters -------------------------------
    # filter1_data = model.module.encoder[0].model.Conv1.conv1.weight.data.clone()
    # filter2_data = model.module.encoder[0].model.Conv2.conv1.weight.data.clone()

    # visTensor("filter1_visual.png", filter1_data, ch=0, allkernels=False)
    # visTensor("filter2_visual.png", filter2_data, ch=0, allkernels=False)

    #visEncoderEnd(model, test_loader)

    # visualise the one or two encoder
    # visEncoder(model, test_loader)

def getData(test_loader):
    examples = enumerate(test_loader)
    batch_idx, (example_data, example_targets) = next(examples)

    rdm_data = example_data[:50]
    rdm_targets = example_targets[:50]

    classes = np.zeros([10,1])

    data = [[torch.empty(32,1,64,64) for _ in range(1)] for _ in range(10)]
    labels = [[] for i in range(10)]

    for idx, (img, label) in enumerate(test_loader):
        for l_idx, l in enumerate(label):
            if (classes[l] == 0):
                lab = torch.tensor([l], dtype=torch.long)
                
                classes[l] = 1

                x = torch.empty(1,1,64,64)
                y = torch.empty(1)
                a = img[l_idx]
                a = a.unsqueeze(0)
                for i in range(32):
                    x = torch.cat([x, a], dim=0)
                    y = torch.cat([y,lab], dim=0)
                x = x[1:, :, :, :]
                y = y[1:]
                data[l] = x
                labels[l] = y

    for i, l in enumerate(labels):
        labels[i] = l.long()
    return data, labels

def getAllRDMs(opt, data, labels):
    # model_nums = [1,5,10,15,20,25,29]
    model_nums = [1,15,29]
    fig, ax = plt.subplots(1,3)

    for idx,num in enumerate(model_nums):
        opt.model_num = num

        model, _ = load_vision_model.load_model_and_optimizer(
            opt, reload_model=True, calc_loss=False
        )

        outputs = []

        # get output for each class
        # for step, (img, label) in enumerate(data):
        for step, img in enumerate(data):
            n_patches_x, n_patches_y = None, None
            loss, _, output, accuracies = model(img, labels[step])
            #output = model.module.encoder[0].model.Conv1.forward(img)
            
            
            #output, _, _, _, _, _ = model.module.encoder[0].forward(img, n_patches_x, n_patches_y, labels[step])
            
            #output = output[1:2, :, :, :].detach()
            output = output.detach()
            outputs.append(output)

        print(len(outputs))
        RDM = torch.empty(1,64,49,49)

        finals = []

        for row in range(10): # needs to be 10
            a = outputs[row]
            a_mean = torch.mean(a)
            a_sub = torch.sub(a, a_mean)
            a_2 = torch.square(a_sub)
            a_sum = torch.sum(a_2)
            one_class = []
            for col in range(10):
                b = outputs[col]
                b_mean = torch.mean(b)
                b_sub = torch.sub(b, b_mean)
                b_2 = torch.square(b_sub)
                b_sum = torch.sum(b_2)
                a_b = a_sub * b_sub
                ab_sum = torch.sum(a_b)
                final = ab_sum / (m.sqrt(a_sum * b_sum))
                final = 1- final
                one_class.append(round(final.item(),1))
            finals.append(one_class)

        classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
        
        # plt.pcolor(finals, cmap='RdBu', vmin=0, vmax=1)
        im = ax[idx].imshow(finals, vmin=0, vmax=1)
        ax[idx].set_xticks(np.arange(len(classes)))
        ax[idx].set_yticks(np.arange(len(classes)))
        ax[idx].set_xticklabels(classes)
        ax[idx].set_yticklabels(classes)   

        plt.setp(ax[idx].get_xticklabels(), rotation=45, ha="right",
            rotation_mode="anchor")

        for i in range(len(classes)):
            for j in range(len(classes)):
                text = ax[idx].text(j, i, finals[i][j],
                            ha="center", va="center", color="w", fontsize="5")

        x[idx].set_title("RDM")
    fig.tight_layout()
        
    plt.savefig("RDM.png",bbox_inches = 'tight',pad_inches = 0.1)
    plt.clf()

def getGradWRTWeights():
    for num in range(30):
        opt.model_num = num

        model, _ = load_vision_model.load_model_and_optimizer(
            opt, reload_model=True, calc_loss=False
        )

        
        grads = torch.sum(model.module.encoder[0].model.Conv1.conv1.weight.grad)


def getLossPlot():
    gim = np.load('gim-logs/vision_experiment/train_loss.npy')
    cpc = np.load('cpc-logs/vision_experiment/train_loss.npy')
    fs = np.load('fs-logs/vision_experiment/train_loss.npy')
    gs = np.load('gs-logs/vision_experiment/train_loss.npy')
    plt.plot(gim[0], label="GIM")
    plt.plot(cpc[0], label="CPC")
    plt.plot(gs[0], label="Greedy Supervised")
    plt.plot(fs[0], label="Supervised")
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc="upper right")
    plt.show()

def draw_grad_weight_heatmap(data, name, title, ax ):
    data = np.array(data)
    data = np.transpose(data)
    kernels = np.arange(1, len(data)+1, 4)
    epochs = np.arange(1, len(data[0])+1, 4)

    #cmap = plt.get_cmap('PuOr')
    # # extract all colors from the .jet map
    # cmaplist = [cmap(i) for i in range(cmap.N)]
    # # create the new map
    # cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)

    # bounds = np.arange(np.min(data),np.max(data),.5)
    # idx=np.searchsorted(bounds,0)
    # bounds=np.insert(bounds,idx,0)
    # norm = BoundaryNorm(bounds, cmap.N)

    
    norm = colors.TwoSlopeNorm(vmin=np.min(data), vcenter=0, vmax=np.max(data))
    
    # im = ax.imshow(data, vmin=0, vmax=4)
    # im = ax.imshow(data, interpolation='none',norm=norm,cmap=cmap)
    im = ax.imshow(data, interpolation='none',norm=norm)
    ax.set_xticks(epochs)
    ax.set_yticks(kernels)
    ax.set_xticklabels(epochs, fontsize=8)
    ax.set_yticklabels(kernels, fontsize=8)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
        rotation_mode="anchor")
    ax.invert_yaxis()
    ax.set_title(title, fontsize='8')
    # plt.xlabel('Epoch')
    # plt.ylabel('Input')
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = ax.figure.colorbar(im, ax=ax,cax=cax)
    cbar.ax.tick_params(labelsize=8)
    
    name = name + ".png"
    return ax, im
    # plt.savefig(name)
    #plt.clf()

if __name__ == "__main__":
    opt = arg_parser.parse_args()

    # model, _ = load_vision_model.load_model_and_optimizer(
    #    opt, reload_model=True, calc_loss=False
    # )

    # _, _, train_loader, train_dataset, test_loader, test_dataset = get_dataloader.get_dataloader(opt)

    #model.eval()

    ## VISUALISE GRADIENTS WRT WEIGHTS OVER EPOCHS ----------------------------------------------------------
    #getGradWRTWeights()


    # getWeightsAndActivations()
    # examples = enumerate(test_loader)
    # batch_idx, (example_data, example_targets) = next(examples)

    # print(batch_idx)
    # print(example_data.shape)
    # idx = (examples.targets==1)
    # examples.targets = examples.targets[idx]
    # examples.data = examples.data[idx]

    #isEncoderGIM(model, test_loader)
    #visEncoderEnd(model, test_loader)


    # RDM --------------------------------------------------------------------------


    # data = []
    # labels = []
    # for i in range(10):
    #     _, _, train_loader, train_dataset, test_loader, test_dataset = get_dataloader.get_dataloader(opt)

    #     #c = torch.tensor(c, dtype=torch.int64)
    #     idx = (train_dataset.targets== i)
    #     train_dataset.targets = train_dataset.targets[idx]
    #     train_dataset.data = train_dataset.data[idx]


    #     t_loader = torch.utils.data.DataLoader(
    #         train_dataset, batch_size=32
    #     )

    #     for step, (img, label) in enumerate(t_loader):
    #         if (step == 1):
    #             data.append(img)
    #             labels.append(label)

    # getAllRDMs(opt, data, labels)
    
    # get Loss ------------------------------------------------------------------------
    # getLossPlot()

    # average grad ----------------------------------------------------------------
    # x = []
    # y = []

    # df = pd.read_csv('gim_avg_grad.csv')
    # df = pd.read_csv('gim-avg-grad_2.csv')
    
    # df = pd.read_csv('gs_conv1_grad.csv')
    # df = pd.read_csv('gs_conv2_grad.csv')

    # df = pd.read_csv('cpc-conv1-avg.csv')
    # df = pd.read_csv('cpc-conv2-avg.csv')

    # df = pd.read_csv('fs-avg-conv1.csv')
    # df = pd.read_csv('fs-avg-conv2.csv')
    
    
    # data = np.genfromtxt("gim_avg_grad.csv", delimiter=",", names=["x", "y"])
    # print(df['Step'])
    # print(df['Value'])
    # plt.plot(df['Step'], df['Value'])
    # plt.ylim([-0.04,0.04])
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.title('CPC Average Gradients')
    # plt.savefig("grad1_avg.png")


    # compare difference in gradients between models - .json ---------------------------
    csv.field_size_limit(sys.maxsize)

    result = {}
    gim_conv1_grads = np.genfromtxt("gim-csv/conv1.csv", delimiter=',')
    gs_conv1_grads = np.genfromtxt("gs-csv/conv1.csv", delimiter=',')
    cpc_conv1_grads = np.genfromtxt("cpc-csv/conv1.csv", delimiter=',')
    fs_conv1_grads = np.genfromtxt("fs-csv/conv1.csv", delimiter=',')

    fig, ax = plt.subplots(2,3)
    plt.set_cmap('PuOr')

    gim_gs = gim_conv1_grads - gs_conv1_grads
    ax[0,0],im = draw_grad_weight_heatmap(gim_gs, "gim_gs","gim vs greedy supervised", ax[0,0])

    gim_cpc = gim_conv1_grads - cpc_conv1_grads
    ax[0, 1], im = draw_grad_weight_heatmap(gim_cpc, "gim_cpc", "gim vs cpc", ax[0,1])

    gim_fs = gim_conv1_grads - fs_conv1_grads
    ax[0, 2], im =draw_grad_weight_heatmap(gim_fs, "gim_fs", "gim vs supervised", ax[0,2])

    cpc_gs = cpc_conv1_grads - gs_conv1_grads
    ax[1,0], im =draw_grad_weight_heatmap(cpc_gs, "cpc_gs", "cpc vs greedy supervised", ax[1,0])

    cpc_fs = cpc_conv1_grads - fs_conv1_grads
    ax[1,1], im =draw_grad_weight_heatmap(cpc_fs, "cpc_fs", "cpc vs supervised", ax[1,1])

    gs_fs = gs_conv1_grads - fs_conv1_grads
    ax[1,2], im =draw_grad_weight_heatmap(gs_fs, "gs_fs", "greedy supervised vs supervised", ax[1,2])

    fig.text(0.5, 0.04, 'Epoch', ha='center')
    fig.text(0.03, 0.5, 'Input', va='center', rotation='vertical')
    plt.subplots_adjust(left=0.1, bottom=None, right=None, top=None, wspace=0.6, hspace=0)

    plt.suptitle("Difference in gradients", y=0.9)
    plt.savefig("grad_compare.png")

    # ----------------- visualise MNSIT














    

