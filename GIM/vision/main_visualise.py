import torch
import numpy as np
import time
import os
import matplotlib
import matplotlib.pyplot as plt
from torchvision import utils as u
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from PIL import Image
import torchvision
import math as m

## own modules
from GIM.vision.data import get_dataloader
from GIM.vision.arg_parser import arg_parser
from GIM.vision.models import load_vision_model
from GIM.utils import logger, utils

def visTensor(name, tensor, ch=0, allkernels=False, nrow=10, padding=1): 
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

    # conv1_out = model.conv1.forward(number.cuda())
    # output = model.module.encoder[0].model.Conv1.conv1.forward(number)
    for step, (img, label) in enumerate(train_loader):
        if (step == 1):
            model_input = img
            visTensor("original.png", img )

            filter1_data = model.module.encoder[0].model.Conv1.forward(img)
            # filter1_data = F.relu(model.module.encoder[0].model.Conv1.conv1.forward(img))
            
            model_input = filter1_data.detach()
            name = "res1_out_1.png" 
            visTensor(name, filter1_data.detach().clone() )

            filter2_data = model.module.encoder[1].model.Conv2.forward(filter1_data)
            # filter2_data = F.relu(model.module.encoder[1].model.Conv2.conv1.forward(filter1_data))
            print(filter1_data)
            model_input = filter2_data.detach()
            name = "res1_out_2.png"
            visTensor(name, filter2_data.detach().clone() )  

def visEncoderGreedy(model, test_loader):
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

    for step, (img, label) in enumerate(train_loader):
        if (step == 1):
            model_input = img
            visTensor("original.png", img )

            # filter1_data = model.module.encoder[0].model.Conv1.forward(img)    
            filter1_data = F.relu(model.module.encoder[0].model.Conv1.conv1.forward(img))
            model_input = filter1_data.detach()
            name = "res1_out_1.png" 
            visTensor(name, filter1_data.detach().clone() )

            # gives black
            print("b")
            print(model.module.encoder[0].model.Conv2)
            filter2_data = F.relu(model.module.encoder[0].model.Conv2.conv1.forward(filter1_data))
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
    model_nums = [1,5,10,15,20,25,29]
    #model_nums = [0]

    for num in model_nums:
        opt.model_num = num

        model, _ = load_vision_model.load_model_and_optimizer(
            opt, reload_model=True, calc_loss=False
        )

        outputs = []

        # get output for each class
        for step, img in enumerate(data):
            n_patches_x, n_patches_y = None, None
            loss, _, output, accuracies = model(img, labels[step])
            output = output[1:2, :, :, :].detach()
            outputs.append(output)

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
        fig, ax = plt.subplots()
        # plt.pcolor(finals, cmap='RdBu', vmin=0, vmax=1)
        im = ax.imshow(finals, vmin=0, vmax=1)
        ax.set_xticks(np.arange(len(classes)))
        ax.set_yticks(np.arange(len(classes)))
        ax.set_xticklabels(classes)
        ax.set_yticklabels(classes)   

        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
            rotation_mode="anchor")

        for i in range(len(classes)):
            for j in range(len(classes)):
                text = ax.text(j, i, finals[i][j],
                            ha="center", va="center", color="w", fontsize="x-small")

        ax.set_title("RDM")
        fig.tight_layout()
        
        plt.savefig("RDM_{}.png".format(opt.model_num))
        plt.clf()



if __name__ == "__main__":
    opt = arg_parser.parse_args()

    model, _ = load_vision_model.load_model_and_optimizer(
        opt, reload_model=True, calc_loss=False
    )

    _, _, train_loader, train_dataset, test_loader, test_dataset = get_dataloader.get_dataloader(opt)

    model.eval()

    # getWeightsAndActivations()


    # RDM --------------------------------------------------------------------------

    data, labels = getData(test_loader)
    getAllRDMs(opt, data, labels)
    
    

