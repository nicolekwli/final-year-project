import torch
import numpy as np
import time
import os
import matplotlib.pyplot as plt
from torchvision import utils as u
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from PIL import Image
import torchvision

## own modules
from GIM.vision.data import get_dataloader
from GIM.vision.arg_parser import arg_parser
from GIM.vision.models import load_vision_model
from GIM.utils import logger, utils


# def visualise(train_loader):
#     for idx, (img, target) in enumerate(train_loader):
#         if (idx == 1):
#             model_input = img.to(opt.device)

#             if opt.model_type == 2:  ## fully supervised training
#                 _, _, z = model(model_input)
#             else:
#                 with torch.no_grad():
#                     _, _, z, _ = model(model_input, target) # z is the loss values
#                 z = z.detach() #double security that no gradients go to representation learning part of model

def visTensor(name, tensor, ch=0, allkernels=False, nrow=10, padding=1): 
    n,c,w,h = tensor.shape

    if allkernels: tensor = tensor.view(n*c, -1, w, h)
    elif c != 3: tensor = tensor[:,ch,:,:].unsqueeze(dim=1)

    rows = np.min((tensor.shape[0] // nrow + 1, 64))    
    grid = u.make_grid(tensor, nrow=nrow, normalize=True, padding=padding)
    plt.figure( figsize=(nrow,rows) )

    #Image.fromarray(grid).convert('RGB').resize((150, 300)).save(name)

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

def getData(test_loader):
    examples = enumerate(test_loader)
    batch_idx, (example_data, example_targets) = next(examples)


    rdm_data = example_data[:50]
    rdm_targets = example_targets[:50]

    classes = np.zeros([10,1])
    #print(classes)

    data = [[torch.empty(32,1,64,64) for _ in range(1)] for _ in range(10)]
    #print(data)

    for idx, (img, label) in enumerate(test_loader):
        for l_idx, l in enumerate(label):
            if (classes[l] == 0):
                classes[l] = 1

                x = torch.empty(1,1,64,64)
                a = img[l_idx]
                a = a.unsqueeze(0)
                for i in range(31):
                    
                    
                    x = torch.cat([x, a], dim=0)
                data[l] = x

    return data



if __name__ == "__main__":
    opt = arg_parser.parse_args()

    model, _ = load_vision_model.load_model_and_optimizer(
        opt, reload_model=True, calc_loss=False
    )

    _, _, train_loader, train_dataset, test_loader, test_dataset = get_dataloader.get_dataloader(opt)

    model.eval()

    # print(model.module.encoder[1].model.Conv2.conv1.weight.data.clone())

    # 0 corresponds fully black color, and 255 corresponds to the white color
    # dark patches have a lower weight than the brighter patches

    # code to visualise weights of each layer - gim  -------------------------------
    # filter1_data = model.module.encoder[0].model.Conv1.conv1.weight.data.clone()
    # filter2_data = model.module.encoder[1].model.Conv2.conv1.weight.data.clone()

    # visTensor("filter1_visual.png", filter1_data, ch=0, allkernels=False)
    # visTensor("filter2_visual.png", filter2_data, ch=0, allkernels=False)

    # visEncoderGIM(model, test_loader)

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


    # RDM --------------------------------------------------------------------------

    # idx = train_dataset.train_labels==1
    # train_dataset.targets = train_dataset.targets[idx]
    # print(train_dataset.targets)
    # train_dataset.data = train_dataset.data[idx]

    # print(train_dataset.data)

    # print(train_dataset.data.shape)

    data = getData(test_loader)

    # thing to store activation results of each class
    # then used to compare and get correlation
    outputs = []

    for c in range(10):

        for step, img in enumerate(data):
            #model_input = img.reshape([1,1,64,64])
            if (step == 1):

                #isTensor("original.png", model_input )
                
                n_patches_x, n_patches_y = None, None

                loss, _, output, accuracies = model(img, step)

        output = output[1:2, :, :, :].detach()
        # output = output.squeeze()# remove dimansion 1

        outputs.append(output)


    visTensor("test0.png", outputs[0])
    visTensor("test4.png", outputs[4])


    RDM = torch.empty(1,64,49,49)

    for row in range(10): # needs to be 10

        for col in range(10):
            a = outputs[row]
            b = outputs[col]

            c = torch.empty(1,1,49,49) 

            for i in range(64):

                # print(a.shape)
                aa = a[:,i,:,:]
                aa = aa.view(49,1)

                bb = b[:,i,:,:]
                # b = b.reshape(-1,1)
                bb = bb.view(1,49)


                corr = torch.matmul(aa,bb)
                corr = corr.reshape([1,1,49,49])

                c = torch.cat([corr, c], dim=0)

            c = c[1:, :, :, :]
            c = c.reshape(1,64,49,49)

            

            RDM = torch.cat([RDM, c], dim=0)

    RDM = RDM[1:, :, :, :]
    print(RDM.shape)
    visTensor("test.png", RDM)
    

    #     RDM.append(temp)


    # # [1, 64, 7, 7]
    # final = torch.empty(10,64,7,7) 
    # for i in range(10):
    #     final = torch.cat([final, a], dim=0)