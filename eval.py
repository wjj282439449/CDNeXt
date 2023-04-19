import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import GradScaler
from torch.autograd import Variable
from torch.utils import data as data_
from tqdm import tqdm
from models.cdnext import get_cdnext
from PIL import Image
from data_utils.data_loader import *

import matplotlib.pyplot as plt
from multiprocessing import Process
from prettytable import PrettyTable
import glob
# ONLY_TRAIN_FC = True
labelNameList = ["change"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#dataset path setting
ROOTDIR = {
    # "LEVIR-CD+": r"datasets/LEVIR-CD+",
    # "S2Looking": r"datasets/S2Looking",
    # "SYSU-CD": r"datasets/SYSU-CD",
    # "DSIFN-CD": r"../../datasets/DSIFN-CD",
    "LEVIR-CD": r"../../datasets/LEVIR-CD",
}
backboneName = "tiny" #'tiny','small','base','resnet18'
SIMULATE_BATCH_SIZE = 32#128
BATCH_SIZE = 16
accumulate_steps = SIMULATE_BATCH_SIZE//BATCH_SIZE
train_num_workers = 2
test_num_workers = 2
saveDatasetName = "-".join(ROOTDIR.keys())
# save_dir = "checkpoint/LEVIR-CD+"
save_dir = "checkpoint/"+saveDatasetName
total_epoch = 400
use_cuda = True
num_classes = len(labelNameList)
result_dir = "results/"+saveDatasetName
lossType = "balance ce" #"ce" means cross entropy, "focal" means focal loss ,"balance ce" means no parameters ce
learning_rate = 4e-5
itersDisplayMetrics = [ "Acc", "Pre", "Rec", "IoU", "TNR", "Loss"]
epochsDisplayMetrics = [ "Acc", "Pre", "Rec", "IoU", "TNR", "F1", "Kappa", "Loss"]
plot_metrics =  ["F1", "IoU", "Loss"]
plot_metrics.sort()
stage = ["train", "val"]
stage.sort()
use_amp = True
model_path = r"..\UConvNext\checkpoint\LEVIR-CD\*.pth"

def setup_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    # torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def data_loader(ROOTDIR, mode="train", taskList=labelNameList, miniScale=1, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, total_fold = 5, valid_fold=5):
    dataset = WSIDataset(root_dir=ROOTDIR, mode=mode, taskList=taskList, total_fold = total_fold, valid_fold=valid_fold, miniScale=miniScale)
    dataloader = data_.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=train_num_workers,
        shuffle=shuffle,
        drop_last=drop_last,
        pin_memory=True#载入内存提高速度
    )
    return dataset, dataloader

def save_checkpoints(model, step):
    if os.path.exists(save_dir) == False: 
        os.makedirs(save_dir)
    filename = "CDNet_" + str(step) + "_" + "-".join(labelNameList) + ".pth"
    torch.save(model.state_dict(), os.path.join(save_dir, filename))
    print("        Save checkpoint {} to {}. \n".format(step.split("_")[0], filename))

def validate(model, index, valDataLoader, criterion, scaler, minival = False):
    showRowLen = num_classes + 1
    epochMetricsList = ["TP", "FP", "TN", "FN", "Acc", "Pre", "Rec", "IoU", "TNR", "F1", "Kappa", "Loss"]
    epochMetrics = {key: [0]*showRowLen for key in epochMetricsList}
    dir_fold = result_dir + os.sep + str(index) 
    model.eval()
    with torch.no_grad():
        for batch_idx, (img1, img2, label1, label2, img_change, dir) in tqdm(enumerate(valDataLoader)): 
            img1, img2, img_change = img1.float(), img2.float(), img_change.float()
            if use_cuda:
                img1 = img1.to(device)
                img2 = img2.to(device)
                img_change = img_change.to(device)

            # reset gradients
            if scaler == None:
                output_change = model(img1, img2)
                loss = criterion_N_with_AdvSup([output_change], [img_change], model.AdvSupResult, criterion)
            else:
                with torch.cuda.amp.autocast():
                    output_change = model(img1, img2)
                    loss = criterion_N_with_AdvSup([output_change], [img_change], model.AdvSupResult, criterion)

            output_change = F.softmax(output_change, dim=1)
            confusionMatrix(epochMetrics, [output_change[:,1,:,:].detach()], [img_change.detach()])
            epochMetrics["Loss"][0] += loss.detach().cpu().item()
            result_change = [result_dir + os.sep + os.sep + i + '_change.png' for i in dir]
            output_change = output_change[:,1,:,:].detach().float()
            output_change[output_change>=0.5] = 255
            output_change[output_change<0.5] = 0
            output_change = output_change.cpu().numpy()
            for i in range(output_change.shape[0]):
                temp_output_change = Image.fromarray(output_change[i])
                temp_output_change.convert("RGB").save(result_change[i])
        epochMetrics["Loss"][0] = round(epochMetrics["Loss"][0] / len(valDataLoader), 5)
        calculateMatrix(epochMetrics)
        printTable(epochMetrics, epochsDisplayMetrics, labelNameList)
    returnMetrics = {key: epochMetrics[key][-1] for key in plot_metrics}
    return returnMetrics

def criterion_N_with_AdvSup(pred, label, AdvSupResult, criterion):
    count = 0
    loss = 0.0
    for name, func in criterion.items():
        if func != None:
            count += 1
            loss_temp = 0.0
            if name == "focalloss":
                loss_temp = func(pred[0], label[0].long())#.detach()
            elif name == "adversialsupervised" and AdvSupResult != []:
                batch_size = label[0].size(0)
                for i in  AdvSupResult:
                    unchange = func(i[:,0], 1-label[0].view(batch_size, -1).mean(dim=1)).mean()
                    change = func(i[:,1], label[0].view(batch_size, -1).mean(dim=1)).mean()
                    loss_temp += (unchange+change)/2
                loss_temp = loss_temp/len(AdvSupResult)
                loss_temp = 0.1*loss_temp
            elif name == "bce":
                loss_temp = func(pred[0],label[0].long()).mean()
            elif name == "diceloss":
                loss_temp = 0.1 * func(pred[0][:,1,:,:], label[0]).mean()#.detach()
            loss += loss_temp
    return  loss/count

def criterion_N(pred, label, criterion):
    loss_c = criterion[0](pred[0].permute(0,2,3,1).reshape(-1,2),label[0].reshape(-1).long()).mean()
    for i in range(1, 2):
        loss_c += criterion[i](pred[0],label[0].long())
    loss = loss_c
    return loss/len(criterion)

def confusionMatrix(metrics, pred, label, threshold=0.5):
    for i in range(len(label)):
        singlePred = (pred[i] >= threshold).byte()
        singleLabel = (label[i] > threshold).byte()
        plus = singlePred + singleLabel
        FN = (singlePred < singleLabel).sum()
        FP = (singlePred > singleLabel).sum()
        TP = (plus == 2).sum()
        TN = (plus == 0).sum()
        metrics["TN"][i] = metrics["TN"][i]+TN.cpu().item()
        metrics["FP"][i] = metrics["FP"][i]+FP.cpu().item()
        metrics["FN"][i] = metrics["FN"][i]+FN.cpu().item()
        metrics["TP"][i] = metrics["TP"][i]+TP.cpu().item()
        return 

def calculateMatrix(metrics):
    for i in range(len(metrics["Acc"])-1):
        TN = metrics["TN"][i]
        FP = metrics["FP"][i]
        FN = metrics["FN"][i]
        TP = metrics["TP"][i]
        metrics["Acc"][i] = (TP+TN)/(TP+TN+FP+FN)
        metrics["Pre"][i] = round(TP/(TP+FP+0.0001)*100, 3)
        metrics["Rec"][i] = round(TP/(TP+FN+0.0001)*100, 3)
        metrics["IoU"][i] = round(TP/(TP+FP+FN)*100, 3)
        metrics["TNR"][i] = round(TN/(TN+FP)*100, 3)
        metrics["F1"][i] = round(2*TP/(2*TP+FP+FN)*100, 3)
        Pe = ((TP+FP)*(TP+FN)+(TN+FN)*(TN+FP))/(TP+FP+TN+FN)/(TP+FP+TN+FN)
        metrics["Kappa"][i] = round((metrics["Acc"][i]-Pe)/(1-Pe)*100, 3)
        metrics["Acc"][i] = round(metrics["Acc"][i]*100, 3)
    for k,v in metrics.items():
        metrics[k][-1] = sum(metrics[k][0:-1])/(len(metrics["Acc"])-1)

def printTable(metrics, displayMetrics, labelName):
    labelNameCopy = labelName + ["average"]
    table = PrettyTable(["",] + displayMetrics)
    for i in range(len(metrics["Acc"])-1):
        row = [labelNameCopy[i]] + [metrics[key][i] for key in displayMetrics]
        table.add_row(row)
    print(table)

def saveMetricsPlot(metrics, plot_metrics, stage):
    stage_name = list(metrics.keys())
    stage_name.sort()
    epochs = len(list(metrics.values())[0])

    x = range(epochs)
    len_stage = 1
    len_metric = len(plot_metrics)
    fig, axs = plt.subplots(len_metric, len_stage, dpi=600, figsize=(10,10))#
    for index, each_subplot in enumerate(plot_metrics):
        if each_subplot == "Loss":
            axs[index].set_ylim(0, 0.05)
        else:
            axs[index].set_ylim(70, 100)
        color = ['b','r','g','c','m','y','k','w']
        for each_stage in stage:
            axs[index].plot(x, metrics[each_subplot + " " + each_stage], label=each_stage)
        axs[index].set_title(each_subplot)
        axs[index].legend()
    fig.savefig(save_dir + os.sep + "results.png", bbox_inches="tight")
    plt.close("all")

def creatPlotProcess(metric_record,plot_metrics,stage):
    plotProcess = Process(target=saveMetricsPlot, args=(metric_record,plot_metrics,stage))
    plotProcess.start()
    plotProcess.join()
 
def main():
    # setup_seed(42)
    # # setup_seed(3407)
    # model = get_cdnext(out_channels=2, backbone_scale=backboneName, pretrained=True, backbone_trainable=True).cuda()
    # modelParams = filter(lambda p: p.requires_grad, model.parameters())
    # # print("hello")
    # print(model)
    # model.load_state_dict(torch.load(model_path))
    # model.to(device).eval()
    # optimizer = optim.AdamW(modelParams, lr=learning_rate, weight_decay=0.05, amsgrad=True)
    # if use_amp == True:
    #     scaler = GradScaler()
    # else:
    #     scaler = None
    # valDataset, valDataloader = data_loader(ROOTDIR, mode="test", taskList=labelNameList, miniScale=1,
    #                                         batch_size=BATCH_SIZE * 2, shuffle=False, drop_last=False,
    #                                         total_fold = 5, valid_fold = 5)
    # if lossType == "balance ce":
    #     criterion = {
    #         "focalloss": None,#FocalLoss().to(device), 
    #         "adversialsupervised": None,#nn.L1Loss().to(device), 
    #         "bce": nn.CrossEntropyLoss().to(device),#None,
    #         "diceloss": None,#DiceLoss().to(device),
    #     }
    # start_i = 0
    # metric_record = {(plot_metrics[i//len(stage)]+" "+stage[i%len(stage)]):[] for i in range(len(stage)*len(plot_metrics))}
    # print(" =====> epoch: {}, learning:{:.7f}, train and valid metrics: ".format(
    #                 90,  optimizer.state_dict()['param_groups'][0]['lr']))

    # val_avg_metric = validate(model, 90, valDataloader, criterion, scaler, minival=False)
    # for key in val_avg_metric.keys():
    #     metric_record[key+" val"].append(val_avg_metric[key])
    ckmodels = glob.glob(model_path)
    for i in ckmodels:
        print(i)
        setup_seed(42)
        model = get_cdnext(out_channels=2, backbone_scale=backboneName, pretrained=True, backbone_trainable=True).cuda()
        modelParams = filter(lambda p: p.requires_grad, model.parameters())
        # print("hello")
        # print(model)
        model.load_state_dict(torch.load(i))
        model.to(device).eval()
        optimizer = optim.AdamW(modelParams, lr=learning_rate, weight_decay=0.05, amsgrad=True)
        if use_amp == True:
            scaler = GradScaler()
        else:
            scaler = None
        valDataset, valDataloader = data_loader(ROOTDIR, mode="test", taskList=labelNameList, miniScale=1,
                                                batch_size=BATCH_SIZE * 2, shuffle=False, drop_last=False,
                                                total_fold = 5, valid_fold = 5)
        if lossType == "balance ce":
            criterion = {
                "focalloss": None,#FocalLoss().to(device), 
                "adversialsupervised": None,#nn.L1Loss().to(device), 
                "bce": nn.CrossEntropyLoss().to(device),#None,
                "diceloss": None,#DiceLoss().to(device),
            }
        start_i = 0
        metric_record = {(plot_metrics[i//len(stage)]+" "+stage[i%len(stage)]):[] for i in range(len(stage)*len(plot_metrics))}
        print(" =====> epoch: {}, learning:{:.7f}, train and valid metrics: ".format(
                        90,  optimizer.state_dict()['param_groups'][0]['lr']))

        val_avg_metric = validate(model, 90, valDataloader, criterion, scaler, minival=False)
        for key in val_avg_metric.keys():
            metric_record[key+" val"].append(val_avg_metric[key])
if __name__ == "__main__":
    main()
