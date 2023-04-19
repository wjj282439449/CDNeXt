from importlib.util import source_hash
import os
# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["OPENBLAS_NUM_THREADS"] = "1"
# os.environ["MKL_NUM_THREADS"] = "1"
# os.environ["NUMEXPR_NUM_THREADS"] = "1"
# os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
import sys
from datetime import datetime
import cv2
from matplotlib.pyplot import axis
import numpy as np
# from tensorboardX import SummaryWriter
import socket
from PIL import Image
import glob
import argparse
from osgeo import gdal, gdal_array
import tqdm
#labeltest labeltrain T1test T2test T1train T2train
#T1 T2 label train test
#python slicingPatch.py -i ..\..\..\datasets\S2Looking\labeltrain -o ..\..\..\datasets\S2Looking\train\label -l
#python slicingPatch.py -i ..\..\..\datasets\S2Looking\T1train -o ..\..\..\datasets\S2Looking\train\T1 
#python slicingPatch.py -i ..\..\..\datasets\S2Looking\T2train -o ..\..\..\datasets\S2Looking\train\T2 
#python slicingPatch.py -i ..\..\..\datasets\S2Looking\labeltest -o ..\..\..\datasets\S2Looking\test\label -l
#python slicingPatch.py -i ..\..\..\datasets\S2Looking\T1test -o ..\..\..\datasets\S2Looking\test\T1 
#python slicingPatch.py -i ..\..\..\datasets\S2Looking\T2test -o ..\..\..\datasets\S2Looking\test\T2 
parser = argparse.ArgumentParser(description='Using effUnet algorithm to detecting the input image, to predict change between 2 images.')
parser.add_argument("-i",  "--input_dir", default=r"E:\ChangeDetection\datasets\HRSCD-xBD\T2train", type=str, help="Path to input tif image set directory")
parser.add_argument("-o",  "--output_dir", default=r"E:\ChangeDetection\datasets\HRSCD-xBD\train\T2", type=str, help="Path to output tif image set directory")
parser.add_argument("-is",  "--img_size", default=256, type=int, help="img size")
parser.add_argument("-ol",  "--overlap_size", default=512, type=int, help="img size")#not implemented for the time being.
parser.add_argument('-c', "--multi_scale_slicing", action='store_true', default=False)
parser.add_argument('-l', "--is_label", action='store_true', default=False)
#切图逻辑，放大0.25/0.5/1/2倍
# multiScale = [0.25, 0.5, 1, 2]
multiScale = [1.0,]
def writeTiff(im_data, path, im_bands, im_height, im_width, im_geotrans, im_proj):
    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16   
    elif 'int32' in im_data.dtype.name:
        datatype = gdal.GDT_UInt32
    else:
        datatype = gdal.GDT_Float32
    if len(im_data.shape) == 3:
        im_bands, im_height, im_width = im_data.shape
    elif len(im_data.shape) == 2:
        im_data = np.array([im_data])
    else:
        im_bands, (im_height, im_width) = 1,im_data.shape
    #创建文件
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(path, im_width, im_height, 1, datatype)
    if(dataset!= None):
        dataset.SetGeoTransform(im_geotrans) #写入仿射变换参数
        dataset.SetProjection(im_proj) #写入投影
    for i in range(im_bands):
        dataset.GetRasterBand(i+1).WriteArray(im_data[i])
    del dataset

def readTiff(filename):
    img_ds = gdal.Open(filename, gdal.GA_ReadOnly)  # 打开文件
    im_width = img_ds.RasterXSize  # 栅格矩阵的列数
    im_height = img_ds.RasterYSize  # 栅格矩阵的行数
    im_bands = img_ds.RasterCount  # 波段数
    im_geotrans = img_ds.GetGeoTransform()  # 仿射矩阵，左上角像素的大地坐标和像素分辨率
    im_proj = img_ds.GetProjection()  # 地图投影信息，字符串表示
    im_data = img_ds.ReadAsArray(0, 0, im_width, im_height)
    im_data = np.zeros((img_ds.RasterYSize, img_ds.RasterXSize, img_ds.RasterCount), gdal_array.GDALTypeCodeToNumericTypeCode(img_ds.GetRasterBand(1).DataType))
    for b in range(im_data.shape[2]):
        im_data[:, :, b] = img_ds.GetRasterBand(b + 1).ReadAsArray()
    # del dataset  # 关闭对象dataset，释放内存
    return im_data, img_ds

def getAxisBoundary(Index, length,  imgsize, totalNums):
    if ((Index+1) == totalNums) and (length % imgsize != 0):
        start = length - imgsize
        end = length
    else:
        start = Index*imgsize
        end = (Index+1)*imgsize
    return start, end

def slicingSingleImg(imgDir, outputDir, imgsize=512, scales=[1.0,], isLabel = False):
    # imgDir = r"E:\ChangeDetection\datasets\HRSCD-xBD\change\D14\14-2012-0420-6905-LA93-0M50-E080.tif"
    # outputDir = r"E:\ChangeDetection\datasets\HRSCD-xBD\change\test"
    
    path, name  = os.path.split(imgDir)
    subpath, datasetName = os.path.split(path)
    _, datasetName = os.path.split(subpath)
    filename, extension = os.path.splitext(name)
    outputPath = outputDir + os.sep + datasetName + "_" + filename
    # input()
    save_extension = ".png"
    if extension in [".png",".jpg",".jpeg",".bmp"]:
        sourceImg = Image.open(imgDir)
    elif extension in ".tiff":
        sourceImg, img_ds1 = readTiff(imgDir)
        # sourceShape = sourceImg.shape
        # print(np.unique(sourceImg))
        # print(sourceImg.dtype)
        # print(np.sum(sourceImg),"/",str(sourceShape[0]*sourceShape[1]*sourceShape[2]), "=", str(np.sum(sourceImg)/(sourceShape[0]*sourceShape[1]*sourceShape[2])))
        # sourceImg[sourceImg!=0] = 255
        # print(np.sum(sourceImg),"/",str(sourceShape[0]*sourceShape[1]*sourceShape[2]), "=", str(np.sum(sourceImg)/(sourceShape[0]*sourceShape[1]*sourceShape[2])/255))
        # return 
        sourceImg = Image.fromarray(sourceImg) if sourceImg.shape[2] != 1 else Image.fromarray(sourceImg.squeeze(axis=2))
    sourceW, sourceH, sourceC = sourceImg.size[0], sourceImg.size[1], len(sourceImg.getbands())
    for singleScale in scales:
        # sourceC = len(sourceImg.getbands())
        #根据label还是image确定采样方法
        resample = Image.NEAREST if isLabel else Image.BICUBIC
        img = sourceImg.resize((int(sourceW*singleScale), int(sourceH*singleScale)), resample=resample)
        img = np.array(img)
        # print(int(sourceW*singleScale),int(sourceH*singleScale))
        # img = sourceImg.resize((int(sourceW*singleScale), int(sourceH*singleScale)), resample=resample)
        if isLabel:
            img[img!=0] = 255
            img = img.astype(np.uint8)
            # print(np.unique(img))
        #convert rgb label to binary
        if sourceC != 1 and isLabel:
            img = img.max(axis=2)
            # print("ssss", img.shape)
            # sourceC = 1
                
        img_h, img_w = img.shape[0], img.shape[1]
        h_nums, w_nums = img.shape[0] // imgsize, img.shape[1] // imgsize
        if img_h % imgsize != 0:
            h_nums = h_nums + 1
        if img_w % imgsize != 0:
            w_nums = w_nums + 1

        if img_h < imgsize or img_w <imgsize:
            continue
        # print(singleScale)
        for hIndex in tqdm.tqdm(range(h_nums)):
            # if ((hIndex+1) == h_nums) and (img_h % imgsize != 0):
            #     start_h = img_h - imgsize
            #     end_h = img_h
            # else:
            #     start_h = hIndex*imgsize
            #     end_h = (hIndex+1)*imgsize
            # print("xx", start_h, end_h)
            start_h, end_h = getAxisBoundary(hIndex, img_h, imgsize, h_nums)
            for wIndex in range(w_nums):
                # if ((wIndex+1) == w_nums) and (img_w % imgsize != 0):
                #     start_w = img_w - imgsize
                #     end_w = img_w
                # else:
                #     start_w = wIndex*imgsize
                #     end_w = (wIndex+1)*imgsize
                start_w, end_w = getAxisBoundary(wIndex, img_w, imgsize, w_nums)
                #name = pathName_x-123_y-234_imgsize-256_scale-05
                outputPathTemp = outputPath + "_scale-"+ str(singleScale) + "_y-"+ str(start_h)\
                                            + "_x-"+ str(start_w) + "_imgsize-" + str(imgsize) + save_extension 
                if os.path.exists(outputPathTemp):
                    continue
                temp_img = img[start_h:end_h, start_w:end_w, :] if sourceC != 1 and isLabel == False else img[start_h:end_h, start_w:end_w]
                # print(temp_img.shape)
                temp_img = Image.fromarray(temp_img)
                temp_img.save(outputPathTemp)
                #切目标占比小的数据集时，需要启用下面的代码，切出有目标的影像
                # if np.sum(temp_img)>0:
                #     temp_img = Image.fromarray(temp_img)
                #     temp_img.save(outputPathTemp)


                # cv2.imwrite(outputPathTemp, np.uint8(temp_img))

if __name__=="__main__":
    args = parser.parse_args()
    if args.output_dir == None:
        print("Error: without ouput dir!")
        exit(0)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    fileList = glob.glob(args.input_dir+os.sep+"*")
    print(args.input_dir)
    print(args.output_dir)
    print("is label: ", args.is_label)
    count=0
    for singleImg in fileList:
        count = count + 1
        slicingSingleImg(singleImg, args.output_dir, scales=multiScale, imgsize=args.img_size, isLabel=args.is_label)
        print("processed ", str(count), "\\", len(fileList))

    
            

        # if fileSuffix in [".png",".jpg",".jpeg",".bmp"]:
        #     result = np.array(output_change)
        #     cv2.imwrite(outputPath, np.uint8(result))
        # elif fileSuffix in ".tiff":
        #     writeTiff(output_change.unsqueeze(0).cpu().numpy(), outputPath, 1, img_ds1.RasterYSize, img_ds1.RasterXSize, img_ds1.GetGeoTransform(), img_ds1.GetProjection())

