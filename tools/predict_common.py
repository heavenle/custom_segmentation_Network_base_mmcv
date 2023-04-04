# -*- coding: UTF-8 -*-
__author__ = "yukun"
__data__ = "2023.4.4"
__description__ = "predict road image and convert to vector"


import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from argparse import ArgumentParser
import mmcv
from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
from mmseg.core.evaluation import get_palette, eval_metrics
import numpy as np
import cv2
from osgeo import ogr, gdal, gdalconst, osr
import glob
import sys
print(os.getcwd())
sys.path.append(os.getcwd())
from mmcv_custom import *

num_classes = 2


def tif2shp(outFile, outputPath):
    outdataset = gdal.Open(outFile)
    inband = outdataset.GetRasterBand(1)
    inband.SetNoDataValue(0)
    maskband = inband.GetMaskBand()
    prj = osr.SpatialReference()
    prj.ImportFromWkt(outdataset.GetProjection())

    outshp = outputPath[:-4] + ".shp"
    drv = ogr.GetDriverByName("ESRI Shapefile")
    if os.path.exists(outshp):
        drv.DeleteDataSource(outshp)
    Polygon = drv.CreateDataSource(outshp)
    Poly_layer = Polygon.CreateLayer(outputPath[:-4], srs=prj, geom_type=ogr.wkbMultiPolygon)

    newField = ogr.FieldDefn('value', ogr.OFTReal)
    Poly_layer.CreateField(newField)
    gdal.FPolygonize(inband, maskband, Poly_layer, 0)
    Polygon.SyncToDisk()
    del outdataset


def main():
    parser = ArgumentParser()
    parser.add_argument('--img',
                        default=r'/media/DATA/liyi/project/MmcvMutilModel/data',
                        help='Image file')
    parser.add_argument('--save_path',
                        default=r'./liyi_20221011_GuoWang3',
                        help='Image file')
    parser.add_argument('--config',
                        default=r'./GuoWangTongBuild_512and120/segformer.b5.512x512.guowangtongBuild.160k.py',
                        help='Config file')
    parser.add_argument('--checkpoint',
                        default=r'./GuoWangTongBuild_512and120/latest.pth',
                        help='Checkpoint file')
    parser.add_argument('--eval_m',
                        action="store_true",
                        help='metrics mIou')
    parser.add_argument(
        '--device', default='cuda:3', help='Device used for inference')
    parser.add_argument(
        '--palette',
        default='voc',
        help='Color palette used for segmentation map')
    parser.add_argument(
        '--need_gd',
        action='store_true',
        help='if need_gd==True, the img will add gradient of img. Channel will be 4')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_segmentor(args.config, args.checkpoint, device=args.device)
    mmcv.mkdir_or_exist(args.save_path)
    img_list = []
    ext = ["*.tif", "*.tiff"]
    for ex in ext:
        img_list += glob.glob(os.path.join(args.img, ex))
    for img_path in img_list:
        print("this img is :", img_path)
        predict_big_test_img(model, img_path, args.palette, args.eval_m, args.save_path, args.need_gd, slide_window_size=512)
        print("Start converting the img[{}] to vector....".format(os.path.splitext(os.path.basename(img_path))[0]+ '.tif'))
        tif2shp(os.path.join(args.save_path, os.path.splitext(os.path.basename(img_path))[0]+ '_pred.tif'), 
                os.path.join(args.save_path, os.path.splitext(os.path.basename(img_path))[0]+ '_pred.shp'))
        print("Finished converting the img[{}] to vector....".format(os.path.splitext(os.path.basename(img_path))[0]+ '.tif'))                


def write_array_to_tif_init(image_path, output_path, band_size):
    dataset = gdal.Open(image_path)
    width = dataset.RasterXSize
    height = dataset.RasterYSize
    #         outbandsize = dataset.RasterCount
    geo_trans = dataset.GetGeoTransform()
    projection = dataset.GetProjection()

    format = "GTiff"
    tiff_driver = gdal.GetDriverByName(format)
    output_ds = tiff_driver.Create(output_path, width, height,
                                   band_size, gdalconst.GDT_Byte)
    output_ds.SetGeoTransform(geo_trans)
    output_ds.SetProjection(projection)
    for band_index in range(band_size):
        output_ds.GetRasterBand(band_index + 1).SetNoDataValue(0)
    return output_ds


def predict_big_test_img(model, img_path, palette, eval_m, save_path, need_gd, slide_window_size=512, overlap_rate=0.25):
    dataset = gdal.Open(img_path)
    width = dataset.RasterXSize
    height = dataset.RasterYSize
    img = dataset.ReadAsArray(0, 0, width, height)
    img = img.transpose(1, 2, 0)
    height, width, band = img.shape

    output_img = np.zeros((height, width))
    file_name = os.path.splitext(os.path.basename(img_path))[0]

    output_tif = write_array_to_tif_init(img_path, os.path.join(save_path, file_name + '_pred.tif'), 1)
    overlap_pixel = int(slide_window_size * (1 - overlap_rate))
    if height - slide_window_size < 0:  # 判断x是否超边界，为真则表示超边界
        x_idx = [0]
    else:
        x_idx = [x for x in range(0, height - slide_window_size + 1, overlap_pixel)]
        if x_idx[-1] + slide_window_size > height:
            x_idx[-1] = height - slide_window_size
        else:
            x_idx.append(height - slide_window_size)
    if width - slide_window_size < 0:
        y_idx = [0]
    else:
        y_idx = [y for y in range(0, width - slide_window_size + 1, overlap_pixel)]
        if y_idx[-1] + slide_window_size > width:
            y_idx[-1] = width - slide_window_size
        else:
            y_idx.append(width - slide_window_size)
    # ----------------------------------------------------------------------#
    #                判断下x,y的尺寸问题，并且设置裁剪大小，方便后续进行padding。
    # ----------------------------------------------------------------------#
    cut_width = slide_window_size
    cut_height = slide_window_size

    if height - slide_window_size < 0 and width - slide_window_size >= 0:  # x小，y正常
        cut_width = slide_window_size
        cut_height = height
        switch_flag = 1
    elif height - slide_window_size < 0 and width - slide_window_size < 0:  # x小， y小
        cut_width = width
        cut_height = height
        switch_flag = 3
    elif height - slide_window_size >= 0 and width - slide_window_size < 0:  # x正常， y小
        cut_height = slide_window_size
        cut_width = width
        switch_flag = 2
    elif height - slide_window_size >= 0 and width - slide_window_size >= 0:
        switch_flag = 0
    # ----------------------------------------------------------------------#
    #                开始滑框取图，并且获取检测框。
    # ----------------------------------------------------------------------#
    total_progress = len(x_idx) * len(y_idx)
    count = 0
    for x_start in x_idx:
        for y_start in y_idx: 
            count += 1
            print("Start predict the img[{}], the prograss is [{}%]".format(file_name+".tif", round((count/total_progress)*100, 2)))
            croped_img = img[x_start:x_start + cut_height, y_start:y_start + cut_width]
            if band > 3:
                croped_img = croped_img[:, :, 0:3]
            # ----------------------------------------------------------------------#
            #                依据switch_flag的设置，进行padding。
            # ----------------------------------------------------------------------#
            temp = np.zeros((slide_window_size, slide_window_size, 3), dtype=np.uint8)
            if switch_flag == 1:
                # temp = np.zeros((croped_img.shape[0], cut_height, croped_img.shape[2]), dtype=np.uint8) #此为遥感图像
                temp[0:cut_height, 0:croped_img.shape[1], :] = croped_img
                croped_img = temp
            elif switch_flag == 2:
                # temp = np.zeros((cut_size, croped_img.shape[1], croped_img.shape[2]), dtype=np.uint8)
                temp[0:croped_img.shape[0], 0:cut_width, :] = croped_img
                croped_img = temp
            elif switch_flag == 3:
                temp[0:cut_height, 0:cut_width, :] = croped_img
                croped_img = temp

            # ----------------------------------------------------------------------#
            #                开始检测。
            # ----------------------------------------------------------------------#
            croped_img = croped_img.astype(np.uint8)
            if need_gd:
                gray_img = cv2.cvtColor(croped_img, cv2.COLOR_BGR2GRAY)
                gx_img = cv2.Sobel(gray_img, -1, 0, 1)
                gy_img = cv2.Sobel(gray_img, -1, 1, 0)
                sobel_img = gx_img + gy_img
                gd_img = np.zeros((croped_img.shape[0], croped_img.shape[1], croped_img.shape[2] + 1)).astype("uint8")
                gd_img[:, :, 0:3] = croped_img
                gd_img[:, :, 3] = sobel_img
                result = inference_segmentor(model, gd_img)
            else:
                result = inference_segmentor(model, croped_img)

            output_img[x_start:x_start + cut_height, y_start:y_start + cut_width] = result[0]
            output_tif.GetRasterBand(1).WriteArray(result[0].astype(np.uint8), y_start, x_start)

    if eval_m:
        l_dataset = gdal.Open(os.path.join(os.path.dirname(os.path.dirname(img_path)), "labels", os.path.basename(img_path)))
        label = l_dataset.ReadAsArray(0, 0, l_dataset.RasterXSize, l_dataset.RasterYSize)
        all_acc, acc, iou = eval_metrics(
            output_img,
            label,
            num_classes,
            ignore_index=255,
            metrics='mIoU',
            nan_to_num=-1)
        print("Img[{}]==>Iou[{}]==>mIou[{}]==>all_acc[{}]".format(os.path.basename(img_path), iou, 
                                                        iou.sum() / num_classes,
                                                        all_acc))
        with open(os.path.join(save_path, "img_iou.txt"), "a") as F:
            F.write("Img[{}]==>Iou[{}]==>mIou[{}]==>all_acc[{}]".format(os.path.basename(img_path), iou,
                                                        iou.sum() / num_classes,
                                                        all_acc))
            F.write("\n")
if __name__ == '__main__':
    main()
