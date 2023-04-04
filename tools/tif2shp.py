from osgeo import osr, ogr, gdal

import os


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


if __name__ == "__main__":
    tif2shp(r"/media/DATA/liyi/project/SegFormer-master/tools/henei_predict/zhanjiang03_gdl_pred_2.tif",
            r"/media/DATA/liyi/project/SegFormer-master/tools/henei_predict/zhanjiang03_gdl_pred_2.shp")
