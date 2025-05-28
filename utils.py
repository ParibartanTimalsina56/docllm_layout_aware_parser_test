import sys
import numpy as np
import os 
from loguru import logger
import uuid
import pandas as pd

import json


from IPython.display import HTML
import cv2
import matplotlib.pyplot as plt

from docsumo_image_util.services.process_image import ProcessFile
from docsumo_image_util.parse import PdfImages
from docsumo_image_util.parse.ocr import ocr_data as parse_img
from docsumo_image_util.parse.ocr import ocr_data
from docsumo_image_util.parse.ocr.google import read_everything
from docsumo_image_util.parse.pdf import parse_pdf, scale_coords
from docsumo_image_util.parse.structures import OCRData

from tabframe.misc.debugging import plot_bboxes

from tabframe.api import BBox

from docsumo_image_util.parse import PdfImages, polygon_search
from docsumo_image_util.parse.ocr import ocr_data as parse_img

def get_img_and_df(file_path, ocr_reader="google", mode=None, is_image=False):

    DPI = 202
    df_list = []
    r_imgs = []
    r_angles = []
    COLUMNS = [
        "x0",
        "y0",
        "x2",
        "y2",
        "Text",
        "index_sort",
        "page",
        "block",
        "confidence",
    ]

    if is_image:
        imgs = [cv2.imread(file_path)]
    else:
        imgs = PdfImages(file_path, dpi=DPI)
        
        
    if mode=="ocr":
        logger.info("Doing OCR to read the file")
        for idx, img in enumerate(imgs):
            (df, cdf), (r_img, r_angle) = parse_img[ocr_reader](img)
            if df.empty:
                df = pd.DataFrame(df, columns=COLUMNS)
            df_list.append(df)
            r_imgs.append(r_img)
            r_angles.append(r_angle)
        return df_list, r_imgs, r_angles
    
    else:
        logger.info("Doing Digitall to read the file")
        try:
            df, cdf = parse_pdf(file_path)
            if df.empty:
                df = pd.DataFrame(df, columns=COLUMNS)
            no_of_pages = df.page.max() + 1
            for page in range(no_of_pages):
                temp_df = scale_coords(df, imgs[page].shape[1], imgs[page].shape[0], page)
                df_list.append(temp_df)
        except Exception as e:
            logger.info("Error on digital reading")
            logger.info("Doing OCR to read the file")
            for idx, img in enumerate(imgs):
                (df, cdf), (r_img, r_angle) = parse_img[ocr_reader](img)
                if df.empty:
                    df = pd.DataFrame(df, columns=COLUMNS)
                df_list.append(df)
                r_imgs.append(r_img)
                r_angles.append(r_angle)
    if not r_angles:
        r_angles = [0.0] * len(imgs)
    return df_list, r_imgs or imgs, r_angles