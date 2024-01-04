import numpy as np
import pandas as pd
import pytesseract as tes


def get_ocr_df(
    cv2_img,
    conf_thresh=0
):
    """Given path to image with text, get a dataframe of OCR results"""

    ocr_raw = tes.image_to_data(cv2_img, output_type=tes.Output.DICT)
    ocr_df = pd.DataFrame(ocr_raw)
    ocr_df = ocr_df.apply(lambda vv: pd.to_numeric(vv, errors='ignore'))
    ocr_df = ocr_df[ocr_df['conf'] >= conf_thresh]
    ocr_df['right'] = ocr_df['left'] + ocr_df['width']
    ocr_df['bottom'] = ocr_df['top'] + ocr_df['height']
    ocr_df['x_mid'] = ocr_df.apply(
        lambda row: int(row['left'] + np.round(row['width'] / 2)),
        axis=1
        )
    ocr_df['y_mid'] = ocr_df.apply(
        lambda row: int(row['top'] + np.round(row['height'] / 2)),
        axis=1
        )

    return ocr_df
