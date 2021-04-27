import os, sys, random, shutil
import xml.etree.ElementTree as ET
import glob
import pandas as pd
from shutil import copyfile
import pandas as pd
from sklearn import preprocessing, model_selection
import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np


# img_width = 640
# img_height = 480

def width(df):
    return int(df.xmax - df.xmin)
def height(df):
    return int(df.ymax - df.ymin)
def x_center(df):
    return int(df.xmin + (df.width/2))
def y_center(df):
    return int(df.ymin + (df.height/2))
def w_norm(df):
    return df/img_width
def h_norm(df):
    return df/img_height
def norm(row, col1, col2):
    return row[col1] / row[col2]

def xml_to_csv(xml,img):
    xml_list = []
    for xml_file in glob.glob(xml + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            value = (img+root.find('filename').text+'.jpg',
                     root.find('filename').text+'.jpg',
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     int(member[4][0].text),
                     int(member[4][1].text),
                     int(member[4][2].text),
                     int(member[4][3].text), 
                     member[0].text
                     )
            # print(value)
            xml_list.append(value)

    column_name = ['path_filename','filename','imgWidth','imgHeight', 'xmin', 'ymin', 'xmax', 'ymax', 'class']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df

def addYOLOdata(df):
    # print(df.columns)
    
    le = preprocessing.LabelEncoder()
    le.fit(df['class']) 
    labels = le.fit_transform(df['class'])
    
    print(le.classes_)
    
    df['labels'] = labels
    df['width'] = df.apply(width, axis=1)
    df['height'] = df.apply(height, axis=1)

    df['x_center'] = df.apply(x_center, axis=1)
    df['y_center'] = df.apply(y_center, axis=1)
    
    '''
    OLD 
    df['x_center_norm'] = df['x_center'].apply(w_norm)
    df['width_norm'] = df['width'].apply(w_norm)

    df['y_center_norm'] = df['y_center'].apply(h_norm)
    df['height_norm'] = df['height'].apply(h_norm)
    '''
    
    df['x_center_norm'] = df.apply(norm, axis=1, col1='x_center', col2='imgWidth')
    df['width_norm']     = df.apply(norm, axis=1, col1='width',     col2='imgWidth')
    
    df['y_center_norm'] = df.apply(norm, axis=1, col1='y_center',col2='imgHeight')
    df['height_norm']    = df.apply(norm, axis=1, col1='height',   col2='imgHeight')

    return df

def genYOLO_txt(df, label_path):
    filenames = []
  
    for filename in df.filename:
        filenames.append(filename)
    
    filenames = set(filenames)

    for filename in filenames:
        yolo_list = []

        for _,row in df[df.filename == filename].iterrows():
          yolo_list.append([row.labels, row.x_center_norm, row.y_center_norm, row.width_norm, row.height_norm])

        yolo_list = np.array(yolo_list)
    
        txt_filename = os.path.join(label_path,str(row.filename.split('.')[0])+".txt")
        print(txt_filename)
        # Save the .img & .txt files to the corresponding train and validation folders
        np.savetxt(txt_filename, yolo_list, fmt=["%d", "%f", "%f", "%f", "%f"])
    print('Generated files! Check folder...')

def main():
    pathImages = "G:/Drives compartilhados/iA - 2021/POC Segmentação - Lucas/Bases Treinamento/train"
    pathXML = "G:/Drives compartilhados/iA - 2021/POC Segmentação - Lucas/Bases Treinamento/train/xml"
    pathYolo = 'G:/Drives compartilhados/iA - 2021/POC Segmentação - Lucas/Bases Treinamento/train/yolo5'

    df = xml_to_csv(pathXML, pathImages)
    
    # print(df.columns)
    
    df_yolo = addYOLOdata(df)

    #df_train, df_valid = model_selection.train_test_split(df, test_size=0.1, random_state=13, shuffle=True)
    
    if not os.path.exists(pathYolo):
        os.makedirs(pathYolo)
    #genYOLO_txt(df_yolo, pathYolo)

main()