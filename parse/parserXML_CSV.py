import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET


def xml_to_csv(xml,img):
    xml_list = []
    for xml_file in glob.glob(xml + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            value = (root.find('filename').text+'.jpg',
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     member[0].text,
                     int(member[4][0].text),
                     int(member[4][1].text),
                     int(member[4][2].text),
                     int(member[4][3].text)                     
                     )
            xml_list.append(value)
#    column_name = ['filename', 'xmin', 'ymin', 'xmax', 'ymax', 'class']
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']

    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df


def main():
    xml_path = '/media/abade/Dados/DataSets/NemaDataSet/ObjectDetection/Val20/data/xml'
    img_path = '/media/abade/Dados/DataSets/NemaDataSet/ObjectDetection/Val20/images/'
    xml_df = xml_to_csv(xml_path,img_path)
    xml_df.to_csv('/media/abade/Dados/DataSets/NemaDataSet/ObjectDetection/Val20/data/tfrecord/annotations.csv', index=None)
    print('Successfully converted xml to csv.')


main()
