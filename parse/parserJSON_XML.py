import os
import xml.etree.ElementTree as ET
import pandas as pd
import cv2
import json

def write_to_xml(image_name, image_dict, data_folder, save_folder, xml_template='pascal_voc_template.xml'):
        # get bboxes
    bboxes = image_dict[image_name]
    
    # print(bboxes)
    
    # read xml file
    tree = ET.parse(xml_template)
    root = tree.getroot()    
    
    # modify
    folder = root.find('folder')
    folder.text = 'Annotations'
    
    fname = root.find('filename')
    fname.text = image_name.split('.')[0] 
    
    src = root.find('source')
    database = src.find('database')
    database.text = 'COCO2017'
    
    h = bboxes[0][5]
    w = bboxes[0][6]
    d = 3
    
    # size
    pathImg = os.path.join(data_folder, image_name)
    if os.path.isfile(pathImg) :        
        size = root.find('size')
        width = size.find('width')
        width.text = str(w)
        height = size.find('height')
        height.text = str(h)
        depth = size.find('depth')
        depth.text = str(d)
        
        for box in bboxes:
            # append object
            obj = ET.SubElement(root, 'object')
            
            name = ET.SubElement(obj, 'name')
            name.text = box[0]
            
            pose = ET.SubElement(obj, 'pose')
            pose.text = 'Unspecified'

            truncated = ET.SubElement(obj, 'truncated')
            truncated.text = str(0)

            difficult = ET.SubElement(obj, 'difficult')
            difficult.text = str(0)

            bndbox = ET.SubElement(obj, 'bndbox')
            
            xmin = ET.SubElement(bndbox, 'xmin')
            xmin.text = str(int(box[1]))
            
            ymin = ET.SubElement(bndbox, 'ymin')
            ymin.text = str(int(box[2]))
            
            xmax = ET.SubElement(bndbox, 'xmax')
            xmax.text = str(int(box[3]))
            
            ymax = ET.SubElement(bndbox, 'ymax')
            ymax.text = str(int(box[4]))
        
        # save .xml to anno_path
        anno_path = os.path.join(save_folder, image_name.split('.')[0] + '.xml')
        print(anno_path)
        tree.write(anno_path)
    
def findElement(listName, searchElement):
    for img in images:
        if img['id'] == searchElement:
            return img['file_name'], img["height"], img["width"]

# main routine
if __name__=='__main__':
    
    # read annotations file
    annotations_path = 'G:/Drives compartilhados/iA - 2021/POC Segmentação - Lucas/Bases Treinamento/train/train.json'
    
    # read coco category list
    # df = pd.read_csv('coco_categories.csv')
    # df.set_index('id', inplace=True)
    
    # specify image locations
    image_folder = 'G:/Drives compartilhados/iA - 2021/POC Segmentação - Lucas/Bases Treinamento/train'
    
    # specify savepath - where to save .xml files
    savepath = 'G:/Drives compartilhados/iA - 2021/POC Segmentação - Lucas/Bases Treinamento/train/xml'
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    
    # read in .json format
    with open(annotations_path,'rb') as file:
        doc = json.load(file)
        
    # get annotations
    annotations = doc['annotations']
    images = doc['images']
    categories = doc['categories']
    
    category_dict = {}
    for i in range(0, len(categories)):
        id = categories[i]["id"]
        name = categories[i]["name"]
        category_dict[id] = name
        # print(id , "- ", name)
    
    # iscrowd allowed? 1 for ok, else set to 0
    iscrowd_allowed = 1
    
    # initialize dict to store bboxes for each image
    image_dict = {}

    # loop through the annotations in the subset
    for anno in annotations:
        # get annotation for image name
        image_id = anno['image_id']
        category_id = anno['category_id']
        image_name, height, width = findElement(images,image_id)  
        
        # get category
        category = category_dict[category_id]
        
        # add as a key to image_dict
        if not image_name in image_dict.keys():
            image_dict[image_name]=[]
        
        # append bounding boxes to it
        box = anno['bbox']
        # since bboxes = [xmin, ymin, width, height]:
        image_dict[image_name].append([category, box[0], box[1], box[0]+box[2], box[1]+box[3],  height, width])
        
    # generate .xml files
    for image_name in image_dict.keys():
        write_to_xml(image_name, image_dict, image_folder, savepath)
        print('generated for: ', image_name)
    