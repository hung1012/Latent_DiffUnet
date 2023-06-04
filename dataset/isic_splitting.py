import os
import pandas as pd 
import shutil



df = pd.read_csv('/home/admin_mcn/thaotlp/data/ISIC_part3B/ISBI2016_ISIC_Part3B_Training_GroundTruth.csv')

name_list = df.iloc[:,0].tolist()
data_path = '/home/admin_mcn/thaotlp/data/ISIC_part3B/ISBI2016_ISIC_Part3B_Training_Data'
dest_image_dir = '/home/admin_mcn/thaotlp/data/ISIC/image'
dest_mask_dir = '/home/admin_mcn/thaotlp/data/ISIC/mask'



for index in range(len(name_list)):
    name = name_list[index]
    img_path = os.path.join(data_path, name) + '.jpg'
    dest_img_path = os.path.join(dest_image_dir, name) + '.jpg'
    shutil.copyfile(img_path, dest_img_path)

    mask_name = name_list[index]
    msk_path = os.path.join(data_path, mask_name) + '_Segmentation.png'
    dest_msk_path = os.path.join(dest_mask_dir, name) + '.jpg'
    shutil.copyfile(msk_path, dest_msk_path)



