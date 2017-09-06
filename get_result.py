import pandas as pd
import numpy as np

df_all = pd.read_csv('/kaggle/facial/fkp_lenet_output.csv',header=None,\
                        names=['left_eye_center_x',\
                               'left_eye_center_y',\
                               'right_eye_center_x',\
                               'right_eye_center_y',\
                               'left_eye_inner_corner_x',\
                               'left_eye_inner_corner_y',\
                               'left_eye_outer_corner_x',\
                               'left_eye_outer_corner_y',\
                               'right_eye_inner_corner_x',\
                               'right_eye_inner_corner_y',\
                               'right_eye_outer_corner_x',\
                               'right_eye_outer_corner_y',\
                               'left_eyebrow_inner_end_x',\
                               'left_eyebrow_inner_end_y',\
                               'left_eyebrow_outer_end_x',\
                               'left_eyebrow_outer_end_y',\
                               'right_eyebrow_inner_end_x',\
                               'right_eyebrow_inner_end_y',\
                               'right_eyebrow_outer_end_x',\
                               'right_eyebrow_outer_end_y',\
                               'nose_tip_x',\
                               'nose_tip_y',\
                               'mouth_left_corner_x',\
                               'mouth_left_corner_y',\
                               'mouth_right_corner_x',\
                               'mouth_right_corner_y',\
                               'mouth_center_top_lip_x',\
                               'mouth_center_top_lip_y',\
                               'mouth_center_bottom_lip_x',\
                               'mouth_center_bottom_lip_y'\
                              ])

df_sub = pd.read_csv('/kaggle/facial/IdLookupTable.csv',header=0)

df_all[df_all < 0] = 0
df_all[df_all > 96] = 96

for i in range(len(df_all)+1):
    if i > 0:
        temp = df_sub[df_sub.ImageId == i]
        for j in temp.loc[:,'FeatureName']:
            df_sub.loc[(df_sub['ImageId']==i) & (df_sub['FeatureName']==j),'Location']=df_all.loc[i-1,j]
        
df_sub.to_csv("/kaggle/facial/lenet_fk_result.csv")
