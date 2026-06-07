import os
import pandas as pd
from collections import Counter


task_name_lst=["rock1_model2"]
training_set=["rock1_model2_training_set.csv"] ##
number_lst=[250000]
#single_genname=["repeat1"] #
for tdx,task_name in enumerate(task_name_lst):
    file_path="/home/zhangky/tool/swit_real/examples/"+task_name+"/RL_practice/"
    os.chdir(file_path)
    file_lst=sorted(os.listdir(file_path))
    print("------------"+task_name+":")
    ##### 2.1. output the generation results of each repeat, for tss weight=1: min score=0.7; for tss weight=2: min score:0.6
    df_lst=[]
    for file in file_lst:
        if os.path.exists(file+"/results/scaffold_memory.csv"):
            df=pd.read_csv(file+"/results/scaffold_memory.csv")
            score_threshold=0.6
            test_df=df[df["total_score"]>=score_threshold]
            print(f"{file}: tss score better than 0.4:{test_df.shape}")
            rs_df=df.sort_values(by="CDock Score",ignore_index=True,ascending=False)
            rs_df["source"]=[file]*rs_df.shape[0]
            df_lst.append(rs_df)
            print(rs_df.shape)
            print(rs_df.loc[0])
            
        #     # # # ### 2.2 select best generated set among all the repeats
        # if file in single_genname:
        #     output_name=f"{task_name}_{single_genname[tdx]}"
        #     if not os.path.exists("/home/zhangky/tool/swit_real/spacelight_search_files/search_job/"+output_name):
        #         cmd="mkdir -p /home/zhangky/tool/swit_real/spacelight_search_files/search_job/"+output_name+"/"
        #         os.system(cmd)

        #     df=pd.read_csv(file+"/results/scaffold_memory.csv")
        #     score_threshold=0.4
        #     test_df=df[df["total_score"]>=score_threshold]
        #     print(f"{file}: tss score better than 0.4:{test_df.shape}")
        #     rs_df=df.sort_values(by="CDock Score",ignore_index=True,ascending=False)
        #     rs_df["source"]=[file]*rs_df.shape[0]
        #     train_df=pd.read_csv("/home/zhangky/tool/swit_real/data/"+training_set[tdx])
        #     #train_df.columns=["SMILES","score"]
        #     rs_df=rs_df.drop_duplicates(subset=["SMILES"],ignore_index=True)
        #     print(f"rs_df:{rs_df.shape}")
        #     only_in_A = rs_df[~rs_df['SMILES'].isin(train_df['SMILES'])].reset_index(drop=True)
        #     print(f"only_in_V1:{only_in_A.shape}")
        #     output_df=only_in_A.loc[0:number_lst[tdx]-1]
        #     output_df.to_csv("single_generated_set_for_searching.csv",index=False)
        #     output_df.to_csv("../../../spacelight_search_files/search_job/"+output_name+"/generated_set_for_searching.csv",index=False,columns=["SMILES","ID"],sep="\t",header=False)
        


