import os


os.chdir("/home/zhangky/tool/swit_real")

# #### prepare configs and submit RL jobs
repeat_names=["repeat1","repeat2","repeat3"]#
rl_py_names=["create_rl_config.py"] 
task_name="rock1_model2" ### Change it to your jobname
high_score=20  ### Change it corresponding to your docking software
high_mw=600    ### Change it corresponding to your target
for rdx,rl_py in enumerate(rl_py_names):
#in weight_names:
    for repeat_n in repeat_names:
        cmd=f"python {rl_py} {repeat_n} {task_name} {high_score} {high_mw}"
        os.system(cmd)
        print(cmd)
        
#       ### submit RL
        subfol_name=f'{task_name}_{repeat_n}'
        cmd=f"sbatch -J {subfol_name} run_sbatch.sh {task_name} {repeat_n}"          ####### check run_sbatch.sh
        print(cmd)
        os.system(cmd)
