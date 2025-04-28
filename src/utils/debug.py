# ============================================================
#  C O P Y R I G H T
# ------------------------------------------------------------
#  Copyright (c) 2025 by Robert Bosch GmbH. All rights reserved.
# 
#  The reproduction, distribution and utilization of this file as
#  well as the communication of its contents to others without express
#  authorization is prohibited. Offenders will be held liable for the
#  payment of damages. All rights reserved in the event of the grant
#  of a patent, utility model or design.
# ============================================================

import time

import os

# try:
def setup_env_paths():
    environment = {}
    for path in os.listdir(os.environ["AZUREML_CR_DATA_CAPABILITY_PATH"]):
        if 'INPUT' in path:
            key = path.replace('INPUT', 'AZUREML_DATAREFERENCE').upper().split('_')
            key = '_'.join(key[:(-len(key))//2+1])
            print(key)
            environment[key] = os.path.join(os.environ["AZUREML_CR_DATA_CAPABILITY_PATH"], path)+'/'
            key = key.lower().replace('AZUREML_DATAREFERENCE'.lower(), 'AZUREML_DATAREFERENCE')
            environment[key] = os.path.join(os.environ["AZUREML_CR_DATA_CAPABILITY_PATH"], path)+'/'
    environment['PYTHONPATH'] = os.environ['AZUREML_CR_EXECUTION_WORKING_DIR_PATH']
    return environment

environment = setup_env_paths()
home_directory = os.path.expanduser('~')
print(home_directory)
if os.path.exists(home_directory+'/.bashrc'):
    with open(home_directory+'/.bashrc', 'r') as f:
        bashrc = f.read()
else:
    bashrc = ''
for key in environment:
    if key not in bashrc:
        if len(bashrc)==0 or bashrc[-1]!='\n':
            bashrc += '\n'
        bashrc += f'export {key}="{environment[key]}"'
# print(bashrc)
with open(home_directory+'/.bashrc', 'w') as f:
    f.write(bashrc)
    
# +
with open('/etc/environment', 'r') as f:
    bashrc = f.read()
    for key in environment:
        if key not in bashrc:
            if bashrc[-1]!='\n':
                bashrc += '\n'
            bashrc += f'{key}="{environment[key]}"'
with open('/etc/environment', 'w') as f:
    f.write(bashrc)

time.sleep(1000000)
# except:
#     pass

