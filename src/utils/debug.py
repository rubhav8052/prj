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
    return environment

environment = setup_env_paths()
home_directory = os.path.expanduser('~')
print(home_directory)
with open(home_directory+'/.bashrc', 'r') as f:
    bashrc = f.read()
    for key in environment:
        if key not in bashrc:
            if bashrc[-1]!='\n':
                bashrc += '\n'
            bashrc += f'export {key}="{environment[key]}"'
with open(home_directory+'/.bashrc', 'w') as f:
    f.write(bashrc)
with open('/etc/environment', 'r') as f:
    bashrc = f.read()
    for key in environment:
        if key not in bashrc:
            if bashrc[-1]!='\n':
                bashrc += '\n'
            bashrc += f'{key}="{environment[key]}"'
with open('/etc/environment', 'w') as f:
    f.write(bashrc)
# except:
#     pass
    
time.sleep(10000000)