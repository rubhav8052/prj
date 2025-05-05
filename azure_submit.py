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

# ============================================================================================================
# C O P Y R I G H T
# ------------------------------------------------------------------------------------------------------------
# \copyright (C) 2023 Robert Bosch GmbH and Cariad SE. All rights reserved.
# ============================================================================================================

import os
os.environ.pop('MSI_ENDPOINT', None)

import yaml
import sys

from azureml.core import Environment, Experiment, ScriptRunConfig
from azureml.core.runconfig import DataReferenceConfiguration, MpiConfiguration, ApplicationEndpointConfiguration
from azureml.core.workspace import Workspace

if __name__ == '__main__':

    run_config_path = sys.argv[1] if len(sys.argv) > 1 else 'run_configs/default.yaml'
    with open(run_config_path, 'r') as stream:
        cfg = yaml.safe_load(stream)

    source_directory = f'{os.path.dirname(os.path.abspath(__file__))}/{cfg["SRC"]}'
    input_ds_names = ['vdeepingestprod']

    ws = Workspace(workspace_name='offline_perception', subscription_id='c4f1c7f3-9206-409f-a333-5b89a516e5dd', resource_group='vdeep-ct-prod')
    ws.get_details()

    version = int(cfg["ENV"].split(':')[1]) if ':' in cfg["ENV"] else None
    curated_env = Environment.get(workspace=ws, name=cfg["ENV"].split(':')[0], version=version)

    exp = Experiment(workspace=ws, name=cfg["EXPERIMENT"])

    script_arguments = []
    if cfg["SCRIPT_ARGUMENTS"]:
        for key, value in cfg["SCRIPT_ARGUMENTS"].items():
            script_arguments.append('--'+key.lower())
            script_arguments.append(value)

    src = ScriptRunConfig(
        source_directory=source_directory,
        script=cfg["ENTRY_SCRIPT"],
        compute_target=cfg["COMPUTE"],
        arguments=script_arguments,
        environment=curated_env,
        distributed_job_config=MpiConfiguration(process_count_per_node=1, node_count=cfg["NODES"]) if cfg["NODES"]>1 else None
    )

    input_ds = [
        ws.datastores[ds].path(path=None, data_reference_name=ds).as_mount()
        for ds in input_ds_names
    ]
    src.run_config.data_references = {
        inp.data_reference_name: inp.to_config()
        for inp in input_ds
    }
    blob_store_conf = DataReferenceConfiguration("workspaceblobstore")
    src.run_config.data_references["workspaceblobstore"] = blob_store_conf
    
    if cfg["DEBUG"]:
        src.run_config.services = {
            # 'VSCode': ApplicationEndpointConfiguration('VSCode'),
            'Jupyter': ApplicationEndpointConfiguration('Jupyter', port=8722),                                  
        }

    run = exp.submit(config=src)
    if cfg["RUN_NAME"]:
        run.display_name = f'{cfg["RUN_NAME"]}'

    details = run.get_details()
    print(f"The run {details['runId']} was deployed successfully.")
    print(f"Portal url: {run.get_portal_url()}")
