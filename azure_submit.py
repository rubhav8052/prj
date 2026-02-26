# # ============================================================
# #  C O P Y R I G H T
# # ------------------------------------------------------------
# #  Copyright (c) 2025 by Robert Bosch GmbH. All rights reserved.
# # 
# #  The reproduction, distribution and utilization of this file as
# #  well as the communication of its contents to others without express
# #  authorization is prohibited. Offenders will be held liable for the
# #  payment of damages. All rights reserved in the event of the grant
# #  of a patent, utility model or design.
# # ============================================================

# # ============================================================================================================
# # C O P Y R I G H T
# # ------------------------------------------------------------------------------------------------------------
# # \copyright (C) 2023 Robert Bosch GmbH and Cariad SE. All rights reserved.
# # ============================================================================================================

# import os
# os.environ.pop('MSI_ENDPOINT', None)

# import yaml
# import sys

# from azureml.core import Environment, Experiment, ScriptRunConfig
# from azureml.core.runconfig import DataReferenceConfiguration, MpiConfiguration, ApplicationEndpointConfiguration
# from azureml.core.workspace import Workspace

# if __name__ == '__main__':

#     run_config_path = sys.argv[1] if len(sys.argv) > 1 else 'run_configs/default.yaml'
#     with open(run_config_path, 'r') as stream:
#         cfg = yaml.safe_load(stream)

#     source_directory = f'{os.path.dirname(os.path.abspath(__file__))}/{cfg["SRC"]}'

#     # /////////////////////////////
#     # input_ds_names = ['workspaceblobstore']
#     input_ds_names = ['azureml_container']

#     ws = Workspace(workspace_name='attribute_labelling', subscription_id='b342ffae-3f1d-4865-9d2d-fd0f6b8061f1', resource_group='new_rg')
#     ws.get_details()

#     # ///////
#     try:
#         azureml_ds = ws.datastores['azureml_container']
#         print(f"✅ Found datastore: azureml_container")
#     except:
#         print(f"⚠️  Datastore 'azureml_container' not found, creating it...")
#         from azureml.core.datastore import Datastore
#         default_ds = ws.get_default_datastore()
#         azureml_ds = Datastore.register_azure_blob_container(
#             workspace=ws,
#             datastore_name='azureml_container',
#             container_name='azureml',
#             account_name=default_ds.account_name,
#             account_key=default_ds.account_key,
#             overwrite=True
#         )
#         print(f"✅ Created datastore: azureml_container")
#     # ///////


#     # version = int(cfg["ENV"].split(':')[1]) if ':' in cfg["ENV"] else None
#     # curated_env = Environment.get(workspace=ws, name=cfg["ENV"].split(':')[0], version=version)

#     version = int(cfg["ENV"].split(':')[1]) if ':' in cfg["ENV"] else None
#     env_name = cfg["ENV"].split(':')[0]

#     # will check whether the docker image exist or not if not it will create and register a new one
#     # //////////////////////////////////////////////////////////////////////////////////////////
#     print("\n📦 Checking for environment...")
#     try:
#         curated_env = Environment.get(workspace=ws, name=env_name, version=version)
#         print(f"✅ Using existing environment: {env_name}:{version if version else 'latest'}")
#     except:
#         print(f"⚠️  Environment not found, building from Dockerfile...")
#         curated_env = Environment.from_dockerfile(
#             name=env_name,
#             dockerfile="deployment/docker-vlm/Dockerfile"
#             # dockerfile="deployment/docker-swift/Dockerfile"
#         )
#         curated_env.register(workspace=ws)
#         print(f"✅ Environment registered from Dockerfile: {env_name}")
#     # //////////////////////////////////////////////////////////////////////////////////////


#     # ///////////////////////////////////////////////////////////////////
#     # ✅ ALWAYS BUILD FROM DOCKERFILE (don't use cached version)
#     # print("\n📦 Building environment from Docker...")

#     # env_name = "envswift"
#     # dockerfile_path = "deployment/docker-swift/Dockerfile"

#     # # Create NEW environment from Dockerfile (always)
#     # curated_env = Environment.from_dockerfile(
#     #     name=env_name,
#     #     dockerfile=dockerfile_path
#     # )
#     # print(f"✅ Environment created from Dockerfile: {dockerfile_path}")
#     # //////////////////////////////////////////////////////////////////

#     exp = Experiment(workspace=ws, name=cfg["EXPERIMENT"])

#     # script_arguments = []
#     # if cfg["SCRIPT_ARGUMENTS"]:
#     #     for key, value in cfg["SCRIPT_ARGUMENTS"].items():
#     #         script_arguments.append('--'+key.lower())
#     #         script_arguments.append(value)

#     # /////////////////////////////
#     script_arguments = []
#     if cfg["SCRIPT_ARGUMENTS"]:
#         for key, value in cfg["SCRIPT_ARGUMENTS"].items():
#             script_arguments.append(f"{key}={value}")
#     # /////////////////////////////

#     # Defines how to run your script on Azure ML compute
#     src = ScriptRunConfig(
#         source_directory=source_directory,
#         script=cfg["ENTRY_SCRIPT"],
#         compute_target=cfg["COMPUTE"],
#         arguments=script_arguments,
#         environment=curated_env,
#         distributed_job_config=MpiConfiguration(process_count_per_node=1, node_count=cfg["NODES"]) if cfg["NODES"]>1 else None
#     )

#     input_ds = [
#         ws.datastores[ds].path(path=None, data_reference_name=ds).as_mount()
#         for ds in input_ds_names
#     ]
#     src.run_config.data_references = {
#         inp.data_reference_name: inp.to_config()
#         for inp in input_ds
#     }

#     # ////////////////////
#     # blob_store_conf = DataReferenceConfiguration("workspaceblobstore")
#     # src.run_config.data_references["workspaceblobstore"] = blob_store_conf
#     # ///////////////////
    
#     src.run_config.environment_variables = {
#     'AZUREML_DATAREFERENCE_azureml_container': str(input_ds[0])
#     }

#     if cfg["DEBUG"]:
#         src.run_config.services = {
#             'VSCode': ApplicationEndpointConfiguration('VSCode'),
#             'Jupyter': ApplicationEndpointConfiguration('Jupyter', port=8722),                                  
#         }

#     run = exp.submit(config=src)
#     if cfg["RUN_NAME"]:
#         run.display_name = f'{cfg["RUN_NAME"]}'

#     details = run.get_details()
#     print(f"The run {details['runId']} was deployed successfully.")
#     print(f"Portal url: {run.get_portal_url()}")
















































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

    # /////////////////////////////
    # input_ds_names = ['workspaceblobstore']
    input_ds_names = ['azureml_container']

    ws = Workspace(workspace_name='attribute_labelling', subscription_id='b342ffae-3f1d-4865-9d2d-fd0f6b8061f1', resource_group='new_rg')
    ws.get_details()

    # ///////
    try:
        azureml_ds = ws.datastores['azureml_container']
        print(f"✅ Found datastore: azureml_container")
    except:
        print(f"⚠️  Datastore 'azureml_container' not found, creating it...")
        from azureml.core.datastore import Datastore
        default_ds = ws.get_default_datastore()
        azureml_ds = Datastore.register_azure_blob_container(
            workspace=ws,
            datastore_name='azureml_container',
            container_name='azureml',
            account_name=default_ds.account_name,
            account_key=default_ds.account_key,
            overwrite=True
        )
        print(f"✅ Created datastore: azureml_container")
    # ///////


    # version = int(cfg["ENV"].split(':')[1]) if ':' in cfg["ENV"] else None
    # curated_env = Environment.get(workspace=ws, name=cfg["ENV"].split(':')[0], version=version)

    version = int(cfg["ENV"].split(':')[1]) if ':' in cfg["ENV"] else None
    env_name = cfg["ENV"].split(':')[0]

    # will check whether the docker image exist or not if not it will create and register a new one
    # //////////////////////////////////////////////////////////////////////////////////////////
    print("\n📦 Checking for environment...")
    try:
        curated_env = Environment.get(workspace=ws, name=env_name, version=version)
        print(f"✅ Using existing environment: {env_name}:{version if version else 'latest'}")
    except:
        print(f"⚠️  Environment not found, building from Dockerfile...")
        curated_env = Environment.from_dockerfile(
            name=env_name,
            dockerfile="deployment/docker-vlm/Dockerfile"
            # dockerfile="deployment/docker-swift/Dockerfile"
        )
        curated_env.register(workspace=ws)
        print(f"✅ Environment registered from Dockerfile: {env_name}")
    # //////////////////////////////////////////////////////////////////////////////////////


    # ///////////////////////////////////////////////////////////////////
    # ✅ ALWAYS BUILD FROM DOCKERFILE (don't use cached version)
    # print("\n📦 Building environment from Docker...")

    # env_name = "envswift"
    # dockerfile_path = "deployment/docker-swift/Dockerfile"

    # # Create NEW environment from Dockerfile (always)
    # curated_env = Environment.from_dockerfile(
    #     name=env_name,
    #     dockerfile=dockerfile_path
    # )
    # print(f"✅ Environment created from Dockerfile: {dockerfile_path}")
    # //////////////////////////////////////////////////////////////////

    exp = Experiment(workspace=ws, name=cfg["EXPERIMENT"])

    # script_arguments = []
    # if cfg["SCRIPT_ARGUMENTS"]:
    #     for key, value in cfg["SCRIPT_ARGUMENTS"].items():
    #         script_arguments.append('--'+key.lower())
    #         script_arguments.append(value)

    # /////////////////////////////
    script_arguments = []
    if cfg["SCRIPT_ARGUMENTS"]:
        for key, value in cfg["SCRIPT_ARGUMENTS"].items():
            script_arguments.append(f"{key}={value}")
    # /////////////////////////////

    # Defines how to run your script on Azure ML compute
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

    # ////////////////////
    # blob_store_conf = DataReferenceConfiguration("workspaceblobstore")
    # src.run_config.data_references["workspaceblobstore"] = blob_store_conf
    # ///////////////////
    
    src.run_config.environment_variables = {
    'AZUREML_DATAREFERENCE_azureml_container': str(input_ds[0])
    }

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
