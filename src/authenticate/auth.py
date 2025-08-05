# ==============================================================================
#  C O P Y R I G H T
# ------------------------------------------------------------------------------
#  Copyright (c) 2022-2024 by Robert Bosch GmbH. All rights reserved.
#
#  The reproduction, distribution and utilization of this file as
#  well as the communication of its contents to others without express
#  authorization is prohibited. Offenders will be held liable for the
#  payment of damages. All rights reserved in the event of the grant
#  of a patent, utility model or design.
# ==============================================================================

import os
import asyncio
from azure.core.credentials import TokenCredential
from azure.identity import (
    ChainedTokenCredential,
    ClientSecretCredential,
)


class Authenticator:
    def __init__(self) -> None:
        """Initializes authenticator"""
        self._tenant_id = os.environ["AZURE_TENANT_ID"]
        self._client_id = os.environ["AZURE_CLIENT_ID"]
        self._client_secret = os.environ["AZURE_CLIENT_SECRET"]
        print("Using ClientSecretCredential...")
        self.scopes = {}
        self.token_credential = None
        # self.token_validator = TokenValidator(self._client_id, tenant_id=self._tenant_id)

    def run(self) -> TokenCredential:
        """Creates credential based on existence of client secret, otherwise creates
        AzureCliCredential
        Returns:
            TokenCredential: Token for authenticating azure
        """
        if self.token_credential is None:
            self.token_credential = ChainedTokenCredential(
                ClientSecretCredential(
                    tenant_id=self._tenant_id,
                    client_id=self._client_id,
                    client_secret=self._client_secret,
                ),
            )
            print(f"[MDM] Credentials have been claimed successfully for {self._client_id}!")
        return self.token_credential  # type: ignore
