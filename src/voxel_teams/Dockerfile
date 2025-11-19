FROM python:3.10

# Create non-root user
ENV USER dockeruser
ENV UID 1000
ENV GID 100
ENV HOME /home/${USER}
RUN adduser --disabled-password --gecos "Non-root user" --uid ${UID} --gid ${GID} --home ${HOME} ${USER}

# Create inputs folder for stakeholder teams
ENV INPUT $HOME/inputs/

RUN mkdir $INPUT
RUN chown $UID:$GID $INPUT

# Migrate file for non-root user
COPY --chown=${UID}:${GID} ./src/voxel_teams ${HOME}/src/voxel_teams
COPY --chown=${UID}:${GID} ./src/utils/dbx_connector.py ${HOME}/src/utils/dbx_connector.py

# Switch to non-root user
USER ${USER}

WORKDIR $HOME

ARG ARTIFACTORY_URL=jfrog.ad-alliance.biz/artifactory/api/pypi/voxel51-python-pypi-external/simple

USER root
RUN --mount=type=secret,id=artifactory_user \
    --mount=type=secret,id=artifactory_token \
    set -ex \
    && ARTIFACTORY_USER="$(cat /run/secrets/artifactory_user)" \
    && ARTIFACTORY_TOKEN="$(cat /run/secrets/artifactory_token)" \
    && PIP_EXTRA_INDEX_URL="https://${ARTIFACTORY_USER}:${ARTIFACTORY_TOKEN}@${ARTIFACTORY_URL}" \
    && pip install --upgrade pip \
    && pip install --no-cache-dir --extra-index-url "$PIP_EXTRA_INDEX_URL" -r src/voxel_teams/requirements.txt \
    && rm -rf /root/.cache/pip

USER ${USER}
