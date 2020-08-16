FROM python:3.6
ARG GID=1010
ARG UID=1010
ARG GROUP=imbd
ARG USER=imbd

RUN groupadd -g ${GID} ${GROUP} && \
    useradd -s /bin/bash -m -r -g ${GID} -u ${UID} ${USER}

WORKDIR /home/${USER}/imbd2020
COPY ./ ./
RUN chown -R ${USER} /home/${USER}
USER ${USER}
RUN pip install --user -r requirements.txt
ENTRYPOINT [ "python" ]