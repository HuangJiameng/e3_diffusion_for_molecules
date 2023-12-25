ARG BASE_IMAGE=jerrikeph/base-cuda113-ubuntu2004-torch
ARG FROM_IMAGE=${FROM_IMAGE:-null}
FROM ${BASE_IMAGE} as public

RUN apt update &&\
    apt install -y unzip \
    autoconf \
    less

RUN chsh -s `which zsh` \
    # && curl https://raw.githubusercontent.com/robbyrussell/oh-my-zsh/master/tools/install.sh | bash -s -- --unattended \
    && wget https://raw.githubusercontent.com/oskarkrawczyk/honukai-iterm/master/honukai.zsh-theme -O ${ZSH:-~/.oh-my-zsh}/themes/honukai.zsh-theme --no-check-certificate \
    && sed -i.bak '/ZSH_THEME/s/\".*\"/\"honukai\"/' ~/.zshrc \
    && sed -i.bak '/plugin/s/(.*)/(git zsh-autosuggestions)/' ~/.zshrc

RUN conda create -c conda-forge -n my-rdkit-env rdkit
RUN conda init zsh \
    && conda init bash \
    && echo "conda activate my-rdkit-env" >> ~/.zshrc \
    && echo "conda activate my-rdkit-env" >> ~/.bashrc

COPY ./asset/requirements.txt /tmp/build/requirements.txt
SHELL ["conda", "run", "-n", "my-rdkit-env", "/bin/zsh", "-c"]
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple 
# RUN conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
RUN pip install -r /tmp/build/requirements.txt

FROM ${FROM_IMAGE} as private
ARG TORCH_HOME
ARG _USER
ARG _UID
ARG _GID
ARG PW

RUN pip install -r /tmp/build/requirements.txt

# Option1: Using unencrypted password/ specifying password
RUN usermod --password $(echo ${PW} | openssl passwd -1 -stdin) root
RUN useradd -m ${_USER} --uid=${_UID} -s /usr/bin/zsh && echo "${_USER}:${PW}" | chpasswd

# make the color of zsh-autosuggestions right
ENV TERM xterm-256color

RUN cp -r /root/.oh-my-zsh /home/${_USER} && chown ${_USER}:${_USER} -R /home/${_USER}/.oh-my-zsh &&\
    cp /root/.zshrc /home/${_USER} && chown ${_USER}:${_USER} -R /home/${_USER}/.zshrc

USER ${_UID}:${_GID}
WORKDIR /home/${_USER}

ENV TORCH_HOME ${TORCH_HOME}

COPY --chown=${_USER}:${_USER} ./asset/ssh .ssh

RUN chmod 0700 .ssh && chmod 600 .ssh/id_rsa && chmod 644 .ssh/id_rsa.pub && \
    ssh-keyscan github.com >> .ssh/known_hosts

# RUN conda init zsh \
#     && conda init bash 

RUN git config --global user.email "jerrikeph@gmail.com" \
    &&git config --global user.name "an"

CMD "zsh"


