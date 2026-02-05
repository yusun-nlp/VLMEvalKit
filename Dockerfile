# FROM registry.h.pjlab.org.cn/ailab-evalservice/vlmevalkit:auto-v0.0.10

# WORKDIR /app/vlmevalkit
# ADD . /app/vlmevalkit
# ADD nltk_data /root/.nltk_data
# ENV NLTK_DATA=/root/.nltk_data

# #RUN apt update && apt install -y openssh-server vim net-tools telnet
# #RUN pip install -i https://mirrors.aliyun.com/pypi/simple/ zss apted lxml tabulate
# RUN apt update
# RUN apt install -y openssh-server vim net-tools telnet gcc curl poppler-utils
# RUN apt clean && rm -rf /var/lib/apt/lists/*
# RUN pip install -i https://mirrors.aliyun.com/pypi/simple/ -r /app/vlmevalkit/requirements.txt
# RUN pip cache purge

# RUN ls -la /root

# ENTRYPOINT ["tail", "-f", "/dev/null"]

FROM registry.h.pjlab.org.cn/ailab-evalservice/vlmevalkit:auto-interns1_1-v0.0.3

WORKDIR /app/vlmevalkit
ADD . /app/vlmevalkit
ENV NLTK_DATA=/mnt/shared-storage-user/auto-eval-pipeline/opencompass/nltk_data

#RUN apt update && apt install -y openssh-server vim net-tools telnet
#RUN pip install -i https://mirrors.aliyun.com/pypi/simple/ zss apted lxml tabulate
# RUN apt update
# RUN apt install -y openssh-server vim net-tools telnet gcc curl poppler-utils
# RUN apt clean && rm -rf /var/lib/apt/lists/*
RUN pip install -i https://mirrors.aliyun.com/pypi/simple/ -r /app/vlmevalkit/requirements.txt
RUN pip cache purge

RUN ls -la /root

ENTRYPOINT ["tail", "-f", "/dev/null"]