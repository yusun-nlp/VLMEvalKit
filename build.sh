#!/bin/bash

VERSION=auto-interns1_1-v0.0.4

docker build --progress=plain -t registry.h.pjlab.org.cn/ailab-evalservice/vlmevalkit:$VERSION .
docker push registry.h.pjlab.org.cn/ailab-evalservice/vlmevalkit:$VERSION