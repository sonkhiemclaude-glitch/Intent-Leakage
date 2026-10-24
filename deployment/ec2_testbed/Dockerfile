FROM ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive
RUN printf '#!/bin/sh\nexit 101\n' > /usr/sbin/policy-rc.d && chmod +x /usr/sbin/policy-rc.d \
 && apt-get update && apt-get install -y --no-install-recommends mininet openvswitch-switch iproute2 iputils-ping tcpdump curl python3 kmod procps ca-certificates \
 && rm -rf /var/lib/apt/lists/*
CMD ["sleep", "infinity"]
