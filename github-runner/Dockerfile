# github-runner — self-hosted runner para amael-lab
# Base: myoung34/github-runner (maneja registro/deregistro automático en GitHub)
# Extra: kubectl para deployments en MicroK8s
FROM myoung34/github-runner:latest

USER root

# Instalar kubectl (versión estable actual)
RUN KUBECTL_VER=$(curl -sSL https://dl.k8s.io/release/stable.txt) && \
    curl -sSLO "https://dl.k8s.io/release/${KUBECTL_VER}/bin/linux/amd64/kubectl" && \
    install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl && \
    rm kubectl && \
    kubectl version --client

USER runner
