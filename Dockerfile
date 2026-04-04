FROM python:3.12.11-slim
LABEL title="ahc"
LABEL description="Adaptive homotopy continuation solver for incompressible two-phase flow in porous media"
LABEL version="0.1"
LABEL maintainer="Peter von Schultzendorff"
LABEL email="peter.schultzendorff@uib.no"

WORKDIR /app

# Install git (not included in python:3.12-slim)
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

# Install PorePy fork
RUN git clone https://github.com/pschultzendorff/porepy_hc \
    && pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -e /app/porepy_hc[development,testing]

# Copy files into container and install
COPY . ./ahc

# Normalize Windows CRLF on the run_all.sh script so Bash in Linux containers
# does not fail with "$'\r': command not found".
RUN sed -i 's/\r$//' /app/ahc/run_all.sh \
    && chmod +x /app/ahc/run_all.sh \
    && pip install --no-cache-dir -e /app/ahc

# Matplotlib backend for .png files
ENV MPLBACKEND=Agg

# Use bash explicitly because the launcher relies on bash features (e.g. [[ ]]).
CMD ["bash", "/app/ahc/run_all.sh"]