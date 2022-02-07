# For more information, please refer to https://aka.ms/vscode-docker-python
FROM pytorch/pytorch

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

# Install pip requirements
COPY requirements.txt .
RUN python -m pip install -r requirements.txt  
# RUN apt update && \ 
    # apt install git && \
RUN rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY . /workspace

# Creates a non-root user with an explicit UID and adds permission to access the /app folder
# For more info, please refer to https://aka.ms/vscode-docker-python-configure-containers
RUN adduser -u 5678 --disabled-password --gecos "" xchen && chown -R xchen /workspace
USER xchen

# During debugging, this entry point will be overridden. For more information, please refer to https://aka.ms/vscode-docker-python-debug
CMD ["python", "main.py"]
