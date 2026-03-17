
### 1. Build the Image
```
docker build -t signature-api .
```
### 2. Run the Container
```
docker run -p 8000:8000 signature-api
docker run --gpus all -p 8000:8000 -v "[your model directory]:/app/model"  signet-api
```
Download the weights from [Link] (https://drive.google.com/file/d/1tUjSG66HDGz0vBBzVnfYQW1b7OvnGkzL/view?usp=sharing) and place them in the /model folder.
