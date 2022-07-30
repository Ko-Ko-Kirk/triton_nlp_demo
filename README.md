This is a Triton Inference Server demo in COSCUP 2022. It used Zero Shot Text Classiifcation of NLP.

Thanks to [sachinsharma](https://github.com/sachinsharma9780/Build-ML-pipelines-for-Computer-Vision-NLP-and-Graph-Neural-Networks-using-Nvidia-Triton-Server) to finish this demo. This demo is simplified by his repo.

1. Install libraries in requirements.txt
2. Run create_model.py
3. Get model.pt and move it into model_repository folder. 
4. Run 
```
docker run --rm -p8000:8000 -p8001:8001 -p8002:8002 -v <your_path>:/models nvcr.io/nvidia/tritonserver:22.06-py3 tritonserver --model-repository=/models
```
5. Run embed.py and get result.
