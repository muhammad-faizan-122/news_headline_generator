# Headline Generator App
Given the detail of news and generate headline using FlanT5 pretrained Language model on new headlines. For detail of FlanT5 and DeepSeek-R1-Distill-Qwen-1.5B (using ollama) pretrained Language model you can refer [here](https://huggingface.co/mrm8488/t5-base-finetuned-summarize-news).
## Demo
![alt text](demo/1.png)
![alt text](demo/2.png)
![alt text](demo/3.png)
## Setup
create conda virtual environment
```
conda create -n env_name python=3.11.11 -y
```
install requirements
```
pip install -r requirements.txt
```

run app
```
streamlit run app.py
```