# Simple Chatbot Implementation using Gemma3 and streamlit

This repo contains a quick and easy implementation of a chatbot based on [Google's Gemma3 model](https://huggingface.co/google/gemma-3-1b-it) and the [streamlit app framework](https://streamlit.io).

It contains less than 90 lines of code but represents a fully functioning chatbot.

<img src="img/chatbot.png" alt="Chatbot preview" width="300"/>


## Getting started


### Prerequisites

First, you need to have a [huggingface](https://huggingface.co) account and a corresponding `HF_TOKEN` (that allows access to public gated repos):
```
Read access to contents of all public gated repos you can access
```

In order to access the Gemma3 model, you would also need to accept their terms on [huggingface](https://huggingface.co/google/gemma-3-1b-it) (login to hugginface, open the [model page](https://huggingface.co/google/gemma-3-1b-it), read and accept the terms).

Keep in mind that you would need to run this chatbot on a GPU. Tests were successful on a 32GB AWS GPU (rather slow for inference but certainly good enough to try things out). Follow the steps below for an Ubuntu based web-server.

(streamlit runs on TCP port `8501` => don't forget to open it on your web-server)



### 1. Clone this repo

```
git clone https://github.com/robgineer/simple_chatbot.git
cd simple_chatbot
```

### 2. Install pip and venv

```
sudo apt-get update && sudo apt-get upgrade
sudo apt-get install python3-pip
sudo apt-get install python3-venv
```

### 3. Create venv

```
python3 -m venv .venv
source .venv/bin/activate
```

### 4. Install dependencies
```
pip install -r requirements.txt
```

### 5. Login to huggingface

```
huggingface-cli login
```
... and enter your `HF_TOKEN` there.

### 6. Run the chatbot

```
streamlit run gemma3_chatbot.py
```

You will see the following message on your terminal:

```
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://<internal ip>:8501
  External URL: http://<external ip>:8501
```

Copy the `External URL` and paste it into your browser.