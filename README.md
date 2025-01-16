
# On-Device or Remote? On the Energy Efficiency of Fetching LLM-Generated Content

This repository contains the replication package of the following publication:
> Vince Nguyen, Hieu Huynh, Vidya Dhopate, Anusha Annengala, Hiba Bouhlal, Gian Luca Scoccia, Matias Martinez, Vincenzo Stoico, Ivano Malavolta. On-Device or Remote? On the Energy Efficiency of Fetching LLM-Generated Content. Proceedings of the IEEE/ACM 4th International Conference on AI Engineering--Software Engineering for AI (CAIN), 2025.
 
This research aims to measure and compare the energy usage of fetching content generated by LLMs from a remote server via HTTP requests and generating content with on-device LLMs, in diverse scenarios with different LLMs and varying generated content lengths.

Our work is of interest to researchers exploring the trade-off of deploying on-device LLMs versus fetching similar generated content from remote server, from the perspective of energy usage of the user's device. The result of our experiment can help software engineers better understand the potential energy impact of integrating LLMs on devices to inform software architecture and design choices of future web and mobile applications.

---
# Background
1. This experiment is created and run on a Macbook Pro M2 (Apple Silicon architecture).
2. The server used in this experiment includes an Nvidia RTX 4070 with 12GB of VRAM. Further settings information can be found in the document under **Experiment Execution** section. 

# Requirements

1. Before you begin, make sure you have Python 3 installed on your system. This project requires Python 3 to run. [Link to Install Python](https://www.python.org/downloads/)
2. Install the project requirement in the **root** directory using the following:
```shell
pip install -r requirement.txt
```
3. Create a new `.env` file in the **root** folder, and add your server's IP address to it like this:
```shell
export SERVER_IP= "<Your Server IP here>"
```
4. Make sure to install [Ollama](https://ollama.com/download) and its corresponding LLMs on **both** on-device Device and Server. 
In this experiment we used the following models: [llama3.1:8b](https://ollama.com/library/llama3.1:8b), [gemma:2b](https://ollama.com/library/gemma:2b), [gemma:7b](https://ollama.com/library/gemma:7b), [phi3:3.8b](https://ollama.com/library/phi3), [qwen2:1.5b](https://ollama.com/library/qwen2:1.5b), [qwen2:7b](https://ollama.com/library/qwen2:7b), [mistral:7b](https://ollama.com/library/mistral:7b). 
5. Make sure your server allows HTTP connection to port `11434`, which is the original port of Ollama.

# Running the project

- For running the experiment, run the following command from the **root** directory:
```shell
        python experiment-runner/ experiment/RunnerConfig.py
```

# Getting the results
The output data is saved in the `run_table.csv` file, which could be found in `/experiment/experiment_output` folder.

# Data Analysis

For performing statistical tests on the data generated from the experiment, run the `.ipynb` file (R runtime)  in folder [Data Analysis](data-analysis/) with the `run_table.csv` file.
