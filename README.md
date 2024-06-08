# gpt4_visual_grounding
This project demonstrates the power of combining Visual-Grounding of large Vision-Language Models with an Agentic Flow to enhance localization abilities.

## Demo Video
You can view a demo of the iterative process in the following video. For each test image, the video of iterations will be saved in `FinalResults`



https://github.com/m-ali-awan/gpt4_visual_grounding/assets/62832721/79d34209-d55d-48fb-8def-c3368faa582c



## Setup
It is recommended to create a virtual environment and install the required dependencies from `requirements.txt`.

### Create Virtual Environment
```sh
python -m venv venv
```

### Activate virtual environment

> On Windows

```
venv\Scripts\activate
```

> On macOS and Linux 

```
source venv/bin/activate
```

### Install Requirements

```
pip install -r requirements.txt
```


## Usage

Use `demo.ipynb` to test this project. Open the Jupyter Notebook and follow the instructions provided.


### Limitations

While this process showcases the potential of Visual-Grounding with an Agentic Flow, it is not foolproof. There is a high tendency for iterations to enter into a wrong flow, resulting in poor bounding boxes. Further refinement and validation are needed to improve the robustness of this approach.
