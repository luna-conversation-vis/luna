# Luna

Luna is a framework that predict various aspects of visualization intent from users' conversations. 

### Env Setup
1. Run ```conda create --name luna --file conda_env.txt```
2. Run ```conda activate luna```
3. Run ```pip install -r pip_requirements.txt```


### Dataset Creation

1. Download [CoSQL](https://yale-lily.github.io/cosql) and unzip into `data/cosql_dataset.`
2. Run ```./prepare-datasets.sh```
3. Run ```./generate-covis-splits.sh```

### Training

Training uses the ```train.py```. Use ```src/config.py``` to change default training parameters (e.g, number of GPUs, accelerator, batch size).

Example: 

```bash
# Generic form
python train.py $MODEL $(ARGS)

# Example
python train.py select_col --train_batch_size==8 --test_batch_size=4
```

For defaults, use ```./group-training.sh```. The models to be trained are 

- Visualized Attribute Count: ```select_count```
- Visualized Attribute Column: ```select_col```
- Filter Count: ```where_count```
- Filter Column: ```where_col```
- Filter Operator: ```where_operator```

To monitor the training process, run Tensorboard with ```tensorboard --logdir tb_logs```.

### Evaluation

Each training run returns a ```v_num``` for model versioning. Recover each v_num for the models above. Then to run evaluation over the whole pipeline, use:

```bash
# Generic form, SPLIT can be train, test, or val
python pipeline.py $SPLIT --simple_select=$1 --where_col=$2 --where_count=$3 --where_operator=$4 --where_value=$5

# Example (assuming default hparams are used and v_nums=0)
python pipeline.py val --select_count=0 --select_col=0 --where_col=0 --where_count=0 --where_operator=0 --where_value=0 --verbosity=0

# To test individual modules, set the other v_nums=-1. For example, to test the attribute count module:
python pipeline.py val --select_count=0 --select_col=-1 --where_col=-1 --where_count=-1 --where_operator=-1 --where_value=-1 --verbosity=0
```

For defaults, use ```./eval.sh``` where both val and test sets will be evaluated.

We provide our trained model checkpoint [here](https://storage.googleapis.com/luna-sigmod24/checkpoint.zip). To use the checkpoints, download and unzip the file under the root directory of the repository. 

### Notebook Extension
If you want to use the notebook extension supported by Luna, please first download the model checkpoint [here](https://storage.googleapis.com/luna-sigmod24/checkpoint.zip) and install Lux following the [instruction](https://github.com/lux-org/lux).

Then you can simply create a new notebook under the root directory and import Luna by calling ```from conversation import Conversation```.

As an experimental version, the extension currently only support pandas.DataFrame as the input. To start the exploration,

```python
from conversation import Conversation

df = pd.read_csv("Please fill in your data directory")
conversation = Conversation(df)
```

Then in each cell, you can ask Luna by calling

```python
conversation.ask("Where do the cases happen?")
```

If you want to remove the last utterance, or remove the entire conversation history, you can call

```python
# Remove the last utterance
conversation.clean_last()

# Clean the conversation history
conversation.clean()
```

We provide the notebook ```exploration-example.ipynb``` about the motivating scenario in the main paper as an example.

### GPT-3.5 and GPT-4 baselines

We provide the files to run GPT models for visualization intent in the directory `gpt/`. It uses the ```litellm``` package so you need to install it additionally.

To test the performance of GPT models, you need to first set up Azure OpenAI APIs. The model (GPT-4 or GPT-3.5) is also set up on Azure OpenAI platform. For more details, please refer to [Azure official documentation](https://learn.microsoft.com/en-us/azure/ai-services/openai/).

After setting up the API service, then you should provide the API key and API base link in ```gpt_inference.py``` and run it with ```python gpt_inference.py test --model_name=GPT-4``` for the test set. The model name can be "GPT-3.5" or "GPT-4" depending on your experiement setup.

We also provide our results of running GPT models on the validation and test sets under `GPT-3.5/` and `GPT-4/` respectively. You can run ```python gpt_inference.py test --re_run=0 --model_name=GPT-4``` or ```python gpt_inference.py test --re_run=0 --model_name=GPT-3.5``` to obtain the performance of these models with our results on the test set using two models.