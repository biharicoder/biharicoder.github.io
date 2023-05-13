# Getting started with transformer models

The AutoModel class is a tool used to create a model from a pre-existing checkpoint. This class is essentially a straightforward wrapper over a range of models within the library. Its intelligent design allows it to automatically identify the best model architecture for the given checkpoint, and create a model based on that architecture.

## Creating a Transformer

You can also use the class from the transformer architecture directly instead of using the Automodel wrapper class.

### Initialize the model
**Initialize BERT model by loading config object**

```
from transformers import BertConfig, BertModel

# Building the config
config = BertConfig()

# Building the model from the config
model = BertModel(config)
```

The configuration for BERT contains the following:

```
BertConfig {
  [...]
  "hidden_size": 768,
  "intermediate_size": 3072,
  "max_position_embeddings": 512,
  "num_attention_heads": 12,
  "num_hidden_layers": 12,
  [...]
}
```

### Load the model

#### A. Using default configuration
Creating a model from the default configuration initializes it with random values:

```
from transformers import BertConfig, BertModel

config = BertConfig()
model = BertModel(config)

# Model is randomly initialized!
```

![weight-initialization](/assets/img/2023-05-03-Getting-started-with-transformer-models/weight-initialization.svg)

The model can be utilized now, but it needs to be trained first because it will generate gibberish output. We could start from scratch and train the model for the given task, but this would take a lot of time and data. So, It's crucial to be able to exchange and reuse learned models in order to prevent needless and redundant work.

#### B. Loading pre-trained model
Loading a pretrained model using the `from_pretrained()` method:

```
from transformers import BertModel

model = BertModel.from_pretrained("bert-base-cased")
```

You can also replace `BertModel` with the equivalent `AutoModel` class to write checkpoint-agnostic code.

Now, all of the checkpoint's weights are used to initialize this model. It can be adjusted on a new task as well as utilized directly for inference on the tasks it was trained on. We can quickly see positive outcomes by using pre-trained weights during exercise rather than starting from blank.

## Save the model

The models can be saved using `save_pretrained()` method as shown below:

```
model.save_pretrained("directory_on_my_computer")
```

Two files will be saved. The `config.json` file will contain the attributes necessary to build the model architecture whereas the `pytorch_model.bin` file will contain the model's weights. 

## Making predictions using transformer

Firstly, encode the input using tokenizer. 

```
import torch

model_inputs = torch.tensor(encoded_sequences)
```

![tokenization](/assets/img/2023-05-03-Getting-started-with-transformer-models/tokenization.svg)

Then pass the encoded input to the model for making prediction/inference

```
output = model(model_inputs)
```

<!-- Learning more about tokenization in the [next blog on Tokenization](/_posts/2023-05-03-Tokenization.md) -->

## Reference
[Hugging Face](https://huggingface.co/learn/nlp-course)