
# Using Transformers

## Pipeline functions

Let's see what happens when we use the sentiment analysis using the Pipeline function.

```
from transformers import pipeline

classifier = pipeline("sentiment-analysis")
classifier(
    [
        "I've been waiting for a HuggingFace course my whole life.",
        "I hate this so much!",
    ]
)
```

### Stages of Pipeline function
There are three stages in a pipeline function: Tokenizer, Model and Post Processing

!['Behind-the-pipeline](/assets/img/2023-05-01-Using-Transformers/behind-pipeline.svg)

#### Tokenizer Stage
1. Text is split into tokens.
2. Tokenizers will add some special tokens: [CLS] and [SEP]
3. Tokenizer matches each token with the unique ID in the vocab of the pre-trained model. `AutoTokenizer` method in HF is used here.

```
from transformers import AutoTokenizer

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
```

4. Tokenizer can add padding and truncation to create tensors of same length.

```
raw_inputs = [
    "I've been waiting for a HuggingFace course my whole life.",
    "I hate this so much!",
]
inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt")
print(inputs)
```

#### Model stage:
1. Download the configuration of the model as well as the pre-trained weights of the models
2. The `AutoModel` class loads a model without its pretraining head which means it will return a high dimensional tensor that is representation of sentences but not directly helpful in classifcation task.

```
from transformers import AutoModel

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModel.from_pretrained(checkpoint)
```

```
outputs = model(**inputs)
print(outputs.last_hidden_state.shape)
```

!['Model-heads](/assets/img/2023-05-01-Using-Transformers/model-heads.svg)


3. Use `AutoModelForSequenceClassification` for the classification task. This returns the logits.

```
from transformers import AutoModelForSequenceClassification

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
outputs = model(**inputs)
```

#### Postprocessing stage: 
1. Apply softmax layer to transform logits into probabilities

```
import torch

predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
print(predictions)
```

2. Use `id2label` method for converting logits into labels

```
model.config.id2label
```