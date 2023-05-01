
## Transformers are language models
All transformer models are language models trained on large amounts of raw text in a self-supervised fashion
Not very useful for specific practical tasks => Transfer learning where the pre-trained model is fine-tuned in a supervised way
Causal language modeling: Predicting next word

Masked language modeling: predict masked word in a sentence


## Transfer Learning
Pretraining is the act of training a model from scratch: the weights are randomly initialized, and the training starts without any prior knowledge.



Fine-tuning, on the other hand, is the training done after a model has been pretrained. The fine-tuning will only require a limited amount of data: the knowledge the pretrained model has acquired is “transferred,” hence the term transfer learning.



## General architecture
Encoder (left): The encoder receives an input and builds a representation of it (its features). This means that the model is optimized to acquire understanding from the input.
Decoder (right): The decoder uses the encoder’s representation (features) along with other inputs to generate a target sequence. This means that the model is optimized for generating outputs.

Encoder-only models: Good for tasks that require understanding of the input, such as sentence classification and named entity recognition.
Decoder-only models: Good for generative tasks such as text generation.
Encoder-decoder models or sequence-to-sequence models: Good for generative tasks that require an input, such as translation or summarization.

## The original architecture

Task:
Write an article in your own words on Transformer models

