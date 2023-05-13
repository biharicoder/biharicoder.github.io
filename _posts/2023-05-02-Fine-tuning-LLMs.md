# Tuning LLMs

As the field of artificial intelligence advances rapidly, it has become increasingly crucial to make the most of large language models (LLMs) in an efficient and effective manner. However, there are many different ways in which LLMs can be used, which can be daunting for beginners.

Basically, we can utilize pretrained LLMs for new tasks in two primary approaches: in-context learning and finetuning. This article will provide a brief overview of what in-context learning entails, followed by a discussion of the different methods available for finetuning LLMs and the steps to create a dataset for fine-tuning LLM for the task of Question Answering specifically.

## In-context learning
Large language models (LLMs) that are trained on a general text corpus have the ability to learn in-context. This means that it is not necessary to further train or fine-tune the pretrained LLMs to perform specific or new tasks that were not explicitly trained on. Instead, by providing a few examples of the target task via the input prompt, the LLM can directly learn the task, as shown in the example below.

!['In-context-classification](/assets/img/2023-05-02-Fine-tuning-LLMs/In-context-sentiment-classification.png)

If we are using the model through an API, in-context learning can be particularly beneficial since we may not have direct access to the model.

!['In-context-translation](/assets/img/2023-05-02-Fine-tuning-LLMs/In-context-translation.png)

## Hard/Discrete prompt tuning

Prompting refers to a method of incorporating additional information into a model's output generation process. This is typically done by adding a set of text tokens to the input text, either before or after the original text, in order to transform the task into a masked language modeling problem. This process of modifying the input text with a prefix or suffix text template is known as hard prompting. The purpose of hard prompting is to provide the model with more context and guidance for generating the desired output, whether it be for classification or generation tasks.

For example: You can modify the prompt from

```
What is of 15^2 / 5?
```

to 
```
Calculate the value of 15^2 / 5
```

![hard-vs-soft](/assets/img/2023-05-02-Fine-tuning-LLMs/hard-vs-soft.png)

The method of prompt tuning is a more cost-effective option compared to fine-tuning the model's parameters. However, its effectiveness is usually not as good as fine-tuning, since it does not update the model's parameters specifically for a given task, which may hinder its ability to adapt to unique characteristics of the task. Additionally, prompt tuning may require significant effort, as it frequently involves human participation in assessing and comparing the quality of different prompts.

## Soft prompt tuning
Soft prompting is a technique of adding an adjustable input embedding to the primary input of the model and optimizing those embeddings. This method offers better outcomes compared to hard prompting, and it is comparable to complete model fine-tuning for natural language processing (NLP) tasks such as paraphrase detection and question-answering generation.

![soft-prompting](/assets/img/2023-05-02-Fine-tuning-LLMs/soft-prompting.png)


## Indexing for IR (Converting into Embeddings)

In the domain of large language models, indexing can be considered a way to use in-context learning to transform LLMs into retrieval systems that can extract information from external sources and websites. This involves using an indexing module to divide a document or website into smaller sections, converting them into vectors that can be stored in a vector database. When a user inputs a query, the indexing module computes the vector similarity between the query and each vector in the database, and returns the top k most similar vectors as the response.

![indexing](/assets/img/2023-05-02-Fine-tuning-LLMs/indexing.png)

![ir-llm-arch](/assets/img/2023-05-02-Fine-tuning-LLMs/ir-llm-arch.png)

## Fine-tuning

Fine-tuning could be of three types:
1. **Feature based fine-tuning:**

    The feature-based approach involves utilizing a pre-trained LLM on the target dataset to generate output embeddings for the training set. These embeddings are used as input features to train a classification model. Although this approach is commonly used for embedding-focused models like BERT, it can also be applied to extract embeddings from generative GPT-style models.

2. **Fine-tuning to update the output layers**

    In this method, we maintain the parameters of the pre-trained LLM as they are and only train the newly added output layers, similar to how we train a small multilayer perceptron or a logistic regression classifier on the embedded features.

![bam-tuning](/assets/img/2023-05-02-Fine-tuning-LLMs/bam-tuning.png)

3. **Updating all layers**
The best but most computationally expensive method of fine-tuning is by updating all the layers of the model. The parameters of the pretrained LLMs are not frozen rather finetuned so it can be a computationally prohibitive solutions in LLMs.


## Comparison of different tuning tasks

![different-types](/assets/img/2023-05-02-Fine-tuning-LLMs/different-types.png)


## Creating dataset for fine-tuning LLM for Question Answering

### Summarization
Leverage summarization to create summary from the documents/passages
1. Extract passages from documents
![html-passage](/assets/img/2023-05-02-Fine-tuning-LLMs/html-passage.png)

2. Summarize passage using FLAN-UL2 (BAM endpoint). The Flan-UL2 is a type of encoder-decoder model that follows the T5 architecture. To improve its performance, it was fine-tuned using a technique called "Flan" prompt tuning and a dataset collection.

![summary](/assets/img/2023-05-02-Fine-tuning-LLMs/summary.png)

### Question Generation
Use passage (context) and answer (summary) to create questions using `t5-small-question-generator` model. The model is designed to generate questions from a given context and answer, following a sequence-to-sequence approach. It takes the answer and context as input and produces the corresponding question as the output.
 
![question-generation](/assets/img/2023-05-02-Fine-tuning-LLMs/question-generation.png)

### Finally Questions, context and answer pair
Convert the data (question, context and answer) into the format required for fine-tuning the BAM model

![bam-input-data-file](/assets/img/2023-05-02-Fine-tuning-LLMs/bam-input-data-file.png)

## References:
1. [Stanford AI lab](https://www.google.com/url?sa=i&url=http%3A%2F%2Fai.stanford.edu%2Fblog%2Fin-context-learning%2F&psig=AOvVaw1aS1lHam_Zq7kXK0zElafr&ust=1683079549131000&source=images&cd=vfe&ved=0CBIQjhxqFwoTCMDU_ZbI1f4CFQAAAAAdAAAAABAE)
2. [How does in-context learning work](https://www.google.com/url?sa=i&url=http%3A%2F%2Fai.stanford.edu%2Fblog%2Funderstanding-incontext%2F&psig=AOvVaw1aS1lHam_Zq7kXK0zElafr&ust=1683079549131000&source=images&cd=vfe&ved=0CBIQjhxqFwoTCMDU_ZbI1f4CFQAAAAAdAAAAABAJ)
3. [Instruction tuning](https://www.google.com/url?sa=i&url=https%3A%2F%2Fsmilegate.ai%2Fen%2F2021%2F09%2F12%2Finstruction-tuning-flan%2F&psig=AOvVaw3_5HnnqNfpGAIFYBlZCgim&ust=1683082947402000&source=images&cd=vfe&ved=0CBIQjhxqFwoTCPD3vdvS1f4CFQAAAAAdAAAAABAb)
4. [Knowledge Retrieval Architecture](https://mattboegner.com/knowledge-retrieval-architecture-for-llms/)
5. [Summarization](https://www.google.com/url?sa=i&url=https%3A%2F%2Fwww.width.ai%2Fpost%2F4-long-text-summarization-methods&psig=AOvVaw2OYXx9QQ-JTFo1JOTKN3dl&ust=1683128146764000&source=images&cd=vfe&ved=0CBIQjhxqFwoTCLCZj9n71v4CFQAAAAAdAAAAABAE)
6. [Question Generation](https://www.google.com/url?sa=i&url=https%3A%2F%2Fwww.mdpi.com%2F2076-3417%2F11%2F21%2F10267&psig=AOvVaw38suC0fCJVvX6jJvaNDB_0&ust=1683128620588000&source=images&cd=vfe&ved=0CBIQjhxqFwoTCPijj-781v4CFQAAAAAdAAAAABA4)