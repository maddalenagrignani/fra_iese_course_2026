# %%

import torch
from transformers import pipeline
from transformers import BertTokenizer, AutoModel
import random
from sklearn.metrics.pairwise import cosine_similarity

# %%

unmasker = pipeline('fill-mask',
                    model='bert-base-uncased')

test = "Every morning last summer in Texas, " + \
       "I visited the [MASK] where I would swim" + \
       "and sunbathe."

result = unmasker(test)
result

# %%

# bert-base-uncased
# textattack/bert-base-uncased-SST-2
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
tokenizer

# %% explore the vocabulary of the tokenizer by looking at some words
vocab = tokenizer.get_vocab()
print(f"Total number of tokens in vocabulary: {len(vocab)} \n---------")
for _ in range(10):
    word, idx = random.choice(list(vocab.items()))
    print(word, idx)

# %%

text1 = "oranges are nicer than apples"
text2 = "I love oranges"
text3 = "apples are tasty"

encoded_input1 = tokenizer(text1,
                           max_length=30,
                           padding="max_length",
                           return_tensors='pt')

print("Tokens:")
temp_tokens = encoded_input1["input_ids"][0]
print(tokenizer.convert_ids_to_tokens(temp_tokens))
print("\n------------------------------------------\n")
print("Tokens IDs:")
print(temp_tokens)

# %% load a model using its name and explore its configuration

model = AutoModel.from_pretrained("bert-base-uncased",
                                  output_hidden_states=True,
                                  output_attentions=True,
                                  attn_implementation="eager"
                                  )

print(model.config)

# %% explore embedding shapes

encoded_example = tokenizer(text1, max_length=30,
                            padding="max_length",
                            return_tensors='pt')

with torch.no_grad():
    output = model(**encoded_example)

last_hidden_state = output.last_hidden_state

print(f"last_hidden_state shape: {last_hidden_state.shape}")
print(f"  -> batch size: {last_hidden_state.shape[0]}  (we passed 1 text)")
print(f"  -> sequence length (max_length): {last_hidden_state.shape[1]}")
print(f"  -> hidden size (embedding dim):  {last_hidden_state.shape[2]}")

print(f"\nNumber of tokens in '{text1}':")
tokens = tokenizer.convert_ids_to_tokens(encoded_example['input_ids'][0])
print(tokens)
print(f"  -> non-padding tokens: {sum(t != '[PAD]' for t in tokens)}")

# %% batch of texts

encoded_batch = tokenizer([text1, text2, text3], 
                          max_length=30, 
                          padding="max_length", 
                          return_tensors='pt')

with torch.no_grad():
    output_batch = model(**encoded_batch)

print(f"last_hidden_state shape: {output_batch.last_hidden_state.shape}")
print(f"  -> batch size:    {output_batch.last_hidden_state.shape[0]}  (we passed 3 texts)")
print(f"  -> sequence length: {output_batch.last_hidden_state.shape[1]}")
print(f"  -> hidden size:     {output_batch.last_hidden_state.shape[2]}")

# %% function for generating embeddings


def get_mean_embeddings(texts):
    encoded = tokenizer(texts, max_length=30,
                        padding="max_length",
                        return_tensors='pt')
    with torch.no_grad():
        output = model(**encoded)
    mask = encoded["attention_mask"]
    return (output.last_hidden_state * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True)


# %% compare similarities

embeddings = get_mean_embeddings([text1, text2, text3])  # shape: (3, 768)

print(f"Similarity 1-2: {cosine_similarity(embeddings[[0]], embeddings[[1]])[0, 0]:.4f}")
print(f"Similarity 1-3: {cosine_similarity(embeddings[[0]], embeddings[[2]])[0, 0]:.4f}")
# %%
