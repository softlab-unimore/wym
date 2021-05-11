
import torch
from transformers import BertModel, BertTokenizer

class WordEmbedding():
    def __init__(self, device='auto'):
        self.model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
        # Set the device to GPU (cuda) if available, otherwise stick with CPU
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        self.model = self.model.to(self.device)
        self.model.eval()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def get_word_embeddings(self, words):
        tokens = self.tokenizer.encode(words, return_tensors='pt')
        detected_tokens = self.tokenizer.convert_ids_to_tokens(tokens[0])
        token_emb = self.get_token_embeddings(tokens)
        word_token_map, words_list = WordEmbedding.map_token_to_word(detected_tokens)
        word_embedding = [torch.mean(token_emb[word_token_map[i]], 0) for i, word in enumerate(words_list)]
        return torch.stack(word_embedding), words_list

    def get_token_embeddings(self, tokens):
        with torch.no_grad():
            outputs = self.model(tokens.to(self.device))
            # Evaluating the model will return a different number of objects based on
            # how it's  configured in the `from_pretrained` call earlier. In this case,
            # becase we set `output_hidden_states = True`, the third item will be the
            # hidden states from all layers. See the documentation for more details:
            # https://huggingface.co/transformers/model_doc/bert.html#bertmodel
            hidden_states = outputs[2]
        token_embeddings = torch.stack(hidden_states, dim=0)
        token_embeddings = torch.squeeze(token_embeddings, dim=1)
        token_embeddings = token_embeddings.permute(1, 0, 2)
        token_vecs_sum = torch.mean(token_embeddings[:, 2:12, :], 1)  # I chose to use the mean instead of sum

        return token_vecs_sum

    @staticmethod
    def map_token_to_word(detected_tokens):
        words_map = {x: [] for x in range(len(detected_tokens))}
        words_list = []
        word_pos = -1
        for i, token in enumerate(detected_tokens[1:-1]):
            if token.startswith('##'):
                words_map[word_pos] += [i + 1]
                words_list[-1] += token[2:]
            else:
                word_pos += 1
                words_map[word_pos] = [i + 1]
                words_list.append(token)
        return words_map, words_list