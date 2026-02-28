import torch
from typing import Type, Union, List
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.distributions.categorical import Categorical
from dataset import TextDataset
from tqdm.notebook import tqdm


class EncoderDecoderRNN(nn.Module):
    def __init__(self, dataset: TextDataset, embed_size: int = 256, hidden_size: int = 256,
                 rnn_type: Type = nn.RNN, dropout: float = 0.3, rnn_layers: int = 1):
        """
        Model for text generation
        :param dataset: text data dataset (to extract vocab_size and max_length)
        :param embed_size: dimensionality of embeddings
        :param hidden_size: dimensionality of hidden state
        :param rnn_type: type of RNN layer (nn.RNN or nn.LSTM)
        :param rnn_layers: number of layers in RNN
        """
        super(EncoderDecoderRNN, self).__init__()
        self.dataset = dataset  # required for decoding during inference
        self.vocab_size = dataset.vocab_size
        self.max_length = dataset.max_length

        """
        Create necessary layers
        """
        self.emb_enc = nn.Embedding(self.vocab_size, embed_size, padding_idx=dataset.pad_id)
        self.emb_dec = nn.Embedding(self.vocab_size, embed_size, padding_idx=dataset.pad_id)
        self.encoder = rnn_type(input_size=embed_size, hidden_size=hidden_size, num_layers=rnn_layers, dropout=dropout, batch_first=True)
        self.decoder = rnn_type(input_size=embed_size, hidden_size=hidden_size, num_layers=rnn_layers, dropout=dropout, batch_first=True)
        self.linear = nn.Linear(hidden_size, self.vocab_size)

    # def forward(self, indices_de: torch.Tensor, lengths_de: torch.Tensor, indices_en: torch.Tensor, lengths_en: torch.Tensor) -> torch.Tensor:
    #     """
    #     Compute forward pass through the model and
    #     return logits for the next token probabilities
    #     :param indices: LongTensor of encoded tokens of size (batch_size, input length)
    #     :param lengths: LongTensor of lengths of size (batch_size, )
    #     :return: FloatTensor of logits of shape (batch_size, output length, vocab_size)
    #     """

    #     """
    #     Convert indices to embeddings, pass them through recurrent layers
    #     and apply output linear layer to obtain the logits
    #     """
    #     embeds_enc = self.emb_enc(indices_de)
    #     packed_embeds_enc = pack_padded_sequence(embeds_enc, lengths_de, batch_first=True, enforce_sorted=False)
    #     _, context = self.encoder(packed_embeds_enc)
        
    #     embeds_dec = self.emb_dec(indices_en)
    #     packed_embeds_dec = pack_padded_sequence(embeds_dec, lengths_en, batch_first=True, enforce_sorted=False)
    #     packed_outputs_dec, _ = self.decoder(packed_embeds_dec, context)
    #     outputs, lengths = pad_packed_sequence(packed_outputs_dec, batch_first=True, padding_value=self.dataset.pad_id)
        
    #     logits = self.linear(outputs)
    #     return logits

    def forward(self, indices_de: torch.Tensor, lengths_de: torch.Tensor, indices_en: torch.Tensor, lengths_en: torch.Tensor) -> torch.Tensor:
        """
        Compute forward pass through the model and return logits
        :param indices_de: (batch_size, input length)
        :param lengths_de: (batch_size, )
        :param indices_en: (batch_size, output length)
        :param lengths_en: (batch_size, )
        """
        device = indices_de.device
        lengths_de = lengths_de.to(device)
        lengths_de_sorted, sort_idx = lengths_de.sort(descending=True)
        
        indices_de_sorted = indices_de[sort_idx]
        indices_en_sorted = indices_en[sort_idx]
        lengths_en_sorted = lengths_en.to(device)[sort_idx]

        # Для восстановления порядка в конце
        _, unsort_idx = sort_idx.sort()

        # 2. Encoder
        print(indices_de_sorted.max())
        embeds_enc = self.emb_enc(indices_de_sorted)
        # pack_padded_sequence требует длины на CPU (int64)
        packed_embeds_enc = pack_padded_sequence(embeds_enc, lengths_de_sorted.cpu(), batch_first=True)
        
        _, hidden = self.encoder(packed_embeds_enc)
        
        embeds_dec = self.emb_dec(indices_en_sorted)
        output_dec, _ = self.decoder(embeds_dec, hidden) 
        output_dec = output_dec[unsort_idx]
        
        logits = self.linear(output_dec)
        return logits

    # def forward(self, indices_de: torch.Tensor, lengths_de: torch.Tensor, indices_en: torch.Tensor, lengths_en: torch.Tensor) -> torch.Tensor:
    #     """
    #     Простой forward без упаковки последовательностей.
    #     """
    #     # 1. ENCODER
    #     embeds_enc = self.emb_enc(indices_de) # (Batch, SeqLen, Emb)
        
    #     # Прогоняем через RNN "как есть", включая паддинги
    #     output_enc, _ = self.encoder(embeds_enc) # output: (Batch, SeqLen, Hidden)
        
    #     idx = (lengths_de - 1).view(-1, 1).expand(output_enc.size(0), output_enc.size(2)).unsqueeze(1)
    #     # idx shape: (Batch, 1, Hidden)
        
    #     # gather выбирает элементы по индексам.
    #     encoder_hidden_last = output_enc.gather(1, idx.to(output_enc.device)).squeeze(1) 
        
    #     # Формируем context для декодера: (Layers, Batch, Hidden)
    #     context = encoder_hidden_last.unsqueeze(0).repeat(1, 1, 1)
        
    #     if isinstance(self.encoder, nn.LSTM):
    #         h_0 = context
    #         c_0 = torch.zeros_like(context)
    #         context = (h_0, c_0)

    #     # 2. DECODER
    #     embeds_dec = self.emb_dec(indices_en) # (Batch, SeqLen, Emb)
        
    #     # Подаем в декодер. Паддинги тоже обработаются, но мы их потом замаскируем в лоссе.
    #     output_dec, _ = self.decoder(embeds_dec, context)
        
    #     logits = self.linear(output_dec)
        
    #     return logits

    # @torch.inference_mode()
    # def inference(self, texts: torch.Tensor, temp: float = 1.) -> str:
    #     """
    #     Generate new text with an optional prefix
    #     :param prefix: prefix to start generation
    #     :param temp: sampling temperature
    #     :return: generated text
    #     """
    #     self.eval()
    #     """
    #     Encode the prefix (do not forget the BOS token!),
    #     pass it through the model to accumulate RNN hidden state and
    #     generate new tokens sequentially, sampling from categorical distribution,
    #     until EOS token or reaching self.max_length.
    #     Do not forget to divide predicted logits by temperature before sampling
    #     """
    #     tokens = texts[:, :1]
    #     embeds = self.embedding(tokens) # (B, L, H)
    #     output, hidden = self.rnn(embeds) 
    #     logits = self.linear(output) # (B, L, V)

    #     new_tokens = Categorical(logits=logits[:, -1:, :] / temp).sample()
    #     tokens = torch.cat([tokens, new_tokens], dim=1)

    #     for i in range(1, texts.shape[1]):
    #         embeds = self.embedding(texts[:, i:i+1])
    #         output, hidden = self.rnn(embeds, hidden)
    #         logits = self.linear(output)

    #         new_tokens = Categorical(logits=logits[:, -1:, :] / temp).sample()
    #         tokens = torch.cat([tokens, new_tokens], dim=1)

    #     generated = self.dataset.ids2text(tokens, lang='en')

    #     return generated

    # @torch.inference_mode()
    # def inference(self, texts: torch.Tensor, temp: float = 1.) -> str:
    #     """
    #     Generate new text with an optional prefix
    #     :param prefix: prefix to start generation
    #     :param temp: sampling temperature
    #     :return: generated text
    #     """
    #     self.eval()
    #     """
    #     Encode the prefix (do not forget the BOS token!),
    #     pass it through the model to accumulate RNN hidden state and
    #     generate new tokens sequentially, sampling from categorical distribution,
    #     until EOS token or reaching self.max_length.
    #     Do not forget to divide predicted logits by temperature before sampling
    #     """
    #     result = []
    #     for i in tqdm(range(len(texts))):
    #         text = texts[i, :]
    #         tokens = torch.tensor([self.dataset.bos_id])
    #         embeds = self.embedding(tokens) # (B, L, H)
    #         output, hidden = self.rnn(embeds) 
    #         logits = self.linear(output) # (B, L, V)

    #         new_tokens = Categorical(logits=logits[-1:, :] / temp).sample()
    #         tokens = torch.cat([tokens, new_tokens], dim=0)

    #         for j in range(1, len(text)):
    #             if text[j] == self.dataset.pad_id:
    #                 torch.cat([tokens, torch.tensor([self.dataset.eos_id])], dim=0)
    #                 break
    #             embeds = self.embedding(text[j:j+1])
    #             output, hidden = self.rnn(embeds, hidden)
    #             logits = self.linear(output)

    #             new_tokens = Categorical(logits=logits[-1:, :] / temp).sample()
    #             tokens = torch.cat([tokens, new_tokens], dim=0)
    #             if new_tokens == self.dataset.eos_id:
    #                 break
    #         result.append(tokens.tolist())

    #     generated = self.dataset.ids2text(result, lang='en')

    #     return generated


    @torch.inference_mode()
    def inference(self, indices: torch.Tensor, lengths: torch.Tensor, temp: float = 1.) -> Union[str, List[str]]:
        """
        Generate new text with an optional prefix
        :param indices: indices of tokens of texts for translating (batch_size, max_length)
        :param lengths: real lengths of indices (batch_size,)
        :param temp: sampling temperature
        :return: generated text
        """
        self.eval()
        """
        Encode the prefix (do not forget the BOS token!),
        pass it through the model to accumulate RNN hidden state and
        generate new tokens sequentially, sampling from categorical distribution,
        until EOS token or reaching self.max_length.
        Do not forget to divide predicted logits by temperature before sampling
        """
        device = next(self.parameters()).device
        indices = indices.to(device)
        batch_size = len(indices)
        embeds_enc = self.emb_enc(indices) # (B, L, H)
        packed_embeds_enc = pack_padded_sequence(embeds_enc, lengths, batch_first=True, enforce_sorted=False)
        _, hidden = self.encoder(packed_embeds_enc) 
        tokens = torch.tensor([self.dataset.bos_id] * batch_size, device=device).unsqueeze(1)

        new_tokens = torch.tensor([self.dataset.bos_id] * batch_size, device=device)
        mask_eos = torch.zeros((batch_size,), dtype=torch.bool, device=device)
        for i in range(self.max_length):
            embeds = self.emb_dec(new_tokens)[:, None, :]
            output, hidden = self.decoder(embeds, hidden)
            logits = self.linear(output)
            new_tokens = Categorical(logits=logits[:, 0, :] / temp).sample()
            new_tokens = torch.where(mask_eos, self.dataset.pad_id, new_tokens)
            mask_eos = mask_eos | (new_tokens == self.dataset.eos_id)
            tokens = torch.cat([tokens, new_tokens[:, None]], dim=1)

        generated = self.dataset.ids2text(tokens, lang='en')

        return generated
