import torch
import torch.nn as nn
from typing import List
from torch.distributions.categorical import Categorical


class PositionalEncoding(nn.Module):
    def __init__(self, max_length=128, embed_size=256, dropout=0.5):
        super().__init__()

        self.dropout = nn.Dropout(dropout)

        values = torch.exp(torch.arange(0, embed_size, 2) * (-torch.log(torch.tensor(10000)) / embed_size))
        positions = torch.arange(0, max_length)
        values = positions[:, None] * values[None, :]

        self.pe = torch.zeros((max_length, embed_size))
        self.pe[:, 0::2] = torch.sin(values)
        self.pe[:, 1::2] = torch.cos(values)
        self.pe = nn.Parameter(self.pe[None, :, :], requires_grad=False) # (1, max_length, embed_dim)

    def forward(self, embeds):
        '''
        param embeds: matrix of embedings (batch_size, length, embeds_size)
        '''
        return self.dropout(embeds + self.pe[:, :embeds.shape[1]])


class Attention(nn.Module):
    def __init__(self, embed_size=256, n_heads=8, dropout=0.5):
        super().__init__()

        assert embed_size % n_heads == 0

        self.hidden_size = embed_size // n_heads

        self.WQ = nn.Linear(embed_size, self.hidden_size, bias=False)
        self.WK = nn.Linear(embed_size, self.hidden_size, bias=False)
        self.WV = nn.Linear(embed_size, self.hidden_size, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None): 
        '''
        params query, key, value: matrices (batch_size, length, embeds_size)
        param mask: matrix of inaccessible indeces (batch_size, length, length)
        return: attention (batch_size, length, hidden_size)
        '''
        Q = self.WQ(query)
        K = self.WK(key)
        V = self.WV(value)
        # Q, K, V - (batch_size, length, hidden_size)

        dk = torch.sqrt(torch.tensor(Q.shape[-1]))

        pairwise_dot = torch.bmm(Q, K.transpose(1, 2)) / dk # (batch_size, length, length)

        if mask is not None:
            pairwise_dot = pairwise_dot.masked_fill(mask, -torch.inf)

        attention_score = nn.functional.softmax(pairwise_dot, dim=-1) # (batch_size, length, length)
        attention = torch.bmm(self.dropout(attention_score), V)  # (batch_size, length, hidden_size)

        return attention


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size=256, n_heads=8, dropout=0.5):
        super().__init__()

        self.heads_list = nn.ModuleList([Attention(embed_size, n_heads, dropout) for _ in range(n_heads)])

        self.WO = nn.Linear(embed_size, embed_size, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        '''
        params query, key, value: matrices (batch_size, length, embeds_size)
        return: multihead_attention (batch_size, length, embed_size)
        '''

        heads = torch.cat([attention(query, key, value, mask) for attention in self.heads_list], dim=-1) # (batch_size, length, embed_size)

        multihead_attention = self.WO(heads) # (batch_size, length, embed_size)

        return self.dropout(multihead_attention)


class EncoderTransformerLayer(nn.Module):
    def __init__(self, embed_size=256, hidden_size=256, n_heads=8, dropout=0.5):
        super().__init__()

        self.attention = MultiHeadAttention(embed_size, n_heads, dropout)

        self.layer_norm1 = nn.LayerNorm(embed_size)
        self.layer_norm2 = nn.LayerNorm(embed_size)

        self.feedforward = nn.Sequential(
            nn.Linear(embed_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, embed_size),
            nn.Dropout(dropout)
        )

    def forward(self, embeds):
        '''
        param embeds: matrix of embeddings (batch_size, length, embeds_size)
        return: (batch_size, length, embeds_size)
        '''

        attention = self.attention(embeds, embeds, embeds)

        norm_attention = self.layer_norm1(attention + embeds)

        output = self.feedforward(norm_attention)
        output = self.layer_norm2(output + norm_attention)

        return output


class EncoderTransformer(nn.Module):
    def __init__(self, max_length=128, embed_size=256, feedforward_hidden_size=256, n_heads=8, n_layers=2, dropout=0.5):
        super().__init__()

        self.pe = PositionalEncoding(max_length, embed_size, dropout)

        self.transformer_layers = nn.Sequential(*[
            EncoderTransformerLayer(embed_size, feedforward_hidden_size, n_heads, dropout)
            for _ in range(n_layers)
            ])

    def forward(self, embeds):
        '''
        param embeds: matrix of embeddings (batch_size, length, embeds_size)
        return: (batch_size, length, embeds_size)
        '''
        pe_embeds = self.pe(embeds)
        output = self.transformer_layers(pe_embeds)

        return output

class DecoderTransformerLayer(nn.Module):
    def __init__(self, embed_size=256, feedforward_hidden_size=256, n_heads=8, dropout=0.5):
        super().__init__()

        self.masked_attention = MultiHeadAttention(embed_size, n_heads, dropout)
        self.cross_attention = MultiHeadAttention(embed_size, n_heads, dropout)

        self.layer_norm1 = nn.LayerNorm(embed_size)
        self.layer_norm2 = nn.LayerNorm(embed_size)
        self.layer_norm3 = nn.LayerNorm(embed_size)

        self.feedforward = nn.Sequential(
            nn.Linear(embed_size, feedforward_hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_hidden_size, embed_size),
            nn.Dropout(dropout)
        )

    def create_mask_(self, batch_size, length):
        '''
        create batch of upper-triangular matrix
        return: (batch_size, length, length)
        '''
        device = next(self.parameters()).device
        mask = torch.triu(torch.ones((length, length), dtype=torch.bool, device=device), diagonal=1)
        return mask.repeat(batch_size, 1, 1)

    def forward(self, embeds, encoder_output):
        '''
        param embeds: matrix of embeddings (batch_size, length, embeds_size)
        param encoder_output: output of encoder (batch_size, length, embeds_size)
        return: (batch_size, length, embeds_size)
        '''
        batch_size, length, _ = encoder_output.shape

        mask = self.create_mask_(batch_size, length)
        masked_attention_output = self.masked_attention(embeds, embeds, embeds, mask)
        norm_masked_output = self.layer_norm1(masked_attention_output + embeds)

        cross_attention_output = self.cross_attention(norm_masked_output, encoder_output, encoder_output)
        norm_cross_output = self.layer_norm2(cross_attention_output + norm_masked_output)

        feedforward_output = self.feedforward(norm_cross_output)
        output = self.layer_norm3(feedforward_output + norm_cross_output)

        return output


class DecoderTransformer(nn.Module):
    def __init__(self, max_length=128, embed_size=256, feedforward_hidden_size=256, n_heads=8, n_layers=2, dropout=0.5):
        super().__init__()

        self.pe = PositionalEncoding(max_length, embed_size, dropout)

        self.transformer_layers = nn.ModuleList([
            DecoderTransformerLayer(embed_size, feedforward_hidden_size, n_heads, dropout)
            for _ in range(n_layers)
            ])

    def forward(self, embeds, encoder_output):
        '''
        param embeds: matrix of embeddings (batch_size, length, embeds_size)
        param encoder_output: output of encoder (batch_size, length, embeds_size)
        return: (batch_size, length, embeds_size)     
        '''

        pe_embeds = self.pe(embeds)

        output = pe_embeds
        for transformer in self.transformer_layers:
            output = transformer(output, encoder_output)

        return output


class EncoderDecoderTransformer(nn.Module):
    def __init__(self, dataset, embed_size=256, feedforward_hidden_size=256, n_heads=8, n_layers=2, dropout=0.5):
        super().__init__()

        vocab_sizes = dataset.vocab_sizes
        self.dataset = dataset
        self.max_length = dataset.max_length

        self.encoder_embedding = nn.Embedding(vocab_sizes[0], embed_size)
        self.decoder_embedding = nn.Embedding(vocab_sizes[1], embed_size)

        self.encoder = EncoderTransformer(self.max_length, embed_size, feedforward_hidden_size, n_heads, n_layers, dropout)
        self.decoder = DecoderTransformer(self.max_length, embed_size, feedforward_hidden_size, n_heads, n_layers, dropout)

        self.linear = nn.Linear(embed_size, vocab_sizes[1])

    def forward(self, encoder_indices, encoder_lengths, decoder_indices, decoder_lengths):
        '''
        params encoder_indices, decoder_indices -- batch of padded tokenized sequences (batch_size, length)
        params encoder_lengths, decoder_lengths -- real lengths of sequences (batch_size,)
        return: logits (batch_size, length, vocab_size)
        '''
        input_embeddings = self.encoder_embedding(encoder_indices)
        output_embeddings = self.decoder_embedding(decoder_indices)

        encoder_output = self.encoder(input_embeddings)

        decoder_output = self.decoder(output_embeddings, encoder_output)

        logits = self.linear(decoder_output)

        return logits

    @torch.inference_mode()
    def inference(self, indices: torch.Tensor, lengths: torch.Tensor, temp: float = 1.) -> List[str]:
        '''
        param indices -- batch of padded tokenized input sequences (batch_size, length)
        params lengths -- real lengths of sequences (batch_size,)
        return: batch of generated strings
        '''

        self.eval()
        device = next(self.parameters()).device

        input_embeddings = self.encoder_embedding(indices)
        encoder_output = self.encoder(input_embeddings)

        batch_size = indices.shape[0]
        tokens = torch.tensor([self.dataset.bos_id] * batch_size, device=device, dtype=torch.int32)[:, None]
        mask_eos = torch.zeros((batch_size,), device=device, dtype=torch.bool)
        token_embeddings = self.decoder_embedding(tokens) # (batch_size, 1, embed_size)

        for i in range(self.max_length - 1):
            decoder_output = self.decoder(token_embeddings, encoder_output)

            logits = self.linear(decoder_output)
            new_tokens = Categorical(logits=logits[:, -1, :] / temp).sample() # (batch_size,)
            new_tokens = torch.where(mask_eos, self.dataset.pad_id, new_tokens)
            mask_eos = mask_eos | (new_tokens == self.dataset.eos_id)
            tokens = torch.cat([tokens, new_tokens[:, None]], dim=1)
            new_embeddings = self.decoder_embedding(new_tokens[:, None])
            token_embeddings = torch.cat([token_embeddings, new_embeddings], dim=1) # (batch_size, length, embed_size)

            if mask_eos.sum() == batch_size:
                break
        
        generated = self.dataset.ids2text(tokens, lang='en')

        return generated
