import math
import torch
import torch.nn as nn
from torch.distributions import Categorical

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Linear projections & split heads
        # (Batch, Len, d_model) -> (Batch, Len, Heads, d_k) -> (Batch, Heads, Len, d_k)
        Q = self.q_linear(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.k_linear(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.v_linear(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled Dot-Product Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k) # (Batch, Heads, Q_Len, K_Len)
        
        if mask is not None:
            # Mask shape expected: (Batch, 1, Q_Len, K_Len) or (Batch, 1, 1, K_Len)
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        output = torch.matmul(attn_weights, V) # (Batch, Heads, Q_Len, d_k)
        
        # Concat heads
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)
        return self.out_linear(output)

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
    def forward(self, x):
        return self.net(x)

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        # Pre-Norm architecture (often more stable) or Post-Norm. Here: Post-Norm standard
        attn_out = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        
        ff_out = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, enc_output, src_mask, tgt_mask):
        # 1. Masked Self-Attention
        attn_out = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_out))
        
        # 2. Cross-Attention (Query=Decoder, Key/Value=Encoder)
        attn_out = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn_out))
        
        # 3. Feed Forward
        ff_out = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_out))
        return x

class Seq2SeqTransformer(nn.Module):
    def __init__(self, 
                 dataset, # Ваш dataset класс
                 d_model=256, 
                 num_heads=4, 
                 num_layers=3, 
                 d_ff=512, 
                 dropout=0.1):
        super().__init__()
        self.dataset = dataset
        self.src_vocab_size = dataset.vocab_size # Предполагаем общий словарь или тот же размер
        self.tgt_vocab_size = dataset.vocab_size
        self.pad_idx = dataset.pad_id
        
        self.enc_embedding = nn.Embedding(self.src_vocab_size, d_model, padding_idx=self.pad_idx)
        self.dec_embedding = nn.Embedding(self.tgt_vocab_size, d_model, padding_idx=self.pad_idx)
        self.pos_encoder = PositionalEncoding(d_model)
        
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        
        self.generator = nn.Linear(d_model, self.tgt_vocab_size)
        
        # Init weights (Xavier) - важно для трансформеров
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def make_src_mask(self, src):
        # src: (Batch, Len)
        # Mask: (Batch, 1, 1, Len)
        return (src != self.pad_idx).unsqueeze(1).unsqueeze(2)

    def make_tgt_mask(self, tgt):
        # tgt: (Batch, Len)
        # Pad mask: (Batch, 1, 1, Len)
        pad_mask = (tgt != self.pad_idx).unsqueeze(1).unsqueeze(2)
        
        # Sequence mask (Causal): (1, 1, Len, Len)
        seq_len = tgt.size(1)
        nopeak_mask = torch.tril(torch.ones((seq_len, seq_len), device=tgt.device)).bool()
        nopeak_mask = nopeak_mask.unsqueeze(0).unsqueeze(0)
        
        return pad_mask & nopeak_mask

    def encode(self, src, src_mask):
        x = self.enc_embedding(src) * math.sqrt(self.enc_embedding.embedding_dim)
        x = self.pos_encoder(x)
        for layer in self.encoder_layers:
            x = layer(x, src_mask)
        return x

    def decode(self, tgt, enc_output, src_mask, tgt_mask):
        x = self.dec_embedding(tgt) * math.sqrt(self.dec_embedding.embedding_dim)
        x = self.pos_encoder(x)
        for layer in self.decoder_layers:
            x = layer(x, enc_output, src_mask, tgt_mask)
        return x

    def forward(self, indices_de, lengths_de, indices_en, lengths_en):
        """
        Соответствует сигнатуре вызова в training_epoch:
        model(indices.to(device), lengths, target[:, :-1].to(device), target_lengths - 1)
        """
        # indices_de: Source (Encoder input)
        # indices_en: Target (Decoder input)
        
        src_mask = self.make_src_mask(indices_de)
        tgt_mask = self.make_tgt_mask(indices_en)
        
        enc_output = self.encode(indices_de, src_mask)
        dec_output = self.decode(indices_en, enc_output, src_mask, tgt_mask)
        
        logits = self.generator(dec_output)
        return logits

    @torch.inference_mode()
    def inference(self, indices, lengths, temp=1.0, max_len=None):
        """
        Greedy decoding для инференса.
        В идеале нужен Beam Search для лучшего качества, но для начала сойдет Greedy.
        """
        self.eval()
        device = indices.device
        batch_size = indices.shape[0]
        
        # 1. Encode source
        src_mask = self.make_src_mask(indices)
        enc_output = self.encode(indices, src_mask)
        
        # 2. Prepare decoder input (start with BOS)
        # (Batch, 1)
        dec_input = torch.full((batch_size, 1), self.dataset.bos_id, dtype=torch.long, device=device)
        
        generated_tokens = []
        
        # Limit generation length
        max_len = max_len if max_len is not None else self.dataset.max_length
        
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for _ in range(max_len):
            tgt_mask = self.make_tgt_mask(dec_input)
            
            # Forward pass decoder
            # Здесь мы пересчитываем все каждый раз (неэффективно, но просто).
            # В проде используют кэширование KV (incremental decoding).
            dec_output = self.decode(dec_input, enc_output, src_mask, tgt_mask)
            
            # Get last token logits
            logits = self.generator(dec_output[:, -1, :]) # (Batch, Vocab)
            
            # Sampling or Greedy
            if temp == 0:
                next_token = logits.argmax(dim=-1)
            else:
                next_token = Categorical(logits=logits / temp).sample()
            
            # Save token
            generated_tokens.append(next_token)
            
            # Append to input for next step
            dec_input = torch.cat([dec_input, next_token.unsqueeze(1)], dim=1)
            
            # Check for EOS
            is_eos = (next_token == self.dataset.eos_id)
            finished = finished | is_eos
            if finished.all():
                break
                
        # Stack tokens -> (Batch, Len)
        # Note: generated_tokens is list of (Batch,) -> need to stack and transpose
        result_tokens = torch.stack(generated_tokens, dim=1)
        
        return self.dataset.ids2text(result_tokens, lang='en')
