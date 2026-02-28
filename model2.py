import torch
from typing import Type, Union, List, Tuple
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.distributions.categorical import Categorical
from dataset import TextDataset

class Attention(nn.Module):
    def __init__(self, hidden_size: int):
        super(Attention, self).__init__()
        # Для объединения Context Vector и Hidden State декодера
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.tanh = nn.Tanh()

    def forward(self, decoder_hidden: torch.Tensor, encoder_outputs: torch.Tensor, mask: torch.Tensor = None):
        """
        :param decoder_hidden: (Batch, Hidden) - текущее скрытое состояние декодера
        :param encoder_outputs: (Batch, SeqLen, Hidden) - все скрытые состояния энкодера
        :param mask: (Batch, SeqLen) - маска паддингов (True там, где паддинг)
        """
        # 1. Считаем scores (Attention energies) через скалярное произведение (Dot-Product)
        # (Batch, 1, Hidden) * (Batch, Hidden, SeqLen) -> (Batch, 1, SeqLen)
        scores = torch.bmm(decoder_hidden.unsqueeze(1), encoder_outputs.transpose(1, 2))
        
        # 2. Маскируем паддинги (ставим -inf, чтобы softmax дал 0 вероятности)
        if mask is not None:
            # mask shape: (Batch, SeqLen). unsqueeze -> (Batch, 1, SeqLen)
            scores.masked_fill_(mask.unsqueeze(1), -1e9)
        
        # 3. Softmax для получения весов внимания
        attn_weights = nn.functional.softmax(scores, dim=-1) # (Batch, 1, SeqLen)
        
        # 4. Считаем Context Vector (взвешенная сумма выходов энкодера)
        # (Batch, 1, SeqLen) * (Batch, SeqLen, Hidden) -> (Batch, 1, Hidden)
        context = torch.bmm(attn_weights, encoder_outputs)
        context = context.squeeze(1) # (Batch, Hidden)
        
        # 5. Объединяем контекст и hidden декодера для получения финального вектора внимания
        # Luong attention: h_tilde = tanh(Wc[c_t; h_t])
        combined = torch.cat((context, decoder_hidden), dim=1)
        output = self.tanh(self.concat(combined)) # (Batch, Hidden)
        
        return output, attn_weights

class EncoderDecoderRNN(nn.Module):
    def __init__(self, dataset: TextDataset, embed_size: int = 256, hidden_size: int = 256,
                 rnn_type: Type = nn.RNN, rnn_layers: int = 1):
        super(EncoderDecoderRNN, self).__init__()
        self.dataset = dataset
        self.vocab_size = dataset.vocab_size
        self.max_length = dataset.max_length
        self.hidden_size = hidden_size
        self.rnn_layers = rnn_layers
        self.rnn_type = rnn_type

        self.emb_enc = nn.Embedding(self.vocab_size, embed_size, padding_idx=dataset.pad_id)
        self.emb_dec = nn.Embedding(self.vocab_size, embed_size, padding_idx=dataset.pad_id)
        
        self.encoder = rnn_type(input_size=embed_size, hidden_size=hidden_size, num_layers=rnn_layers, batch_first=True)
        self.decoder = rnn_type(input_size=embed_size, hidden_size=hidden_size, num_layers=rnn_layers, batch_first=True)
        
        self.attention = Attention(hidden_size)
        
        # Линейный слой теперь принимает output от Attention (размера hidden_size)
        self.linear = nn.Linear(hidden_size, self.vocab_size)

    def _get_last_hidden(self, hidden):
        """Извлекает скрытое состояние последнего слоя для Attention"""
        if isinstance(hidden, tuple): # LSTM (h, c)
            # hidden[0] shape: (Layers, Batch, Hidden)
            return hidden[0][-1] 
        else: # RNN/GRU (h)
            return hidden[-1]

    def forward(self, indices_de: torch.Tensor, lengths_de: torch.Tensor, indices_en: torch.Tensor, lengths_en: torch.Tensor) -> torch.Tensor:
        device = indices_de.device
        batch_size = indices_de.size(0)
        
        # --- ENCODER ---
        # Сортируем для pack_padded_sequence
        lengths_de_gpu = lengths_de.to(device)
        lengths_de_sorted, sort_idx = lengths_de_gpu.sort(descending=True)
        indices_de_sorted = indices_de[sort_idx]
        _, unsort_idx = sort_idx.sort()
        
        embeds_enc = self.emb_enc(indices_de_sorted)
        # lengths must be on CPU for pack_padded_sequence
        packed_enc = pack_padded_sequence(embeds_enc, lengths_de_sorted.cpu(), batch_first=True)  

        packed_out, hidden = self.encoder(packed_enc)
        
        # Распаковываем output энкодера (нам нужны ВСЕ состояния для attention)
        encoder_outputs, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=indices_de.size(1))
        
        # Восстанавливаем порядок
        encoder_outputs = encoder_outputs[unsort_idx]
        
        if isinstance(hidden, tuple):
            h, c = hidden
            # (Layers, Batch, Hidden) -> переставляем Batch измерение (dim=1)
            h = h.index_select(1, unsort_idx)
            c = c.index_select(1, unsort_idx)
            decoder_hidden = (h, c)
        else:
            decoder_hidden = hidden.index_select(1, unsort_idx)

        # Маска для паддингов энкодера (True, где паддинг)
        # indices_de в исходном порядке
        enc_mask = (indices_de == self.dataset.pad_id)

        # --- DECODER ---
        # indices_en: (Batch, SeqLen) - целевые токены
        embeds_dec = self.emb_dec(indices_en) # (Batch, SeqLen, Emb)
        
        seq_len = indices_en.size(1)
        # Тензор для сбора логитов
        logits = torch.zeros(batch_size, seq_len, self.vocab_size, device=device)
        
        for t in range(seq_len):
            # Вход на текущем шаге (Teacher Forcing): берем t-й токен из target
            input_step = embeds_dec[:, t, :].unsqueeze(1) # (Batch, 1, Emb)
            
            # Один шаг декодера
            # decoder_out: (Batch, 1, Hidden)
            decoder_out, decoder_hidden = self.decoder(input_step, decoder_hidden)
            
            # Применяем Attention
            # Берем скрытое состояние с последнего слоя
            last_hidden = self._get_last_hidden(decoder_hidden)
            
            # attn_output: (Batch, Hidden) - вектор контекста + скрытое состояние
            attn_output, _ = self.attention(last_hidden, encoder_outputs, enc_mask)
            
            # Предсказываем
            step_logits = self.linear(attn_output) # (Batch, Vocab)
            logits[:, t, :] = step_logits

        return logits

    @torch.inference_mode()
    def inference(self, indices: torch.Tensor, lengths: torch.Tensor, temp: float = 1.) -> Union[str, List[str]]:
        self.eval()
        device = next(self.parameters()).device
        indices = indices.to(device)
        lengths_gpu = lengths.to(device)
        batch_size = indices.size(0)
        
        # --- ENCODER ---
        lengths_sorted, sort_idx = lengths_gpu.sort(descending=True)
        indices_sorted = indices[sort_idx]
        _, unsort_idx = sort_idx.sort()
        
        embeds_enc = self.emb_enc(indices_sorted)
        packed_enc = pack_padded_sequence(embeds_enc, lengths_sorted.cpu(), batch_first=True)
        
        packed_out, hidden = self.encoder(packed_enc)
        encoder_outputs, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=indices.size(1))
        
        # Восстанавливаем порядок
        encoder_outputs = encoder_outputs[unsort_idx]
        if isinstance(hidden, tuple):
            h, c = hidden
            decoder_hidden = (h.index_select(1, unsort_idx), c.index_select(1, unsort_idx))
        else:
            decoder_hidden = hidden.index_select(1, unsort_idx)

        enc_mask = (indices == self.dataset.pad_id)
        
        # --- DECODER LOOP ---
        input_token = torch.tensor([self.dataset.bos_id] * batch_size, device=device).unsqueeze(1)
        generated_tokens = [] 
        mask_eos = torch.zeros((batch_size,), dtype=torch.bool, device=device)
        
        for i in range(self.max_length):
            embeds = self.emb_dec(input_token)
            
            # Шаг декодера
            decoder_out, decoder_hidden = self.decoder(embeds, decoder_hidden)
            
            # Attention
            last_hidden = self._get_last_hidden(decoder_hidden)
            attn_output, _ = self.attention(last_hidden, encoder_outputs, enc_mask)
            
            # Логиты
            logits = self.linear(attn_output) # (Batch, Vocab)
            
            # Сэмплирование
            new_tokens = Categorical(logits=logits / temp).sample()
            
            # Обработка EOS и PAD
            new_tokens = torch.where(mask_eos, torch.tensor(self.dataset.pad_id, device=device), new_tokens)
            mask_eos = mask_eos | (new_tokens == self.dataset.eos_id)
            
            generated_tokens.append(new_tokens.unsqueeze(1))
            input_token = new_tokens.unsqueeze(1)
            
            if mask_eos.all():
                break

        generated_tokens = torch.cat(generated_tokens, dim=1)
        generated = self.dataset.ids2text(generated_tokens, lang='en')

        return generated