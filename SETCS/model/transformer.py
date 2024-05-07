import torch
import torch.nn as nn
import torch.nn.functional as f

from prettytable import PrettyTable
from SETCS.modules.embeddings import Embeddings
from SETCS.model.encoder import TransformerEncoder
from SETCS.model.decoder import TransformerDecoder
from SETCS.inputters import constants
from SETCS.utils.misc import sequence_mask


class Embedder(nn.Module):
    def __init__(self, args):
        super(Embedder, self).__init__()

        self.enc_input_size = 0
        self.dec_input_size = 0

        self.use_src_word = args.use_src_word
        self.use_tgt_word = args.use_tgt_word
        if self.use_src_word:
            self.src_word_embeddings = Embeddings(args.emsize,
                                                  args.src_vocab_size,
                                                  constants.PAD)
            self.enc_input_size += args.emsize
        if self.use_tgt_word:
            self.tgt_word_embeddings = Embeddings(args.emsize,
                                                  args.tgt_vocab_size,
                                                  constants.PAD)
            self.dec_input_size += args.emsize

        self.use_tokenized = args.use_tokenized_code
        if self.use_tokenized:
            self.src_word_tokenized_embeddings = Embeddings(args.emsize,
                                                  args.src_vocab_size,
                                                  constants.PAD)

        self.use_type = args.use_code_type
        if self.use_type:
            self.type_embeddings = nn.Embedding(len(constants.TOKEN_TYPE_MAP),
                                                self.enc_input_size)

        # Fully connected layer for difference representation
        # self.fc_diff = nn.Linear(512, 512)

        self.fc1 = nn.Sequential(nn.Linear(1024, 512), nn.ReLU())

        self.src_pos_emb = args.src_pos_emb
        self.tgt_pos_emb = args.tgt_pos_emb
        self.no_relative_pos = all(v == 0 for v in args.max_relative_pos)

        if self.src_pos_emb and self.no_relative_pos:
            self.src_pos_embeddings = nn.Embedding(args.max_src_len,
                                                   self.enc_input_size)

        if self.tgt_pos_emb:
            self.tgt_pos_embeddings = nn.Embedding(args.max_tgt_len + 2,
                                                   self.dec_input_size)

        self.dropout = nn.Dropout(args.dropout_emb)

    def forward(self,
                sequence,
                sequence_tokenized=None,
                sequence_type=None,
                mode='encoder',
                step=None):

        if mode == 'encoder':
            word_rep = None
            if self.use_src_word:
                word_rep = self.src_word_embeddings(sequence.unsqueeze(2))  # B x P x d
                # print(word_rep.shape) # 64 * 92 * 512
                # duplicate_word_rep = word_rep.clone()
                # word_rep = torch.cat((word_rep, duplicate_word_rep), dim=2)
                # word_rep = self.fc1(word_rep)
            if self.use_tokenized:
                word_tokenized_rep = self.src_word_tokenized_embeddings(sequence_tokenized.unsqueeze(2))

                # Calculate difference between tokenized and original code embeddings
                # diff = word_tokenized_rep - word_rep

                # Apply fully connected layer to difference representation
                # diff_rep = self.fc_diff(diff)
                # word_rep = torch.cat((word_rep, word_tokenized_rep), dim=1)
                word_rep = torch.cat((word_rep, word_tokenized_rep), dim=-1)
                # word_rep = torch.cat((word_rep, diff_rep), dim=-1)
                word_rep = self.fc1(word_rep)
            
            if self.use_type:
                type_rep = self.type_embeddings(sequence_type)
                word_rep = word_rep + type_rep

            if self.src_pos_emb and self.no_relative_pos:
                pos_enc = torch.arange(start=0,
                                       end=word_rep.size(1)).type(torch.LongTensor)
                pos_enc = pos_enc.expand(*word_rep.size()[:-1])
                if word_rep.is_cuda:
                    pos_enc = pos_enc.cuda()
                pos_rep = self.src_pos_embeddings(pos_enc)
                word_rep = word_rep + pos_rep

        elif mode == 'decoder':
            word_rep = None
            if self.use_tgt_word:
                word_rep = self.tgt_word_embeddings(sequence.unsqueeze(2))  # B x P x d
            if self.tgt_pos_emb:
                if step is None:
                    pos_enc = torch.arange(start=0,
                                           end=word_rep.size(1)).type(torch.LongTensor)
                else:
                    pos_enc = torch.LongTensor([step])  # used in inference time

                pos_enc = pos_enc.expand(*word_rep.size()[:-1])
                if word_rep.is_cuda:
                    pos_enc = pos_enc.cuda()
                pos_rep = self.tgt_pos_embeddings(pos_enc)
                word_rep = word_rep + pos_rep

        else:
            raise ValueError('Unknown embedder mode!')

        word_rep = self.dropout(word_rep)
        return word_rep


class Encoder(nn.Module):
    def __init__(self,
                 args,
                 input_size):
        super(Encoder, self).__init__()

        self.transformer = TransformerEncoder(num_layers=args.nlayers,
                                              d_model=input_size,
                                              heads=args.num_head,
                                              d_k=args.d_k,
                                              d_v=args.d_v,
                                              d_ff=args.d_ff,
                                              dropout=args.trans_drop,
                                              max_relative_positions=args.max_relative_pos,
                                              use_neg_dist=args.use_neg_dist)
        self.use_all_enc_layers = args.use_all_enc_layers
        if self.use_all_enc_layers:
            self.layer_weights = nn.Linear(input_size, 1, bias=False)

    def count_parameters(self):
        return self.transformer.count_parameters()

    def forward(self,
                input,
                input_len):
        layer_outputs, _ = self.transformer(input, input_len)  # B x seq_len x h
        if self.use_all_enc_layers:
            output = torch.stack(layer_outputs, dim=2)  # B x seq_len x nlayers x h
            layer_scores = self.layer_weights(output).squeeze(3)
            layer_scores = f.softmax(layer_scores, dim=-1)
            memory_bank = torch.matmul(output.transpose(2, 3),
                                       layer_scores.unsqueeze(3)).squeeze(3)
        else:
            memory_bank = layer_outputs[-1]
        return memory_bank, layer_outputs


class Decoder(nn.Module):
    def __init__(self, args, input_size):
        super(Decoder, self).__init__()

        self.input_size = input_size

        self.transformer = TransformerDecoder(
            num_layers=args.nlayers,
            d_model=self.input_size,
            heads=args.num_head,
            d_k=args.d_k,
            d_v=args.d_v,
            d_ff=args.d_ff,
            coverage_attn=args.coverage_attn,
            dropout=args.trans_drop
        )

    def count_parameters(self):
        return self.transformer.count_parameters()

    def init_decoder(self,
                     src_lens,
                     max_src_len):
        return self.transformer.init_state(src_lens, max_src_len)

    def decode(self,
               tgt_words,
               tgt_emb,
               memory_bank,
               state,
               step=None,
               layer_wise_coverage=None):
        decoder_outputs, attns = self.transformer(tgt_words,
                                                  tgt_emb,
                                                  memory_bank,
                                                  state,
                                                  step=step,
                                                  layer_wise_coverage=layer_wise_coverage)

        return decoder_outputs, attns

    def forward(self,
                memory_bank,
                memory_len,
                tgt_pad_mask,
                tgt_emb):

        max_mem_len = memory_bank[0].shape[1] \
            if isinstance(memory_bank, list) else memory_bank.shape[1]
        state = self.init_decoder(memory_len, max_mem_len)
        return self.decode(tgt_pad_mask, tgt_emb, memory_bank, state)


class Transformer(nn.Module):
    """Module that writes an answer for the question given a passage."""

    def __init__(self, args, tgt_dict):
        """"Constructor of the class."""
        super(Transformer, self).__init__()

        self.name = 'Transformer'
        if len(args.max_relative_pos) != args.nlayers:
            assert len(args.max_relative_pos) == 1
            args.max_relative_pos = args.max_relative_pos * args.nlayers

        self.embedder = Embedder(args)
        self.encoder = Encoder(args, self.embedder.enc_input_size)
        self.decoder = Decoder(args, self.embedder.dec_input_size)
        self.layer_wise_attn = args.layer_wise_attn

        self.generator = nn.Linear(self.decoder.input_size, args.tgt_vocab_size)
        if args.share_decoder_embeddings:
            if self.embedder.use_tgt_word:
                assert args.emsize == self.decoder.input_size
                self.generator.weight = self.embedder.tgt_word_embeddings.word_lut.weight

        self.criterion = nn.CrossEntropyLoss(reduction='none')

    def _run_forward_ml(self,
                        code_word_rep,
                        code_word_tokenized_rep,
                        code_type_rep,
                        code_len,
                        summ_word_rep,
                        summ_len,
                        tgt_seq,
                        **kwargs):

        batch_size = code_len.size(0)
        # embed and encode the source sequence
        code_rep = self.embedder(code_word_rep,
                                 code_word_tokenized_rep,
                                 code_type_rep,
                                 mode='encoder')
        memory_bank, layer_wise_outputs = self.encoder(code_rep, code_len)  # B x seq_len x h

        # embed and encode the target sequence
        summ_emb = self.embedder(summ_word_rep,
                                 mode='decoder')
        summ_pad_mask = ~sequence_mask(summ_len, max_len=summ_emb.size(1))
        enc_outputs = layer_wise_outputs if self.layer_wise_attn else memory_bank
        layer_wise_dec_out, attns = self.decoder(enc_outputs,
                                                 code_len,
                                                 summ_pad_mask,
                                                 summ_emb)
        decoder_outputs = layer_wise_dec_out[-1]

        loss = dict()
        target = tgt_seq[:, 1:].contiguous()
        scores = self.generator(decoder_outputs)  # `batch x tgt_len x vocab_size`
        scores = scores[:, :-1, :].contiguous()  # `batch x tgt_len - 1 x vocab_size`
        ml_loss = self.criterion(scores.view(-1, scores.size(2)),
                                 target.view(-1))

        ml_loss = ml_loss.view(*scores.size()[:-1])
        ml_loss = ml_loss.mul(target.ne(constants.PAD).float())
        ml_loss = ml_loss.sum(1) * kwargs['example_weights']
        loss['ml_loss'] = ml_loss.mean()
        loss['loss_per_token'] = ml_loss.div((summ_len - 1).float()).mean()

        return loss

    def forward(self,
                code_word_rep,
                code_word_tokenized_rep,
                code_type_rep,
                code_len,
                summ_word_rep,
                summ_len,
                tgt_seq,
                **kwargs):
        """
        Input:
            - code_word_rep: ``(batch_size, max_doc_len)``
            - code_len: ``(batch_size)``
            - summ_word_rep: ``(batch_size, max_que_len)``
            - summ_len: ``(batch_size)``
            - tgt_seq: ``(batch_size, max_len)``
        Output:
            - ``(batch_size, P_LEN)``, ``(batch_size, P_LEN)``
        """
        if self.training:
            return self._run_forward_ml(code_word_rep,
                                        code_word_tokenized_rep,
                                        code_type_rep,
                                        code_len,
                                        summ_word_rep,
                                        summ_len,
                                        tgt_seq,
                                        **kwargs)

        else:
            return self.decode(code_word_rep,
                               code_word_tokenized_rep,
                               code_type_rep,
                               code_len,
                               **kwargs)

    def __tens2sent(self,
                    t,
                    tgt_dict,
                    src_vocabs):

        words = []
        for idx, w in enumerate(t):
            widx = w[0].item()
            if widx < len(tgt_dict):
                words.append(tgt_dict[widx])
            else:
                widx = widx - len(tgt_dict)
                words.append(src_vocabs[idx][widx])
        return words

    def __generate_sequence(self,
                            params,
                            choice='greedy',
                            tgt_words=None):

        batch_size = params['memory_bank'].size(0)
        use_cuda = params['memory_bank'].is_cuda

        if tgt_words is None:
            tgt_words = torch.LongTensor([constants.BOS])
            if use_cuda:
                tgt_words = tgt_words.cuda()
            tgt_words = tgt_words.expand(batch_size).unsqueeze(1)  # B x 1

        dec_preds = []
        copy_info = []
        attentions = []
        dec_log_probs = []
        acc_dec_outs = []

        max_mem_len = params['memory_bank'][0].shape[1] \
            if isinstance(params['memory_bank'], list) else params['memory_bank'].shape[1]
        dec_states = self.decoder.init_decoder(params['src_len'], max_mem_len)

        attns = {"coverage": None}
        enc_outputs = params['layer_wise_outputs'] if self.layer_wise_attn \
            else params['memory_bank']

        # +1 for <EOS> token
        for idx in range(params['max_len'] + 1):
            tgt = self.embedder(tgt_words,
                                mode='decoder',
                                step=idx)

            tgt_pad_mask = tgt_words.data.eq(constants.PAD)
            layer_wise_dec_out, attns = self.decoder.decode(tgt_pad_mask,
                                                            tgt,
                                                            enc_outputs,
                                                            dec_states,
                                                            step=idx,
                                                            layer_wise_coverage=attns['coverage'])
            decoder_outputs = layer_wise_dec_out[-1]
            acc_dec_outs.append(decoder_outputs.squeeze(1))
            prediction = self.generator(decoder_outputs.squeeze(1))
            prediction = f.softmax(prediction, dim=1)

            if choice == 'greedy':
                tgt_prob, tgt = torch.max(prediction, dim=1, keepdim=True)
                log_prob = torch.log(tgt_prob + 1e-20)
            elif choice == 'sample':
                tgt, log_prob = self.reinforce.sample(prediction.unsqueeze(1))
            else:
                assert False

            dec_log_probs.append(log_prob.squeeze(1))
            dec_preds.append(tgt.squeeze(1).clone())
            if "std" in attns:
                # std_attn: batch_size x num_heads x 1 x src_len
                std_attn = torch.stack(attns["std"], dim=1)
                attentions.append(std_attn.squeeze(2))

            words = self.__tens2sent(tgt, params['tgt_dict'], params['source_vocab'])

            words = [params['tgt_dict'][w] for w in words]
            words = torch.Tensor(words).type_as(tgt)
            tgt_words = words.unsqueeze(1)

        return dec_preds, attentions, copy_info, dec_log_probs

    def decode(self,
               code_word_rep,
               code_word_tokenized_rep,
               code_type_rep,
               code_len,
               **kwargs):

        word_rep = self.embedder(code_word_rep,
                                 code_word_tokenized_rep,
                                 code_type_rep,
                                 mode='encoder')
        memory_bank, layer_wise_outputs = self.encoder(word_rep, code_len)  # B x seq_len x h

        params = dict()
        params['memory_bank'] = memory_bank
        params['layer_wise_outputs'] = layer_wise_outputs
        params['src_len'] = code_len
        params['source_vocab'] = kwargs['source_vocab']
        params['src_mask'] = kwargs['code_mask_rep']
        params['fill'] = kwargs['fill']
        params['blank'] = kwargs['blank']
        params['src_dict'] = kwargs['src_dict']
        params['tgt_dict'] = kwargs['tgt_dict']
        params['max_len'] = kwargs['max_len']
        params['src_words'] = code_word_rep

        dec_preds, attentions, copy_info, _ = self.__generate_sequence(params, choice='greedy')
        dec_preds = torch.stack(dec_preds, dim=1)
        copy_info = torch.stack(copy_info, dim=1) if copy_info else None
        # attentions: batch_size x tgt_len x num_heads x src_len
        attentions = torch.stack(attentions, dim=1) if attentions else None

        return {
            'predictions': dec_preds,
            'copy_info': copy_info,
            'memory_bank': memory_bank,
            'attentions': attentions
        }

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_encoder_parameters(self):
        return self.encoder.count_parameters()

    def count_decoder_parameters(self):
        return self.decoder.count_parameters()

    def layer_wise_parameters(self):
        table = PrettyTable()
        table.field_names = ["Layer Name", "Output Shape", "Param #"]
        table.align["Layer Name"] = "l"
        table.align["Output Shape"] = "r"
        table.align["Param #"] = "r"
        for name, parameters in self.named_parameters():
            if parameters.requires_grad:
                table.add_row([name, str(list(parameters.shape)), parameters.numel()])
        return table
