
import os
import torch
from torch.utils.data import (
    IterableDataset,
    DataLoader,
)
from torch.nn import (
    Module,
)
from torch.optim import (
    AdamW,
)
from torch.optim.lr_scheduler import (
    OneCycleLR,
)
import transformers

import logging
from rich.logging import RichHandler
logging.basicConfig(
    level="INFO", format="%(message)s", datefmt="[%X]", handlers=[RichHandler()]
)
logger = logging.getLogger("rich")


def select_parameters_fortuning(models, args):
    return models.parameters()

def create_optimizer(parameters, args):
    return AdamW(parameters, lr=args.lr)

def create_scheduler(optimizer, args):
    return OneCycleLR(optimizer, max_lr=args.lr, total_steps=args.n_iters+5)

def nlines(filepath):
    res = os.popen(f'wc -l {filepath}').read()
    nlines = int(res.split()[0])
    return nlines

class LinePairDataset(IterableDataset):
    def __init__(self, source_path, target_path):
        self.source_path = source_path
        self.target_path = target_path

        logger.info('Counting lines of the source/target language dataset...')
        source_nlines, target_nlines = nlines(source_path), nlines(target_path)
        logger.info(f'Source: {source_nlines}; target: {target_nlines}')
        assert source_nlines == target_nlines, \
            f'source ({source_nlines}) has different number of lines with target ({target_nlines})'
        self.source_nlines, self.target_nlines = source_nlines, target_nlines

    def __len__(self):
        return self.nlines
    
    def __iter__(self):
        if hasattr(self, 'fsrc'):
            self.fsrc.close()
        if hasattr(self, 'ftgt'):
            self.ftgt.close()
        self.fsrc = open(self.source_path, 'rt')
        self.ftgt = open(self.target_path, 'rt')

        for srcline, tgtline in zip(self.fsrc, self.ftgt):
            yield srcline, tgtline


def make_collate_fn(source_tok, target_tok):
    def collate_fn(linepairs):
        srclines, tgtlines = list(zip(*linepairs))
        srcids = source_tok(list(srclines), return_tensors='pt', padding=True, truncation=True, add_special_tokens=False)
        tgtids = target_tok(list(tgtlines), return_tensors='pt', padding=True, truncation=True)
        return srcids, tgtids
    
    return collate_fn


def load_model_pair(args, bos_token_id, eos_token_id):
    encoder = transformers.BertModel.from_pretrained(args.encoder_name)
    encoder = transformers.BertGenerationEncoder.from_pretrained(args.encoder_name, )
    decoder = transformers.BertGenerationDecoder.from_pretrained(args.decoder_name)
    return encoder, decoder

def load_tokenizer_pair(args):
    encoder_tok = transformers.BertTokenizerFast.from_pretrained(args.encoder_name, max_len=args.max_len)
    decoder_tok = transformers.BertTokenizerFast.from_pretrained(args.decoder_name, max_len=args.max_len)
    return encoder_tok, decoder_tok

class Translator(Module):
    def __init__(self, args):
        super().__init__()
        self.encoder, self.decoder = load_model_pair(args)
    
    def forward(self, source_ids, source_attention_mask,
                target_ids, target_attention_mask):
        encoder_output = self.encoder(
            input_ids=source_ids, attention_mask=source_attention_mask,
        )
        decoder_output = self.decoder(
            input_ids=target_ids,
            attention_mask=target_attention_mask,
            encoder_hidden_states=encoder_output.last_hidden_state,
            encoder_attention_mask=source_attention_mask,
            labels=target_ids,
        )
        return decoder_output


def train(args):
    if args.cpu or (not torch.cuda.is_available()):
        device = 'cpu'
    else:
        device = 'cuda'
    if not os.path.exists(args.savedir):
        os.makedirs(args.savedir)

    # ---------------- create model, optimizer -----------------
    source_tok, target_tok = load_tokenizer_pair(args)

    encoder = transformers.BertGenerationEncoder.from_pretrained(
        args.encoder_name,
        bos_token_id=source_tok.cls_token_id,
        eos_token_id=source_tok.sep_token_id,
    )
    # add cross attention layers and use BERT's cls token as BOS token and sep token as EOS token
    decoder = transformers.BertGenerationDecoder.from_pretrained(
        args.decoder_name, 
        add_cross_attention=True, 
        is_decoder=True, 
        bos_token_id=target_tok.cls_token_id,
        eos_token_id=target_tok.sep_token_id,
    )
    model = transformers.EncoderDecoderModel(encoder=encoder, decoder=decoder)
    model = model.to(device)
    print(model)

    parameters = select_parameters_fortuning(model, args)
    optimizer = create_optimizer(parameters, args)
    scheduler = create_scheduler(optimizer, args)
    logger.info('Model, optimizer, scheduler initialization finished.')

    # ------------------- prepare dataloader -------------------
    dataset = LinePairDataset(args.source_path, args.target_path)
    collate_fn = make_collate_fn(source_tok, target_tok)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_fn)

    prev_encoder_wordemb = model.encoder.embeddings.word_embeddings.weight.clone()
    prev_decoder_wordemb = model.decoder.bert.embeddings.word_embeddings.weight.clone()

    for i, (source_tokenized, target_tokenized) in enumerate(dataloader):
        if i > args.n_iters:
            break
        if i % args.save_interval == 0:
            if args.debug:
                savepath = f'{args.savedir}/{i:0>5}.pt'
                torch.save(model, savepath)
            else:
                savepath1 = f'{args.savedir}/{i:0>5}.encoder_wordemb.pt'
                torch.save(model.encoder.embeddings.word_embeddings, savepath1)
                savepath2 = f'{args.savedir}/{i:0>5}.decoder_wordemb.pt'
                torch.save(model.decoder.bert.embeddings.word_embeddings, savepath2)
                logger.info(f'Saved model to {savepath1} and {savepath2}')

        optimizer.zero_grad()
        input_ids = source_tokenized.input_ids.to(device)
        labels = target_tokenized.input_ids.to(device)
        output = model(input_ids=input_ids, decoder_input_ids=labels, labels=labels)
        loss = output.loss
        
        loss.backward()
        optimizer.step()
        scheduler.step()

        logger.info(f'Iter {i+1:0>5}. loss={loss.item():5.3f}. lr={scheduler.get_last_lr()[0]:6.4f}')
        encoder_wordemb = model.encoder.embeddings.word_embeddings.weight
        decoder_wordemb = model.decoder.bert.embeddings.word_embeddings.weight
        logger.info(f'Mean difference of encoder word embedding: {(encoder_wordemb - prev_encoder_wordemb).abs().mean().item():.3g}')
        logger.info(f'Mean difference of decoder word embedding: {(decoder_wordemb - prev_decoder_wordemb).abs().mean().item():.3g}')
        prev_encoder_wordemb = encoder_wordemb.clone()
        prev_decoder_wordemb = decoder_wordemb.clone()

    # ------------------- training & eval ----------------------


def parse_args():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--source_path', type=str, default='data/en-zh/UNv1.0.en-zh.en')
    parser.add_argument('--target_path', type=str, default='data/en-zh/UNv1.0.en-zh.zh')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--n_iters', type=int, default=1000)
    parser.add_argument('--save_interval', type=int, default=100)
    parser.add_argument('--savedir', type=str, default='ckp/test/')
    parser.add_argument('--encoder_name', type=str, default='bert-base-uncased')
    parser.add_argument('--decoder_name', type=str, default='bert-base-chinese')
    parser.add_argument('--max_len', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    train(args)