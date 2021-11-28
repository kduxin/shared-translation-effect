
import os
import torch
from torch.utils.data import (
    IterableDataset,
    DataLoader,
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


class DummyScheduler:
    def __init__(self, lr=0): self.lr = [lr]
    def step(self): pass
    def get_last_lr(self): return self.lr


def select_parameters_fortuning(models, args):
    return models.parameters()

def create_optimizer(parameters, args):
    return AdamW(parameters, lr=args.lr)

def create_scheduler(optimizer, args):
    return transformers.optimization.get_linear_schedule_with_warmup(optimizer, int(0.2*args.n_iters), args.n_iters)
    # return OneCycleLR(optimizer, max_lr=args.lr, total_steps=args.n_iters+5)
    # return DummyScheduler(lr=args.lr)

def nlines(filepath):
    res = os.popen(f'wc -l {filepath}').read()
    nlines = int(res.split()[0])
    return nlines

class LinePairDataset(IterableDataset):
    def __init__(self, source_path, target_path, skiplines):
        self.source_path = source_path
        self.target_path = target_path

        logger.info('Counting lines of the source/target language dataset...')
        source_nlines, target_nlines = nlines(source_path), nlines(target_path)
        logger.info(f'Source: {source_nlines}; target: {target_nlines}')
        assert source_nlines == target_nlines, \
            f'source ({source_nlines}) has different number of lines with target ({target_nlines})'
        self.source_nlines, self.target_nlines = source_nlines, target_nlines

        self.skiplines = skiplines

    def __len__(self):
        return self.nlines
    
    def __iter__(self):
        if hasattr(self, 'fsrc'):
            self.fsrc.close()
        if hasattr(self, 'ftgt'):
            self.ftgt.close()
        self.fsrc = open(self.source_path, 'rt')
        self.ftgt = open(self.target_path, 'rt')

        for _ in range(self.skiplines):
            self.fsrc.readline()
            self.ftgt.readline()

        for srcline, tgtline in zip(self.fsrc, self.ftgt):
            yield srcline, tgtline


def make_collate_fn(source_tok, target_tok):
    def collate_fn(linepairs):
        srclines, tgtlines    = list(zip(*linepairs))
        srcids                = source_tok(list(srclines), return_tensors='pt', padding=True, truncation=True, add_special_tokens=False)
        # srcids = source_tok(list(srclines), return_tensors='pt', padding=True, truncation=True)
        tgtids                = target_tok(list(tgtlines), return_tensors='pt', padding=True, truncation=True)
        return srcids, tgtids
    
    return collate_fn


def load_tokenizer_pair(args):
    encoder_tok = transformers.AutoTokenizer.from_pretrained(args.encoder_name)
    encoder_tok.model_max_length = args.max_len
    encoder_tok.pad_token = encoder_tok.unk_token

    decoder_tok = transformers.AutoTokenizer.from_pretrained(args.decoder_name)
    decoder_tok.model_max_length = args.max_len
    return encoder_tok, decoder_tok

def train(args):
    args.debug = True
    if args.cpu or (not torch.cuda.is_available()):
        device = 'cpu'
    else:
        device = 'cuda'
    if not os.path.exists(args.savedir):
        os.makedirs(args.savedir)

    # ---------------- create model, optimizer -----------------
    source_tok, target_tok = load_tokenizer_pair(args)

    if args.pretrained_path is not None:
        model = torch.load(args.pretrained_path, map_location=device)
    else:
        model = transformers.EncoderDecoderModel.from_encoder_decoder_pretrained(args.encoder_name, args.decoder_name)
        model.config.decoder_start_token_id = target_tok.cls_token_id
        model.config.eos_token_id = target_tok.eos_token_id
        model.config.pad_token_id = target_tok.pad_token_id
        model.decoder.config.use_cache = False
        
        model = model.to(device)
    
    if args.tune_wordemb_only:
        model.encoder.requires_grad_(False)
        model.encoder.wte.requires_grad_(True)

    # model.encoder.init_weights()
    # model.encoder.requires_grad_(False)
    # model.decoder.init_weights()

    print(model)

    parameters = select_parameters_fortuning(model, args)
    optimizer = create_optimizer(parameters, args)
    scheduler = create_scheduler(optimizer, args)
    logger.info('Model, optimizer, scheduler initialization finished.')

    # ------------------- prepare dataloader -------------------
    dataset = LinePairDataset(args.source_path, args.target_path, skiplines=args.skiplines)
    collate_fn = make_collate_fn(source_tok, target_tok)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_fn)

    testsrc = 'Adopted by the Security Council at its 3377th meeting, on 17 May 1994'
    test_input = source_tok(testsrc, return_tensors='pt', add_special_tokens=False)
    test_input_ids = test_input.input_ids.to(device)
    test_attention_mask = test_input.attention_mask.to(device)

    # ------------------- training & eval ----------------------
    for i, (source_tokenized, target_tokenized) in enumerate(dataloader):
        logger.debug(f'source ids:{source_tokenized.input_ids}')
        logger.debug(f'target ids:{target_tokenized.input_ids}')
        i = i + args.pretrained_niters
        if i > args.n_iters + args.pretrained_niters:
            break
        if i % args.save_interval == 0:
            savepath = f'{args.savedir}/{i:0>5}.pt'
            torch.save(model, savepath)

        optimizer.zero_grad()
        labels = target_tokenized.input_ids.to(device)
        output = model(input_ids=source_tokenized.input_ids.to(device), 
                       attention_mask=source_tokenized.attention_mask.to(device),
                       labels=labels[:, 1:].contiguous(),
                    #    decoder_attention_mask=target_tokenized.attention_mask.to(device),
                       output_hidden_states=True,
        )
        loss = output.loss
        
        loss.backward()
        optimizer.step()
        scheduler.step()

        if args.debug and i % args.log_interval == 0:
            logger.info(f'Iter {i:0>5}. loss={loss.item():5.3f}. lr={scheduler.get_last_lr()[0]:6.4f}')
            logger.debug(f'Encoder last hidden states:\n{output.encoder_last_hidden_state[0, :5]}')
            with torch.no_grad():
                genoutput = model.generate(input_ids=test_input_ids,
                                        attention_mask=test_attention_mask,
                                        decoder_start_token_id=target_tok.cls_token_id)
                testgen = target_tok.decode(genoutput[0])
            logger.info(f'Test: {testsrc} -> {testgen}')


def parse_args():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--source_path', type=str, default='data/en-zh/UNv1.0.en-zh.en')
    parser.add_argument('--target_path', type=str, default='data/en-zh/UNv1.0.en-zh.zh')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--n_iters', type=int, default=10000)
    parser.add_argument('--save_interval', type=int, default=1000)
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--savedir', type=str, default='ckp/test/')
    parser.add_argument('--encoder_name', type=str, default='bert-base-uncased')
    parser.add_argument('--decoder_name', type=str, default='bert-base-chinese')
    parser.add_argument('--max_len', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--pretrained_path', type=str, default=None)
    parser.add_argument('--pretrained_niters', type=int, default=0)
    parser.add_argument('--tune_wordemb_only', action='store_true')
    parser.add_argument('--skiplines', type=int, default=0)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    train(args)