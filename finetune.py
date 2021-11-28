
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
    return transformers.optimization.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.2*args.n_iters), num_training_steps=args.n_iters)
    # return OneCycleLR(optimizer, max_lr=args.lr, total_steps=args.n_iters+5)
    # return DummyScheduler(lr=args.lr)

def nlines(filepath):
    res = os.popen(f'wc -l {filepath}').read()
    nlines = int(res.split()[0])
    return nlines

class LineDataset(IterableDataset):
    def __init__(self, source_path, skiplines):
        self.source_path = source_path

        logger.info('Counting lines of the dataset...')
        source_nlines      = nlines(source_path)
        logger.info(f'Source: {source_nlines} lines')
        self.source_nlines = source_nlines

        self.skiplines = skiplines

    def __len__(self):
        return self.nlines
    
    def __iter__(self):
        if hasattr(self, 'fsrc'):
            self.fsrc.close()
        self.fsrc = open(self.source_path, 'rt')

        for _ in range(self.skiplines):
            self.fsrc.readline()

        for srcline in self.fsrc:
            yield srcline


def make_collate_fn(source_tok):
    def collate_fn(lines):
        srcids = source_tok(list(lines), return_tensors='pt', padding=True, truncation=True, add_special_tokens=False)
        return srcids
    
    return collate_fn


def train(args):
    if args.cpu or (not torch.cuda.is_available()):
        device = 'cpu'
    else:
        device = 'cuda'
    if not os.path.exists(args.savedir):
        os.makedirs(args.savedir)

    # ---------------- create model, optimizer -----------------
    source_tok                  = transformers.AutoTokenizer.from_pretrained('gpt2')
    source_tok.pad_token        = source_tok.unk_token
    source_tok.model_max_length = args.max_len

    if args.pretrained_path is not None:
        model = torch.load(args.pretrained_path, map_location=device)
    else:
        model = transformers.GPT2LMHeadModel.from_pretrained('gpt2')
        model = model.to(device)
    
    if args.tune_wordemb_only:
        model.requires_grad_(False)
        model.transformer.wte.requires_grad_(True)
        model.lm_head.requires_grad_(True)

    print(model)

    parameters = select_parameters_fortuning(model, args)
    optimizer  = create_optimizer(parameters, args)
    scheduler  = create_scheduler(optimizer, args)
    logger.info('Model, optimizer, scheduler initialization finished.')

    # ------------------- prepare dataloader -------------------
    dataset    = LineDataset(args.source_path, skiplines=args.skiplines)
    collate_fn = make_collate_fn(source_tok)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_fn)

    # ------------------- training & eval ----------------------
    for i, source_tokenized in enumerate(dataloader):
        logger.debug(f'source ids:{source_tokenized.input_ids}')
        i = i + args.pretrained_niters
        if i > args.n_iters + args.pretrained_niters:
            break
        if i % args.save_interval == 0:
            savepath = f'{args.savedir}/{i:0>5}.pt'
            torch.save(model, savepath)

        optimizer.zero_grad()
        input_ids = source_tokenized.input_ids.to(device)
        output    = model(input_ids            = input_ids,
                          attention_mask       = source_tokenized.attention_mask.to(device),
                          labels               = input_ids,
                          output_hidden_states = True,
                          )
        loss = output.loss
        
        loss.backward()
        optimizer.step()
        scheduler.step()

        if i % args.log_interval == 0:
            logger.info(f'Iter {i:0>5}. loss={loss.item():5.3f}. lr={scheduler.get_last_lr()[0]:6.4f}')
            logger.debug(f'Encoder last hidden states:\n{output.hidden_states[-1][0, :5]}')

def parse_args():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--source_path', type=str, default='data/en-zh/UNv1.0.en-zh.en')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--n_iters', type=int, default=100000)
    parser.add_argument('--save_interval', type=int, default=5000)
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--savedir', type=str, default='ckp/finetuned.gpt2/')
    parser.add_argument('--max_len', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--pretrained_path', type=str, default=None)
    parser.add_argument('--pretrained_niters', type=int, default=0)
    parser.add_argument('--skiplines', type=int, default=0)
    parser.add_argument('--tune_wordemb_only', action='store_true')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    train(args)
