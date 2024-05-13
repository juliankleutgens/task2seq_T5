import argparse

parser = argparse.ArgumentParser(description="CodeT5+ finetuning on Seq2Seq LM task")
parser.add_argument('--data-num', default=-1, type=int)
parser.add_argument('--max-source-len', default=320, type=int)
parser.add_argument('--max-target-len', default=128, type=int)
parser.add_argument('--cache-data', default='cache_data/summarize_python', type=str)
parser.add_argument('--load', default='Salesforce/codet5p-220m', type=str)

# Training
parser.add_argument('--epochs', default=10, type=int)
parser.add_argument('--lr', default=5e-5, type=float)
parser.add_argument('--lr-warmup-steps', default=200, type=int)
parser.add_argument('--batch-size-per-replica', default=8, type=int)
parser.add_argument('--grad-acc-steps', default=4, type=int)
parser.add_argument('--local_rank', default=-1, type=int)
parser.add_argument('--deepspeed', default=None, type=str)
parser.add_argument('--fp16', default=False, action='store_true')

# Logging and stuff
parser.add_argument('--save-dir', default="saved_models/summarize_python", type=str)
parser.add_argument('--log-freq', default=10, type=int)
parser.add_argument('--save-freq', default=500, type=int)