# coding:utf-8
import torch
import numpy as np
import json
import sys
import os
import argparse
import logging
import framework
import encoder
import model1
import model2 

parser = argparse.ArgumentParser()
parser.add_argument('--pretrain_path', default='bert-base-uncased',
        help='Pre-trained ckpt path / model name (hugginface)')
parser.add_argument('--ckpt', default='verified_nyt10_Passage_Level',
        help='Checkpoint name')
parser.add_argument('--only_test', action='store_true',
        help='Only run test')
parser.add_argument('--mask_entity', action='store_true',
        help='Mask entity mentions')
parser.add_argument('--metric', default='auc', choices=['micro_f1', 'auc','p@10','p@30'],
        help='Metric for picking up best checkpoint')
parser.add_argument('--train_file', default='nyt10/nyt10_train.txt', type=str,
        help='Training data file')
parser.add_argument('--val_file', default='nyt10/nyt10_test.txt', type=str,
        help='Validation data file')
parser.add_argument('--test_file', default='nyt10/nyt10_test.txt', type=str,
        help='Test data file')
parser.add_argument('--rel2id_file', default='nyt10/nyt10_rel2id.json', type=str,
        help='Relation to ID file')
parser.add_argument('--batch_size', default=16, type=int,
        help='Batch size')
parser.add_argument('--lr', default=2e-5, type=float,
        help='Learning rate')
parser.add_argument('--optim', default='adamw', type=str,
        help='Optimizer')
parser.add_argument('--weight_decay', default=1e-5, type=float,
        help='Weight decay')
parser.add_argument('--max_length', default=512, type=int,
        help='Maximum sentence length')
parser.add_argument('--max_epoch', default=5, type=int,
        help='Max number of training epochs')
parser.add_argument('--save_name', default='', type=str,
        help='name for saving checkpoint')
parser.add_argument('--seed', default=772, type=int,
        help='random seed')
parser.add_argument(
  "--devs",  
  nargs="*",
  type=int,
  default=[0,1],
  help='list of gpu ids on which model needs to be run'
)

args = parser.parse_args()
import os
import random
    
def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

seed_everything(seed=args.seed)

# Some basic settings
root_path = '.'
sys.path.append(root_path)
if not os.path.exists('ckpt'):
    os.mkdir('ckpt')
if args.ckpt.endswith('.pth.tar') or '/' in args.ckpt:
    ckpt = args.ckpt
else:
    ckpt = 'ckpt/{}.pth.tar'.format(args.ckpt)
print(ckpt)
if args.only_test:
    if not (os.path.exists(args.test_file) and os.path.exists(args.rel2id_file)):
        raise Exception('--test_file and --rel2id_file are not specified or files do not exist. Or specify --dataset')
else:
    if not (os.path.exists(args.train_file) and os.path.exists(args.val_file) and os.path.exists(args.test_file) and os.path.exists(args.rel2id_file)):
        raise Exception('--train_file, --val_file, --test_file and --rel2id_file are not specified or files do not exist. Or specify --dataset')

logging.info('Arguments:')
for arg in vars(args):
    logging.info('    {}: {}'.format(arg, getattr(args, arg)))

rel2id = json.load(open(args.rel2id_file))

# Define the passage encoder
passage_encoder = encoder.PassageEncoder(
    pretrain_path=args.pretrain_path,
    batch_size = args.batch_size,
    mask_entity=args.mask_entity
)
# Define the model
# For NYT, use separate fully connected layers for NA and non-NA (a shared layer for all the non-NA labels)
if 'nyt' in args.test_file:
	model_ = model2.PassageAttention(passage_encoder, len(rel2id), rel2id)
else:
	model_ = model1.PassageAttention(passage_encoder, len(rel2id), rel2id)

if args.only_test:
    from framework.data_loader import PassageRELoader
    test_loader = PassageRELoader(
        path=args.test_file,
        rel2id=model_.rel2id,
        tokenizer=model_.passage_encoder.tokenize,
        batch_size=args.batch_size,
        shuffle=False)
    framework_ = framework.PassageRE(
        model=model_,
        train_path=None,
        val_path=None,
        test_path=args.test_file,
        ckpt=ckpt,
        batch_size=args.batch_size,
        max_epoch=args.max_epoch,
        lr=args.lr,
        weight_decay=args.weight_decay,
        opt='adamw',
        warmup_step = 30000 // args.batch_size,
        devices = args.devs)
    framework_.test_loader = test_loader
else:
    framework_ = framework.PassageRE(
        model=model_,
        train_path=args.train_file,
        val_path=args.val_file,
        test_path=args.test_file,
        ckpt=ckpt,
        batch_size=args.batch_size,
        max_epoch=args.max_epoch,
        lr=args.lr,
        weight_decay=args.weight_decay,
        opt='adamw',
        warmup_step = 30000 // args.batch_size,
        devices = args.devs)

if not args.only_test:
    framework_.train_model(args.metric)

# Test
if os.path.exists(ckpt):
    framework_.load_state_dict(torch.load(ckpt, map_location='cuda:0')['state_dict'])
else:
    raise Exception('Checkpoint not found: ' + str(ckpt))
result = framework_.eval_model(framework_.test_loader)
# Print the result
#print(result)
#pred = result['pred']
#with open("prediction.txt", "w") as f:
#	for key in pred:
#		f.write(key)
#		preds = pred[key]['prediction']
#		for p in preds:
#			f.write("\t"+p)
#		f.write("\n")
print('Test set results for ckpt = ' +str(ckpt)+ ' are:')
print("AUC: %.4f" % result['auc'])
print("Average P@100: %.4f" % result['p@100'])
print("Average P@200: %.4f" % result['p@200'])
print("Average P@300: %.4f" % result['p@300'])
print("Average P@M: %.4f" % result['avg_p300'])
print("Recall@5 (micro): %.4f" % result['recall@5_micro'])
print("Recall@5 (macro): %.4f" % result['recall@5_macro'])
print("MaxMicro F1: %.4f" % (result['max_micro_f1']))
print("MaxMacro F1: %.4f" % (result['max_macro_f1']))
print("Micro F1: %.4f" % (result['micro_f1']))
print("Macro F1: %.4f" % (result['macro_f1']))
