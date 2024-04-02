import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import torch
import argparse
from tqdm import tqdm
import torch.backends.cudnn as cudnn
from config import cfg
from common.base import Tester


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='0', dest='gpu_ids')
    parser.add_argument('--test_epoch', default='25', dest='test_epoch')
    args = parser.parse_args()



    if not args.gpu_ids:
        assert 0, "Please set propoer gpu ids"

    if '-' in args.gpu_ids:
        gpus = args.gpu_ids.split('-')
        gpus[0] = int(gpus[0])
        gpus[1] = int(gpus[1]) + 1
        args.gpu_ids = ','.join(map(lambda x: str(x), list(range(*gpus))))

    assert args.test_epoch, 'Test epoch is required.'
    return args

def main():

    args = parse_args()
    cfg.set_args(args.gpu_ids)
    cudnn.benchmark = True
    test_epoch = args.test_epoch

    tester = Tester(test_epoch)
    tester._make_batch_generator()
    tester._make_model()

    cur_sample_idx = 0
    for itr, (inputs, targets, meta_info) in enumerate(tqdm(tester.batch_generator)):

        # forward
        with torch.no_grad():
            out = tester.model(inputs)

        # save output
        out = {k: v.cpu().numpy() for k,v in out.items()}
        for k,v in out.items(): batch_size = out[k].shape[0]
        out = [{k: v[bid] for k,v in out.items()} for bid in range(batch_size)]

        # evaluate
        tester._evaluate(out, cur_sample_idx)
        cur_sample_idx += len(out)

    tester._print_eval_result(test_epoch)

if __name__ == "__main__":
    main()