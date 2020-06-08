from argparse import ArgumentParser

from mmdet.core import coco_eval
import os


def main():
    parser = ArgumentParser(description='COCO Evaluation')
    parser.add_argument('result', help='result file path')
    parser.add_argument('--ann', help='annotation file path')
    parser.add_argument(
        '--types',
        type=str,
        nargs='+',
        choices=['proposal_fast', 'proposal', 'bbox', 'segm', 'keypoint'],
        default=['bbox'],
        help='result types')
    parser.add_argument(
        '--max-dets',
        type=int,
        nargs='+',
        default=[100, 300, 1000],
        help='proposal numbers, only used for recall evaluation')
    args = parser.parse_args()
  #  result_files = {'bbox':'results.pkl.bbox.json','segm':'results.pkl.segm.json'}
    result_files = {'bbox':args.result}
    coco_eval(result_files, args.types, args.ann, args.max_dets)


if __name__ == '__main__':
    main()
