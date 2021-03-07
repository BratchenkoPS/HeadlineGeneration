import torch
import logging

from rouge import Rouge
from train_eval import generate_headline
from tqdm import tqdm


def calculate_avg_rouge_f(test_data, SRC, TRG, model, device):
    pred_list = []
    ref_list = []
    src_list = []
    model.eval()
    with torch.no_grad():
        for example in tqdm(test_data.examples[1:]):
            src = example.src
            src_list.append(src)
            prediction = generate_headline(src, SRC, TRG, model, device)
            prediction = ' '.join(prediction)

            reference = example.trg
            reference = ' '.join(reference)

            pred_list.append(prediction)
            ref_list.append(reference)

    rouge = Rouge()
    scores = rouge.get_scores(pred_list, ref_list, avg=True)
    logging.info('Prediction title: {}'.format(pred_list[0]))
    logging.info('Reference title: {}'.format(ref_list[0]))
    logging.info('Reference text: {}'.format(src_list[0]))
    f_scores = []
    for metric in scores.values():
        f_scores.append(metric['f'])
    avg_f = sum(f_scores) / len(f_scores)
    return avg_f
