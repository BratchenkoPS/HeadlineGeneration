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
    logging.info('Calculating metrics')
    with torch.no_grad():
        for example in tqdm(test_data.examples[1:]):
            src = example.src
            src_list.append(src)

            prediction = generate_headline(src, SRC, TRG, model, device)
            if len(prediction) > 1:
                prediction = ' '.join(prediction[:-1])
            else:
                prediction = ' '.join(prediction)

            reference = example.trg
            reference = ' '.join(reference)

            pred_list.append(prediction)
            ref_list.append(reference)

    rouge = Rouge()
    scores = rouge.get_scores(pred_list, ref_list)

    metrics = []
    for example in scores:
        metric_ex = []
        for metric in example.values():
            metric_ex.append(metric['f'])
        metrics.append(sum(metric_ex) / len(metric_ex))

    n_example = metrics.index(max(metrics))  # cherry picking

    logging.info('Best prediction title: {}'.format(pred_list[n_example]))
    logging.info('Reference title: {}'.format(ref_list[n_example]))
    logging.info('Reference text: {}'.format(src_list[n_example]))

    avg_f = sum(metrics) / len(metrics)
    return avg_f
