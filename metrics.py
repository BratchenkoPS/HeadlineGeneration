from rouge import Rouge
from train_eval import generate_headline


def calculate_avg_rouge_f(test_data, SRC, TRG, model, device):
    pred_list = []
    ref_list = []

    for example in test_data.examples[1:]:
        src = example.src

        prediction = generate_headline(src, SRC, TRG, model, device)
        prediction = ' '.join(prediction)

        reference = example.trg
        reference = ' '.join(reference)

        pred_list.append(prediction)
        ref_list.append(reference)

    rouge = Rouge()
    scores = rouge.get_scores(pred_list, ref_list, avg=True)
    return scores
