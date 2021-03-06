import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import time

from dataloader import DataLoader
from embeddings import Embedder
from clustering import Clustering
from splitter import Splitter
from encoder import Encoder
from decoder import Decoder
from model import Seq2Seq
from train_eval import train, evaluate
from utils import count_parameters, epoch_time
from metrics import calculate_avg_rouge_f

if __name__ == '__main__':
    with open('config.yml', 'r') as file:
        config = yaml.load(file)

    loader = DataLoader(config['Dataloader']['url'],
                        config['Dataloader']['path'],
                        config['Dataloader']['name'])

    data = loader.get_data(config['Dataloader']['max_txt_length'],
                           config['Dataloader']['samples'])

    emb = Embedder(config['Embedder'])

    embeddings = emb.get_embeddings(data['title'])

    clustering = Clustering(data,
                            config['Clustering']['directory'],
                            config['Clustering']['cluster_picture_name'],
                            config['Clustering']['result_data_file_name'],
                            config['Clustering']['center_replics_file_name'],
                            config['Clustering']['part_to_plot'],
                            config['Clustering']['bgm_config'])

    df = clustering.get_clusters_and_final_data(embeddings)
    # import pandas as pd
    # df = pd.read_csv('result/result_df.csv')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    splitter = Splitter(df,
                        config['Splitter']['path_to_save_data'],
                        config['Splitter']['min_freq'],
                        config['Splitter']['test_size'],
                        config['Splitter']['batch_size'],
                        device)

    train_iterator, test_iterator, train_data, test_data, SRC, TRG = splitter.get_iterators_and_fields()

    input_dim = len(SRC.vocab)
    output_dim = len(TRG.vocab)
    trg_pad_idx = TRG.vocab.stoi[TRG.pad_token]

    enc = Encoder(input_dim,
                  config['model']['EMB_DIM'],
                  config['model']['HID_DIM'],
                  config['model']['ENC_LAYERS'],
                  config['model']['ENC_KERNEL_SIZE'],
                  config['model']['ENC_DROPOUT'],
                  device)

    dec = Decoder(output_dim,
                  config['model']['EMB_DIM'],
                  config['model']['HID_DIM'],
                  config['model']['DEC_LAYERS'],
                  config['model']['DEC_KERNEL_SIZE'],
                  config['model']['DEC_DROPOUT'],
                  trg_pad_idx,
                  device)

    model = Seq2Seq(enc, dec).to(device)

    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss(ignore_index=trg_pad_idx)
    count_parameters(model)

    N_EPOCHS = 10
    CLIP = 0.1

    best_valid_loss = float('inf')

    for epoch in range(N_EPOCHS):
        start_time = time.time()

        train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
        test_loss = evaluate(model, test_iterator, criterion)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        if epoch > 5:
            metrics_epoch = calculate_avg_rouge_f(test_data, SRC, TRG, model, device)
            print(f'\t Val. Loss: {test_loss:.3f} |  Metrics_val: {metrics_epoch}')
        if test_loss < best_valid_loss:
            best_valid_loss = test_loss
            torch.save(model.state_dict(), 'result/tut5-model.pt')

        print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\t Val. Loss: {test_loss:.3f}')
