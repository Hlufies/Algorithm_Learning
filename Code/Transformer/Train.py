import torch
import torch.nn as nn
import torch.utils.data as Data
from torch import optim
from Transformer import Transformer
from utils import make_data
from utils import MyDataSet
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
if __name__ == "__main__":
    #参数设置
    batch_size = 2
    epochs = 30
    workers = 4
    lr = 1e-3
    saveResult = 'D:\FUDAN-MAS\demo4\pth'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_layers = 3    # encoder layers = decoder layers 数量
    d_model = 512   # outputs 模型的输出
    d_k = 64
    d_q = 64
    d_v = 64
    d_ff = 512
    n_heads = 4
    """                                
    d_k => the keys of dimension           
    d_q => the queries of dimension     
    d_v => the values of dimension   
    d_ff=> the FeedForward of dimension   
    """
    sentences = [
        # enc_input                 dec_input             dec_output
        ['ich mochte ein bier P', 'S i want a beer .', 'i want a beer . E'],
        ['ich mochte ein cola P', 'S i want a coke .', 'i want a coke . E'],
    ]

    src_vocab = {'P' : 0, 'ich' : 1, 'mochte' : 2, 'ein' : 3, 'bier' : 4, 'cola' : 5}
    src_vocab_size = len(src_vocab)

    tgt_vocab = {'P' : 0, 'i' : 1, 'want' : 2, 'a' : 3, 'beer' : 4, 'coke' : 5, 'S' : 6, 'E' : 7, '.' : 8}
    tgt_vocab_size = len(tgt_vocab)

    src_len = 5 # encoder inputs max sequence length
    tgt_len = 6 # decoder inputs max sequence length = decoder outputs max sequence length

    encoder_inputs, decoder_inputs, decoder_outputs = make_data(sentences,src_vocab,tgt_vocab)
    #
    # print(f'encoder_inputs shape {encoder_inputs.shape}')
    # print(encoder_inputs)
    #
    # print(f'decoder_inputs shape {decoder_inputs.shape}')
    # print(decoder_inputs)
    # print(f'')
    """
    encoder inputs  shape: torch.Size([2, 5])
    encoder inputs  shape: torch.Size([2, 6])
    encoder outputs shape: torch.Size([2, 6])  
    """

    train_loader = Data.DataLoader(dataset=MyDataSet(encoder_inputs=encoder_inputs,decoder_inputs=decoder_inputs,decoder_outputs=decoder_outputs),
                                   batch_size=batch_size,
                                   shuffle=True,
                                   num_workers=workers)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    model = Transformer(src_vocab_size=src_vocab_size,
                        tgt_vocab_size=tgt_vocab_size,
                        n_layers=n_layers,
                        n_heads=n_heads,
                        d_model=d_model,
                        d_k=d_k,
                        d_v=d_v,
                        d_ff=d_ff)
    model.apply(weights_init)
    # print(model)
    model = model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.99)

    outputs, encoder_self_attentions, decoder_self_attentions, decoder_encoder_attentions = model(encoder_inputs, decoder_inputs)
    loss = criterion(outputs,decoder_outputs.view(-1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # for epoch in range(epochs):
    #     for index,[enc_inputs, dec_inputs, dec_outputs] in enumerate(train_loader):
    #         """
    #         enc_inputs : [num_experts,src_len]
    #         dec_inputs : [num_experts,tgt_len]
    #         dec_outputs: [num_experts,tgt_len]
    #         """
    #         enc_inputs = enc_inputs.to(device)
    #         dec_inputs = dec_inputs.to(device)
    #         dec_outputs= dec_outputs.to(device)
    #
    #         outputs, encoder_self_attentions, decoder_self_attentions, decoder_encoder_attentions = model(enc_inputs, dec_inputs)
    #         loss = criterion(outputs,dec_outputs.view(-1))
    #         print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(loss))
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()