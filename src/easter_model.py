import config
import torch
from torch import nn

def ctc_loss(args):
    y_pred, labels, input_length, label_length = args
    ctc_loss = nn.CTCLoss()
    return ctc_loss(y_pred,
                    labels,
                    input_length,
                    label_length)

def ctc_custom(args):
    
    # Custom CTC Loss

    y_pred, labels, input_length, label_length = args
    ctc_loss = nn.CTCLoss()
    loss = ctc_loss(y_pred,
                    labels,
                    input_length,
                    label_length)
    p = torch.exp(-loss)
    gamma = 0.5
    alpha = 0.25
    return alpha * (torch.pow((1 - p), gamma)) * loss

def batch_norm(inputs):
    return nn.BatchNorm1d(
        eps = config.BATCH_NORM_EPSILON,
        momentum = config.BATCH_NORM_DECAY
    )(inputs)

def add_global_context(data, filters):

    # 1D Squeeze and Excitation Layer

    SHAPE = data.shape[-1]
    AGC = nn.Sequential(
        nn.AvgPool1d(),
        nn.Linear(in_features = SHAPE, out_features = filters // 8),
        nn.ReLU(),
        nn.Linear(in_features = filters // 8, out_features = filters),
        nn.Sigmoid()
    )
    pool = AGC(data)
    final = torch.mul(data, pool)
    return final

def easter_unit(old, data, filters, kernel, stride, dropouts):

    # Easter unint with dense residual connections

    SHAPE_data = data.shape[-1]
    SEQ_1 = nn.Conv1d(
        in_channels = SHAPE_data,
        out_channels = filters,
        kernel_size = 1,
        stride = 1,
        padding = 'same'
    )
    old = SEQ_1(old)
    old = batch_norm(old)
    this = SEQ_1(data)
    this = batch_norm(this)
    old = old + this

    #FIRST BLOCK
    CONV_BLOCK_1 = nn.Conv1d(
            in_channels = SHAPE_data,
            out_channels = filters,
            stride = stride,
            kernel_size = kernel,
            padding = 'same'        
    )
    data = CONV_BLOCK_1(data)
    data = batch_norm(data)
    ACT_BLOCK_1 = nn.Sequential(
        nn.ReLU(),
        nn.Dropout(dropouts)
    )
    data = ACT_BLOCK_1(data)

    #SECOND BLOCK
    data = CONV_BLOCK_1(data)
    data = batch_norm(data)
    data = ACT_BLOCK_1(data)

    #THIRD BLOCK
    data = CONV_BLOCK_1(data)
    data = batch_norm(data)

    #SE
    data = add_global_context(data, filters)
    final = old + data
    RELU = nn.ReLU()
    DROP = nn.Dropout(dropouts)
    data = RELU(final)
    data = DROP(data)
    
    return data, old

class Easter2(nn.Module):
    def __init__(self, INPUT_SHAPE, VOCAB_SIZE):
        super().__init__()
        self.conv_1 = nn.Conv1d(
                in_channels = INPUT_SHAPE,
                out_channels = 128,
                kernel_size = 3,
                strides = 2,
                padding = 'same'
            )
        self.RELU_activate = nn.ReLU()
        self.dropout_02 = nn.Dropout(0.2)
        self.conv_2 = nn.Conv1d(
                in_channels = 128,
                out_channels = 512,
                kernel_size = 11,
                strides = 1,
                padding = 'same',
                dilation = 2
            )
        self.dropout_04 = nn.dropout(0.4)
        self.conv_3 = nn.Conv1d(
            in_channels = 512,
            out_channels = 512,
            kernel_size = 1,
            stride = 1,
            padding = 'same'
        )
        self.conv_4 = nn.Conv1d(
            in_channels = 512,
            out_channels = VOCAB_SIZE,
            kernel_size = 1,
            stride = 1,
            padding = 'same'
        )
    def forward(self, x):
        x = batch_norm(self.conv_1(x))
        x = self.dropout_02(self.RELU_activate(x))
        x = batch_norm(self.conv_1(x))
        x = self.dropout_02(self.RELU_activate(x))
        old = x
        x, old = easter_unit(old, x, 256, 5, 1, 0.2)
        x, old = easter_unit(old, x, 256, 7, 1, 0.2)
        x, old = easter_unit(old, x, 256, 9, 1, 0.3)
        x = batch_norm(self.conv_2(x))
        x = self.dropout_04(self.RELU_activate(x))
        x = batch_norm(self.conv_3(x))
        x = self.dropout_04(self.RELU_activate(x))
        x = self.conv_4(x)
        return x    

train_data = torch.utils.data.DataLoader(dataset = config.DATA_PATH,
                                         batch_size = config.BATCH_SIZE)
validation_data = torch.utils.data.DataLoader(dataset = config.DATA_PATH,
                                         batch_size = config.BATCH_SIZE)
test_data = torch.utils.data.DataLoader(dataset = config.DATA_PATH,
                                         batch_size = config.BATCH_SIZE)

model = Easter2(INPUT_SHAPE = config.INPUT_SHAPE,
                VOCAB_SIZE = config.VOCAB_SIZE)
optimizer = torch.optim.Adam(params = model.parameters(),
                             lr = config.LEARNING_RATE)

def train(model: torch.nn.Module,
          data_loader: torch.utils.data.DataLoader,
          ctc_custom: ctc_custom,
          optimizer = torch.optim.Optimizer):
    #Creating Easter2 object
    
    # steps per epoch calculation based on number of samples and batch size
    # STEPS_PER_EPOCH = len(training_data.samples)//config.BATCH_SIZE
    # VALIDATION_STEPS = len(validation_data.samples)//config.BATCH_SIZE

    # Train
    for epoch in range(config.EPOCHS):
        train_loss = 0
        model.train()
        for batch, (X, y) in enumerate(data_loader):
            preds = model(X)
            loss = ctc_custom(preds, y)
            train_loss += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if batch % 10 == 0:
                print(f"Looked at {batch * len(X)} / {len(data_loader.dataset)} samples.")
        
        # Test
        test_loss = 0
        model.eval()
        with torch.inference_mode():
            for X, y in data_loader:
                preds = model(X)
                test_loss = loss_fn(preds, y)
            print(f"Test loss: {test_loss:.5f}")
