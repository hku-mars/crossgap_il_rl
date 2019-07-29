import torch
import torch.nn

class Rnn_net(torch.nn.Module):
    def __init__(self):
        super(Rnn_net, self).__init__()
        hidden_size = 20
        input_size = 9
        output_size = 4

        # self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        # self.i2o = nn.Linear(input_size + hidden_size, output_size)

        self.rnn = torch.nn.RNN(
            input_size = input_size,
            hidden_size= hidden_size,
            num_layers= 2,
            batch_first= True,
            )

        self.out = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden_state):
        # combined = torch.cat((input, hidden), 1)
        # hidden_state = self.i2h(combined)
        # rnn_output = self.i2o(combined)

        # print("Run forward")
        rnn_output, hidden_state = self.rnn(x, hidden_state)

        outs = []
        for time_step in range(rnn_output.size(1)):
            outs.append(self.out(rnn_output[:, time_step, :]))
        return torch.stack(outs, dim=1), hidden_state
