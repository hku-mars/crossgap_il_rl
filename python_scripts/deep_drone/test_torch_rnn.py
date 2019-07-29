import torch.nn as nn
import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1" # Disable GPU using
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np


def numpy_test():
    arr_a = np.array([[1.0, 2.0], [3.0, 4.0]])
    arr_b = np.array([[1.0, 2.0], [3.0, 4.0]])
    print(arr_a)
    print("array shape = ", arr_a.shape, arr_b.shape)
    print("arr_a * arr_b = \n", arr_a * arr_b)
    print("arr_a dot arr_b = \n", arr_a.dot( arr_b))

    mat_a = np.matrix([[1.0, 2.0], [3.0, 4.0]])
    mat_b = np.matrix([[1.0, 2.0], [3.0, 4.0]])
    print("matrix shape = ", mat_a.shape, mat_b.shape)
    print("mat_a * mat_b = \n", mat_a * mat_b)

def test_torch_rnn():
    print("test_torch_rnn");
    torch.random.manual_seed(1)
    print("This is a temp test file")
    input_size = 2
    output_size = 4
    hidder_size = 2
    rnn = nn.RNN(input_size, output_size, hidder_size)
    input = torch.randn(1, 1, input_size)
    # h0 = None
    h0 = torch.randn(hidder_size, 1, output_size)

    output, hn = rnn(input, h0)
    print("output is: ", output)
    print("hidden state is: ", hn.shape, hn)
    x_in = np.matrix(input[0, :, :].data.numpy())
    h0_in = np.matrix(h0[:, :, :].data.numpy())
    y_out = np.matrix(output[0, :, :].data.numpy())
    h_out = np.matrix(hn[:, :, :].data.numpy())
    print("In  :", x_in, h0_in)
    print("Out :", output, h_out)
    w_hh_l0 = np.matrix(rnn.weight_hh_l0.data.numpy())
    w_ih_l0 = np.matrix(rnn.weight_ih_l0.data.numpy())
    w_hh_l1 = np.matrix(rnn.weight_hh_l1.data.numpy())
    w_ih_l1 = np.matrix(rnn.weight_ih_l1.data.numpy())

    b_hh_l0 = np.matrix(rnn.bias_hh_l0.data.numpy())
    b_ih_l0 = np.matrix(rnn.bias_ih_l0.data.numpy())
    b_hh_l1 = np.matrix(rnn.bias_hh_l1.data.numpy())
    b_ih_l1 = np.matrix(rnn.bias_ih_l1.data.numpy())

    print(x_in.shape, h0_in.shape)
    print(w_ih_l0.shape, b_ih_l0.shape)
    print(w_hh_l0.shape, b_hh_l0.shape)

    print(w_ih_l1.shape, b_ih_l1.shape)
    print(w_hh_l1.shape, b_hh_l1.shape)

    # print(x_in * w_ih_l0.T + b_ih_l0)
    # print(h0_in* w_hh_l0 )
    _h0 = (x_in * w_ih_l0.T + b_ih_l0 + h0_in[0, :] * w_hh_l0.T + b_hh_l0)
    _h2 = (np.tanh(_h0) * w_ih_l1.T + b_ih_l1 + h0_in[1, :] * w_hh_l1.T + b_hh_l1)


    # TF_RNN_2

    np.set_printoptions(precision= 10)
    print(_h0.shape, " --- ", np.tanh(_h0))
    print(_h2.shape, " --- ", np.tanh(_h2))
    print("Torch RNN output = :\n", output.data.numpy(), " ,should equal to h2")

    # TF_RNN1
    tf_input_x = np.concatenate((x_in, h0_in[0, :]), 1)
    tf_kernel_x = np.concatenate((w_ih_l0, w_hh_l0), 1).T
    tf_bias = b_ih_l0 + b_hh_l0
    tf_out = np.tanh(tf_input_x*tf_kernel_x+tf_bias)
    print("TF_rnn1 output = ",tf_out)


    tf_input_x = np.concatenate((tf_out, h0_in[1, :]), 1)
    tf_kernel_x = np.concatenate((w_ih_l1, w_hh_l1), 1).T
    tf_bias = b_ih_l1 + b_hh_l1
    tf_out = np.tanh(tf_input_x*tf_kernel_x+tf_bias)
    print("TF_rnn2 output = ",tf_out)

def test_tf_rnn():
    print("test_tf_rnn")
    np.random.seed(1)
    batch_size = 5
    vector_size = 3
    inputs = tf.placeholder(tf.float32, [batch_size, vector_size])

    num_units = 4
    state = tf.zeros([batch_size, num_units], tf.float32)

    cell = tf.contrib.rnn.BasicRNNCell(num_units=num_units)

    output, newstate = cell(inputs=inputs, state=state)

    print("Output of cell.variables is a list of Tensors:")
    print(cell.variables)
    kernel, bias = cell.variables
    print("kernel =  ", kernel)
    print("bias = " , bias )
    # X = np.zeros([batch_size, vector_size])
    X = np.random.random([batch_size, vector_size])
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    output_, newstate_, k_, b_ = sess.run(
            [output, newstate, kernel, bias], feed_dict={inputs: X})
    # output_, newstate_, k_, b_ = sess.run(
    #         [output, newstate, kernel, bias], feed_dict={inputs: X})
    print("Output:")
    print(output_.shape, output_)
    print("New State == Output:")
    print(newstate_.shape, newstate_)
    print("\nKernel:")
    print(k_.shape, k_)
    print("\nBias:")
    print(b_.shape, b_)

    # tf.summary.FileWriter("rnn_log/", sess.graph)

    input_x = np.concatenate((X, state.eval()), 1)
    rnn_output=  np.tanh(input_x.dot(k_) + b_)

    print("numpy rnn_output = ", rnn_output)
    print("tf    rnn_output = ", output_)
    print("test_tf_rnn finish")


if __name__ == "__main__":

    # numpy_test()
    # test_torch_rnn()
    test_tf_rnn();

    # exit(0)



    # print(w_hh_l0)
