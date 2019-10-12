import tensorflow as tf
import numpy as np
import matplotlib
import os
import matplotlib.pyplot as plt
import csv

tf.set_random_seed(777)
np.random.seed(777)

data_min = 0
data_max = 0

def plot_test_data(test_predict, test_y):
    real_predict = ReverseScaler(test_predict)
    real_real = ReverseData(test_y)
    print("real_predict")
    print(real_predict)
    print("real_real")
    print(real_real)
    print("min: ", data_min)
    print("max: ", data_max)
    plt.plot(real_real, 'r')
    plt.plot(real_predict, 'b')
    plt.xlabel("Time Period")
    plt.ylabel("Demand Level")
    plt.show()


def MinSet(data):
    return np.min(data, 0)


def MaxSet(data):
    return np.max(data, 0)


def MinMaxScaler(data):
    numerator = data - data_min

    denominator = data_max - data_min

    out = numerator / (denominator + 1e-7)

    return out


def ReverseScaler(data):
    temp = (data_max[1] - data_min[1] + (1e-7)) * data

    out = temp + data_min[1]

    return out


def ReverseData(data):
    for i in range(0, len(data)):
        data[i] = (data_max[1] - data_min[1] + (1e-7)) * data[i]

    out = data + data_min[1]

    return out


xy = np.loadtxt('TenYearDemand.csv', delimiter=',', dtype=np.float32)
futureXY = np.loadtxt('FutureDemand.csv', delimiter=',', dtype=np.float32)

##xy = xy[::-1]
print("xy[0][0]: ", xy[0][0])

data_min = MinSet(xy)
data_max = MaxSet(xy)
xy = MinMaxScaler(xy)
futureXY = MinMaxScaler(futureXY)
print("xy[0][0]: ", xy[0][0])
x = xy
y = xy[:, [-1]]
futureX = futureXY
futureY = futureXY[:, [-1]]
print("x[0]: ", x[0])
print("y[0]: ", y[0])

dataX = []
dataY = []

# Parameters
seq_length = 12
data_dim = 2
layer_dim = 2  ##changeable
hidden_dim = 12  ##changeable
output_dim = 1
epoch_num = 40000
batch_size = 12
learning_rate = 0.0001  ##changeable

for i in range(0, len(y) - seq_length):
    _x = x[i: i + seq_length]
    _y = y[i + seq_length]
    if i is 0:
        print(_x, "->", _y)
    dataX.append(_x)
    dataY.append(_y)
dataX = np.array(dataX)
dataY = np.array(dataY)

tempX = np.concatenate((x, futureX), axis=0)
tempY = np.concatenate((y, futureY), axis=0)

print(tempY)
inputX = []
inputY = []
for i in range(0, len(futureY)):
    _x = tempX[len(y)-seq_length+i:len(y)+i]
    _y = tempY[len(y)+i]
    if i is 0:
        print(_x, "->", _y)
    inputX.append(_x)
    inputY.append(_y)

# Make Train/Test Dataset
train_size = int(len(dataY) * 0.8)
test_size = len(dataY) - train_size

rand_perm = np.random.permutation(len(dataY))
train_mask = rand_perm[:train_size]
test_mask = rand_perm[train_size:train_size + test_size]

trainX = np.array(dataX[train_mask])
trainY = np.array(dataY[train_mask])
testX = np.array(dataX[test_mask])
testY = np.array(dataY[test_mask])

# Tensor model set
X = tf.placeholder(tf.float32, [None, seq_length, data_dim])
print("X: ", X)
Y = tf.placeholder(tf.float32, [None, 1])
print("Y: ", Y)

targets = tf.placeholder(tf.float32, [None, 1])
print("targets: ", targets)
predictions = tf.placeholder(tf.float32, [None, 1])
print("predictions: ", predictions)


def lstm_cell():
    cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_dim, forget_bias=0.8, state_is_tuple=True, activation=tf.tanh)
    return cell


multi_cells = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(layer_dim)], state_is_tuple=True)

hypothesis, _states = tf.nn.dynamic_rnn(multi_cells, X, dtype=tf.float32)
print("hypothesis: ", hypothesis)

Y_pred = tf.contrib.layers.fully_connected(hypothesis[:, -1], output_dim, activation_fn=None)
endingInventory = tf.constant(73.)

# if Y_pred > Y
Diff_1 = tf.subtract(Y_pred, Y)
loss_1 = tf.reduce_sum(tf.square(tf.maximum(Diff_1, 0)))
# elif Y_pred < Y
Diff_2 = tf.subtract(Y, Y_pred)
loss_2 = tf.reduce_sum(9 * tf.square(tf.maximum(Diff_2, 0)))
# elif Y_pred - Y > 90
Diff_3 = Y_pred - Y - MinMaxScaler(90)
loss_3 = tf.reduce_sum(3 * tf.square(tf.maximum(Diff_3, 0)))

loss = loss_1 + loss_2 + loss_3

# loss = tf.reduce_sum(tf.square(Y_pred - Y))                                          ##maybe the penalty

optimizer = tf.train.AdamOptimizer(learning_rate)
train = optimizer.minimize(loss)

# rmse = tf.sqrt(tf.reduce_mean(tf.squared_difference(targets, predictions)))         ##maybe the penalty
Diff_real_1 = tf.subtract(tf.maximum(ReverseScaler(predictions), endingInventory), ReverseScaler(targets))
# Diff_real_1 = tf.subtract(Y_pred, Y)
loss_real_1 = tf.reduce_sum(tf.maximum(Diff_real_1, 0))

Diff_real_2 = tf.subtract(ReverseScaler(targets), tf.maximum(ReverseScaler(predictions), endingInventory))
# Diff_real_2 = tf.subtract(Y, Y_pred)
loss_real_2 = tf.reduce_sum(3 * tf.maximum(Diff_real_2, 0))

Diff_real_3 = tf.maximum(ReverseScaler(predictions), endingInventory) - ReverseScaler(targets) - 90
# Diff_real_3 = Y_pred - Y - MinMaxScaler(90)
loss_real_3 = tf.reduce_sum(tf.maximum(Diff_real_3, 0))

OF = tf.reduce_sum(loss_real_1 + loss_real_2 + loss_real_3)
OF_holdingCost_total = tf.reduce_sum(loss_real_1+loss_real_3)
OF_backorderCost_total = tf.reduce_sum(loss_real_2)
OF_holdingCost_avg = tf.reduce_mean(loss_real_1+loss_real_3)
OF_backorderCost_avg = tf.reduce_mean(loss_real_2)
# weighted avg <-

init_op = tf.initialize_all_variables()

SAVER_DIR = "model"
saver = tf.train.Saver()
checkpoint_path = os.path.join(SAVER_DIR, "model")
ckpt = tf.train.get_checkpoint_state(SAVER_DIR)

list_step_loss = []
list_write = []
list_write_attribute = ["Target Level (Focasted Next Month Demand)", "Actual Demand Level", "Available Balance (Ending inventory)", "Net Requirments", "Order Receipts", "Order Release", "Holding cost", "Backorder cost"]

list_write.append(["Attribute||time"])
for i in range(len(futureX)+1):
    list_write[0].append(i)
for i in range(len(list_write_attribute)):
    list_write.append([list_write_attribute[i]])
list_write[1].append("NaN")
list_write[2].append("NaN")
list_write[3].append(73.)
list_write[4].append("NaN")
list_write[5].append("NaN")
list_write[7].append("NaN")
list_write[8].append("NaN")
#print(list_write)

list_summary = []
list_summary.append(["holding cost total"])
list_summary.append(["holding cost average"])
list_summary.append(["backorder cost total"])
list_summary.append(["backorder cost average"])

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    if ckpt and ckpt.model_checkpoint_path:
        print("restore")
        saver.restore(sess, ckpt.model_checkpoint_path)

    else:
        print("train")
        for epoch in range(epoch_num):
            batch_mask = np.random.choice(train_size, batch_size)
            trainX_batch = trainX[batch_mask]
            trainY_batch = trainY[batch_mask]
            _, step_loss = sess.run([train, loss], feed_dict={X: trainX_batch, Y: trainY_batch})
            list_step_loss.append(step_loss)
            if epoch % 50 == 0:
                print("step: {} avg_loss: {}".format(epoch, np.average(list_step_loss[epoch - 100:epoch])))
            # print("[step: {} loss: {}".format(epoch, step_loss))
        saver.save(sess, checkpoint_path, global_step=epoch)

    #test_predict = sess.run(Y_pred, feed_dict={X: testX})
    #plot_test_data(test_predict, testY)

    #test_all_predict = sess.run(Y_pred, feed_dict={X: dataX})
    #plot_test_data(test_all_predict, dataY)

    #print(OF)
    #OFV = sess.run(OF, feed_dict={targets: testY, predictions: test_predict})
    #print("OFV: ", OFV)

    #future_predict = sess.run(Y_pred, feed_dict={X: inputX})
    #plot_test_data(future_predict, inputY)

    PAB = 0
    GAP = 0
    for i in range(len(inputX)):
        future_predict = sess.run(Y_pred, feed_dict={X: [inputX[i]]})
        list_write[1].append(tf.maximum(ReverseScaler(future_predict)[0][0], endingInventory).eval())
        list_write[2].append(ReverseScaler(inputY[i][0]))
        PAB = list_write[1][i+2] - list_write[2][i+2]
        list_write[3].append(PAB)
        endingInventory = PAB
        GAP = np.maximum(list_write[1][i+2]-list_write[3][i+1], 0)
        list_write[4].append(GAP)
        list_write[6].append(GAP)
        list_write[5].append(GAP)
        list_write[7].append(np.maximum(1*list_write[3][i+2] + np.maximum((list_write[3][i+2]-90), 0), 0))
        list_write[8].append(np.maximum((-3)*list_write[3][i+2], 0))

    holding = np.array(list_write[7][2:len(list_write[7])])
    backorder = np.array(list_write[8][2:len(list_write[8])])

    list_summary[0].append(np.sum(holding))
    list_summary[1].append(np.mean(holding))
    list_summary[2].append(np.sum(backorder))
    list_summary[3].append(np.mean(backorder))

    f_1 = open('output.csv', 'w', encoding='utf-8', newline='')
    wr = csv.writer(f_1)
    for i in list_write:
        wr.writerow(i)
    f_1.close()

    f_2 = open('summary.csv', 'w', encoding='utf-8', newline='')
    wr = csv.writer(f_2)
    for i in list_summary:
        wr.writerow(i)
    f_2.close()

    sess.close()
