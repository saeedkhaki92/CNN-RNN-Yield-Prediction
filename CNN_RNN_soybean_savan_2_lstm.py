import numpy as np
import time
import tensorflow as tf

# include for logging
import pandas as pd
log_df = pd.DataFrame(columns={'Iteration', 'training_RMSE', 'Cor_train','test_RMSE', 'Cor_is'})


def conv_res_part_P(P_t,f,is_training,var_name):
    print("----- Flatten layer on p ----", P_t.shape)
    X=tf.contrib.layers.flatten(P_t)
    print('conv2 out P', X.shape)
    return X

def conv_res_part_E(E_t,f,is_training,var_name):
    s0 = 1
    # First Conv Layer
    print("----Conv Layer 1 ------ Start E_t shape :", E_t.shape)
    X = tf.layers.conv1d(E_t, filters=8, kernel_size=9, strides=1, padding='valid',
                         kernel_initializer=tf.contrib.layers.xavier_initializer(), activation=None,
                         name='Conv00' + var_name, data_format='channels_last', reuse=tf.AUTO_REUSE)
    X = tf.nn.relu(X)
    print('conv1 out E', X)
    print('conv1 out E, X shape', X.shape)
    X = tf.layers.average_pooling1d(X, pool_size=2, strides=2, data_format='channels_last', name='average_pool')
    print('conv1 out E, with average pooling, X shape: ', X.shape)

    # Second Conv Layer
    print("------Conv Layer 2 ----- Start X shape :",X.shape)
    X = tf.layers.conv1d(X, filters=12, kernel_size=3, strides=1, padding='valid',
                         kernel_initializer=tf.contrib.layers.xavier_initializer(), activation=None,
                         name='Conv0' + var_name, data_format='channels_last', reuse=tf.AUTO_REUSE)
    X = tf.nn.relu(X)
    print('conv2 out E, X shape', X.shape)
    
    X = tf.layers.average_pooling1d(X, pool_size=2, strides=2, data_format='channels_last', name='average_pool')
    print('conv2 out E, with average pooling, X shape :', X.shape)


    # Third Conv Layer
    print("------Conv Layer 3 ----- Start X shape :",X.shape)
    X = tf.layers.conv1d(X, filters=16, kernel_size=3, strides=1, padding='valid',
                         kernel_initializer=tf.contrib.layers.xavier_initializer(), activation=None,
                         name='Conv1'+var_name, data_format='channels_last',reuse=tf.AUTO_REUSE)

    X = tf.nn.relu(X)
    print('conv3 out E, X shape', X.shape)
    X = tf.layers.average_pooling1d(X, pool_size=2, strides=2, data_format='channels_last', name='average_pool')
    print('conv3 out E, with average pooling, X shape :', X.shape)


    # Fouth Conv Layer
    print("------Conv Layer 4 ----- Start X shape :",X.shape)
    X = tf.layers.conv1d(X, filters=20, kernel_size=3, strides=s0, padding='valid',
                         kernel_initializer=tf.contrib.layers.xavier_initializer(), activation=None,
                         name='Conv2'+var_name, data_format='channels_last',reuse=tf.AUTO_REUSE)
    X = tf.nn.relu(X)
    print('conv4 out E, X shape', X.shape)
    X = tf.layers.average_pooling1d(X, pool_size=2, strides=2, data_format='channels_last', name='average_pool')
    print('conv4 out E, with average pooling, X shape :', X.shape)
    print('conv4',X)
    return X




def conv_res_part_S(S_t,f,is_training,var_name):

    # First Conv Layer
    print("---------conv_res_part_S layer 1 input : ", S_t.shape)
    X = tf.layers.conv1d(S_t, filters=4, kernel_size=3, strides=1, padding='valid',
                         kernel_initializer=tf.contrib.layers.xavier_initializer(), activation=None,
                         name='Conv1'+var_name, data_format='channels_last',reuse=tf.AUTO_REUSE)
    X = tf.nn.relu(X)
    print("conv_res_part_S layer 1 output : ", X.shape)
    print('conv1 out S',X)
    # X = tf.layers.average_pooling1d(X, pool_size=2, strides=2, data_format='channels_last', name='average_pool')
    print("conv_res_part_S layer 1, average pooling : ", X.shape)

    print("---------conv_res_part_S layer 2 input : ", X.shape)
    X = tf.layers.conv1d(X, filters=8, kernel_size=3, strides=1, padding='valid',
                         kernel_initializer=tf.contrib.layers.xavier_initializer(), activation=None,
                         name='Conv2'+var_name, data_format='channels_last',reuse=tf.AUTO_REUSE)
    X = tf.nn.relu(X)
    print("conv_res_part_S layer 1 output : ", X.shape)
    print('conv2 out S', X)

    print("---------conv_res_part_S layer 3 input : ", X.shape)
    X = tf.layers.conv1d(X, filters=12, kernel_size=2, strides=1, padding='valid',
                         kernel_initializer=tf.contrib.layers.xavier_initializer(), activation=None,
                         name='Conv3'+var_name, data_format='channels_last',reuse=tf.AUTO_REUSE)
    X = tf.nn.relu(X)
    print("conv_res_part_S layer 3 output : ", X.shape)
    print('conv3 out S', X)

    return X






def main_proccess(E_t1,E_t2,E_t3,E_t4,E_t5,E_t6,S_t1,S_t2,S_t3,S_t4,S_t5,S_t6,S_t7,S_t8,S_t9,S_t10,P_t,Ybar,S_t_extra,f,is_training,num_units,num_layers,dropout):



    e_out1 = conv_res_part_E(E_t1, f, is_training=is_training,var_name='v1')
    e_out1 = tf.contrib.layers.flatten(e_out1)
    e_out2 = conv_res_part_E(E_t2, f, is_training=is_training, var_name='v1')
    e_out2 = tf.contrib.layers.flatten(e_out2)
    e_out3 = conv_res_part_E(E_t3, f, is_training=is_training, var_name='v1')
    e_out3 = tf.contrib.layers.flatten(e_out3)
    e_out4 = conv_res_part_E(E_t4, f, is_training=is_training, var_name='v1')
    e_out4 = tf.contrib.layers.flatten(e_out4)
    e_out5 = conv_res_part_E(E_t5, f, is_training=is_training, var_name='v1')
    e_out5 = tf.contrib.layers.flatten(e_out5)
    e_out6 = conv_res_part_E(E_t6, f, is_training=is_training, var_name='v1')
    e_out6 = tf.contrib.layers.flatten(e_out6)



    e_out=tf.concat([e_out1,e_out2,e_out3,e_out4,e_out5,e_out6],axis=1)
    print('after concatenate',e_out)

    e_out = tf.contrib.layers.fully_connected(inputs=e_out, num_outputs=60, activation_fn=tf.nn.relu,
                                              weights_initializer=tf.contrib.layers.xavier_initializer(),
                                              biases_initializer=tf.zeros_initializer())


    s_out1 = conv_res_part_S(S_t1, f, is_training=is_training, var_name='v1S')
    s_out1 = tf.contrib.layers.flatten(s_out1)
    s_out2 = conv_res_part_S(S_t2, f, is_training=is_training, var_name='v1S')
    s_out2 = tf.contrib.layers.flatten(s_out2)
    s_out3 = conv_res_part_S(S_t3, f, is_training=is_training, var_name='v1S')
    s_out3 = tf.contrib.layers.flatten(s_out3)
    s_out4 = conv_res_part_S(S_t4, f, is_training=is_training, var_name='v1S')
    s_out4 = tf.contrib.layers.flatten(s_out4)
    s_out5 = conv_res_part_S(S_t5, f, is_training=is_training, var_name='v1S')
    s_out5 = tf.contrib.layers.flatten(s_out5)
    s_out6 = conv_res_part_S(S_t6, f, is_training=is_training, var_name='v1S')
    s_out6 = tf.contrib.layers.flatten(s_out6)

    #s_out7 = conv_res_part_S(S_t7, f, is_training=is_training, var_name='v1S')
    #s_out7 = tf.contrib.layers.flatten(s_out7)
    s_out7 = conv_res_part_S(S_t7, f, is_training=is_training, var_name='v1S')
    s_out7 = tf.contrib.layers.flatten(s_out7)
    s_out8 = conv_res_part_S(S_t8, f, is_training=is_training, var_name='v1S')
    s_out8 = tf.contrib.layers.flatten(s_out8)
    s_out9 = conv_res_part_S(S_t9, f, is_training=is_training, var_name='v1S')
    s_out9 = tf.contrib.layers.flatten(s_out9)
    s_out10 = conv_res_part_S(S_t10, f, is_training=is_training, var_name='v1S')
    s_out10 = tf.contrib.layers.flatten(s_out10)

    p_out=conv_res_part_P(P_t,f,is_training,var_name='P')
    p_out=tf.contrib.layers.flatten(p_out)

    print('p outtttttt',p_out)
    # print('E output----', e_out1)
    # e_out = tf.contrib.layers.flatten(e_out)

    s_out = tf.concat([s_out1, s_out2, s_out3, s_out4, s_out5, s_out6,s_out7,s_out8,s_out9,s_out10], axis=1)
    print('soil after concatenate', s_out)

    s_out = tf.contrib.layers.fully_connected(inputs=s_out, num_outputs=40, activation_fn=None,
                                               weights_initializer=tf.contrib.layers.xavier_initializer(),
                                               biases_initializer=tf.zeros_initializer())



    s_out = tf.nn.relu(s_out)


    print('soil after FC layer', s_out)
    print(s_out,'*****S_out')
    print(e_out,'*****e_out')
    print(p_out,'******p_out')
    e_out=tf.concat([e_out,s_out,p_out],axis=1)

    print('soil + Weather after concatante', e_out)

    time_step=5

    print(e_out,'e_out1111111')
    e_out=tf.reshape(e_out,shape=[-1,time_step,e_out.get_shape().as_list()[-1]])

    print('e_out_after_reshapeeeee',e_out)
    S_t_extra=tf.reshape(S_t_extra,shape=[-1,time_step,4])
    e_out=tf.concat([e_out,Ybar],axis=-1)
    # e_out = tf.concat([e_out, Ybar,S_t_extra], axis=-1)





    cells = []

    for _ in range(num_layers):
        #cell = tf.contrib.rnn.LSTMCell(num_units)

        cell = tf.contrib.rnn.LSTMCell(num_units)

        cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=1.0 - dropout)

        cells.append(cell)

    # It stack multiple type RNN models. like LSTM, GRU etc.
    cell = tf.contrib.rnn.MultiRNNCell(cells)



    output, _= tf.nn.dynamic_rnn(cell, e_out, dtype=tf.float32)

    print('RNN output',output)



    output=tf.reshape(output,shape=[-1,output.get_shape().as_list()[-1]])

    print('my RNN output before FC layer',output)
    output = tf.contrib.layers.fully_connected(inputs=output, num_outputs=1, activation_fn=None,
                                               weights_initializer=tf.contrib.layers.xavier_initializer(),
                                               biases_initializer=tf.zeros_initializer())

    print(output)

    output = tf.reshape(output, shape=[-1,5])
    print("output of all time steps", output)
    Yhat1 = tf.gather(output, indices=[4], axis=1)

    print('Yhat1111111111', Yhat1)

    Yhat2 = tf.gather(output, indices=[0,1,2,3], axis=1)
    print('Yhat222222', Yhat2)

    return Yhat1,Yhat2



def Cost_function(Y, Yhat):

    E = Y - Yhat
    E2 = tf.pow(E, 2)

    MSE = tf.squeeze(tf.reduce_mean(E2))
    RMSE = tf.pow(MSE, 0.5)
    Loss = tf.losses.huber_loss(Y, Yhat, weights=1.0, delta=5.0)

    return RMSE, MSE, E, Loss





def get_sample(dic,L,avg,batch_size,time_steps,num_features):


    L_tr=L[:-1,:]



    out=np.zeros(shape=[batch_size,time_steps,num_features])

    for i in range(batch_size):

        r1 = np.squeeze(np.random.randint(L_tr.shape[0], size=1))

        years = L_tr[r1, :]

        for j, y in enumerate(years):
            X = dic[str(y)]
            ym=avg[str(y)]
            r2 = np.random.randint(X.shape[0], size=1)
            #n=X[r2, :]
            out[i, j, :] = np.concatenate((X[r2, :],np.array([[ym]])),axis=1)


    return out



def get_sample_te(dic,mean_last,avg,batch_size_te,time_steps,num_features):

    out = np.zeros(shape=[batch_size_te, time_steps, num_features]) # num_features should be 398

    X = dic[str(2018)]

  #  r1 = np.random.randint(X.shape[0], size=batch_size_te)
    # out[:, 0:4, :] += mean_last.reshape(1,4,3+6*52+1+100+16+4)
    out[:, 0:4, :] += mean_last.reshape(1,4,6*52+66+14+4)  # 398
    #n=X[r1, :]
    #print(n.shape)
    ym=np.zeros(shape=[batch_size_te,1])+avg['2018']

    out[:,4,:]=np.concatenate((X,ym),axis=1)

    return out







def main_program(X, Index,num_units,num_layers,Max_it, learning_rate, batch_size_tr,le,l):
    with tf.device('/cpu:0'):
        E_t1 = tf.placeholder(shape=[None, 52,1], dtype=tf.float32, name='E_t1')
        E_t2 = tf.placeholder(shape=[None, 52, 1], dtype=tf.float32, name='E_t2')
        E_t3 = tf.placeholder(shape=[None, 52, 1], dtype=tf.float32, name='E_t3')
        E_t4 = tf.placeholder(shape=[None, 52, 1], dtype=tf.float32, name='E_t4')
        E_t5 = tf.placeholder(shape=[None, 52, 1], dtype=tf.float32, name='E_t5')
        E_t6 = tf.placeholder(shape=[None, 52, 1], dtype=tf.float32, name='E_t6')

        # S_t1 = tf.placeholder(shape=[None, 10,1], dtype=tf.float32, name='S_t1')
        # S_t2 = tf.placeholder(shape=[None, 10,1], dtype=tf.float32, name='S_t2')
        # S_t3 = tf.placeholder(shape=[None, 10, 1], dtype=tf.float32, name='S_t3')
        # S_t4 = tf.placeholder(shape=[None, 10, 1], dtype=tf.float32, name='S_t4')
        # S_t5 = tf.placeholder(shape=[None, 10, 1], dtype=tf.float32, name='S_t5')
        # S_t6 = tf.placeholder(shape=[None, 10, 1], dtype=tf.float32, name='S_t6')
        # S_t7 = tf.placeholder(shape=[None, 10, 1], dtype=tf.float32, name='S_t7')
        # S_t8 = tf.placeholder(shape=[None, 10, 1], dtype=tf.float32, name='S_t8')
        # S_t9 = tf.placeholder(shape=[None, 10, 1], dtype=tf.float32, name='S_t9')
        # S_t10 = tf.placeholder(shape=[None, 10, 1], dtype=tf.float32, name='S_t10')
        # S_t_extra=tf.placeholder(shape=[None, 4,1], dtype=tf.float32, name='S_t_extra')

        S_t1 = tf.placeholder(shape=[None, 6,1], dtype=tf.float32, name='S_t1')
        S_t2 = tf.placeholder(shape=[None, 6,1], dtype=tf.float32, name='S_t2')
        S_t3 = tf.placeholder(shape=[None, 6, 1], dtype=tf.float32, name='S_t3')
        S_t4 = tf.placeholder(shape=[None, 6, 1], dtype=tf.float32, name='S_t4')
        S_t5 = tf.placeholder(shape=[None, 6, 1], dtype=tf.float32, name='S_t5')
        S_t6 = tf.placeholder(shape=[None, 6, 1], dtype=tf.float32, name='S_t6')
        S_t7 = tf.placeholder(shape=[None, 6, 1], dtype=tf.float32, name='S_t7')
        S_t8 = tf.placeholder(shape=[None, 6, 1], dtype=tf.float32, name='S_t8')
        S_t9 = tf.placeholder(shape=[None, 6, 1], dtype=tf.float32, name='S_t9')
        S_t10 = tf.placeholder(shape=[None, 6, 1], dtype=tf.float32, name='S_t10')
        S_t_extra=tf.placeholder(shape=[None, 6,1], dtype=tf.float32, name='S_t_extra')

        P_t = tf.placeholder(shape=[None, 14, 1], dtype=tf.float32, name='P_t')  #Plant Data

        Ybar=tf.placeholder(shape=[None,5,1], dtype=tf.float32, name='Ybar')

        Y_t = tf.placeholder(shape=[None, 1], dtype=tf.float32,name='Y_t')

        Y_t_2 = tf.placeholder(shape=[None, 4], dtype=tf.float32, name='Y_t_2')


        is_training=tf.placeholder(dtype=tf.bool, name="is_training")
        lr=tf.placeholder(shape=[],dtype=tf.float32,name='learning_rate')
        dropout = tf.placeholder(tf.float32,name='dropout')

        f=3
        Yhat1,Yhat2= main_proccess(E_t1,E_t2,E_t3,E_t4,E_t5,E_t6,S_t1,S_t2,S_t3,S_t4,S_t5,S_t6,S_t7,S_t8,S_t9,S_t10,P_t,Ybar,S_t_extra,f,is_training,num_units,num_layers,dropout)
        Yhat1=tf.identity(Yhat1,name='Yhat1')
        # Yhat2 is the prediction we got before the final time step (year t)
        print('Yhatttttttttt',Yhat1)

        total_parameters = 0
        for variable in tf.trainable_variables():
            # shape is an array of tf.Dimension
            print(variable)
            shape = variable.get_shape()
            #print(shape)
            # print(len(shape))
            variable_parameters = 1
            for dim in shape:
                #   print(dim)
                variable_parameters *= dim.value
            # print(variable_parameters)
            total_parameters += variable_parameters
        print("total_parameters",total_parameters)

        with tf.name_scope('loss_function'):

            RMSE,_,_,Loss1=Cost_function(Y_t, Yhat1)

            _, _, _, Loss2 = Cost_function(Y_t_2, Yhat2)


            Tloss=tf.constant(l,dtype=tf.float32)*Loss1+tf.constant(le,dtype=tf.float32)*Loss2

        RMSE=tf.identity(RMSE,name='RMSE')
        with tf.name_scope('train'):

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(Tloss)


        init = tf.global_variables_initializer()

        sess = tf.Session()

        sess.run(init)
        writer = tf.summary.FileWriter("./tensorboard")
        writer.add_graph(sess.graph)

        t1=time.time()

        A = []

        for i in range(4, 39):
            A.append([ i - 4, i - 3, i - 2, i - 1, i])

        A = np.vstack(A)
        A += 1980
        print(A.shape)

        dic = {}

        for i in range(39):
            dic[str(i + 1980)] = X[X[:, 1] == i + 1980]



        avg = {}
        avg2 = []
        for i in range(39):
            avg[str(i + 1980)] = np.mean(X[X[:, 1] == i + 1980][:, 2])
            avg2.append(np.mean(X[X[:, 1] == i + 1980][:, 2]))

        print('avgggggg', avg)

        mm = np.mean(avg2)
        ss = np.std(avg2)


        avg = {}

        for i in range(39):
            avg[str(i + 1980)] = (np.mean(X[X[:, 1] == i + 1980][:, 2]) - mm) / ss

        avg['2018'] = avg['2017']
        #avg['2017']=avg['2016']


        #a2 = np.concatenate((np.mean(dic['2008'], axis=0), [avg['2008']]))

        #a3 = np.concatenate((np.mean(dic['2009'], axis=0), [avg['2009']]))

        #a4 = np.concatenate((np.mean(dic['2010'], axis=0), [avg['2010']]))

        #a5 = np.concatenate((np.mean(dic['2011'], axis=0), [avg['2011']]))

        #a6 = np.concatenate((np.mean(dic['2012'], axis=0), [avg['2012']]))

        #a7 = np.concatenate((np.mean(dic['2013'], axis=0), [avg['2013']]))

        a8 = np.concatenate((np.mean(dic['2014'], axis=0), [avg['2014']]))

        a9 = np.concatenate((np.mean(dic['2015'], axis=0), [avg['2015']]))
        a10 = np.concatenate((np.mean(dic['2016'], axis=0), [avg['2016']]))

        a11 = np.concatenate((np.mean(dic['2017'], axis=0), [avg['2017']]))

        mean_last = np.concatenate((a8, a9, a10,a11))



        loss_validation=[]

        loss_train=[]

        for i in range(Max_it):

            # out_tr = get_sample(dic, A, avg,batch_size_tr, time_steps=5, num_features=316+100+16+4)
            out_tr = get_sample(dic, A, avg,batch_size_tr, time_steps=5, num_features=312+66+14+3+1)



            Ybar_tr=out_tr[:, :, -1].reshape(-1,5,1)

            # Batch_X_e = out_tr[:, :, 3:-1].reshape(-1,6*52+100+16+4)
            Batch_X_e = out_tr[:, :, 3:-1].reshape(-1,6*52+66+14) # 392


            Batch_X_e=np.expand_dims(Batch_X_e,axis=-1)
            Batch_Y = out_tr[:, -1, 2]
            Batch_Y = Batch_Y.reshape(len(Batch_Y), 1)

            Batch_Y_2 = out_tr[:, np.arange(0,4), 2]


            if i==60000:
                learning_rate=learning_rate/2
                print('learningrate1',learning_rate)
            elif i==120000:
                learning_rate = learning_rate/2
                print('learningrate2', learning_rate)
            elif i==180000:
                learning_rate = learning_rate/2
                print('learningrate3', learning_rate)



            # sess.run(train_op, feed_dict={ E_t1: Batch_X_e[:,0:52,:],E_t2: Batch_X_e[:,52*1:2*52,:],E_t3: Batch_X_e[:,52*2:3*52,:],
            #                                E_t4: Batch_X_e[:, 52 * 3:4 * 52, :],E_t5: Batch_X_e[:,52*4:5*52,:],E_t6: Batch_X_e[:,52*5:52*6,:],
            #                                S_t1:Batch_X_e[:,312:322,:],S_t2:Batch_X_e[:,322:332,:],S_t3:Batch_X_e[:,332:342,:],
            #                                S_t4:Batch_X_e[:,342:352,:],S_t5:Batch_X_e[:,352:362,:],S_t6:Batch_X_e[:,362:372,:],
            #                                S_t7: Batch_X_e[:, 372:382, :], S_t8: Batch_X_e[:, 382:392, :],
            #                                S_t9: Batch_X_e[:, 392:402, :], S_t10: Batch_X_e[:, 402:412, :],P_t: Batch_X_e[:, 412:428, :],S_t_extra:Batch_X_e[:, 428:, :],
            #                                Ybar:Ybar_tr,Y_t: Batch_Y,Y_t_2: Batch_Y_2,is_training:True,lr:learning_rate,dropout:0.0})

            sess.run(train_op, feed_dict={ E_t1: Batch_X_e[:,0:52,:],E_t2: Batch_X_e[:,52*1:2*52,:],E_t3: Batch_X_e[:,52*2:3*52,:],
                                           E_t4: Batch_X_e[:, 52 * 3:4 * 52, :],E_t5: Batch_X_e[:,52*4:5*52,:],E_t6: Batch_X_e[:,52*5:52*6,:],
                                           S_t1:Batch_X_e[:, 312:312 + 6*1, :],S_t2:Batch_X_e[:, 312 + 6*1:312 + 6*2 ,:],S_t3:Batch_X_e[:,312 + 6*2:312 + 6*3,:],
                                           S_t4:Batch_X_e[:,312 + 6*3:312 + 6*4,:],S_t5:Batch_X_e[:,312 + 6*4:312 + 6*5,:],S_t6:Batch_X_e[:,312 + 6*5:312 + 6*6,:],
                                           S_t7: Batch_X_e[:, 312 + 6*6:312 + 6*7, :], S_t8: Batch_X_e[:, 312 + 6*7:312 + 6*8, :],
                                           S_t9: Batch_X_e[:, 312 + 6*8:312 + 6*9, :], S_t10: Batch_X_e[:, 312 + 6*9:312 + 6*10, :],P_t: Batch_X_e[:, 378:378 + 14, :],
                                           S_t_extra:Batch_X_e[:, 312 + 6*10:312 + 6*11, :],
                                           Ybar:Ybar_tr, 
                                           Y_t: Batch_Y,
                                           Y_t_2: Batch_Y_2,
                                           is_training:True, 
                                           lr:learning_rate,dropout:0.0})


            if i%1000==0:

                # out_tr = get_sample(dic, A, avg, batch_size=1000, time_steps=5, num_features=316 + 100 + 16 + 4)
                out_tr = get_sample(dic, A, avg, batch_size=1000, time_steps=5, num_features=312+66+14+3+1)

                Ybar_tr = out_tr[:, :, -1].reshape(-1, 5, 1)

                # Batch_X_e = out_tr[:, :, 3:-1].reshape(-1, 6 * 52 + 100 + 16 + 4)
                Batch_X_e = out_tr[:, :, 3:-1].reshape(-1, 6 * 52 + 66 + 14)  # 394


                Batch_X_e = np.expand_dims(Batch_X_e, axis=-1)
                Batch_Y = out_tr[:, -1, 2]
                Batch_Y = Batch_Y.reshape(len(Batch_Y), 1)

                Batch_Y_2 = out_tr[:, np.arange(0, 4), 2]

                # out_te = get_sample_te(dic, mean_last, avg,np.sum(Index), time_steps=5, num_features=316+100+16+4)
                out_te = get_sample_te(dic, mean_last, avg,np.sum(Index), time_steps=5, num_features=312+66+14+4) # num_features should be 398
                print(out_te.shape)
                Ybar_te = out_te[:, :, -1].reshape(-1, 5, 1)
                # Batch_X_e_te = out_te[:, :, 3:-1].reshape(-1,6*52+100+16+4)
                Batch_X_e_te = out_te[:, :, 3:-1].reshape(-1,6*52+66+14)
                Batch_X_e_te = np.expand_dims(Batch_X_e_te, axis=-1)
                Batch_Y_te = out_te[:, -1, 2]
                Batch_Y_te = Batch_Y_te.reshape(len(Batch_Y_te), 1)
                Batch_Y_te2 = out_te[:, np.arange(0,4), 2]


                # rmse_tr,yhat1_tr,loss_tr = sess.run([RMSE,Yhat1,Tloss], feed_dict={ E_t1: Batch_X_e[:,0:52,:],E_t2: Batch_X_e[:,52*1:2*52,:],E_t3: Batch_X_e[:,52*2:3*52,:],
                #                            E_t4: Batch_X_e[:, 52 * 3:4 * 52, :],E_t5: Batch_X_e[:,52*4:5*52,:],E_t6: Batch_X_e[:,52*5:52*6,:],
                #                            S_t1:Batch_X_e[:,312:322,:],S_t2:Batch_X_e[:,322:332,:],S_t3:Batch_X_e[:,332:342,:],
                #                            S_t4:Batch_X_e[:,342:352,:],S_t5:Batch_X_e[:,352:362,:],S_t6:Batch_X_e[:,362:372,:],
                #                            S_t7: Batch_X_e[:, 372:382, :], S_t8: Batch_X_e[:, 382:392, :],
                #                            S_t9: Batch_X_e[:, 392:402, :], S_t10: Batch_X_e[:, 402:412, :],P_t: Batch_X_e[:, 412:428, :],S_t_extra:Batch_X_e[:, 428:, :],
                #                            Ybar:Ybar_tr,Y_t: Batch_Y,Y_t_2: Batch_Y_2,is_training:True,lr:learning_rate,dropout:0.0})
                
                rmse_tr,yhat1_tr,loss_tr = sess.run([RMSE,Yhat1,Tloss], feed_dict={ E_t1: Batch_X_e[:,0:52,:],E_t2: Batch_X_e[:,52*1:2*52,:],E_t3: Batch_X_e[:,52*2:3*52,:],
                                           E_t4: Batch_X_e[:, 52 * 3:4 * 52, :],E_t5: Batch_X_e[:,52*4:5*52,:],E_t6: Batch_X_e[:,52*5:52*6,:],
                                           S_t1:Batch_X_e[:, 312:312+6*1, :],S_t2:Batch_X_e[:, 312+6*1:312+6*2, :],
                                           S_t3:Batch_X_e[:, 312+6*2:312+6*3, :],
                                           S_t4:Batch_X_e[:, 312+6*3:312+6*4,:],
                                           S_t5:Batch_X_e[:, 312+6*4:312+6*5, :],
                                           S_t6:Batch_X_e[:, 312+6*5:312+6*6,:],
                                           S_t7: Batch_X_e[:, 312+6*6:312+6*7, :],
                                           S_t8: Batch_X_e[:, 312+6*7:312+6*8, :],
                                           S_t9: Batch_X_e[:, 312+6*8:312+6*9, :], 
                                           S_t10: Batch_X_e[:, 312+6*9:312+6*10, :],
                                           P_t: Batch_X_e[:, 378:378+14, :],
                                           S_t_extra:Batch_X_e[:, 312+6*10:312+6*11, :],
                                           Ybar:Ybar_tr,Y_t: Batch_Y,Y_t_2: Batch_Y_2,is_training:True,lr:learning_rate,dropout:0.0})


                rc_tr = np.corrcoef(np.squeeze(Batch_Y), np.squeeze(yhat1_tr))[0, 1]


                # rmse_te,yhat1_te,loss_val = sess.run([RMSE,Yhat1,Tloss], feed_dict={ E_t1: Batch_X_e_te[:,0:52,:],E_t2: Batch_X_e_te[:,52*1:2*52,:],E_t3: Batch_X_e_te[:,52*2:3*52,:],
                #                            E_t4: Batch_X_e_te[:, 52 * 3:4 * 52, :],E_t5: Batch_X_e_te[:,52*4:5*52,:],E_t6: Batch_X_e_te[:,52*5:52*6,:],
                #                            S_t1:Batch_X_e_te[:,312:322,:],S_t2:Batch_X_e_te[:,322:332,:],S_t3:Batch_X_e_te[:,332:342,:],
                #                            S_t4:Batch_X_e_te[:,342:352,:],S_t5:Batch_X_e_te[:,352:362,:],S_t6:Batch_X_e_te[:,362:372,:],
                #                            S_t7: Batch_X_e_te[:, 372:382, :], S_t8: Batch_X_e_te[:, 382:392, :],
                #                            S_t9: Batch_X_e_te[:, 392:402, :], S_t10: Batch_X_e_te[:, 402:412, :],P_t: Batch_X_e_te[:, 412:428, :],S_t_extra:Batch_X_e_te[:, 428:, :],
                #                            Ybar:Ybar_te,Y_t: Batch_Y_te,Y_t_2: Batch_Y_te2,is_training:True,lr:learning_rate,dropout:0.0})
                rmse_te,yhat1_te,loss_val = sess.run([RMSE,Yhat1,Tloss], feed_dict={ E_t1: Batch_X_e_te[:,0:52,:],E_t2: Batch_X_e_te[:,52*1:2*52,:],E_t3: Batch_X_e_te[:,52*2:3*52,:],
                                           E_t4: Batch_X_e_te[:, 52 * 3:4 * 52, :],E_t5: Batch_X_e_te[:,52*4:5*52,:],E_t6: Batch_X_e_te[:,52*5:52*6,:],
                                           S_t1:Batch_X_e_te[:, 312:312+6*1, :],
                                           S_t2:Batch_X_e_te[:, 312+6*1:312+6*2, :],
                                           S_t3:Batch_X_e_te[:, 312+6*2:312+6*3, :],
                                           S_t4:Batch_X_e_te[:, 312+6*3:312+6*4, :],
                                           S_t5:Batch_X_e_te[:, 312+6*4:312+6*5, :],
                                           S_t6:Batch_X_e_te[:, 312+6*5:312+6*6, :],
                                           S_t7: Batch_X_e_te[:, 312+6*6:312+6*7, :],
                                           S_t8: Batch_X_e_te[:, 312+6*7:312+6*8, :],
                                           S_t9: Batch_X_e_te[:, 312+6*8:312+6*9, :], 
                                           S_t10: Batch_X_e_te[:, 312+6*9:312+6*10, :],
                                           P_t: Batch_X_e_te[:, 378:378+16, :],
                                           S_t_extra:Batch_X_e_te[:, 312+6*10:312+6*11, :],
                                           Ybar:Ybar_te,Y_t: Batch_Y_te,Y_t_2: Batch_Y_te2,is_training:True,lr:learning_rate,dropout:0.0})


                rc=np.corrcoef(np.squeeze(Batch_Y_te),np.squeeze(yhat1_te))[0,1]
                loss_validation.append(loss_val)

                loss_train.append(loss_tr)

                print(loss_tr,loss_val)
                print("Iteration %d , The training RMSE is %f and Cor train is %f  and test RMSE is %f and Cor is %f " % (i, rmse_tr,rc_tr, rmse_te,rc))
                # Append the data to logs
                log_df.append({"Iteration":i, "training_RMSE":rmse_tr, "Cor_train": rc_tr, "test_RMSE":rmse_te, "Cor_is":rc }, ignore_index=True)




    out_te = get_sample_te(dic, mean_last, avg,np.sum(Index), time_steps=5, num_features=316+100+16+4)

    Batch_X_e_te = out_te[:, :, 3:-1].reshape(-1,6*52+100+16+4)
    Ybar_te = out_te[:, :, -1].reshape(-1, 5, 1)
    Batch_X_e_te = np.expand_dims(Batch_X_e_te, axis=-1)
    Batch_Y_te = out_te[:, -1, 2]
    Batch_Y_te = Batch_Y_te.reshape(len(Batch_Y_te), 1)
    Batch_Y_te2 = out_te[:, np.arange(0, 4), 2]

    rmse_te,yhat1 = sess.run([RMSE,Yhat1], feed_dict={ E_t1: Batch_X_e_te[:,0:52,:],E_t2: Batch_X_e_te[:,52*1:2*52,:],E_t3: Batch_X_e_te[:,52*2:3*52,:],
                                           E_t4: Batch_X_e_te[:, 52 * 3:4 * 52, :],E_t5: Batch_X_e_te[:,52*4:5*52,:],E_t6: Batch_X_e_te[:,52*5:52*6,:],
                                           S_t1:Batch_X_e_te[:,312:322,:],S_t2:Batch_X_e_te[:,322:332,:],S_t3:Batch_X_e_te[:,332:342,:],
                                           S_t4:Batch_X_e_te[:,342:352,:],S_t5:Batch_X_e_te[:,352:362,:],S_t6:Batch_X_e_te[:,362:372,:],
                                           S_t7: Batch_X_e_te[:, 372:382, :], S_t8: Batch_X_e_te[:, 382:392, :],
                                           S_t9: Batch_X_e_te[:, 392:402, :], S_t10: Batch_X_e_te[:, 402:412, :],P_t: Batch_X_e_te[:, 412:428, :],S_t_extra:Batch_X_e_te[:, 428:, :],
                                           Ybar:Ybar_te,Y_t: Batch_Y_te,Y_t_2: Batch_Y_te2,is_training:True,lr:learning_rate,dropout:0.0})


    print("The training RMSE is %f  and test RMSE is %f " % (rmse_tr, rmse_te))
    t2=time.time()

    print('the training time was %f' %(round(t2-t1,2)))
    saver = tf.train.Saver()
    saver.save(sess, './model_corn', global_step=i)  # Saving the model

    return  rmse_tr,rmse_te,loss_train,loss_validation



BigX = np.load('./soyabean_fianl_npz.npz') ##order W(52*6) S(100) P(16) S_extra(4)

X=BigX['data']


X_tr=X[X[:,1]<=2017]

X_tr=X_tr[:,3:]



# mean with axis=0 means respect to columns
M=np.mean(X_tr,axis=0,keepdims=True)
S=np.std(X_tr,axis=0,keepdims=True)
# normalization step, z score calculation for all the variable
X[:,3:]=(X[:,3:]-M)/S



index_low_yield=X[:,2]<10
print('low yield observations',np.sum(index_low_yield))
print(X[index_low_yield][:,1])
X=np.nan_to_num(X)
X=X[np.logical_not(index_low_yield)]


del BigX

Index=X[:,1]==2018  #validation year


print("train data",np.sum(np.logical_not(Index)))
print("test data",np.sum(Index))


# Validation year crop yeild mean and std.
print('Std %.2f and mean %.2f  of test ' %(np.std(X[Index][:,2]),np.mean(X[Index][:,2])))



Max_it=350000      # 150000 could also be used
learning_rate=0.0003   # Learning rate
batch_size_tr=25  # traning batch size
le=0.0  # Weight of loss for prediction using times before final time steps
l=1.0    # Weight of loss for prediction using final time s tep
num_units=64  # Number of hidden units for LSTM celss
num_layers=2  # Number of layers of LSTM cell


rmse_tr,rmse_te,train_loss,validation_loss=main_program(X, Index,num_units,num_layers,Max_it, learning_rate, batch_size_tr,le,l)
# store the csv file
log_df.to_csv('logs/lstm_layer_2_unit_64.csv')




