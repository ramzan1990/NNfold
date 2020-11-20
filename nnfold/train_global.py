import os
#os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import numpy as np
from math import sqrt
from numpy import zeros
import re
import math
from .batch_object import batch_object
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pdb
import random
import time
from numpy import genfromtxt
#from scipy.spatial import distance
#from PIL import Image
#import pickle
from pathlib import Path
import time
from random import randint
import csv
from . import datadir
    
def main():
    import apetype as at

    class CLI_Settings(at.ConfigBase):
        seq_len: int # Sequence length
        tr_dir: str # Directory with training data '.ct' files
        
    settings = CLI_Settings()

    # Import tensorflow, after processing CLI_Settings as it prints warnings
    import tensorflow as tf
    from tensorflow.python.saved_model import builder as saved_model_builder
    from tensorflow.contrib import rnn
    #import nnfold.tf_model_component as tmc

    def conv1d(input, pname, name, kshape, stride=1):
        with tf.name_scope(name):
            W = tf.get_variable(name=pname+'w_'+name,
                                shape=kshape)
            b = tf.get_variable(name=pname+'bias_' + name,
                                shape=[kshape[2]])
            out = tf.nn.conv1d(input,W,stride=stride, padding='SAME')###############
            out = tf.nn.bias_add(out, b)
            out = tf.nn.leaky_relu(out)
            #out = tf.nn.relu(out)
            return out
    
    def fullyConnected(input, name, output_size):
        with tf.name_scope(name):
            input_size = input.shape[1:]
            input_size = int(np.prod(input_size))
            W = tf.get_variable(name='w_'+name,
                                shape=[input_size, output_size])
            b = tf.get_variable(name='bias_'+name,
                                shape=[output_size])
            input = tf.reshape(input, [-1, input_size])
            out = tf.add(tf.matmul(input, W), b)
            out = tf.nn.leaky_relu(out)
            #out = tf.maximum(out, 0.01 * out, name = "forsoft")
            return out
    
    def model(x, y, keep_ratio1, keep_ratio2):  
        dpi = tf.nn.dropout(x, keep_ratio1)
        filter_size = 30
        lc1 = conv1d(dpi, 'fre_', 'lc1', [filter_size, max_features, 8])
        lc2 = conv1d(lc1, 'fre_', 'lc2', [filter_size, 8, 16])
        lc3 = conv1d(lc2, 'fre_', 'lc3', [filter_size, 16, 32])
        lc4 = conv1d(lc3, 'fre_', 'lc4', [filter_size, 32, 64])
        lc5 = conv1d(lc4, 'fre_', 'lc5', [filter_size, 64, 128])
    
        ml = tf.contrib.layers.flatten(lc5)
    
        dp = tf.nn.dropout(ml, keep_ratio2) 
    
        out = fullyConnected(dp, "output_p", seq_len)
        loss = tf.reduce_mean(tf.squared_difference( y, out)) 
        #vars   = tf.trainable_variables() 
        #l2_loss =  tf.add_n([ tf.nn.l2_loss(v) for v in vars if 'bias' not in v.name ])
        #loss = tf.add(loss, 0.000001*l2_loss) 
        return out, loss
    
    def clean_seq(s):
        ns = s.upper()    
        pattern = re.compile(r'\s+')
        ns = re.sub(pattern, '', ns)
        ns = re.sub(r'[^a-zA-Z]{1}', 'N', ns)
        return ns
    
    def encode(ns):
        ns = ns.replace("A", "1,0,0,0,")
        ns = ns.replace("U", "0,1,0,0,")
        ns = ns.replace("G", "0,0,1,0,")
        ns = ns.replace("C", "0,0,0,1,")
        if re.search('[a-zA-Z]', ns):
            #print(s)
            #print('Non-standard symbol in sequence - changed to A.')
            ns = re.sub("[a-zA-Z]", "0,0,0,0,", ns)
        return ns[:-1]
    
    def brun(sess, x, y, a, keep_prob1, keep_prob2):
        preds = []
        batch_size = 128
        number_of_full_batch=int(math.ceil(float(len(a))/batch_size))
        for i in range(number_of_full_batch):
            preds += list(sess.run(y, feed_dict={x: np.asarray(a[i*batch_size:(i+1)*batch_size]),
             keep_prob1: 1.0, keep_prob2: 1.0}))
        return preds
    
    
    max_features = 4  
    seq_len = settings.seq_len
    tr_dir = settings.tr_dir
    pos_seq = []
    raw_seq = []
    used_seq = set(['N'*seq_len + '_' + 'N'*seq_len])
    dup = 0
    seq = ""
    
    
    directory = os.fsencode(tr_dir)
    
    small_classes = []
    with open(os.path.join(datadir,'small_classes.csv')) as file:
        for line in file:
            small_classes.append(line.strip())
    
    
    t_classes = []
    with open(os.path.join(datadir,'t_classes.csv')) as file:
        for line in file:
            t_classes.append(line.strip())
    
    x_data = []
    y_data = []
    y_raw = []
    ctp = 0
    rms = 0
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        skip_file = False
        if filename.endswith(".ct"): 
            seq=""
            prev_nuc = 0
            pairs = []  
            first = True
            with open(tr_dir + "/" + filename) as fp:  
                for cnt, line in enumerate(fp):
                    if(line.startswith("#")):
                        continue
                    if(first):
                        first = False
                        continue
                    values = line.split()
                    seq = seq + values[1]
                    if(int(values[0]) != prev_nuc + 1):
                        rms = rms + 1
                        skip_file = True
                        break
                    prev_nuc = int(values[0])
                    if(int(values[4])>0):
                        pairs.append([int(values[2]), int(values[4]) - 1])
                        
            if(skip_file):
                continue
            seq = clean_seq(seq)
            #if(len(seq) <= 140):
            #    continue
    
            start = 0
    
            filename = filename[filename.index("_") + 1 : ]
            if(filename in small_classes): 
                maxstep = 1
            elif(filename in t_classes): 
                maxstep = 1
            else:
                if(len(seq) <= 140):
                    maxstep = 2
                else:
                    maxstep = 16
    
            while(start + len(seq) <= seq_len): 
                if(start > len(seq)):
                    break
                nseq = 'N'*(start) + seq + 'N'*(seq_len - start - len(seq))
                x_data.append(np.fromstring(encode(nseq), dtype=int, sep=",").reshape(-1, 4))
                y_cm = zeros((seq_len))
                for j in range(len(pairs)):
                    y_cm[start + pairs[j][0]] = 1.0
                    y_cm[start + pairs[j][1]] = 1.0
                y_data.append(y_cm)
                if(start<=40):
                    start = start + 1
                else:
                    start = start + randint(1, maxstep)
            ctp = ctp + 1
            if(ctp%10 == 0):
                print(str(ctp) + " ("+ str(len(y_data)) +") - ", end='', flush=True)
            #if(len(x_data) > 1000):
            #    break
    
        else:
            continue
    
    
    print("")
    print("----------------------------------------------------------------", flush=True)
    print("Done generating", flush=True)
    print("Final size: " + str(len(y_data)), flush=True)
    print("Skipped files: " + str(rms), flush=True)
    print("----------------------------------------------------------------", flush=True)
    
    
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.01, random_state=2504)
    batch_size = 1024
    nb_epoch = 10001
    # initialize inputs
    x = tf.placeholder(tf.float32, shape=[None, seq_len, max_features], name="input_rna")
    y = tf.placeholder(tf.float32, shape=[None, seq_len])
    keep_prob1 = tf.placeholder(tf.float32, name="kr_rna1")
    keep_prob2 = tf.placeholder(tf.float32, name="kr_rna2")
    # build the model
    out, loss = model(x, y, keep_prob1, keep_prob2)
    out =  tf.identity(out, name="output_rna")
    # initialize optimizer
    train_step = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss)
    # run the training loop
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        total = int(len(x_train)/batch_size)+1
        for epoch in range(nb_epoch):
            my_file = Path("trainm")
            if not my_file.is_file():
                break
            rng_state = np.random.get_state()
            np.random.shuffle(x_train)
            np.random.set_state(rng_state)
            np.random.shuffle(y_train)
    
            rng_state = np.random.get_state()
            np.random.shuffle(x_test)
            np.random.set_state(rng_state)
            np.random.shuffle(y_test)
    
            x_train_obj = batch_object(x_train, batch_size)
            y_train_obj = batch_object(y_train, batch_size)
            for i in range(total):
                x_train_batch = x_train_obj.next_batch()
                y_train_batch = y_train_obj.next_batch()
                #train_batch = tmc.generate_random_batch([x_train], y_train, batch_size)
                #feed = {x : x_train_batch, y_: y_train_batch, keep_prob : 1.0}
                #print(np.shape(np.squeeze(np.split(x_train_batch, 2, axis=1)[0])))
                feed = {x : x_train_batch, y: y_train_batch, keep_prob1 : 1.0, keep_prob2 : 1.0}
                train_step.run(feed_dict=feed)
            if epoch % 1 == 0:
                ts = 1001
                pred = brun(sess, x, out, x_train[:ts], keep_prob1, keep_prob2)            
                orig = np.reshape(np.asarray(y_train[:ts]), (ts, seq_len))
                pred = np.asarray(pred)          
                pred = np.around(pred, 0).astype(int) 
                pred = np.reshape(pred, (ts, seq_len)) 
    
                ae = 0.0
                at = 0.0
                ac = 0.0
                for u in range(ts):
                    o1 = orig[u]
                    p1 = pred[u]
                    for q in range(len(o1)):
                        if(o1[q] == 1):
                            at = at + 1
                            if(p1[q] == 1):
                                ac = ac + 1
                            else:
                                ae = ae + 1
                        elif(p1[q] == 1):
                            ae = ae + 1
    
                sn = 0
                if(at>0):
                    sn = ac/at
                print("Epoch: %d. Train  --- %g - %g" % (epoch, sn, ae), end='', flush=True)
                
                ts = 1001
                pred = brun(sess, x, out, x_test[:ts], keep_prob1, keep_prob2)            
                orig = np.reshape(np.asarray(y_test[:ts]), (ts, seq_len))
                pred = np.asarray(pred)         
                pred = np.reshape(pred, (ts, seq_len))
                pred = np.around(pred, 0).astype(int)
    
    
                ae = 0.0
                at = 0.0
                ac = 0.0
                for u in range(ts):
                    o1 = orig[u]
                    p1 = pred[u]
                    for q in range(len(o1)):
                        if(o1[q] == 1):
                            at = at + 1
                            if(p1[q] == 1):
                                ac = ac + 1
                            else:
                                ae = ae + 1
                        elif(p1[q] == 1):
                            ae = ae + 1
    
                sn = 0
                if(at>0):
                    sn = ac/at
                print(". Test  --- %g - %g" % (sn, ae), flush=True)            
                out_dir = "model_rna_check_m_" + str(epoch)
                if(os.path.exists(out_dir)):
                    out_dir = out_dir + str(time.time())
                builder = tf.saved_model.builder.SavedModelBuilder(out_dir)
                predict_tensor_inputs_info = tf.saved_model.utils.build_tensor_info(x)
                predict_tensor_scores_info = tf.saved_model.utils.build_tensor_info(y)
                prediction_signature = (tf.saved_model.signature_def_utils.build_signature_def(inputs={"input_rna": predict_tensor_inputs_info}, outputs={"output_rna": predict_tensor_scores_info}, method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))
                builder.add_meta_graph_and_variables(sess, [tf.saved_model.tag_constants.SERVING], signature_def_map={"model": prediction_signature})
                builder.save(True)
    
    open("trainm", 'a').close()

if __name__ == '__main__':
    main()
