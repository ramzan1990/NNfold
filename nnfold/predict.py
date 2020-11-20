#!/usr/bin/env python
import os
#os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import numpy as np
from math import sqrt
import numpy as np
from numpy import zeros
import sys
import re
import math
import time
from random import shuffle
#from PIL import Image, ImageOps

def main():
    import apetype as at
    
    class CLI_Settings(at.ConfigBase):
        inp: str     # Input fasta file
        output: str  # Output directory
        seq_len: int # Local model length
        c_len: int   # Global model length
        tag: str     # tag
        
    settings = CLI_Settings()

    # Import tensorflow, after processing CLI_Settings as it prints warnings
    import tensorflow as tf
    from tensorflow.python.saved_model import builder as saved_model_builder

    cpg1 = 0
    cpg2 = 0
    cpg3 = 0
    cpg4 = 0
    
    
    def clean_seq(s):
        ns = s.upper()    
        pattern = re.compile(r'\s+')
        ns = re.sub(pattern, '', ns)
        ns = re.sub(r'[^a-zA-Z]{1}', 'N', ns)
        return ns
    
    def brun(sess, x, y, a, keep_prob1, keep_prob2):
        preds = []
        batch_size = 128
        number_of_full_batch=int(math.ceil(float(len(a))/batch_size))
        for i in range(number_of_full_batch):
            preds += list(sess.run(y, feed_dict={x: np.asarray(a[i*batch_size:(i+1)*batch_size]),
             keep_prob1: 1.0, keep_prob2: 1.0}))
        return preds
    
    def encode(s):
        ns = s.upper()    
        pattern = re.compile(r'\s+')
        ns = re.sub(pattern, '', ns)
        ns = re.sub(r'[^a-zA-Z]{1}', 'N', ns)
        ns = ns.replace("A", "1,0,0,0,")
        ns = ns.replace("U", "0,1,0,0,")
        ns = ns.replace("G", "0,0,1,0,")
        ns = ns.replace("C", "0,0,0,1,")
        ns = ns.replace("N", "0,0,0,0,")
        if re.search('[a-zA-Z]', ns):
            print(ns)
            print('Non-standard symbol in sequence - changed to A.')
            ns = re.sub("[a-zA-Z]", "0,0,0,0,", ns)
        return ns[:-1]
    
    
    def toBP(m, sequence, binding):  
        size = len(sequence)
        pairs = []
        used = []
        pairs = []
        summ = m + m.T
        index = np.argwhere(np.triu(summ, k=1)>=0.5)   
        pairs = list(zip(index[:, 0], index[:, 1], summ[index[:,0], index[:,1]]))
        jl = reversed(range(len(pairs)))
        for j in jl:
            if(pairs[j][0] >= size or pairs[j][1] >= size):
                del pairs[j]
            else:
                if(pairs[j][0] < len(binding) and pairs[j][1] < len(binding)):
                    if(binding[pairs[j][0]] < 0.5 or binding[pairs[j][1]] < 0.5):
                        del pairs[j]
    
        pairs.sort(key=lambda x: x[2], reverse=True)
        current_pairs = []
        used = []
        for j in range(len(pairs)):
            if(pairs[j][0] in used or pairs[j][1] in used):
                continue
            used.append(pairs[j][0])
            used.append(pairs[j][1])
            current_pairs.append(pairs[j])
    
        pairs_map = {}
        for i in range(len(current_pairs)):
            pairs_map[current_pairs[i][0]] = current_pairs[i][1]
            pairs_map[current_pairs[i][1]] = current_pairs[i][0]
    
        str_list = []
        for i in range(size):
            p = 0
            if(i in pairs_map):
                p = pairs_map[i] + 1
            str_list.append(str(i+1) + " " + sequence[i] + " " + str(p))
            str_list.append("\n")
    
        return ''.join(str_list) 
    
    def checkBP(a, b):
        if( not ((a == "A" and b == "U")
                 or (a == "U" and b == "A")
                 or (a == "G" and b == "C")
                 or (a == "C" and b == "G")
                 or (a == "G" and b == "U")
                 or (a == "U" and b == "G"))
                 ): 
            return False
        return True
    
    np.random.seed(2504) 
    
    max_features = 4
    inp = settings.inp
    output =  settings.output
    seq_len = settings.seq_len
    c_len = settings.c_len
    tag = settings.tag
    pos_seq = []
    pos_seq2 = []
    origs = []
    names = []
    seq = ""
    with open(inp) as f:
        for line in f:
            if(line.startswith(">")):
                names.append(line.strip())
                if(len(seq)!=0):
                    seq = clean_seq(seq)
                    origs.append(seq)
                    en = seq_len - (len(seq) % seq_len)
                    seq1 = seq + 'N'*en
                    pos_seq.append(np.fromstring(encode(seq1), dtype=int, sep=",").reshape(-1, 4))
                    seq2 = seq + 'N'*(c_len - len(seq))
                    pos_seq2.append(np.fromstring(encode(seq2), dtype=int, sep=",").reshape(-1, 4))
                    seq=""                    
                continue                
            else:
                seq+=line
    
    if(len(seq)!=0):
        seq = clean_seq(seq)
        origs.append(seq)
        en = seq_len - (len(seq) % seq_len)
        seq1 = seq + 'N'*en
        pos_seq.append(np.fromstring(encode(seq1), dtype=int, sep=",").reshape(-1, 4))
        seq2 = seq + 'N'*(c_len - len(seq))
        pos_seq2.append(np.fromstring(encode(seq2), dtype=int, sep=",").reshape(-1, 4))
    
    
    
    print("------------------------------------------------------------------------")
    print("------------------------------------------------------------------------")
    
    binding = []
    new_graph = tf.Graph()
    #config = tf.ConfigProto(
    #        device_count = {'GPU': 0}
    #    )
    #,config=config
    with tf.Session(graph=new_graph) as sess:
        # Import the previously export meta graph.
        tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], "model_rna_check_" + tag)
        # Restore the variables
        saver = tf.train.Saver()
        saver.restore(sess, "model_rna_check_" + tag + "/variables/variables")
        input_x = tf.get_default_graph().get_tensor_by_name("input_rna:0")
        y = tf.get_default_graph().get_tensor_by_name("output_rna:0")
        kr1 = tf.get_default_graph().get_tensor_by_name("kr_rna1:0")
        kr2 = tf.get_default_graph().get_tensor_by_name("kr_rna2:0")    
        for i in range(len(pos_seq)):
            pred = sess.run(y, feed_dict={input_x: [pos_seq2[i][0:c_len]], kr1: 1.0, kr2: 1.0})
            binding.append(pred)
    
    print("------------------------------------------------------------------------")
    print("------------------------------------------------------------------------")
    
    cp1 = 0
    cp2 = 0
    cp3 = 0
    str_list = []
    str_list_DP = []
    new_graph = tf.Graph()
    with tf.Session(graph=new_graph) as sess:
        # Import the previously export meta graph.
        tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], "model_rna_" + tag)
        # Restore the variables
        saver = tf.train.Saver()
        saver.restore(sess, "model_rna_" + tag + "/variables/variables")
        input_x = tf.get_default_graph().get_tensor_by_name("input_rna:0")
        y = tf.get_default_graph().get_tensor_by_name("output_rna:0")
        kr1 = tf.get_default_graph().get_tensor_by_name("kr_rna1:0")
        kr2 = tf.get_default_graph().get_tensor_by_name("kr_rna2:0")
        #predict = brun(sess, y, sequences)
        skip_step = seq_len
        min_mat_score = 1
        for i in range(len(pos_seq)):
            num = math.ceil(float(len(pos_seq[i]))/seq_len)
            if(i%10 == 0):
                print(str(i) + " - ", end='', flush=True)
            seq_mat = zeros((num*seq_len, num*seq_len)) 
            num = len(pos_seq[i]) - seq_len + 1  
            r_i = 0
            while r_i < num:
                c_i = 0
                skip = True
                while c_i < num: 
                    zos1 = np.squeeze(pos_seq[i][r_i : r_i + seq_len])
                    zos2 = np.squeeze(pos_seq[i][c_i : c_i + seq_len])
                    
                    pred = sess.run(y, feed_dict={input_x : [np.concatenate((zos1, zos2), axis=1)],
                        kr1 : 1.0, kr2 : 1.0})
                    pred = np.asarray(pred) 
                    mat = np.reshape(pred[0], (seq_len, seq_len))
                    #mat_round = mat + 0.3
                    #mat_round = np.around(mat, 0).astype(int) 
                    #if(mat_round.sum()<min_mat_score):
                    #    c_i = c_i + skip_step
                    #    continue
                    #else:
                    #    skip = False
                    seq_mat[r_i : r_i + seq_len, c_i : c_i + seq_len] = mat
                    #pix = mat.clip(min=0, max=1)   
                    #img1 = ImageOps.invert(Image.fromarray(np.uint8(pix * 255) , 'L'))
                    #img1.save("pics_p/img" + str(r_i) + "_" + str(c_i) + ".bmp","BMP")
                    c_i = c_i + seq_len
                if(skip):
                    r_i = r_i + skip_step
                else:
                    r_i = r_i + seq_len 
    
            #pix = seq_mat.clip(min=0, max=1)   
            #img1 = ImageOps.invert(Image.fromarray(np.uint8(pix * 255) , 'L'))
            #img1.save("pics_t/img" + str(i) + ".bmp","BMP")
            with open(output + "/" + str(i), 'w+') as f:
                f.write(toBP(seq_mat, list(origs[i]), np.squeeze(binding[i])))
    
  
if __name__ == '__main__':
    main()
