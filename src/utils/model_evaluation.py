from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import poisson

def auprc_score(true, pred):

    precision, recall, _ = precision_recall_curve(true, pred)

    return auc(recall, precision)

def auroc_score(true, pred):

    return roc_auc_score(true, pred)

def AP_score(true, pred):

    return average_precision_score(true, pred)

def metrics(true, pred, index = None):

    assert true.shape == pred.shape

    if index is None:
        auprc = auprc_score(true.flatten(), pred.flatten())
        auroc = auroc_score(true.flatten(), pred.flatten())
        ap = AP_score(true.flatten(), pred.flatten())
    else:
        auprc = auprc_score(true[index, :, :].flatten(), pred[index, :, :].flatten())
        auroc = auroc_score(true[index, :, :].flatten(), pred[index, :, :].flatten())
        ap = AP_score(true[index, :, :].flatten(), pred[index, :, :].flatten())
        
    return auprc, auroc, ap

def bp_eval(true, pred, base_true = None):

    assert true.shape == pred.shape

    if base_true is not None:

        assert true.shape == base_true.shape
        
        base_true_copy = base_true.copy()
        base_true_copy[base_true_copy == 2] = 1
        base_BP_index = np.where(base_true_copy == 1)[0]
        base_value, base_counts = np.unique(base_BP_index, return_counts=True)
        base_unique_index = np.where(base_counts == 1)[0]
        base_multiple_index = np.where(base_counts > 1)[0]

        all_true = true[true == base_true]
        all_pred = pred[true == base_true]
        all = metrics(all_true, all_pred, index = None)

        if len(base_unique_index) == 0:
            unique = ([],[],[])
        else:
            unique_true = true[base_unique_index, :, :]
            unique_true = unique_true[true[base_unique_index, :, :] == base_true[base_unique_index, :, :]]
            unique_pred = pred[base_unique_index, :, :]
            unique_pred = unique_pred[true[base_unique_index, :, :] == base_true[base_unique_index, :, :]]
            unique = metrics(unique_true, unique_pred, index = None)

        if len(base_multiple_index) == 0:
            multiple = ([],[],[])
        else:
            multiple_true = true[base_multiple_index, :, :]
            multiple_true = multiple_true[true[base_multiple_index, :, :] == base_true[base_multiple_index, :, :]]
            multiple_pred = pred[base_multiple_index, :, :]
            multiple_pred = multiple_pred[true[base_multiple_index, :, :] == base_true[base_multiple_index, :, :]]
            multiple = metrics(multiple_true, multiple_pred, index = None)

        return all, unique, multiple

    BP_index = np.where(true == 1)[0]
    value, counts = np.unique(BP_index, return_counts=True)
    unique_index = np.where(counts == 1)[0]
    multiple_index = np.where(counts > 1)[0]

    all = metrics(true, pred, index = None)

    if len(unique_index) == 0:
        unique = ([],[],[])
    else:
        unique = metrics(true, pred, index = unique_index)

    if len(multiple_index) == 0:
        multiple = ([],[],[])
    else:
        multiple = metrics(true, pred, index = multiple_index)

    return all, unique, multiple

def model_pred(model_path, input, ensemble = False, ensemble_weight = None, **kwargs):

    if ensemble:
        assert len(model_path) == 3 and len(input) == 3 and ensemble_weight != None

    if ensemble:
        pred = []
        for i in range(0, 3):
            model = tf.keras.models.load_model(model_path[i], **kwargs)
            sub_pred = model.predict(input[i], verbose = 0)
            pred.append(sub_pred)
        pred = ensemble_weight * pred[0] + ensemble_weight * pred[1] + ensemble_weight * pred[2]

    else:
        model = tf.keras.models.load_model(model_path, **kwargs)
        pred = model.predict(input, verbose = 0)

    return pred

def model_eval(model_path, input, true, ensemble = False, ensemble_weight = None, base_true = None, **kwargs):

    pred = model_pred(model_path, input, ensemble = ensemble, ensemble_weight = ensemble_weight, **kwargs)

    all, unique, multiple = bp_eval(true, pred, base_true)    

    return pred, all, unique, multiple      

def num_model_eval(path_prefix, path_suffix, num, input, true, ensemble = False, ensemble_weight = None, base_true = None, **kwargs):

    if ensemble:
        assert len(path_suffix) == 3 and len(input) == 3 and ensemble_weight != None

    pred_l = []
    all_auprc_l = []
    all_auroc_l = []
    all_ap_l = []
    unique_auprc_l = []
    unique_auroc_l = []
    unique_ap_l = []
    multiple_auprc_l = []
    multiple_auroc_l = []
    multiple_ap_l = []
    for i in range(0, num):

        if ensemble:
            model_path = [path_prefix[0] + str(i + 1) + path_suffix[0], 
                          path_prefix[1] + str(i + 1) + path_suffix[1],
                          path_prefix[2] + str(i + 1) + path_suffix[2]]
        else:
            model_path = path_prefix + str(i + 1) + path_suffix

        pred, all, unique, multiple = model_eval(model_path = model_path, input = input, true = true, ensemble = ensemble, ensemble_weight = ensemble_weight, base_true = base_true, **kwargs)
        all_auprc, all_auroc, all_ap = all
        unique_auprc, unique_auroc, unique_ap = unique
        multiple_auprc, multiple_auroc, multiple_ap = multiple

        pred_l.append(pred)
        all_auprc_l.append(all_auprc)
        all_auroc_l.append(all_auroc)
        all_ap_l.append(all_ap)
        unique_auprc_l.append(unique_auprc)
        unique_auroc_l.append(unique_auroc)
        unique_ap_l.append(unique_ap)
        multiple_auprc_l.append(multiple_auprc)
        multiple_auroc_l.append(multiple_auroc)
        multiple_ap_l.append(multiple_ap)
        
    return pred_l, all_auprc_l, all_auroc_l, all_ap_l, unique_auprc_l, unique_auroc_l, unique_ap_l, multiple_auprc_l, multiple_auroc_l, multiple_ap_l

def model_eval_plot(*args, name, title):
    
    value = np.array([])
    metric = np.array([])
    bp_class = np.array([])
    method = np.array([])
    for eval_list, eval_name in zip(args, name):
        pred_l, all_auprc_l, all_auroc_l, all_ap_l, unique_auprc_l, unique_auroc_l, unique_ap_l, multiple_auprc_l, multiple_auroc_l, multiple_ap_l = eval_list
        
        value = np.append(value, all_auprc_l)
        metric = np.append(metric, ['auPRC'] * len(all_auprc_l))
        bp_class = np.append(bp_class, ['All'] * len(all_auprc_l))
        
        value = np.append(value, unique_auprc_l)
        metric = np.append(metric, ['auPRC'] * len(unique_auprc_l))
        bp_class = np.append(bp_class, ['Unique'] * len(unique_auprc_l))
        
        value = np.append(value, multiple_auprc_l)
        metric = np.append(metric, ['auPRC'] * len(multiple_auprc_l))
        bp_class = np.append(bp_class, ['Multiple'] * len(multiple_auprc_l))

        method = np.append(method, [eval_name] * (len(all_auprc_l) + len(unique_auprc_l) + len(multiple_auprc_l)))
    
    stack = pd.DataFrame({'auPRC': value, 'Metric': metric, 'BP_class': bp_class, 'Method': method})
    sns.set_theme(style="whitegrid", palette="pastel")
    plot = sns.barplot(data=stack, x="auPRC", y="BP_class", hue="Method")
    # sns.move_legend(plot, "lower center", bbox_to_anchor=(.5, 1), ncol=2, title=None, frameon=False)
    sns.move_legend(plot, "upper left", bbox_to_anchor=(1, 1))
    plot.set_xlim(0.60, 1)
    for i in plot.containers:
        plot.bar_label(i,fmt='%.4f', label_type = "center", fontsize = 6)
    plot.set(title=title)
    # return(stack)

def num_model_eval_window(input, true, y_fit, q, base_true = None):

    model_pred_l  = input[0]

    rows, cols = np.nonzero(y_fit)
    dist_mean = np.mean(70 - cols)
    quantile = poisson.ppf([q / 2, 1 - q / 2], dist_mean)
    window_start = int(70 - quantile[1])
    window_end = int(70 - quantile[0])

    true = true[:, window_start:window_end, :]
    BP_index = np.where(true == 1)[0]
    value, counts = np.unique(BP_index, return_counts=True)
    unique_index = np.where(counts == 1)[0]
    multiple_index = np.where(counts > 1)[0]
    row_index = np.append(unique_index, multiple_index)
    true = true[row_index, :, :]

    if base_true is not None:
        base_true = base_true[row_index, window_start:window_end, :]

    pred_l = []
    all_auprc_l = []
    all_auroc_l = []
    all_ap_l = []
    unique_auprc_l = []
    unique_auroc_l = []
    unique_ap_l = []
    multiple_auprc_l = []
    multiple_auroc_l = []
    multiple_ap_l = []
    for pred in model_pred_l:
    
        pred = pred[row_index, window_start:window_end, :]
        assert pred.shape == true.shape

        all, unique, multiple = bp_eval(true, pred, base_true)
        all_auprc, all_auroc, all_ap = all
        unique_auprc, unique_auroc, unique_ap = unique
        multiple_auprc, multiple_auroc, multiple_ap = multiple

        pred_l.append(pred)
        all_auprc_l.append(all_auprc)
        all_auroc_l.append(all_auroc)
        all_ap_l.append(all_ap)
        unique_auprc_l.append(unique_auprc)
        unique_auroc_l.append(unique_auroc)
        unique_ap_l.append(unique_ap)
        multiple_auprc_l.append(multiple_auprc)
        multiple_auroc_l.append(multiple_auroc)
        multiple_ap_l.append(multiple_ap)

    return pred_l, all_auprc_l, all_auroc_l, all_ap_l, unique_auprc_l, unique_auroc_l, unique_ap_l, multiple_auprc_l, multiple_auroc_l, multiple_ap_l

def show_statistics(input, metric_show):

    mean = np.mean(input) * 100
    error = 1.96 * np.std(input * 100) / np.sqrt(len(input))

    print(metric_show + ":")
    if any(input):
        print(f"{mean:.2f}", "error:", f"{error:.4f}")
    else:
        print("No such data")

    return mean, error
    
def summary_statistics(input, method_show):

    pred_l, all_auprc_l, all_auroc_l, all_ap_l, unique_auprc_l, unique_auroc_l, unique_ap_l, multiple_auprc_l, multiple_auroc_l, multiple_ap_l = input

    print("===========================================================")
    print(method_show, "All BPs:")
    all_auprc_mean, all_auprc_error = show_statistics(all_auprc_l, "auPRC")
    all_auroc_mean, all_auroc_error = show_statistics(all_auroc_l, "auROC")
    all_ap_mean, all_ap_error = show_statistics(all_ap_l, "AP")
    print("")
    print(method_show, "Unique BPs:")
    unique_auprc_mean, unique_auprc_error = show_statistics(unique_auprc_l, "auPRC")
    unique_auroc_mean, unique_auroc_error = show_statistics(unique_auroc_l, "auROC")
    unique_ap_mean, unique_ap_error = show_statistics(unique_ap_l, "AP")
    print("")
    print(method_show, "Multiple BPs:")
    multiple_auprc_mean, multiple_auprc_error = show_statistics(multiple_auprc_l, "auPRC")
    multiple_auroc_mean, multiple_auroc_error = show_statistics(multiple_auroc_l, "auROC")
    multiple_ap_mean, multiple_ap_error = show_statistics(multiple_ap_l, "AP")
    print("===========================================================")

    return all_auprc_mean, all_auprc_error, all_auroc_mean, all_auroc_error, all_ap_mean, all_ap_error, unique_auprc_mean, unique_auprc_error, unique_auroc_mean, unique_auroc_error, unique_ap_mean, unique_ap_error, multiple_auprc_mean, multiple_auprc_error, multiple_auroc_mean, multiple_auroc_error, multiple_ap_mean, multiple_ap_error

def threshold_determined(input, true, method_show, base_true = None):

    pred_l, all_auprc_l, all_auroc_l, all_ap_l, unique_auprc_l, unique_auroc_l, unique_ap_l, multiple_auprc_l, multiple_auroc_l, multiple_ap_l = input

    threshold_l = []
    fscore_l = []
    for pred in pred_l:
        if base_true is not None:
            true_adj = true[true == base_true]
            pred_adj = pred[true == base_true]
            precision, recall, threshold = precision_recall_curve(true_adj, pred_adj)
        else:    
            precision, recall, threshold = precision_recall_curve(true.flatten(), pred.flatten())
        a = 2 * precision * recall
        b = precision + recall
        fscore = np.divide(a, b, out = np.zeros_like(a), where = b!= 0)
        ix = np.argmax(fscore)
        threshold_l.append(threshold[ix])
        fscore_l.append(fscore[ix])

    print("===========================================================")
    print(method_show, "All BPs:")
    show_statistics(fscore_l, "F1 score")
    print("===========================================================")

    return threshold_l, fscore_l

def metrics_with_threshold(input, true, threshold, method_show):

    pred_l, all_auprc_l, all_auroc_l, all_ap_l, unique_auprc_l, unique_auroc_l, unique_ap_l, multiple_auprc_l, multiple_auroc_l, multiple_ap_l = input

    fscore_l = []
    for i in range(0, len(pred_l)):
        pred = pred_l[i]
        pred[pred >= threshold[i]] = 1
        pred[pred < threshold[i]] = 0

        precision = precision_score(true.flatten(), pred.flatten())
        recall = recall_score(true.flatten(), pred.flatten())

        fscore = (2 * precision * recall) / (precision + recall)
        fscore_l.append(fscore)

    print("===========================================================")
    print(method_show, "All BPs:")
    show_statistics(fscore_l, "F1 score")
    print("===========================================================")

    return fscore_l

def plot_attention(input_oh_1, input_oh_2, input_kmers_1, input_kmers_2, input_word2vec_1, input_word2vec_2, output, model, layer_index):
    
    oh_pred = model[0]((tf.expand_dims(input_oh_1, axis = 0), tf.expand_dims(input_oh_2, axis = 0)))
    kmers_pred = model[1]((tf.expand_dims(input_kmers_1, axis = 0), tf.expand_dims(input_kmers_2, axis = 0)))
    word2vec_pred = model[2]((tf.expand_dims(input_word2vec_1, axis = 0), tf.expand_dims(input_word2vec_2, axis = 0)))

    oh_attention_weight = model[0].layers[layer_index[0]].last_attention_weights[0]
    kmers_attention_weight = model[1].layers[layer_index[1]].last_attention_weights[0]
    word2vec_attention_weight = model[2].layers[layer_index[2]].last_attention_weights[0]
 
    str_input = tf.argmax(input_oh_1, axis = 1) + 1
    string2int_layer = tf.keras.layers.StringLookup(vocabulary = ["A", "T", "C", "G"], output_mode='int', invert = True)
    str_input = string2int_layer(str_input).numpy().astype(str)
  
    fig = plt.figure(figsize=(30, 30))
    fontdict = {'fontsize': 10}
    bp_index = np.where(output == 1)[0]

    ax1 = fig.add_subplot(1, 3, 1)
    ax2 = fig.add_subplot(1, 3, 2)
    ax3 = fig.add_subplot(1, 3, 3)

    ax1.matshow(oh_attention_weight, cmap='viridis', vmin=0.0)
    ax2.matshow(kmers_attention_weight, cmap='viridis', vmin=0.0)
    ax3.matshow(word2vec_attention_weight, cmap='viridis', vmin=0.0)

    ax1.set_xticks(ticks = np.arange(len(str_input)), labels = str_input, fontdict=fontdict)
    # ax1.set_yticks(ticks = np.arange(len(str_input)), fontdict=fontdict)
    ax2.set_xticks(ticks = np.arange(len(str_input)), labels = str_input, fontdict=fontdict)
    # ax2.set_yticks(ticks = np.arange(len(str_input)), fontdict=fontdict)
    ax3.set_xticks(ticks = np.arange(len(str_input)), labels = str_input, fontdict=fontdict)
    # ax3.set_yticks(ticks = np.arange(len(str_input)), fontdict=fontdict)

    for i in bp_index:
        ax1.get_xticklabels()[i].set_color("red")
        # ax1.get_yticklabels()[i].set_color("red")
        ax2.get_xticklabels()[i].set_color("red")
        # ax2.get_yticklabels()[i].set_color("red")
        ax3.get_xticklabels()[i].set_color("red")
        # ax3.get_yticklabels()[i].set_color("red")

    ax1.set_xlabel('Sequence output')
    ax1.set_ylabel('Position')
    ax2.set_xlabel('Sequence output')
    ax2.set_ylabel('Position')
    ax3.set_xlabel('Sequence output')
    ax3.set_ylabel('Position')

    ax1.set_title("One Hot")
    ax2.set_title("Kmers embedding")
    ax3.set_title("Word2vec embedding")