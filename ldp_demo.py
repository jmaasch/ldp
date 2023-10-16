# General importations.
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import platform
import time
import argparse
import multiprocessing
from scipy import stats

# Custom scripts.
from ldp import LDP
from ldp_utils import LDPUtils
from data_generation import DataGeneration

'''
# View versioning.
print("python version     :", platform.python_version())
print("numpy version      :", np.__version__)
print("pandas version     :", pd.__version__)
print("matplotlib version :", matplotlib.__version__)
print("seaborn version    :", sns.__version__)
'''

def main():

    # Parse arguments.
    # Example call:
    # python ldp_demo.py -x=1 -m=0 -b=1 -n=5000 -a=0.005 -r=10 -e=0
    parser = argparse.ArgumentParser()
    parser.add_argument("-x",
                        type = int,
                        help = "boolean for whether x causes y",
                        default = 1)
    parser.add_argument("-n",
                        type = int,
                        help = "total observations in synethic dataset",
                        default = 5000)
    parser.add_argument("-a",
                        type = float,
                        help = "alpha", 
                        default = 0.05)
    parser.add_argument("-m",
                        type = int,
                        help = "boolean for whether to include m-structures",
                        default = 0)
    parser.add_argument("-b",
                        type = int,
                        help = "boolean for whether to include butterfly structures", 
                        default = 0)
    parser.add_argument("-r",
                        type = int,
                        help = "total replicate runs of the algorithm", 
                        default = 10)
    parser.add_argument("-e",
                        type = int,
                        help = "export results", 
                        default = 0)
    args = parser.parse_args()
    
    # Error checking.
    if args.x not in [0,1]:
        raise ValueError("Argument -x must be in [0,1].")
    if args.b not in [0,1]:
        raise ValueError("Argument -b must be in [0,1].")
    if args.m not in [0,1]:
        raise ValueError("Argument -m must be in [0,1].")
    if args.e not in [0,1]:
        raise ValueError("Argument -e must be in [0,1].")

    results_df_list = []
    accuracy_list   = []
    z1_precision    = []
    z1_recall       = []
    z1_f1           = []
    total_times     = []
    total_tests     = []
    utils = LDPUtils()
    
    start_total = time.time()
    for replicate in range(args.r):
        
        # Generate data.
        dg = DataGeneration()
        df_vars, df_noise = dg.generate_linear_gaussian(n = args.n,
                                                        x_causes_y = args.x,
                                                        m_structure = args.m,
                                                        butterfly_structure = args.b,
                                                        coefficient_range = (1,2),
                                                        verbose = False)

        # Test.
        ldp = LDP(data = df_vars, 
                  independence_test = "fisher")
        start = time.time()
        print("Begin.")
        results = ldp.partition_z(exposure = "X",
                                  outcome = "Y",
                                  alpha = args.a,
                                  verbose = True)
        total_time = time.time() - start
        total_times.append(total_time)
        total_tests.append(ldp.total_tests)
        print("TOTAL TESTS:", ldp.total_tests)
        print("Partition obtained in {} seconds.".format(round(total_time, 3)))

        # Align predictions with ground truth.
        true_labels = ["Z1", "Z2", "Z3", "Z4",
                       "Z5", "Z6", "Z7", "Z8"]
        if args.m:
            true_labels = true_labels + ["Z5 (M1)", "Z4 (M2)", "Z2 (M3)"]
        if args.b:
            true_labels = true_labels + ["Z1 (B1)", "Z1 (B2)", "Z1 (B3)"]
        pred_labels = results.get("Predicted label")
        
        df_results = pd.DataFrame({"True label": true_labels, 
                                   "Pred label": pred_labels})
        df_results["Label concordant"] = [True if x[:2] in y else False \
                                          for x,y in zip(df_results["True label"],df_results["Pred label"])]
        df_results["Pred bool"] = results.get("Predicted boolean")
        df_results["True bool"] = [True if "Z1" in x else False for x in true_labels]

        [acc, f1, precision, recall, roc] = ldp.score(df_results["True bool"],
                                                      df_results["Pred bool"], 
                                                      verbose = True, 
                                                      plot_confusion = False)
        z1_precision.append(precision)
        z1_recall.append(recall)
        z1_f1.append(f1)
        
        if args.r == 1:
            print(df_results)
            overall_accuracy = df_results["Label concordant"].sum() / df_results["Label concordant"].shape[0]
            print("\n--*-- OVERALL ACCURACY: {} --*--".format(overall_accuracy))
            if args.e:
                df_results.to_csv("results_discrete_xy-{}_noise-{}_fun-{}_n-{}.csv".format(args.x,
                                                                                           args.d,
                                                                                           args.f,
                                                                                           args.n), index = False)
            return
        else:
            df_results["Replicate"] = replicate
            results_df_list.append(df_results)
            accuracy = df_results["Label concordant"].sum() / df_results["Label concordant"].shape[0]
            accuracy_list.append(accuracy)

    # Aggregate results.
    df_all = pd.concat(results_df_list)
    print(df_all.head(24))
    if args.e:
        df_all.to_csv("results_xy-{}_n-{}_m-{}_b-{}_a-{}.csv".format(args.x,
                                                                     args.n,
                                                                     args.m,
                                                                     args.b,
                                                                     args.a), index = False)
    
    # Get all performance metrics for Z1.
    print("\nZ1 metrics for all tested variables:")
    [acc, f1, precision, recall, roc] = ldp.score(df_all["True bool"],
                                                 df_all["Pred bool"], 
                                                 verbose = True, 
                                                 plot_confusion = False)
    
    # Get 95% confidence intervals.
    mean_accuracy, ci_accuracy = utils.get_ci([x * 100 for x in accuracy_list])
    mean_precision, ci_precision = utils.get_ci([x * 100 for x in z1_precision])
    mean_recall, ci_recall = utils.get_ci([x * 100 for x in z1_recall])
    mean_time, ci_time = utils.get_ci(total_times)
    mean_test, ci_test = utils.get_ci(total_tests)
    
    
    # Report performance.
    print(df_all["Label concordant"].sum(), df_all["Label concordant"].shape[0])
    print("\n--*-- OVERALL ACCURACY: {} ({}-{}) --*--".format(round(mean_accuracy, 1), 
                                                              round(ci_accuracy[0], 1),
                                                              min(round(ci_accuracy[1], 1), 100)))
    print(accuracy_list)
    
    print("--*-- Z1 PRECISION: {} ({}-{}) --*--".format(round(mean_precision, 1), 
                                                        max(round(ci_precision[0], 1), 0),
                                                        min(round(ci_precision[1], 1), 100)))
    print("--*-- Z1 RECALL: {} ({}-{}) --*--".format(round(mean_recall, 1), 
                                                     max(round(ci_recall[0], 1), 0),
                                                     min(round(ci_recall[1], 1), 100)))
    print("\nAll replicates obtained in {} minutes.".format(round((time.time() - start_total) / 60, 3)))
    print("--*-- MEAN RUNTIME: {} ({}-{}) --*--".format(round(mean_time, 4), 
                                                        round(ci_time[0], 4),
                                                        max(round(ci_time[1], 4), 0)))
    print("--*-- MEAN TOTAL TESTS: {} ({}-{}) --*--".format(round(mean_test, 1), 
                                                            ci_test[0],
                                                            max(ci_test[1], 0)))
    quad_tests = (df_vars.shape[1]-2)**2
    print("--*-- QUADRATIC TOTAL TESTS: {} ({} true / {} quad = {}) --*--".format(quad_tests,
                                                                                  round(mean_test, 1), 
                                                                                  quad_tests,
                                                                                  round(mean_test, 1) / quad_tests))

    print()
    return

# Protecting "main" necessary for multiprocessing.
if __name__ == "__main__": 
    main()