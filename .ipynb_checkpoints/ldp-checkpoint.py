# General imports.
import pandas as pd
import numpy as np
import warnings
import time
import matplotlib.pyplot as plt
import seaborn as sns
import random
import itertools
import math
from scipy import stats

# Independence testing imports.
from causallearn.utils.cit import CIT
import networkx as nx

# sklearn imports.
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix


class LDP():


    def __init__(self,
                 data: pd.DataFrame,
                 independence_test: str = "chi"):

        # Store data.
        self.data = data

        # Instantiate objects for independence testing.
        if independence_test == "chi":
            self.test = CIT(self.data.to_numpy(), "chisq")
        elif independence_test == "fisher":
            self.test = CIT(self.data.to_numpy(), "fisherz")
        elif independence_test == "kci":
            self.test = CIT(self.data.to_numpy(), "kci")
        elif independence_test == "gsq":
            self.test = CIT(self.data.to_numpy(), "gsq")
        elif independence_test == "oracle":
            self.test = "oracle"

        # Init data structures for memoization.
        self.x_ind_z_dict = dict()
        self.y_ind_z_dict = dict()

        # Track how many tests were performed.
        self.total_tests = 0


    def ind_test(self,
                 var_0: str,
                 var_1: str,
                 conditioning_set: list = None) -> float:

        '''
        Wrapper for independence test that takes variable names as strings
        for easier user interpretability.

        Return:
        -------
        p_value
        '''

        if self.test != "oracle":
            # Obtain column indices (independence test references array columns).
            df_cols = list(self.data.columns)
            var_0_idx = df_cols.index(var_0)
            var_1_idx = df_cols.index(var_1)
            if conditioning_set is not None:
                cond_idx = [df_cols.index(x) for x in conditioning_set]
            else:
                cond_idx = None
            p_value = self.test(var_0_idx, var_1_idx, cond_idx)
        else:
            # Oracle test using ground truth DAG.
            p_value = self.oracle(var_0, var_1, conditioning_set = conditioning_set)

        # Increment total tests performed.
        self.total_tests += 1

        return p_value


    def oracle(self,
               var_0,
               var_1,
               conditioning_set = None) -> float:

        '''
        Oracle independence test given ground truth DAG.
        This method only works for variants of the 10-node DAG, where each covariate
        is directly adjacent to {X,Y} and has no paths to other covariates.

        This test is used for the experiments reported in Figure 2 and Table B.1.
        '''

        if self.dag is None:
            raise ValueError("self.dag is None; must supply ground truth DAG as numpy array to use oracle.")
        if self.var_names is None:
            raise ValueError("self.var_names is None; must supply ground truth variable names as list to use oracle.")

        # Obtain column indices.
        var_0 = var_0.split(sep = ".")[0]
        var_1 = var_1.split(sep = ".")[0]
        var_0_idx = self.var_names.index(var_0)
        var_1_idx = self.var_names.index(var_1)
        if conditioning_set is not None:
            conditioning_set = set([x.split(sep = ".")[0] for x in conditioning_set])
            if var_0 in conditioning_set:
                conditioning_set.remove(var_0)
            if var_1 in conditioning_set:
                conditioning_set.remove(var_1)
            cond_idx = set([self.var_names.index(x) for x in conditioning_set])
        else:
            cond_idx = set()

        # Get p-value using ground truth DAG.
        graph = nx.from_numpy_array(self.dag, create_using = nx.DiGraph)
        p_val = 1 if nx.d_separated(graph, {var_0_idx}, {var_1_idx}, cond_idx) else 0
        return p_val


    def partition_z(self,
                    exposure: str = "X",
                    outcome: str = "Y",
                    instrument: str = None,
                    causes_outcome: list = None,
                    discrete: bool = False,
                    use_random_z4: bool = False,
                    use_random_z5: bool = False,
                    alpha: float = 0.005,
                    alpha_step_5: float = None,
                    scale: bool = True,
                    verbose: bool = False) -> dict:


        '''
        This method partitions dataset Z.
        '''

        # Track whether a valid adjustment set exists in Z.
        self.vas_exists = False

        # Extract variable names from columns.
        self.z_names = list(self.data.columns)
        self.z_names.remove(exposure)
        self.z_names.remove(outcome)

        # Initialize data structures that will store results.
        self.pred_bool_list   = [False] * (len(self.z_names))
        self.pred_bool_dict   = dict(zip(self.z_names, self.pred_bool_list))
        self.pred_label_dict  = dict()
        self.z_prime          = []
        self.z1               = [] # Confounders.
        self.z4               = [] # Parents of outcome.
        self.z5               = [] # Instrumental variables.
        self.z7               = [] # Children of exposure.
        self.z8               = [] # Isolated variables.
        self.z_mix            = []
        self.z_post           = []
        self.z1_z5            = []
        self.z5_adj_x      = []
        self.z5_z7            = [] 

        #---------------------------------------
        # Steps 1–3: Test for Z8, Z4, and Z5/Z7.
        #---------------------------------------

        start = time.time()
        self.partition_z_steps_1_2_3(exposure = exposure,
                                     outcome = outcome,
                                     alpha = alpha,
                                     verbose = verbose)
        if verbose:
            print("** Steps 1–3 complete in {} seconds.".format(round(time.time() - start, 4)))

        #---------------------------------------
        # Step 4: Identify a fraction of Z_post.
        #---------------------------------------

        if len(self.z4) > 0:
            start = time.time()
            self.partition_z_step_4(exposure = exposure,
                                    outcome = outcome,
                                    causes_outcome = causes_outcome,
                                    use_random_z4 = use_random_z4,
                                    alpha = alpha,
                                    verbose = verbose)
            if verbose:
                print("** Step 4 complete in {} seconds.".format(round(time.time() - start, 4)))
        else:
            if verbose:
                print("!! WARNING: Condition 2 unsatisfied (no Z4 discovered). Z_post that are descended from Y might be unidentifiable. A valid adjustment set might still be identifiable.")

        #---------------------------------------
        # Step 5: Identify Z_mix.
        #---------------------------------------

        start = time.time()
        if alpha_step_5 is None:
            self.partition_z_step_5(exposure = exposure,
                                    outcome = outcome,
                                    alpha = alpha,
                                    verbose = verbose)
        else:
            self.partition_z_step_5(exposure = exposure,
                                    outcome = outcome,
                                    alpha = alpha_step_5,
                                    verbose = verbose)
        if verbose:
            print("** Step 5 complete in {} seconds.".format(round(time.time() - start, 4)))

        #---------------------------------------
        # Step 6: Resolve Z_post and Z_mix.
        #---------------------------------------

        start = time.time()
        self.partition_z_step_6(exposure = exposure,
                                instrument = instrument,
                                use_random_z5 = use_random_z5,
                                alpha = alpha,
                                verbose = verbose)
        if verbose:
            print("** Step 6 complete in {} seconds.".format(round(time.time() - start, 4)))

        #---------------------------------------
        # Step 7: Resolve Z1 and Z5.
        #---------------------------------------

        start = time.time()
        self.partition_z_step_7(exposure = exposure,
                                outcome = outcome,
                                alpha = alpha,
                                verbose = verbose)

        if verbose:
            print("** Step 7 complete in {} seconds.".format(round(time.time() - start, 4)))

        #---------------------------------------
        # Step 8: Test Z5 criterion.
        #---------------------------------------

        start = time.time()
        self.test_z5_criterion(exposure = exposure,
                               outcome = outcome,
                               alpha = alpha,
                               verbose = verbose)

        if verbose:
            print("** Step 8 complete in {} seconds.".format(round(time.time() - start, 4)))

        #---------------------------------------
        # Return results.
        #---------------------------------------

        # Construct results dictionary.
        results = {"Predicted label": list(self.pred_label_dict.values()),
                   "Predicted boolean": list(self.pred_bool_dict.values()),
                   "Z1": self.z1,
                   "Z4": self.z4,
                   "Z5": self.z5,
                   "Z7": self.z7,
                   "Z8": self.z8,
                   "Z_post": self.z_post}
        return results


    def partition_z_steps_1_2_3(self,
                                exposure: str = "X",
                                outcome: str = "Y",
                                alpha: float = 0.005,
                                verbose: bool = False):

        for candidate in self.z_names:

            #-------------------------------------------
            # Step 1: Test for isolated variables (Z8).
            #-------------------------------------------


            is_isolated = self.test_z8(exposure = exposure,
                                             outcome = outcome,
                                             candidate = candidate,
                                             alpha = alpha,
                                             verbose = verbose)

            if is_isolated:
                self.z8.append(candidate)
                self.pred_label_dict[candidate] = "Z8"
                continue

            #-----------------------------------------
            # Step 2: Test for causes of outcome (Z4).
            #-----------------------------------------

            causes_y = self.test_z4(exposure = exposure,
                                                outcome = outcome,
                                                candidate = candidate,
                                                alpha = alpha,
                                                verbose = verbose)

            if causes_y:
                self.z4.append(candidate)
                self.pred_label_dict[candidate] = "Z4"
                continue

            #---------------------------------------------------------------------------
            # Step 3: Test for children of exposure (Z7) and sometimes instruments (Z5).
            #---------------------------------------------------------------------------

            # Conditioning only on the exposure is often sufficient in practice,
            # even in the presence of open backdoor paths.
            conditioning_set = [exposure]
            is_z5_z7 = self.test_z5_z7(conditioning_set = conditioning_set,
                                       exposure = exposure,
                                       outcome = outcome,
                                       candidate = candidate,
                                       alpha = alpha,
                                       verbose = verbose)

            if is_z5_z7:
                self.z5_z7.append(candidate)
                self.pred_label_dict[candidate] = "Z5 or Z7"
                continue

            # If no tests pass, declare as not identifiable.
            self.z_prime.append(candidate)
            self.pred_label_dict[candidate] = "not identifiable"


    def partition_z_step_4(self,
                           exposure: str = "X",
                           outcome: str = "Y",
                           causes_outcome: list = None,
                           use_random_z4: bool = False,
                           alpha: float = 0.005,
                           verbose: bool = False):

        #--------------------------------------------------------
        # Step 4: Identify a fraction of Z_post.
        #--------------------------------------------------------

        # Consider all candidates that have not been labeled.
        for candidate in self.z_prime:

            if isinstance(causes_outcome, str):
                causes_outcome = [causes_outcome]
            if causes_outcome is None:
                if use_random_z4 and len(self.z4) > 0:
                    causes_outcome = [random.choice(self.z4)]
                else:
                    causes_outcome = self.z4.copy()
            is_post = self.test_zpost(exposure = exposure,
                                      outcome = outcome,
                                      candidate = candidate,
                                      causes_outcome = causes_outcome,
                                      alpha = alpha,
                                      verbose = verbose)
            if is_post:
                if verbose:
                    print(candidate, "is in Z_post (STEP 4).")
                self.pred_label_dict[candidate] = "Z2 or Z3 or Z6"
                self.z_post.append(candidate)

        # Remove Z_post from remaining candidates.
        self.z_prime = [x for x in self.z_prime if x not in self.z_post]


    def partition_z_step_5(self,
                           exposure: str = "X",
                           outcome: str = "Y",
                           alpha: float = 0.005,
                           verbose: bool = False):

        #---------------------------
        # Step 5: Identify Z_mix.
        #---------------------------

        # Consider all candidates that have not been labeled.
        for candidate in self.z_prime:

            # Get conditioning set.
            conditioning_set = self.z_prime.copy()
            conditioning_set.remove(candidate)
            if len(conditioning_set) == 0:
                conditioning_set = None

            # Identify instrumental variables and children of the exposure.
            is_parent_of_x = self.test_z5_z7(conditioning_set = conditioning_set,
                                             exposure = exposure,
                                             outcome = outcome,
                                             candidate = candidate,
                                             alpha = alpha,
                                             verbose = verbose)
            if is_parent_of_x:
                self.pred_label_dict[candidate] = "Z1 or Z2 or Z3 or Z5"
                #print(candidate, "is d-sep from Y by", conditioning_set)
                self.z_mix.append(candidate)

        # Remove Z1, Z2, Z3, Z5 from remaining candidates.
        self.z_prime = [x for x in self.z_prime if x not in self.z_mix]


    def partition_z_step_6(self,
                           exposure: str,
                           instrument: str = None,
                           use_random_z5: bool = False,
                           alpha: float = 0.005,
                           verbose: bool = False):

        #--------------------------------------
        # Step 6: Resolve Z_mix and Z_post.
        #--------------------------------------

        # Init memoization structure.
        self.ind_dictionary = dict()

        # Evaluate all variables that are still unlabeled.
        if len(self.z_prime) > 0:
            z_mix = self.z_mix.copy() + self.z5_z7.copy()
            for candidate in self.z_prime:
                if len(z_mix) == 0:
                    if verbose:
                        print("!! WARNING: Condition 3 unsatisfied (no potential Z1_Z5 discovered). Z1/Z3 not identifiable.")
                else:
                    identified_confounder = False
                    ind_results = dict()
                    for var in z_mix:
    
                        # Test marginal independence between z_mix and candidate.
                        m_ind = self.ind_test(var_0 = var,
                                              var_1 = candidate,
                                              conditioning_set = None)
                        ind_results[var] = m_ind > alpha
    
                        # Test conditional independence given the exposure.
                        c_ind = self.ind_test(var_0 = var,
                                              var_1 = candidate,
                                              conditioning_set = [exposure])
    
                        # If a v-structure is identified, the candidate is in Z1
                        # and the z_mix is in Z1 or Z5.
                        if m_ind > alpha and c_ind <= alpha:
                            identified_confounder = True
                            if verbose:
                                print("{} -> X <- {}: {} is a confounder (STEP 6.1).".format(var, candidate, candidate))
                            # Add instrument to Z1 or Z5.
                            if var not in self.z1_z5:
                                self.z1_z5.append(var)
                                self.pred_label_dict[var] = "Z1 or Z5"
                                if var in self.z_mix:
                                    self.z_mix.remove(var)
                            # Add confounder to Z1 list.
                            if candidate not in self.z1:
                                self.z1.append(candidate)
                                self.pred_label_dict[candidate] = "Z1"
                                self.pred_bool_dict[candidate] = True
    
                    # Store independence results for each variable pair.
                    # 1 = independent, 0 = dependent.
                    self.ind_dictionary[candidate] = ind_results
    
                    if not identified_confounder:
                        # If the candidate is not in Z1, then it is in Z_post.
                        self.z_post.append(candidate)
                        self.pred_label_dict[candidate] = "Z2 or Z3 or Z6" # Previously, just "Z3"
    
            # Now all Z7 are differentiated from Z5.
            if len(self.z1_z5) > 0:
                self.z7 = [x for x in self.z5_z7 if x not in self.z1_z5]
                for z7 in self.z7:
                    self.pred_label_dict[z7] = "Z7"

            # Pass through all remaining z_mix one more time to remove remaining Z1.
            # By now, all Z5 should have been placed in self.z1_z5.
            z_mix_set = self.z_mix.copy()
            for z_mix in z_mix_set:
                for z1_z5 in self.z1_z5:
                #for z1_z5 in self.z1_z5 + self.z1:
                    # Test marginal independence between z_mix and z1_z5.
                    m_ind = self.ind_test(var_0 = z_mix,
                                          var_1 = z1_z5,
                                          conditioning_set = None)
                    # Test conditional independence given the exposure.
                    #c_ind = self.ind_test(var_0 = z_mix,
                    #                      var_1 = z1_z5,
                    #                      conditioning_set = [exposure])
                    # If a v-structure is identified, then z_mix is in Z1.
                    #if m_ind > alpha and c_ind <= alpha:
                    if m_ind > alpha:
                        if verbose:
                            print("{} -> X <- {}: {} is a confounder (STEP 6.2).".format(z1_z5, z_mix, z_mix))
                        if z_mix not in self.z1:
                            self.z1.append(z_mix)
                            self.pred_label_dict[z_mix] = "Z1"
                            self.pred_bool_dict[z_mix] = True
                            self.z_mix.remove(z_mix)
                        break
            
            # Now all remaining z_mix are in z_post.
            self.z_post = self.z_mix + self.z_post
            for z_post in self.z_post:
                self.pred_label_dict[z_post] = "Z2 or Z3 or Z6"
            

    def partition_z_step_7(self,
                           exposure: str = "X",
                           outcome: str = "Y",
                           alpha: float = 0.005,
                           verbose: bool = False):

        #---------------------------------------
        # Step 7: Resolve Z1 and Z5.
        #---------------------------------------

        if len(self.z1_z5) == 0 or len(self.z1) == 0:
            if verbose:
                print("!! WARNING: Cannot perform Step 7: len(self.z1_z5) == 0 or len(self.z1) == 0. Consequently, a valid adjustment set could not be identified.")
            return
        else:
            # Test whether candidate is marginally dependent on a known confounder.
            # Previously known confounders are those that are directly adjacent to Y.
            # Confounders left to be discovered are those with indirect active paths to Y.
            # No Z5 will ever be dependent on a confounder that is directly adjacent to Y.
            previously_known_confounders = self.z1.copy()
            for candidate in self.z1_z5:
                for confounder in previously_known_confounders:
                    ind_results = self.ind_dictionary.get(confounder)
                    if ind_results is None:
                        continue
                    else:
                        ind = ind_results.get(candidate)
                        if not ind:
                            if verbose:
                                print("{} vs {}: {} is in Z1, remove from Z5.".format(candidate,
                                                                                      confounder,
                                                                                      candidate))
                            self.z1.append(candidate)
                            self.pred_label_dict[candidate] = "Z1"
                            self.pred_bool_dict[candidate] = True
                            break

            # All Z5 are those variables that were not detected as Z1.
            self.z5 = [x for x in self.z1_z5 if x not in self.z1]
            for z5 in self.z5:
                self.pred_label_dict[z5] = "Z5"


    def test_z5_criterion(self,
                          exposure: str = "X",
                          outcome: str = "Y",
                          candidate: str = "Z",
                          alpha: float = 0.005,
                          verbose: bool = False):

        # Test for Z5 that is d-separable from Y given Z and Z1.
        for z5 in self.z5:
            ind = self.ind_test(var_0 = z5,
                                var_1 = outcome,
                                conditioning_set = self.z1 + [exposure])
            if ind > alpha:
                self.vas_exists = True
                if verbose:
                    print(z5, "is d-separable from Y given Z1.")
                    print("** A valid adjustment set exists in Z.")
                return
        if not self.vas_exists and verbose:
            print("!! WARNING: A valid adjustment set does not exist in Z, or it is unidentifiable (e.g., due to assumption violations).")
            return

    def test_z8(self,
                exposure: str = "X",
                outcome: str = "Y",
                candidate: str = "Z",
                alpha: float = 0.005,
                verbose: bool = False) -> bool:

        # Test marginal independence of X and Z.
        if candidate in self.x_ind_z_dict:
            x_ind_z = self.x_ind_z_dict.get(candidate)
        else:
            x_ind_z = self.ind_test(var_0 = exposure,
                                    var_1 = candidate,
                                    conditioning_set = None)
            self.x_ind_z_dict[candidate] = x_ind_z

        # Test marginal independence of Y and Z.
        if candidate in self.y_ind_z_dict:
            y_ind_z = self.y_ind_z_dict.get(candidate)
        else:
            y_ind_z = self.ind_test(var_0 = outcome,
                                    var_1 = candidate,
                                    conditioning_set = None)
            self.y_ind_z_dict[candidate] = y_ind_z

        # Test for Z8.
        #if x_ind_z and y_ind_z:
        if x_ind_z > alpha and y_ind_z > alpha:
            return True
        return False


    def test_z4(self,
                exposure: str = "X",
                outcome: str = "Y",
                candidate: str = "Z",
                alpha: float = 0.005,
                verbose: bool = False) -> bool:

        # Test marginal independence of X and Z.
        if candidate in self.x_ind_z_dict:
            x_ind_z = self.x_ind_z_dict.get(candidate)
        else:
            x_ind_z = self.ind_test(var_0 = exposure,
                                    var_1 = candidate,
                                    conditioning_set = None)
            self.x_ind_z_dict[candidate] = x_ind_z

        # Test conditional independence of X and Z given Y.
        x_ind_z_given_y = self.ind_test(var_0 = exposure,
                                        var_1 = candidate,
                                        conditioning_set = [outcome])

        # Test for Z4, which induces a v-structure X -> Y <- Z4.
        #if x_ind_z and not x_ind_z_given_y:
        if x_ind_z > alpha and x_ind_z_given_y <= alpha:
            return True
        return False


    def test_z5_z7(self,
                   conditioning_set: list = None,
                   exposure: str = "X",
                   outcome: str = "Y",
                   candidate: str = "Z",
                   alpha: float = 0.005,
                   verbose: bool = False) -> bool:

        # Update conditioning set.
        if conditioning_set is not None:
            # Add exposure to conditioning set.
            if exposure not in conditioning_set:
                conditioning_set.append(exposure)
        else:
            conditioning_set = list(self.data.columns)
            conditioning_set.remove(outcome)
            conditioning_set.remove(candidate)

        # Test marginal independence of Y and Z.
        if candidate in self.y_ind_z_dict:
            y_ind_z = self.y_ind_z_dict.get(candidate)
        else:
            y_ind_z = self.ind_test(var_0 = outcome,
                                    var_1 = candidate,
                                    conditioning_set = None)
            self.y_ind_z_dict[candidate] = y_ind_z


        # Test conditional independence of Y and Z given conditioning set, which contains at least X.
        # Addresses Case 5 or Case 7.
        y_ind_z_given_x = self.ind_test(var_0 = outcome,
                                        var_1 = candidate,
                                        conditioning_set = conditioning_set)

        # Marginally dependent and conditionally independent.
        #if not y_ind_z and y_ind_z_given_x:
        if y_ind_z <= alpha and y_ind_z_given_x > alpha:
            return True
        return False


    def test_zpost(self,
                   exposure: str = "X",
                   outcome: str = "Y",
                   candidate: str = "Z",
                   causes_outcome: list = None,
                   alpha: float = 0.005,
                   verbose: bool = False) -> bool:

        '''
        This method also returns True if the candidate is a child of the outcome.
        '''

        if causes_outcome is None:
            raise ValueError("`causes_outcome` cannot be None.")
        if isinstance(causes_outcome, str):
            causes_outcome = [causes_outcome]
        if len(causes_outcome) == 0:
            return False

        # Null hypothesis = independent. p_value <= alpha indicates dependence.
        for var in causes_outcome:
            m_ind = self.ind_test(var_0 = var,
                                  var_1 = candidate,
                                  conditioning_set = None)
            m_ind_given_exposure_outcome = self.ind_test(var_0 = var,
                                                         var_1 = candidate,
                                                         conditioning_set = [exposure, outcome])

            # Eliminate colliders (Z2) and children of outcome (Z6) by testing for chain structure.
            if m_ind <= alpha or m_ind_given_exposure_outcome > alpha:
                return True

        return False


    def test_v_structure(self,
                         collider: str,
                         causes: list,
                         discrete: bool = False,
                         alpha: float = 0.005,
                         verbose: bool = False) -> bool:

        # Test marginal independence.
        m_ind = self.ind_test(var_0 = causes[0],
                              var_1 = causes[1],
                              conditioning_set = None)

        # Test conditional independence.
        c_ind = self.ind_test(var_0 = causes[0],
                              var_1 = causes[1],
                              conditioning_set = [collider])

        # Null hypothesis = independent. p_value <= alpha indicates dependence.
        # if m_ind and not c_ind:
        if m_ind > alpha and c_ind <= alpha:
            return True
        return False


    def score(self,
              y_true,
              y_pred,
              verbose: bool = True,
              plot_confusion: bool = True) -> list:

        '''
        Compute performance metrics.
        '''

        # Compute performance metrics.
        confusion = confusion_matrix(y_true, y_pred)
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, zero_division = 0)
        precision = precision_score(y_true, y_pred, zero_division = 0)
        recall = recall_score(y_true, y_pred, zero_division = 0)
        try:
            roc = roc_auc_score(y_true, y_pred)
        except:
            roc = -1.0

        if verbose:
            print("\n---------------------------------------------")
            print("tn, fp, fn, tp =", confusion.ravel())
            print("F1             =", f1)
            print("Accuracy       =", acc)
            print("Precision      =", precision)
            print("Recall         =", recall)
            print("ROC AUC        =", roc)
            print("---------------------------------------------\n")

        # Plot confusion matrix as heatmap, if specified.
        if plot_confusion and (len(confusion) > 0):
            plt.rcParams["figure.figsize"] = (5, 4)
            ax = plt.subplot()
            # (annot = True) to annotate cells.
            # (ftm = "g") to disable scientific notation.
            sns.heatmap(confusion, annot = True, fmt = "g", ax = ax, cmap = "crest");
            # Labels, title, and ticks.
            ax.set_xlabel("\nPredicted labels");
            ax.set_ylabel("True labels\n");
            ax.set_title("Confusion Matrix\n");
            ax.xaxis.set_ticklabels(["-1 (negative)", "1 (positive)"]);
            ax.yaxis.set_ticklabels(["-1 (negative)", "1 (positive)"]);
            plt.show()
            plt.close()

        return [acc, f1, precision, recall, roc]
