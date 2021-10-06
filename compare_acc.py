import nengo
import nengo.spa as spa
import numpy as np

from nengo.spa.module import Module
from nengo.spa.compare import Compare

class CompareAccumulator(Compare):
    """A module for computing the dot product of two inputs, now including:
    a) cleanup memories before comparison
    b) accumulator to judge outcome. Accumulates negatively when inputs are dissimilar.

    Parameters - for accumulation & cleanup memories
    ----------
    vocab : 
        The vocabulary to use to interpret the vector. If None,
        the default vocabulary for the given dimensionality is used.
    
    status_scale = .8 #speed of accumulation
    status_feedback = .8 #feedback transformation on status
    status_feedback_synapse = .1 #feedback synapse on status
    threshold_input_detect = .5 #threshold for input to be counted
    pos_bias = 0, added weight for positive evidence 
    
    threshold_cleanup=.3, threshold cleanup memories
    wta_inhibit_scale_cleanup=3, wta inhibit scale of cleanup memories
    
    
    Parameters - for compare
    ----------

    neurons_per_multiply : int, optional (Default: 200)
        Number of neurons to use in each product computation.
    input_magnitude : float, optional (Default: 1.0)
        The expected magnitude of the vectors to be multiplied.
        This value is used to determine the radius of the ensembles
        computing the element-wise product.

    label : str, optional (Default: None)
        A name for the ensemble. Used for debugging and visualization.
    seed : int, optional (Default: None)
        The seed used for random number generation.
    add_to_container : bool, optional (Default: None)
        Determines if this Network will be added to the current container.
        If None, will be true if currently within a Network.
    """

    def __init__(self, vocab_compare, status_scale=.8, status_feedback=.6, status_feedback_synapse=.1, pos_bias=0, neg_bias=0, threshold_input_detect=.5, threshold_cleanup=.3, wta_inhibit_scale_cleanup=3,threshold=0, direct_compare = True, **kwargs):
        
        dimensions = vocab_compare.dimensions
        super(CompareAccumulator, self).__init__(dimensions, vocab=vocab_compare, **kwargs)
        
        with self:
        
            if direct_compare:
                for net in self.all_networks:
                    #print('compfor')
                    #print(net)
                    if net.label == 'Product':
                        #print(net)
                        for ens in net.all_ensembles:
                            ens.neuron_type = nengo.Direct()
                        
            #clean up memories for input, to enable clean comparison
            self.cleanup_inputA = spa.AssociativeMemory(input_vocab=vocab_compare, wta_output=True, threshold=threshold_cleanup, wta_inhibit_scale=wta_inhibit_scale_cleanup)
            self.cleanup_inputB = spa.AssociativeMemory(input_vocab=vocab_compare, wta_output=True, threshold=threshold_cleanup, wta_inhibit_scale=wta_inhibit_scale_cleanup)
            nengo.Connection(self.cleanup_inputA.output,self.inputA)
            nengo.Connection(self.cleanup_inputB.output,self.inputB)

            #output ensemble (instead of node)
            self.output_ens = nengo.Ensemble(n_neurons=1000,dimensions=1)
            nengo.Connection(self.output,self.output_ens, synapse=None)
                                   
            #compare status indicator
            self.compare_status = nengo.Ensemble(100,1)
            nengo.Connection(self.compare_status,self.compare_status, transform=status_feedback, synapse=status_feedback_synapse)
        
            #print('threshold')
            #print(threshold)

            def dec_func(x):
                grey = 0
                if x > threshold + grey:
                    return .5
                elif x < threshold - grey:
                    return -.5
                else:
                    return x - threshold
             
                      
            nengo.Connection(self.output_ens, self.compare_status, function=dec_func,transform = status_scale)


            #detect when input present on both inputs
            n_signal = min(50,dimensions) #number of random dimensions used to determine whether input is present
    
            #inputA
            self.switch_inA = nengo.Ensemble(n_neurons=n_signal*50,dimensions=n_signal,radius=1) #50 neurons per dimension
            nengo.Connection(self.inputA[0:n_signal], self.switch_inA, transform=10, synapse=None) #no synapse, as input signal directly 

            #inputB
            self.switch_inB = nengo.Ensemble(n_neurons=n_signal*50,dimensions=n_signal,radius=1) #50 neurons per dimension
            nengo.Connection(self.inputB[0:n_signal], self.switch_inB, transform=10, synapse=None) #no synapse, as input signal directly 


            self.switch_detect = nengo.Ensemble(n_neurons=500,dimensions=2,radius=1.4)
            nengo.Connection(self.switch_inA, self.switch_detect[0],function=lambda x: np.sqrt(np.sum(x*x)),eval_points=nengo.dists.Gaussian(0,.1))
            nengo.Connection(self.switch_inB, self.switch_detect[1],function=lambda x: np.sqrt(np.sum(x*x)),eval_points=nengo.dists.Gaussian(0,.1)) 
            
            
            nengo.Connection(self.switch_detect,self.output_ens.neurons, function=lambda x: 0 if x[0]*x[1] > threshold_input_detect else -1,transform=np.ones((1000,1)),synapse=.01)
            nengo.Connection(self.switch_detect,self.compare_status.neurons, function=lambda x: 0 if x[0]*x[1] > threshold_input_detect else -1,transform=np.ones((100,1)),synapse=.02)

   
            #outputs we can use on LHS
            self.outputs['status'] = (self.compare_status,1)
            
            #input for RHS
            self.inputs['cleanA'] = (self.cleanup_inputA.input, vocab_compare)
            self.inputs['cleanB'] = (self.cleanup_inputB.input, vocab_compare )            
            


