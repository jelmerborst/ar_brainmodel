import numpy as np

import nengo
import nengo.spa.action_build
from nengo.spa.action_objects import Symbol, Source, Convolution
from nengo.spa.module import Module
from nengo.utils.compat import iteritems


class Cortical(Module):
    """A SPA module for forming connections between other modules.

    Parameters
    ----------
    actions : Actions
        The actions to implement.
    synapse : float, optional (Default: 0.01)
        The synaptic filter to use for the connections.
    neurons_cconv : int, optional (Default: 200)
        Number of neurons per circular convolution dimension.
    synapse_channel_dict : dict, optional (Default: None, uses synapse for all channels)
        Set synaptic filter per state or channel between two states. 
        E.g. dict(vision=.01, goal_memory=.05, memory=.1) means:
        1) synaptic filters *to* the object 'vision' are set to .01
        2) the synaptic filter from 'goal' to 'memory' are set to .05
        3) synaptic filters to 'memory' not originating in 'goal' are set to .1
        4) all other synaptic filters on channels are set to "synapse"

    label : str, optional (Default: None)
        A name for the ensemble. Used for debugging and visualization.
    seed : int, optional (Default: None)
        The seed used for random number generation.
    add_to_container : bool, optional (Default: None)
        Determines if this Network will be added to the current container.
        If None, will be true if currently within a Network.
    """
    def __init__(self, actions, synapse=0.01, neurons_cconv=200,synapse_channel_dict=None,
                 label=None, seed=None, add_to_container=None):
        super(Cortical, self).__init__(label, seed, add_to_container)
        self.actions = actions
        self.synapse = synapse
        self.neurons_cconv = neurons_cconv
        self.synapse_channel_dict = synapse_channel_dict
        self.spa = None


    def on_add(self, spa):
        Module.on_add(self, spa)
        self.spa = spa

        # parse the provided class and match it up with the spa model
        self.actions.process(spa)
        for action in self.actions.actions:
            if action.condition is not None:
                raise NotImplementedError("Cortical actions do not support "
                                          "conditional expressions: %s." %
                                          action.condition)
            for name, effects in iteritems(action.effect.effect):
                for effect in effects.expression.items:
                    if isinstance(effect, Symbol):
                        self.add_direct_effect(name, effect.symbol)
                    elif isinstance(effect, Source):
                        self.add_route_effect(name, effect.name,
                                              effect.transform.symbol,
                                              effect.inverted)
                    elif isinstance(effect, Convolution):
                        self.add_conv_effect(name, effect)
                    else:
                        raise NotImplementedError(
                            "Subexpression '%s' from action '%s' is not "
                            "supported by the cortex." % (effect, action))

    def add_direct_effect(self, target_name, value):
        """Make a fixed constant input to a module.

        Parameters
        ----------
        target_name : str
            The name of the module input to use.
        value : str
            A semantic pointer to be sent to the module input.
        """
        target_module = self.spa.get_module(target_name)
        sink, vocab = self.spa.get_module_input(target_name)
        transform = np.array([vocab.parse(value).v]).T

        with target_module:
            if not hasattr(target_module, 'bias'):
                target_module.bias = nengo.Node([1],
                                                label=target_name + " bias")
            nengo.Connection(target_module.bias, sink, transform=transform,
                             synapse=self.synapse)

    def add_route_effect(self, target_name, source_name, transform, inverted):
        """Connect a module output to a module input.

        Parameters
        ----------
        target_name : str
            The name of the module input to effect.
        source_name : str
            The name of the module output to read from. If this output uses
            a different vocabulary than the target, a linear transform
            will be applied to convert from one to the other.
        transform : str
            A semantic pointer to convolve with the source value before
            sending it into the target. This transform takes
            place in the source vocabulary.
        inverted : bool
            Whether to invert the transform.
        """
        target, target_vocab = self.spa.get_module_input(target_name)
        source, source_vocab = self.spa.get_module_output(source_name)

        t = source_vocab.parse(transform).get_convolution_matrix()
        if inverted:
            D = source_vocab.dimensions
            t = np.dot(t, np.eye(D)[-np.arange(D)])

        if target_vocab is not source_vocab:
            t = np.dot(source_vocab.transform_to(target_vocab), t)


        ### JPB - use different synapses for different connections
        #print('\nNew route effect')
        print(source_name + '=>' + target_name)

        # by default we use self.synapse channel
        syn = self.synapse      
        #if dict, check if specified connection exists
        if type(self.synapse_channel_dict) is dict:
        
            #check if source_target is in dict:
            source_target = source_name + '_' + target_name
            if source_target in self.synapse_channel_dict:
                  syn = self.synapse_channel_dict[source_target]
            #check if target is in set
            elif target_name in self.synapse_channel_dict:
                  syn = self.synapse_channel_dict[target_name]

        print('synapse used: %.3f s' % syn)
        ### JPB end
        
        with self.spa:
            nengo.Connection(source, target, transform=t, synapse=syn)

    def add_conv_effect(self, target_name, effect):
        """Convolve the output of two modules and send result to target.

        Parameters
        ----------
        target_name : str
            The name of the module input to affect
        effect : Convolution
            The details of the convolution to implement.
        """
        
         ###JPB - use different synapses for different connections
        # by default we use self.synapse channel
        
        syn = self.synapse_channel       
        #if dict, check if specified connection exists
        if type(self.synapse_channel_dict) is dict:
            #check if target is in set
            if target_name in self.synapse_channel_dict:
                  syn = self.synapse_channel_dict[target_name]
        ###
        
        nengo.spa.action_build.convolution(self, target_name, effect,
                                           self.neurons_cconv, syn)
