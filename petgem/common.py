#!/usr/bin/env python3
# Author:  Octavio Castillo Reyes
# Contact: octavio.castillo@bsc.es
''' Define common operations for **PETGEM**.
'''

# ---------------------------------------------------------------
# Load python modules
# ---------------------------------------------------------------
import time
import yaml
import sys
import os
import pprint
import numpy as np
from functools import wraps
from colorama import Fore
from singleton_decorator import singleton

# ---------------------------------------------------------------
# Load petgem modules (BSC)
# ---------------------------------------------------------------
from .parallel import MPIEnvironment

# ###############################################################
# ################     CLASSES DEFINITION      ##################
# ###############################################################

# ---------------------------------------------------------------
# Class Print definition
# ---------------------------------------------------------------
class Print(object):
    ''' This class provides methods for pretty print.

    :param object str: string to be printed.
    :return: None.
    :rtype: None.
    '''

    # Options for Gauss points computation (switch case)
    _options = {
        1: Fore.BLACK,
        2: Fore.BLUE,
        3: Fore.CYAN,
        4: Fore.GREEN,
        5: Fore.LIGHTBLACK_EX,
        6: Fore.LIGHTBLUE_EX,
        7: Fore.LIGHTCYAN_EX,
        8: Fore.LIGHTGREEN_EX,
        9: Fore.LIGHTMAGENTA_EX,
        10: Fore.LIGHTRED_EX,
        11: Fore.LIGHTWHITE_EX,
        12: Fore.LIGHTYELLOW_EX,
        13: Fore.MAGENTA,
        14: Fore.RED,
        15: Fore.WHITE,
        16: Fore.YELLOW
    }

    # Constructor
    def __init__(self, text, color_code=None):
        ''' Constructor
        '''
        self._log(text, color_code)

    # Logging method
    def _log(self, text, color_code=None):
        ''' This function configures and prints a text.

        :param str text: text to be printed.
        :param int color_code: text color code.
        :return: None.
        '''

        # Verify if color_code is None, then use black color
        if color_code is None:
            color_code = int(16)

        set_color = self._options[color_code]
        print(set_color + text)
        sys.stdout.flush()
        return

    @classmethod
    def header(self):
        ''' This functions prints the header.

        :param: None.
        :return: None.
        :rtype: None.
        '''

        # Specific color code for printing the header
        color_code = 5

        if( MPIEnvironment().rank == 0 ):
            self._log(self, '%'*75, color_code)
            self._log(self, '%%%' + ' '*69 + '%%%', color_code)
            self._log(self, '%%%'+  'PETGEM'.center(69) + '%%%', color_code)
            self._log(self, '%%%'+  'Parallel Edge-based Tool for Electromagnetic Modelling'.center(69) + '%%%', color_code)
            self._log(self, '%%%' + ' '*69 + '%%%', color_code)
            self._log(self, '%'*75, color_code)
            self._log(self, '%%%' + ' '*69 + '%%%', color_code)
            self._log(self, '%%%   (c) Octavio Castillo-Reyes' +
                       ' '*40 + '%%%', color_code)
            self._log(self, '%%%   Barcelona Supercomputing Center, 2020' +
                       ' '*29 + '%%%', color_code)
            self._log(self, '%%%' + ' '*69 + '%%%', color_code)
            self._log(self, '%'*75, color_code)

        return


    @classmethod
    def master(self, text, color_code=None):
        ''' If the caller is the master process, this method prints a message.

        :param: None.
        :return: None.
        :rtype: None.
        '''
        if( MPIEnvironment().rank == 0 ):
            self._log(self, text, color_code)

        return

# ---------------------------------------------------------------
# Class InputParameters definition
# ---------------------------------------------------------------
class InputParameters(object):
    ''' This class provides a methods to import a yaml parameter file.

    :param dict object: user params yaml file.
    :return: user parameters as object view.
    :rtype: object.
    '''

    def __init__(self, params, parEnv):
        ''' Class constructor.

        :param str params: yaml parameters file.
        :param object parEnv: parallel environment object.
        :return: InputParameters object.
        :rtype: object
        '''

        # ---------------------------------------------------------------
        # Read the input parameters file
        # ---------------------------------------------------------------
        with open(params, 'r') as f:
            # The FullLoader parameter handles the conversion from YAML
            # scalar values to Python the dictionary format
            inputs = yaml.load(f, Loader=yaml.FullLoader)

        # Store the datasets as a dict  object
        self.model = Dictionary2Object(inputs['model'])
        self.model.src_position = np.asarray(self.model.src_position, dtype=np.float)
        self.run = Dictionary2Object(inputs['run'])
        self.output = Dictionary2Object(inputs['output'])

        # ---------------------------------------------------------------
        # Validations
        # ---------------------------------------------------------------
        if self.model.basis_order < 1 or self.model.basis_order > 6:
            Print.master('     Vector finite element basis order not supported')
            exit(-1)

        #input(type(self.run.cuda))

        if self.run.cuda == False:
            self.run.cuda = False
        elif self.run.cuda == True:
            self.run.cuda = True
        else:
            Print.master('     Cuda option not supported')
            exit(-1)

        # ---------------------------------------------------------------
        # Create output directory
        # ---------------------------------------------------------------
        if(parEnv.rank == 0):
            if not os.path.exists(self.output.directory):
                 os.mkdir(self.output.directory)

            # Create temporal directory
            if not os.path.exists(self.output.directory_scratch):
                os.mkdir(self.output.directory_scratch)

        return



# ---------------------------------------------------------------
# Class dictionary to object definition
# ---------------------------------------------------------------
class Dictionary2Object(object):
    '''
    Turns a dictionary into a class
    '''
    #----------------------------------------------------------------------
    def __init__(self, dictionary):
        '''Constructor'''
        for key in dictionary:
            setattr(self, key, dictionary[key])


# ---------------------------------------------------------------
# Class Timer definition
# ---------------------------------------------------------------
class Timer():

    # Attributes
    elapsed = None

    # Initializing
    def __init__(self, elapsed = 0):
        self._start = 0
        self.elapsed = elapsed
        pass

    # Start timer
    def start(self):
        self._start = time.time()

    # Stop timer
    def stop(self):
        if(self._start > 0):
            self.elapsed += time.time() - self._start
            self._start = 0

    # Reset timer
    def reset(self):
        self.elapsed = 0

# ---------------------------------------------------------------
# Class Timers definition
# ---------------------------------------------------------------
@singleton
class Timers():

    # Initializing
    def __init__(self, opath = None):
        # Create private dict for storing the timers
        self._elems = {}

        # Set the output path for writting the times report
        self._opath = opath

        # Obtain the process rank once
        self.rank = MPIEnvironment().rank

        # Prepare the output file
        self._out = None
        self._header = False
        if  MPIEnvironment().rank == 0 and self._opath:
            # Save times for parallel scalability test
            out_filename = self._opath + '/times.txt'
            if os.path.isfile(out_filename):
                self._out = open(out_filename, 'a')
            else:
                self._out = open(out_filename, 'w')
                self._header = True

        pass

    # Get an specific timer
    def __getitem__(self, key):
        return self._elems.setdefault(key, Timer())

    # Set a specific
    def __setitem__(self, key, value):
        self._elems[key] = Timer(value)

    # Write report
    def _write(self):
        # Obtain the MPI environment
        parEnv = MPIEnvironment()

        # Create empty lists
        block = []
        time = []

        for key, value in self._elems.items():
            total = parEnv.comm.allreduce(value.elapsed, op=parEnv.MPI.SUM) / parEnv.num_proc
            block.append(key)
            time.append(total)

        if( parEnv.rank == 0):
            if( self._header):
                self._out.write("  |  # proc  |" + "".join('%20s  | ' % t for t in block) + "\n")
            self._out.write("  |  %-6s  |" % parEnv.num_proc + "".join('%20.3f  | ' % t for t in time) + "\n")
            self._out.close()


    # Deleting
    def __del__(self):
        rank = MPIEnvironment().rank
        # For each stored element
        for key, value in self._elems.items():
            msg = ' [Process {:2d}] | {:40s} | {:4.3f} seconds'.format(rank, key, value.elapsed)
            Print(msg, color_code=4)

        self._write()

# ###############################################################
# ################     DECORATORS DEFINITION    #################
# ###############################################################

# ---------------------------------------------------------------
# Decorators for code instrumentation
# ---------------------------------------------------------------

def measure_time(f = None, group = None, split = False):
    ''' This function implement a decorator for obtaining the decorated method
        execution time.

    :param function f: the decorated function
    :param str group: the group name
    :param bool split: decides if all blocks in a group contribute to the same timer
    :return: a function wrap
    :rtype: function
    '''
    def inner_function(function):
        @wraps(function)
        def wrapper(*args, **kwargs):

            # No group was specified
            if not group:
                name = function.__name__
            # Each method is measured separately
            elif split:
                name = group + "." + function.__name__
            # Each method contribute to the group time measure
            else:
                name = group

            # Timer start
            Timers()[name].start()

            # Execute the decorated method
            ret = function(*args, **kwargs)

            # Timer stop
            Timers()[name].stop()

            return ret
        return wrapper

    # For the decorate a method without providing arguments
    if f:
        return inner_function(f)
    return inner_function

def measure_all_class_methods(Cls):
    ''' This function implement a decorator for obtaining execution times for
        each method implemented on the decorated class.

    :param class f: the decorated class
    :return: a class wrap
    :rtype: class
    '''

    class DecoratedClass(object):

        def __init__(self,*args,**kwargs):
            self.oInstance = Cls(*args,**kwargs)
            self.className = self.oInstance.__class__.__name__
        def __getattribute__(self,s):
            try:
                x = super(DecoratedClass, self).__getattribute__(s)
            except AttributeError:
                pass
            else:
                return x
            x = self.oInstance.__getattribute__(s)
            if type(x) == type(self.__init__): # it is an instance method
                return measure_time(x, group=self.className, split=True)
            else:
                return x
    return DecoratedClass

# ###############################################################
# ################     FUNCTIONS DEFINITION     #################
# ###############################################################

def unitary_test():
    ''' Unitary test for common.py script.
    '''
    # TODO

# ###############################################################
# ################             MAIN             #################
# ###############################################################
if __name__ == '__main__':
    # ---------------------------------------------------------------
    # Run unitary test
    # ---------------------------------------------------------------
    unitary_test()
