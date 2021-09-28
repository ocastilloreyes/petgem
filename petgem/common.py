#!/usr/bin/env python3
# Author:  Octavio Castillo Reyes
# Contact: octavio.castillo@bsc.es
"""Define common operations for **PETGEM**."""

# ---------------------------------------------------------------
# Load python modules
# ---------------------------------------------------------------
import time
import yaml
import sys
import os
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
    """
    This class provides methods for pretty print.

    :param object str: string to be printed.
    :return: None.
    :rtype: None.

    """
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
        """Constructor."""
        self._log(text, color_code)

    # Logging method
    def _log(self, text, color_code=None):
        """
        Configure and prints a text.

        :param str text: text to be printed.
        :param int color_code: text color code.
        :return: None.

        """
        # Verify if color_code is None, then use black color
        if color_code is None:
            color_code = int(16)

        set_color = self._options[color_code]
        print(set_color + text)
        sys.stdout.flush()
        return

    @classmethod
    def header(self):
        """Print the header.

        :param: None.
        :return: None.
        :rtype: None.
        """
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
            self._log(self, '%%%   Barcelona Supercomputing Center (BSC-CNS), 2021' +
                       ' '*19 + '%%%', color_code)
            self._log(self, '%%%' + ' '*69 + '%%%', color_code)
            self._log(self, '%'*75, color_code)

        return

    @classmethod
    def master(self, text, color_code=None):
        """
        If the caller is the master process, this method prints a message.

        :param: None.
        :return: None.
        :rtype: None.

        """
        if( MPIEnvironment().rank == 0 ):
            self._log(self, text, color_code)

        return

# ---------------------------------------------------------------
# Class InputParameters definition
# ---------------------------------------------------------------
class InputParameters(object):
    """Method to import a yaml parameter file.

    :param dict object: user params yaml file.
    :return: user parameters as object view.
    :rtype: object.
    """

    def __init__(self, params, parEnv):
        """Class constructor.

        :param str params: yaml parameters file.
        :param object parEnv: parallel environment object.
        :return: InputParameters object.
        :rtype: object
        """
        # ---------------------------------------------------------------
        # Read the input parameters file
        # ---------------------------------------------------------------
        with open(params, 'r') as f:
            # The FullLoader parameter handles the conversion from YAML
            # scalar values to the Python dictionary format
            inputs = yaml.safe_load(f)

        # Get set of parameters
        self.model = inputs['model']
        self.run = inputs['run']
        self.output = inputs['output']

        # ---------------------------------------------------------------
        # Check modeling mode
        # ---------------------------------------------------------------
        if not ('mode' in self.model.keys()):
            Print.master('     Modeling mode not provided. Please, verify the parameter file consistency.')
            exit(-1)
        else:
            if not ((self.model.get('mode') == 'csem') or (self.model.get('mode') == 'mt')):
                Print.master('     Modeling mode not supported.')
                exit(-1)

        # ---------------------------------------------------------------
        # Check parameters consistency for csem mode
        # ---------------------------------------------------------------
        if (self.model.get('mode') == 'csem'):
            if not ('csem' in self.model.keys()):
                Print.master('     csem parameters not provided. Please, verify the parameter file consistency.')
                exit(-1)
            else:
                # Check consistency of csem params
                conductivity_from_file, num_polarizations = self.__verify_CSEM_params__(self.model)

        # ---------------------------------------------------------------
        # Check parameters consistency for mt mode
        # ---------------------------------------------------------------
        elif (self.model.get('mode') == 'mt'):
            if not ('mt' in self.model.keys()):
                Print.master('     mt parameters not provided. Please, verify the parameter file consistency.')
                exit(-1)
            else:
                # Check consistency of mt params
                conductivity_from_file, num_polarizations = self.__verify_MT_params__(self.model)

        # Update number of models, interpolation strategy and
        # polarization modes
        self.run.update({'conductivity_from_file': conductivity_from_file})
        self.run.update({'num_polarizations': num_polarizations})

        # ---------------------------------------------------------------
        # Check consistency of common parameters
        # ---------------------------------------------------------------
        # Mesh
        if not ('mesh' in self.model.keys()):
            Print.master('     mesh parameter not provided. Please, verify the parameter file consistency.')
            exit(-1)

        # Receivers
        if not ('receivers' in self.model.keys()):
            Print.master('     receivers parameter not provided. Please, verify the parameter file consistency.')
            exit(-1)

        # Basis order
        if not ('nord' in self.run.keys()):
            Print.master('     nord parameter not provided. Please, verify the parameter file consistency.')
            exit(-1)
        else:
            if ((self.run.get('nord') < 1) or (self.run.get('nord') > 6)):
                Print.master('     Vector finite element basis order not supported. Please, select a valid order (1,2,3,4,5,6).')
                exit(-1)

        # Cuda support
        if not ('cuda' in self.run.keys()):
            self.run.update({'cuda': False})
        else:
            if not ((self.run.get('cuda') is False) or (self.run.get('cuda') is True)):
                Print.master('     cuda option not supported. Please, select a valid order (True/False).')
                exit(-1)

        # Output
        if not ('vtk' in self.output.keys()):
            self.output.update({'vtk': False})

        if not ('directory' in self.output.keys()):
            Print.master('     output directory parameter not provided. Please, verify the parameter file consistency.')
            exit(-1)
        else:
            if(parEnv.rank == 0):
                if not os.path.exists(self.output.get('directory')):
                    os.mkdir(self.output.get('directory'))
        # If not scratch directory, use output directory
        if not ('directory_scratch' in self.output.keys()):
            self.output.update({'directory_scratch': self.output.get('directory')})
            self.output.update({'remove_scratch': False})
        else:
            if(parEnv.rank == 0):
                if not os.path.exists(self.output.get('directory_scratch')):
                    os.mkdir(self.output.get('directory_scratch'))
                self.output.update({'remove_scratch': True})

        return

    def __verify_CSEM_params__(self, data):
        """Verify consistency of CSEM parameters

        :param dict data: csem dictionary
        :return: input conductivity model from file or array.
        :rtype: bool
        """
        # Get csem parameters
        csem_params = data.get('csem')
        # One "polarization mode" per csem model
        num_polarizations = np.int(1)

        # Check consistency for csem modeling
        # Check sigma consistency
        if not ('sigma' in csem_params.keys()):
            Print.master('     csem parameters not provided. Please, verify the parameter file consistency.')
            exit(-1)
        else:
            # Get sigma parameters
            i_sigma = csem_params.get('sigma')
            # Conductivity file
            if ('file' in i_sigma.keys()):
                # No vectors conductivity
                if (('horizontal' in i_sigma.keys()) or ('vertical' in i_sigma.keys())):
                    Print.master('     sigma parameters invalid. Please, verify the parameter file consistency.')
                    exit(-1)
                else:
                    conductivity_from_file = True
            # Vector conductivity
            elif (('horizontal' in i_sigma.keys()) and ('vertical' in i_sigma.keys())):
                # No file conductivity
                if ('file' in i_sigma.keys()):
                    Print.master('     sigma parameters invalid. Please, verify the parameter file consistency.')
                    exit(-1)
                else:
                    conductivity_from_file = False
            else:
                Print.master('     sigma parameters invalid. Please, verify the parameter file consistency.')
                exit(-1)

            # Check source consistency
            if not ('source' in csem_params.keys()):
                Print.master('     source parameters not provided. Please, verify the parameter file consistency.')
                exit(-1)
            else:
                # Get source parameters
                i_source = csem_params.get('source')
                # Check number of source parameters
                if not (len(i_source) == 6):
                    Print.master('     number of source parameters is not consistent. Please, verify the parameter file consistency.')
                    exit(-1)
                else:
                    base_params = ['frequency', 'position', 'azimuth', 'dip', 'current', 'length']
                    for i in np.arange(6):
                        if not (base_params[i] in i_source.keys()):
                            m = '     ' + base_params[i] + ' parameter not provided. Please, verify the parameter file consistency.'
                            Print.master(m)
                            exit(-1)

        return conductivity_from_file, num_polarizations


    def __verify_MT_params__(self, data):
        """Verify consistency of MT parameters

        :param dict data: mt dictionary
        :return: input conductivity model from file or array.
        :rtype: bool
        """
        # Get mt parameters
        mt_params = data.get('mt')

        # Check consistency for all mt models
        # Get model parameters
        # Check sigma consistency
        if not ('sigma' in mt_params.keys()):
            Print.master('     mt parameters not provided. Please, verify the parameter file consistency.')
            exit(-1)
        else:
            # Get sigma parameters
            i_sigma = mt_params.get('sigma')
            # Conductivity file
            if ('file' in i_sigma.keys()):
                # No vectors conductivity
                if (('horizontal' in i_sigma.keys()) or ('vertical' in i_sigma.keys())):
                    Print.master('     sigma parameters invalid. Please, verify the parameter file consistency.')
                    exit(-1)
                else:
                    conductivity_from_file = True
            # Vector conductivity
            elif (('horizontal' in i_sigma.keys()) and ('vertical' in i_sigma.keys())):
                # No file conductivity
                if ('file' in i_sigma.keys()):
                    Print.master('     sigma parameters invalid. Please, verify the parameter file consistency.')
                    exit(-1)
                else:
                    conductivity_from_file = False
            else:
                Print.master('     sigma parameters invalid. Please, verify the parameter file consistency.')
                exit(-1)

        # Check frequency and polarization parameters
        base_params = ['frequency', 'polarization']
        for i in np.arange(2):
            if not (base_params[i] in mt_params.keys()):
                m = '     ' + base_params[i] + ' parameter not provided for model. Please, verify the parameter file consistency.'
                Print.master(m)
                exit(-1)

        # Determine number of polarization modes
        num_polarizations = len(mt_params.get('polarization'))

        return conductivity_from_file, num_polarizations


# ---------------------------------------------------------------
# Class Timer definition
# ---------------------------------------------------------------
class Timer():
    """
    Definition of timer class.

    """

    # Attributes
    elapsed = None

    # Initializing
    def __init__(self, elapsed = 0):
        """Constructor."""
        self._start = 0
        self.elapsed = elapsed

    # Start timer
    def start(self):
        """Start timer."""
        self._start = time.time()

    # Stop timer
    def stop(self):
        """Stop timer."""
        if(self._start > 0):
            self.elapsed += time.time() - self._start
            self._start = 0

    # Reset timer
    def reset(self):
        """Reset timer."""
        self.elapsed = 0

# ---------------------------------------------------------------
# Class Timers definition
# ---------------------------------------------------------------
@singleton
class Timers():
    """
    Defintion of timers class.

    """

    # Initializing
    def __init__(self, opath = None):
        """Constructor."""
        # Create private dict for storing the timers
        self._elems = {}

        # Set the output path for writting the times report
        #self._opath = opath

        # Obtain the process rank once
        self.rank = MPIEnvironment().rank

        # Prepare the output file
        self._out = None
        self._header = False
        #if  MPIEnvironment().rank == 0 and self._opath:
            # Save times for parallel scalability test
        #    out_filename = self._opath + '/times.txt'
        #    if os.path.isfile(out_filename):
        #        self._out = open(out_filename, 'a')
        #    else:
        #        self._out = open(out_filename, 'w')
        #        self._header = True

    # Get an specific timer
    def __getitem__(self, key):
        """Get item name for timer."""
        return self._elems.setdefault(key, Timer())

    # Set a specific
    def __setitem__(self, key, value):
        """Set item name for timer."""
        self._elems[key] = Timer(value)

    # Write report
    def _write(self):
        """Write timer."""
        # Obtain the MPI environment
        parEnv = MPIEnvironment()

        # Create empty lists
        block = []
        time = []

        for key, value in self._elems.items():
            total = parEnv.comm.allreduce(value.elapsed, op=parEnv.MPI.SUM) / parEnv.num_proc
            block.append(key)
            time.append(total)

        #if( parEnv.rank == 0):
        #    if( self._header):
        #        self._out.write("  |  # proc  |" + "".join('%20s  | ' % t for t in block) + "\n")
        #    self._out.write("  |  %-6s  |" % parEnv.num_proc + "".join('%20.3f  | ' % t for t in time) + "\n")
        #    self._out.close()

    # Deleting
    def __del__(self):
        """Delete timers."""
        #rank = MPIEnvironment().rank
        # For each stored element
        #for key, value in self._elems.items():
        #    msg = ' [Process {:2d}] | {:40s} | {:4.3f} seconds'.format(rank, key, value.elapsed)
        #    Print(msg, color_code=4)

        #self._write()

# ###############################################################
# ################     DECORATORS DEFINITION    #################
# ###############################################################

# ---------------------------------------------------------------
# Decorators for code instrumentation
# ---------------------------------------------------------------
def measure_time(f = None, group = None, split = False):
    """"Implement method to measure execution time.

    Args:
        f: the decorated function
        group: the group name
        split: decides if all blocks in a group contribute to the same timer
    Returns:
        a function wrap

    """
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
    """"Implement a decorator to measure execution time for each method.

    Args:
        f: the decorated function
    Returns:
        a function wrap

    """

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
    """Unitary test for common.py script."""
    # TODO


# ###############################################################
# ################             MAIN             #################
# ###############################################################
if __name__ == '__main__':
    # ---------------------------------------------------------------
    # Run unitary test
    # ---------------------------------------------------------------
    unitary_test()
