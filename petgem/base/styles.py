#!/usr/bin/env python3
# Author:  Octavio Castillo Reyes
# Contact: octavio.castillo@bsc.es
''' Define styles for **PETGEM** screen-output such as: string formats,
headers, and footers.
'''


def set_str_format(in_string, FORMAT=None):
    ''' Setup a string with a specific format

    :param str in_string: string to be formated.
    :param str format: format to be applied.
    :return: formated string.
    :rtype: str.

    .. note:: valid formats are: Warning, Error, OkGreen and OkBlue
    '''
    printOut = Terminal()
    out_string = None
    if FORMAT is None:
        out_string = in_string
    elif FORMAT == 'Warning':
        out_string = printOut.yellow(in_string)
    elif FORMAT == 'Error':
        out_string = printOut.red(in_string)
    elif FORMAT == 'OkGreen':
        out_string = printOut.green(in_string)
    elif FORMAT == 'OkBlue':
        out_string = printOut.blue(in_string)
    elif FORMAT == 'OkCyan':
        out_string = printOut.cyan(in_string)

    return out_string


def printPetgemHeader(rank):
    ''' Setup the **PETGEM** header to be printed in screen.

    '''

    if rank == 0:
        Header = ['*' * 75,
                  'PETGEM'.center(75),
                  'Parallel Edge-based Tool for'.center(75),
                  'Geophysical Electromagnetic Modelling'.center(75),
                  '*' * 75]
        for H in Header:
            PETSc.Sys.Print(H)

    return


def printPetgemFooter(rank):
    ''' Setup the **PETGEM** footer to be printed in screen.
    '''

    if rank == 0:
        Footer = ['*' * 75,
                  ('PETGEM execution on: ' + Gettime('%c')).center(75),
                  'Requests and contributions are welcome:'.center(75),
                  'octavio.castillo@bsc.es'.center(75),
                  '*' * 75]
        for F in Footer:
            PETSc.Sys.Print(F)

    return


def test_header(caller):
    ''' Print the header of a unitary test.

    :param str caller: name of caller (test owner).
    '''
    msg = 'This is a unitary test for ' + caller + ' script.'
    msg = set_str_format(msg, FORMAT='OkBlue')
    print(msg)


def test_footer(pass_test):
    ''' Print the footer of a unitary test.

    :param bool pass_test: boolean that express if a test is, or not, passed.
    '''
    if pass_test:
        msg = 'Passed test!!.'
        msg = set_str_format(msg, FORMAT='OkGreen')
        print(msg)
    else:
        msg = 'The test not passed. Check warnings'
        msg = set_str_format(msg, FORMAT='Error')
        print(msg)


def unitary_test():
    ''' Unitary test for styles.py script.
    '''
    MASTER = 0
    printPetgemHeader(MASTER)
    test_header('styles.py')
    print('Testing styles for screen output: HEADER and footer')
    print('PETGEM styles are defined as follows:')
    ok1_msg = ('function(): Blue color for a correct '
               'work-flow description.')
    ok2_msg = ('function(): Green color for a correct '
               'function or task description.')
    ok3_msg = ('function(): Cyan color for a correct '
               'iterative task.')
    warn_msg = 'function(): Yellow color for a warning description.'
    error_msg = 'function(): Red color for an error description.'
    msg = [set_str_format(ok1_msg, FORMAT='OkBlue'),
           set_str_format(ok2_msg, FORMAT='OkGreen'),
           set_str_format(ok3_msg, FORMAT='OkCyan'),
           set_str_format(warn_msg, FORMAT='Warning'),
           set_str_format(error_msg, FORMAT='Error')]

    for M in msg:
        print(M)

    pass_test = True
    test_footer(pass_test)
    printPetgemFooter(MASTER)

    return


if __name__ == '__main__':
    # Standard module import
    from time import strftime as Gettime
    from blessings import Terminal as Terminal
    from petsc4py import PETSc
    unitary_test()
else:
    # Standard module import
    from time import strftime as Gettime
    from blessings import Terminal as Terminal
    from petsc4py import PETSc
