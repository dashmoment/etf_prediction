import hparam
config = hparam.configuration()


def print_c(print_content, verbose = config.verbose_state):
    
    if verbose == True:
        print(print_content)