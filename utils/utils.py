def print_to_output(message=None, title=None, overoneline=True, verbose=True):
    """
    This function is used to simplify the printing task. You can
    print a message with a title, over one line or not for lists.
    """
    if verbose:
        # We print the title between '----'
        if title is not None:
            print('\n' + title.center(70, '-') + '\n')
        # We print the message
        if message is not None:
            # Lists have particular treatment
            if isinstance(message, list):
                # Either printed over one line
                if overoneline:
                    to_print = ''
                    for i in message:
                        to_print += '%s | ' % str(i)
                    print(to_print[:-3])
                # Or not
                else:
                    for i in message:
                        print(i)
            else:
                print(message)
