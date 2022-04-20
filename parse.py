import argparse

global_args = None


def parse():
    """
    Parses the command line arguments
    :return:
    """

    global global_args

    try:
        parser = argparse.ArgumentParser()
        parser.add_argument('-s', "--images-source", help="path to the images directory location", required=False,
                            default='C:/Users/tomer/OneDrive/שולחן העבודה/home exercise/images/all_images')
        parser.add_argument('--show-results', help="show results", required=False, action='store_true')
        global_args = parser.parse_args()

    except SystemExit as e:
        print("Argument parsing failed")
        raise e
    except argparse.ArgumentError as e:
        print(f"Argument parsing failed: {e}")
        raise e


def args():
    """
    The parsed command-line arguments.
    This function is accessible from any file
    :return:
    """
    global global_args
    return global_args
