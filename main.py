from defect_detector import DefectsDetector
from parse import parse

"""
Main program for detecting defect

For usage, see README.md
"""


if __name__ == '__main__':

    parse()

    detector = DefectsDetector()

    detector.load()

    detector.detect()

