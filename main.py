from defect_detector import DefectsDetector
from parse import parse

if __name__ == '__main__':

    parse()

    detector = DefectsDetector()

    detector.load()

    detector.detect()

