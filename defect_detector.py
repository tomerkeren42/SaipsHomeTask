import numpy as np

from discrete_anomaly_find import AnomalyFindDiscrete
from image_align import ImageAligner
from typing import Dict, List
from parse import args
import pathlib
import cv2
import os

class DefectsDetector:
	def __init__(self):
		self._images: Dict[int, List[str]] = dict()
		self._image_aligner = ImageAligner()
		self._discrete_anomaly_finder = AnomalyFindDiscrete()
		self._show_results = args().show_results
		self._de_noise_window_size = 10
		self._de_noise_h = 7

	def load(self):
		"""

		:return:
		"""

		path: str = pathlib.PureWindowsPath(args().images_source).as_posix()

		# noinspection PyUnresolvedReferences
		images: Set[str] = {f.path for f in os.scandir(path) if (f.is_file() and ".txt" not in f.name)}

		for image in images:
			case_number: int = image.split("case")[1].split("_")[0]

			if case_number not in self._images:
				self._images.update({case_number: [image, ""]})
			else:
				self._images[case_number][1] = image

	def _preprocess(self, img):
		"""
		:param img:
		:return:
		"""
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		img = cv2.fastNlMeansDenoising(src=img,
		                               h=self._de_noise_h,
		                               templateWindowSize=self._de_noise_window_size)
		return img

	def detect(self):
		"""

		:return:
		"""
		for images in self._images.values():
			inspected_path, reference_path = images if "inspect" in images[0] else reversed(images)

			inspected = self._preprocess(cv2.imread(inspected_path))
			reference = self._preprocess(cv2.imread(reference_path))

			inspected_aligned, reference_aligned = self._image_aligner.align(inspected, reference)

			mask = self._discrete_anomaly_finder.get_mask(inspected_aligned, reference_aligned)
			if self._show_results:
				cv2.imshow("mask", mask)
				cv2.imshow('inspected', inspected_aligned)
				cv2.imshow('reference', reference_aligned)
				if cv2.waitKey(0) == ord('q'):
					cv2.destroyAllWindows()
					exit(-1)
