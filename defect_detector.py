from image_align import ImageAligner
from typing import Dict, List
from parse import args
import numpy as np
import pathlib
import cv2
import os

class DefectsDetector:
	def __init__(self):
		self._images: Dict[int, List[str]] = dict()
		self._image_aligner = ImageAligner()
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

	@staticmethod
	def _get_buckets(img):
		"""

		:param img:
		:return:
		"""
		# TODO: Find better STD - Maybe before cleaning up
		std = np.std(img)

		n_buckets = np.floor(np.max(img) / std).astype(np.uint) * 2

		buckets_size = np.floor(np.max(img) / n_buckets).astype(np.uint)

		buckets = [buckets_size * bucket for bucket in range(n_buckets)]

		return buckets

	@staticmethod
	def _get_discrete_img_values(img, buckets):
		"""

		:param img:
		:param buckets:
		:return:
		"""
		discrete_inspection = np.zeros_like(img)

		for i in range(img.shape[0]):
			for j in range(img.shape[1]):
				for b in range(len(buckets)):
					if img[i][j] < buckets[b]:
						discrete_inspection[i][j] = b
						break
				else:
					discrete_inspection[i][j] = len(buckets)
		return discrete_inspection

	@staticmethod
	def _get_anomalies_mask(disc_img_a, disc_img_b, buckets_thresh=4):
		"""

		:param disc_img_a:
		:param disc_img_b:
		:param buckets_thresh:
		:return:
		"""

		diff = cv2.absdiff(disc_img_a, disc_img_b)
		mask = np.zeros_like(diff)

		masked = np.where(diff > buckets_thresh, 255, mask)
		return masked

	def detect(self):
		"""

		:return:
		"""
		for images in self._images.values():
			inspected, reference = images if "inspect" in images[0] else reversed(images)

			inspected = cv2.imread(inspected)
			reference = cv2.imread(reference)

			inspected = self._preprocess(inspected)
			reference = self._preprocess(reference)

			x, y = self._image_aligner.get_delta(inspected, reference)

			inspected, reference = self._image_aligner.align_frames(x, y, inspected, reference)

			inspected_buckets = self._get_buckets(inspected)
			reference_buckets = self._get_buckets(reference)

			discrete_inspection = self._get_discrete_img_values(inspected, inspected_buckets)
			discrete_reference = self._get_discrete_img_values(reference, reference_buckets)

			mask = self._get_anomalies_mask(discrete_inspection, discrete_reference)

			# mean pooling
			# window_size = 2
			#
			# inspected_blank = np.zeros((inspected.shape[0], inspected.shape[1]), dtype=np.uint8)
			# reference_blank = np.zeros((inspected.shape[0], inspected.shape[1]), dtype=np.uint8)
			#
			# for i in range(inspected.shape[0]):
			# 	if i % 2 == 0:
			# 		for j in range(inspected.shape[1]):
			# 			if j % 2 == 0:
			# 				inspected_window = inspected[i:i+window_size, j:j+window_size]
			# 				reference_window = reference[i:i + window_size, j:j + window_size]
			# 				inspected_blank[i][j] = np.mean(inspected_window)
			# 				reference_blank[i][j] = np.mean(reference_window)
			# cv2.imshow('w_i', inspected_blank)
			# cv2.imshow('w_r', reference_blank)
			if self._show_results:
				cv2.imshow("mask", mask)
				cv2.imshow('inspected', inspected)
				cv2.imshow('reference', reference)

				cv2.waitKey(0)
