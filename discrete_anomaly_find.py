from typing import List

import cv2
import numpy as np


class AnomalyFindDiscrete:
	def __init__(self):
		self._number_of_buckets_multiplier = 4

	def _get_buckets(self, img: np.ndarray) -> List[int]:
		"""

		:param img:
		:return:
		"""
		# TODO: Find better STD - Maybe before cleaning up
		std = np.std(img)

		n_buckets: int = np.floor(np.max(img) / std).astype(np.uint) * self._number_of_buckets_multiplier

		buckets_size: int = np.floor(np.max(img) / n_buckets).astype(np.uint)

		buckets = [buckets_size * bucket for bucket in range(n_buckets)]

		return buckets

	@staticmethod
	def _get_bucket_maps(img, buckets):
		"""

		:param img:
		:param buckets:
		:return:
		"""
		bucket_map = np.zeros_like(img)

		for i in range(img.shape[0]):
			for j in range(img.shape[1]):

				for b in range(len(buckets)):
					if img[i][j] < buckets[b]:
						bucket_map[i][j] = b
						break
				else:
					bucket_map[i][j] = len(buckets)

		return bucket_map

	def _get_anomalies_mask_from_buckets(self, disc_img_a, disc_img_b):
		"""

		:param disc_img_a:
		:param disc_img_b:
		:return:
		"""

		buckets_diff = cv2.absdiff(disc_img_a, disc_img_b)

		mask = np.zeros_like(buckets_diff)

		masked = np.where(buckets_diff > (len(self._img_a_buckets) / 2), 255, mask)

		return masked

	def get_mask(self, img_a, img_b):

		copy_image_a = img_a.copy()
		copy_image_b = img_b.copy()

		self._img_a_buckets = self._get_buckets(copy_image_a)
		self._img_b_buckets = self._get_buckets(copy_image_b)

		discrete_a = self._get_bucket_maps(copy_image_a, self._img_a_buckets)
		discrete_b = self._get_bucket_maps(copy_image_b, self._img_b_buckets)

		mask = self._get_anomalies_mask_from_buckets(discrete_a, discrete_b)
		return mask