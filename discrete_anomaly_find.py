from typing import List

import cv2
import numpy as np


class AnomalyFindDiscrete:
	"""
	Class for finding defects from aligned images, using discrete mapping of pixel values and comparing differences
	"""
	def __init__(self):
		self._number_of_buckets_multiplier: int = 4

	def _get_buckets(self, img: np.ndarray) -> List[int]:
		"""
		Calculating image STD, which lead us for calculating number of buckets we want to sort our pixels into.
		Get each bucket size and make the histogram themselves.
		:param img: Image to measure STD
		:return: Buckets histogram
		"""
		# TODO: Find better STD - Maybe before cleaning up
		std = np.std(img)

		n_buckets: int = np.floor(np.max(img) / std).astype(np.uint) * self._number_of_buckets_multiplier

		# TODO: Dynamic buckets size - most common pixel value should have smaller buckets,
		#  less common areas should have wider buckets, for better classification
		buckets_size: int = int(np.floor(np.max(img) / n_buckets).astype(np.uint))

		buckets: List[int] = [buckets_size * bucket for bucket in range(n_buckets)]

		return buckets

	@staticmethod
	def _get_bucket_maps(img: np.ndarray, buckets: List[int]) -> np.ndarray:
		"""
		Assign each pixel to a bucket
		:param img: Original image pixels
		:param buckets: buckets ordering
		:return: Classified image - each pixel assigned to the i`th bucket
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

	def _get_anomalies_mask_from_buckets(self, disc_img_a: np.ndarray, disc_img_b: np.ndarray) -> np.ndarray:
		"""
		Calculate distance range between buckets at the same pixel from each image.
		If distance over some threshold (number of buckets / 2), then mask it.
		:param disc_img_a: Bucket classified image A
		:param disc_img_b: Bucket classified image B
		:return: Mask of defects
		"""

		buckets_diff = cv2.absdiff(disc_img_a, disc_img_b)

		mask = np.zeros_like(buckets_diff)

		masked = np.where(buckets_diff > (len(self._img_a_buckets) / 2), 255, mask)

		return masked

	def get_mask(self, img_a: np.ndarray, img_b: np.ndarray) -> np.ndarray:
		"""
		Main function for finding the defects mask
		:param img_a: Image A input (aligned)
		:param img_b: Image B input (aligned)
		:return:
		"""
		copy_image_a = img_a.copy()
		copy_image_b = img_b.copy()

		self._img_a_buckets = self._get_buckets(copy_image_a)
		self._img_b_buckets = self._get_buckets(copy_image_b)

		discrete_a = self._get_bucket_maps(copy_image_a, self._img_a_buckets)
		discrete_b = self._get_bucket_maps(copy_image_b, self._img_b_buckets)

		mask = self._get_anomalies_mask_from_buckets(discrete_a, discrete_b)
		return mask
