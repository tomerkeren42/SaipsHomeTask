from typing import List, Tuple

import cv2
import numpy as np


class ImageAligner:
	"""

	"""
	def __init__(self):
		self._sift = cv2.SIFT_create(nfeatures=100, nOctaveLayers=5, contrastThreshold=0.05, edgeThreshold=10,
		                                     sigma=1.6)
		self._matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
		self._lower_slope_bound = 0.15
		self._upper_slope_bound = 3

	def _detect_and_compute_sift(self, image):
		"""

		:param image:
		:return:
		"""
		kp, desc = self._sift.detectAndCompute(image, None)
		return kp, desc

	def _find_good_matches_knn(self, desc_a, desc_b, ratio_thresh = 0.5) -> List:
		"""

		:param desc_a:
		:param desc_b:
		:param ratio_thresh:
		:return:
		"""
		knn_matches = self._matcher.knnMatch(desc_a, desc_b, 2)
		matches = [m for (m, n) in knn_matches if m.distance < ratio_thresh * n.distance]

		return matches

	def _dilate_points_by_median_slope(self, all_points: List) -> List:
		"""

		:param all_points:
		:return:
		"""
		median_slope = np.median([(abs(points[0][1] - points[1][1])) / (abs(points[0][0] - points[1][0])) for points in all_points])

		good_slope_points: List = []
		for points in all_points:
			slope = (abs(points[0][1] - points[1][1])) / (abs(points[0][0] - points[1][0]))
			if self._lower_slope_bound * median_slope < slope < self._upper_slope_bound * median_slope:
				good_slope_points += [[points[0], points[1]]]

		return good_slope_points

	@staticmethod
	def _is_point_already_calculated(p_set, point_a, point_b) -> bool:
		"""

		:param p_set:
		:param point_a:
		:param point_b:
		:return:
		"""
		if point_a[0] in p_set and point_b[0] in p_set and point_b[1] in p_set and point_a[1] in p_set:
			return True

		p_set.add(point_a[0])
		p_set.add(point_a[1])
		p_set.add(point_b[0])
		p_set.add(point_b[1])
		return False

	def _get_median_delta_translates(self, slope_points: List) -> Tuple[float, float]:
		"""

		:param slope_points:
		:return:
		"""
		points_set = set()
		all_delta_x: List[float] = [(point_b[0] - point_a[0]) for (point_b, point_a) in slope_points if not self._is_point_already_calculated(points_set, point_a, point_b)]

		points_set = set()
		all_delta_y: List[float] = [(point_b[1] - point_a[1]) for (point_b, point_a) in slope_points if not self._is_point_already_calculated(points_set, point_a, point_b)]

		return np.median(all_delta_x), np.median(all_delta_y)

	def get_delta(self, img_a: np.array, img_b: np.array) -> Tuple[float, float]:
		"""

		:param img_a:
		:param img_b:
		:return:
		"""
		# Sift section
		kp1, descriptors1 = self._detect_and_compute_sift(img_a)
		kp2, descriptors2 = self._detect_and_compute_sift(img_b)

		# KNN section
		knn_matches = self._find_good_matches_knn(descriptors1, descriptors2)

		all_matching_points: List = [[[int(np.floor(kp1[match.queryIdx].pt[0])),
		                               int(np.floor(kp1[match.queryIdx].pt[1]))],
		                              [int(np.floor(kp2[match.trainIdx].pt[0])),
		                               int(np.floor(kp2[match.trainIdx].pt[1]))]] for match in knn_matches]

		# Slopes section
		slope_points = self._dilate_points_by_median_slope(all_matching_points)

		median_delta_x, median_delta_y = self._get_median_delta_translates(slope_points)

		# img_matches = np.empty((max(img_a.shape[0], img_b.shape[0]), img_a.shape[1] + img_b.shape[1], 3), dtype=np.uint8)
		# cv2.drawMatches(img_a, kp1, img_b, kp2, knn_matches, img_matches, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
		# cv2.imshow('Good Matches', img_matches)
		# cv2.waitKey(0)
		return median_delta_x, median_delta_y

	@staticmethod
	def align_frames(x: float, y: float, img_a: np.array, img_b: np.array) -> Tuple[np.array, np.array]:
		"""

		:param x:
		:param y:
		:param img_a:
		:param img_b:
		:return:
		"""
		x, y = int(x), int(y)

		if x > 0:
			img_b = img_b[:, x:]
			img_a = img_a[:, :-x]
		else:
			img_b = img_b[:, :x]
			img_a = img_a[:, -x:]

		if y > 0:
			img_b = img_b[y:, :]
			img_a = img_a[:-y, :]
		else:
			img_b = img_b[:y, :]
			img_a = img_a[-y:, :]

		return img_b, img_a
