from typing import List, Tuple
import cv2
import numpy as np

from parse import args


class ImageAligner:
	"""
	Class object to align couple of images by key points detection
	"""
	def __init__(self):
		self._sift = cv2.SIFT_create(nfeatures=100, nOctaveLayers=5, contrastThreshold=0.05, edgeThreshold=10, sigma=1.6)
		self._matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)

		self._lower_slope_bound: float = 0.15
		self._upper_slope_bound: float = 3.

		self._delta_x: float = 0.
		self._delta_y: float = 0.

	def _detect_and_compute_sift(self, image) -> Tuple:
		"""
		Use SIFT algorithm to get key points and their descriptions
		:param image: Image to detect Key Points
		:return: Tuple[Key points, Key points description]
		"""
		kp, desc = self._sift.detectAndCompute(image, None)
		return kp, desc

	def _find_good_matches_knn(self, desc_a, desc_b, ratio_thresh = 0.5) -> List:
		"""
		Match i`th description from image A with i`th description from image B. compare distance after KNN clustering.
		If distance stands out the ratio test (with parameter given from Low`s paper), keep it.
		:param desc_a: Key points description of image A
		:param desc_b: Key points description of image B
		:param ratio_thresh: Low`s ratio test threshold value
		:return: i`th descriptions that are close to each other in both images.
		"""
		knn_matches = self._matcher.knnMatch(desc_a, desc_b, 2)
		matches = [m for (m, n) in knn_matches if m.distance < ratio_thresh * n.distance]

		return matches

	def _dilate_points_by_median_slope(self, all_points: List[List[List[int]]]) -> List[List[List[int]]]:
		"""
		Another technique I created for cleaning noisy alignment issues.
		`slope` - the slope of the line when you connect same key point in both images.
		At general, and only in X-Y Axis shifting, those slopes of the real key points should have the same value.
		Therefore I look for the outlier slopes (of false key points) and remove it by another ratio test.
		:param all_points: All key points couples
		:return: List of all points couple (List) of points in images(List)
		"""
		median_slope = np.median([(abs(points[0][1] - points[1][1])) / (abs(points[0][0] - points[1][0])) for points in all_points])

		good_slope_points: List[List[List[int]]] = []
		for points in all_points:
			slope = (abs(points[0][1] - points[1][1])) / (abs(points[0][0] - points[1][0]))
			if self._lower_slope_bound * median_slope < slope < self._upper_slope_bound * median_slope:
				good_slope_points += [[points[0], points[1]]]

		return good_slope_points

	@staticmethod
	def _is_point_already_calculated(p_set, point_a, point_b) -> bool:
		"""
		Sometimes, same key point (or a very near one) is occurring twice in the same key points list, I don`t want to calculate it again -
		because using the median for calculating the constant shifting, double occurrences will damage my calculation.
		:param p_set: Set to hold all point already existed
		:param point_a: Point from image A
		:param point_b: Point from image B
		:return: Coule of points already existed decision
		"""
		if point_a[0] in p_set and point_b[0] in p_set and point_b[1] in p_set and point_a[1] in p_set:
			return True

		p_set.add(point_a[0])
		p_set.add(point_a[1])
		p_set.add(point_b[0])
		p_set.add(point_b[1])
		return False

	def _get_median_delta_translates(self, slope_points: List[List[List[int]]]) -> Tuple[float, float]:
		"""
		Calculate median shifting in both X and Y axis (get only couple of points once)
		:param slope_points: All couples of points
		:return: Tuple[median X axis shifting, median Y axis shifting]
		"""
		points_set = set()
		all_delta_x: List[float] = [(point_b[0] - point_a[0]) for (point_b, point_a) in slope_points if not self._is_point_already_calculated(points_set, point_a, point_b)]

		points_set = set()
		all_delta_y: List[float] = [(point_b[1] - point_a[1]) for (point_b, point_a) in slope_points if not self._is_point_already_calculated(points_set, point_a, point_b)]

		return np.median(all_delta_x), np.median(all_delta_y)

	def _get_delta(self, img_a: np.array, img_b: np.array) -> None:
		"""
		Main method for calculating the shifted distance in X-Y axis between image A and image B
		:param img_a: Image A
		:param img_b: Image B
		:return: None
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

		if bool(args().show_results):
			img_matches = np.empty((max(img_a.shape[0], img_b.shape[0]), img_a.shape[1] + img_b.shape[1], 3), dtype=np.uint8)
			cv2.drawMatches(img_a, kp1, img_b, kp2, knn_matches, img_matches, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
			cv2.imshow('Good Matches', img_matches)

		self._delta_x = median_delta_x
		self._delta_y = median_delta_y

	def _align_frames(self, img_a: np.array, img_b: np.array) -> Tuple[np.array, np.array]:
		"""
		Method for aligning images - slice off the contrary sides of the alignment.
		If image A is aligned to the left, then we can not compare its left side which is not shown on image B,
		neither compare image B`s right side which is not shown on image A. So we cut both sides. Same for Y axis.
		:param img_a: Image A
		:param img_b: Image B
		:return: Tuple [ Slices image A, Sliced image B]
		"""
		x, y = int(self._delta_x), int(self._delta_y)

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

	def align(self, img_a, img_b) -> Tuple[np.ndarray, np.ndarray]:
		"""
		Main function for aligning images A and B
		:param img_a: Image A
		:param img_b: Image B
		:return: Tuple[ Aligned image A, Aligned image B ]
		"""
		self._get_delta(img_b.copy(), img_a.copy())
		img_a_aligned, img_b_aligned = self._align_frames(img_a, img_b)
		return img_a_aligned, img_b_aligned