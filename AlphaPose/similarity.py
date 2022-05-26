import math 
import numpy as np 
from typing import List, Tuple


def weight_distance(pose1: List[Tuple], pose2: List[Tuple], conf1: List[Tuple]):
	sum1 = 1 / np.sum(conf1)

	sum2 = 0
	for i in range(len(pose1)):
		conf_ind = math.floor(i / 2)
		sum2 += conf1[conf_ind] * abs(pose1[i] - pose2[i])

	weighted_dist = sum1 * sum2

	return weighted_dist

def cosine_distance(pose1: List[Tuple], pose2: List[Tuple]):
	cossim = pose1.dot(np.transpose(pose2)) / (np.linalg.norm(pose1) * np.linalg.norm(pose2))

	cosdist = (1 - cossim)
	return cosdist