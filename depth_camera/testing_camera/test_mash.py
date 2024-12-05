import numpy as np
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

x = 1280  # x 범위 (0 ~ 9)
y = 720  # y 범위 (0 ~ 19)
z = 60 # z 범위 (0 ~ 29)
n = 200  # 생성할 좌표 개수

# 각 차원의 최댓값을 리스트로 지정합니다.
high = [x, y, z]

# 랜덤 (x, y, z) 좌표 생성
points = np.random.randint(0, high, size=(n, 3))


# 2. Convex Hull 계산 (삼각형 메쉬 생성)
hull = ConvexHull(points)

# 3. 부피 계산 함수
def calculate_volume(hull):
    total_volume = 0
    for simplex in hull.simplices:  # 삼각형의 꼭짓점 인덱스
        vertices = hull.points[simplex]  # 삼각형 꼭짓점 (3개)
        a, b, c = vertices[0], vertices[1], vertices[2]
        volume = np.dot(a, np.cross(b, c)) / 6  # 삼각뿔 부피 공식
        total_volume += volume
    return abs(total_volume)  # 부피는 절댓값으로 반환

# 부피 계산
volume = calculate_volume(hull)

# 4. 3D 시각화
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# 포인트 클라우드 시각화
ax.scatter(points[:, 0], points[:, 1], points[:, 2], color='blue', label='Points')

# Convex Hull 삼각형 메쉬 시각화
for simplex in hull.simplices:
    vertices = hull.points[simplex]  # 삼각형 꼭짓점 (3개)
    tri = Poly3DCollection([vertices], alpha=0.5, edgecolor='k')
    tri.set_facecolor((0, 1, 0, 0.5))  # 녹색 반투명
    ax.add_collection3d(tri)

# 시각화 설정
ax.set_title(f'3D Convex Hull with Volume = {volume:.4f}')
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')
ax.legend()
plt.show()
