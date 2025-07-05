import numpy as np

n = int(input("Enter the order of the square matrix (e.g., 3 for 3x3): "))
# 1
matrix_values = []

print(f"Enter {n*n} values row-wise:")

for i in range(n):
    row = list(map(int, input(f"Enter row {i+1} values separated by space: ").split()))
    if len(row) != n:
        print("Please enter exactly", n, "values!")
        exit()
    matrix_values.append(row)

matrix = np.array(matrix_values)

print("\nMatrix is:")
print(matrix)

diag_sum = 0
for i in range(n):
    diag_sum += matrix[i][i]

print("\nSum of diagonal elements is:", diag_sum)