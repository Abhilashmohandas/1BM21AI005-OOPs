
import math

def main():
    n = int(input("Enter length of a side: "))
    area = find_area(n)
    print("Area of a hexagon is", area)

def find_area(n):
    area = (3/2) * math.sqrt(3) * math.pow(n, 2)
    return area

if __name__ == "__main__":
    main()
