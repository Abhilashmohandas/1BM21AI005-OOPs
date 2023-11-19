
def find_largest(numbers):
    if not numbers:
        return "List is empty"
    
    largest = numbers[0]
    for i in range(1, len(numbers)):
        if numbers[i] > largest:
            largest = numbers[i]
    return largest

def main():
    
    a = []
    n = int(input("Enter the number of elements: "))
    for i in range(n):
        element = int(input("Enter element"))
        a.append(element)
    
    result = find_largest(a)
    print("The largest number is:", result)

if __name__ == "__main__":
    main()
