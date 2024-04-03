def thumb(n):
    if n < 2:
        return 1
    else:
        return n * thumb(n-1)

def is_prime(num):
    if num < 2:
        return False
    for i in range(2, int(num**0.5) + 1):
        if num % i == 0:
            return False
    return True

def factorial_prime(num):
    if is_prime(num):
        return thumb(num)
    else:
        return "Number is not prime"

# Test the function
num = 5
print(factorial_prime(num))  # Output: 120