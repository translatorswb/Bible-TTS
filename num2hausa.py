def number_to_hausa(number):
    if not 0 <= number < 1000000:
        print("Number must be between 0 and 999999")
        return None
    
    if number == 0:
        return "sifili"

    # Basic numbers dictionary
    ones = {
        1: "daya", 2: "biyu", 3: "uku", 4: "hudu", 5: "biyar",
        6: "shida", 7: "bakwai", 8: "takwas", 9: "tara"
    }
    
    tens = {
        10: "goma", 20: "ashirin", 30: "talatin", 40: "arba'in",
        50: "hamsin", 60: "sittin", 70: "saba'in", 80: "tamanin",
        90: "casa'in"
    }
    
    def convert_hundreds(n):
        if n == 0:
            return ""
        elif n == 1:
            return "dari "
        else:
            return f"dari {ones[n]} "
    
    def convert_tens_and_ones(n):
        if n == 0:
            return ""
        elif n <= 9:
            return ones[n]
        elif n == 10:
            return "goma"
        elif n < 20:
            return f"goma sha {ones[n-10]}"
        elif n % 10 == 0:
            return tens[n]
        else:
            return f"{tens[n - (n % 10)]} da {ones[n % 10]}"

    def convert_thousands(n):
        if n == 0:
            return ""
        elif n == 1:
            return "dubu daya "
        elif n < 10:
            return f"dubu {ones[n]} "
        elif n == 10:
            return "dubu goma "
        elif n < 20:
            return f"dubu goma sha {ones[n-10]} "
        elif n < 100:
            tens_part = n - (n % 10)
            ones_part = n % 10
            if ones_part == 0:
                return f"dubu {tens[tens_part]} "
            else:
                return f"dubu {tens[tens_part]} da {ones[ones_part]} "
        else:
            # Added 'da' here
            return f"dubu {convert_hundreds(n//100)}da {convert_tens_and_ones(n%100)}"

    # Break down the number
    thousands = number // 1000
    hundreds = (number % 1000) // 100
    remainder = number % 100

    # Build the result
    result = []
    
    if thousands > 0:
        result.append(convert_thousands(thousands).strip())
    if hundreds > 0:
        result.append(convert_hundreds(hundreds).strip())
    if remainder > 0 or (number != 0 and thousands == 0 and hundreds == 0):
        result.append(convert_tens_and_ones(remainder))

    return " da ".join(part for part in result if part)

if __name__ == "__main__":
    # Test examples
    # test_numbers = [0, 1, 11, 21, 100, 101, 111, 1000, 1100, 1111, 10000, 11111, 144000]
    test_numbers = [int(l.strip().split('\t')[0]) for l in open('numberdict.tsv', 'r').readlines()]
    for num in test_numbers:
        print(f"{num}\t{number_to_hausa(num)}")
